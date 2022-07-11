# Modified by Sukmin Yun (sukmin.yun@kaist.ac.kr)
#
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Train a video classification model."""
import pprint
import numpy as np
import torch
from torch.cuda.amp import autocast, GradScaler
import torch.nn.functional as F
import torch.nn as nn

import research_platform.utils.losses as losses
import research_platform.utils.optimizer as optim
import research_platform.utils.loader as loader
import research_platform.utils.checkpoint as cu
import research_platform.utils.distributed as du
import research_platform.utils.logging as logging
import research_platform.utils.metrics as metrics
import research_platform.utils.misc as misc
import research_platform.visualization.tensorboard_vis as tb

from research_platform.models import build_model
from research_platform.utils.meters import TrainMeter, ValMeter
from einops import rearrange

logger = logging.get_logger(__name__)

def train(cfg):
    """
    Train a video model for many epochs on train set and evaluate it on val set.
    Args:
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
    """
    # Set up environment.
    du.init_distributed_training(cfg)
    # Set random seed from configs.
    misc.set_random_seed(cfg.RNG_SEED)

    # Setup logging format.
    logging.setup_logging(cfg.OUTPUT_DIR, cfg.PRINT_ALL_PROCS)

    # Print config.
    logger.info("Train with config:")
    logger.info(pprint.pformat(cfg))

    # Build the video model and print model statistics.
    model = build_model(cfg)
    if du.is_master_proc() and cfg.LOG_MODEL_INFO:
        misc.log_model_info(model, cfg, use_train_input=True)

    # Construct the optimizer.
    optimizer = optim.construct_optimizer(model, cfg)

    # Load a checkpoint to resume training if applicable.
    if not cfg.ERM_TRAIN.FINETUNE:
      start_epoch = cu.load_train_checkpoint(cfg, model, optimizer)
    else:
      start_epoch = 0
      cu.load_checkpoint(cfg.ERM_TRAIN.CHECKPOINT_FILE_PATH, model)

    # Create the video train and val loaders.
    train_loader = loader.construct_loader(cfg, "train")
    val_loader = loader.construct_loader(cfg, "val")

    train_meter = TrainMeter(len(train_loader), cfg)
    val_meter = ValMeter(len(val_loader), cfg)

    # set up writer for logging to Tensorboard format.
    if cfg.TENSORBOARD.ENABLE and du.is_master_proc(
        cfg.NUM_GPUS * cfg.NUM_SHARDS
    ):
        writer = tb.TensorboardWriter(cfg)
    else:
        writer = None

    # Perform the training loop.
    logger.info("Start epoch: {}".format(start_epoch + 1))
    scaler = GradScaler(enabled=cfg.ERM_TRAIN.FP16)
    for cur_epoch in range(start_epoch, cfg.SOLVER.MAX_EPOCH):
        # Shuffle the dataset.
        loader.shuffle_dataset(train_loader, cur_epoch)

        # Train for one epoch.
        train_epoch(
            train_loader, model, optimizer, scaler, train_meter, cur_epoch, cfg, writer
        )

        is_checkp_epoch = cu.is_checkpoint_epoch(
            cfg,
            cur_epoch,
            None
        )
        is_eval_epoch = misc.is_eval_epoch(
            cfg, cur_epoch, None
        )

        # Save a checkpoint.
        if is_checkp_epoch:
            cu.save_checkpoint(cfg.OUTPUT_DIR, model, optimizer, cur_epoch, cfg)
        # Evaluate the model on validation set.
        if is_eval_epoch:
            eval_epoch(val_loader, model, val_meter, cur_epoch, cfg, writer)

    if writer is not None:
        writer.close()

def train_epoch(
    train_loader, model, optimizer, scaler, train_meter, cur_epoch, cfg, writer=None
):
    """
    Perform the video training for one epoch.
    Args:
        train_loader (loader): video training loader.
        model (model): the video model to train.
        optimizer (optim): the optimizer to perform optimization on the model's
            parameters.
        train_meter (TrainMeter): training meters to log the training performance.
        cur_epoch (int): current epoch of training.
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
        writer (TensorboardWriter, optional): TensorboardWriter object
            to writer Tensorboard log.
    """
    # Enable train mode.
    model.train()
    train_meter.iter_tic()
    data_size = len(train_loader)

    num_iters = cfg.ERM_TRAIN.NUM_ITERS
    logger.info("Gradient accumulation enabled!")
    logger.info(f"cur_global_batch_size: {cfg.ERM_TRAIN.BATCH_SIZE}, target_global_batch_size: {cfg.ERM_TRAIN.BATCH_SIZE * cfg.ERM_TRAIN.NUM_ITERS}")

    for cur_iter, (inputs, labels, meta) in enumerate(train_loader):
        # Transfer the data to the current GPU device.        
        if cfg.NUM_GPUS:
            if isinstance(inputs, (list,)):
                for i in range(len(inputs)):
                    inputs[i] = inputs[i].cuda(non_blocking=True)
            else:
                inputs = inputs.cuda(non_blocking=True)

            labels = labels.cuda()

        online_batch_size = inputs.size(0)

        # Update the learning rate.
        lr = optim.get_epoch_lr(cur_epoch + float(cur_iter) / data_size, cfg)
        optim.set_lr(optimizer, lr)

        train_meter.data_toc()

        loss_fun = losses.get_loss_func(cfg.MODEL.LOSS_FUNC)(reduction="mean")

        if cfg.SOLVER.SMOOTHING > 0.0:
            loss_fun = losses.get_loss_func("label_smoothing_cross_entropy")(
                smoothing=cfg.SOLVER.SMOOTHING)

        def closure():
            debias_loss = 0
            order_loss = 0
            flow_loss = 0
            p_, t_, f_ = model(inputs)

            f_ = rearrange(f_, 'b t n c -> (b t n) c')
            f_label = rearrange(meta['flow'], 'b t n -> (b t n)').cuda()
            flow_loss = loss_fun(f_, f_label)

            B, T = t_.shape[:2]
            t_label = torch.LongTensor(list(range(T))).unsqueeze(0).repeat(B,1).cuda()
            order_loss = loss_fun(t_.view(B*T, -1), t_label.view(-1))

            shuffled_indices = np.random.permutation(inputs.shape[1])
            shuffled_inputs = inputs[:, shuffled_indices]
            sh_preds = model(shuffled_inputs, True)
            debias_loss = -torch.mean(torch.sum(torch.nn.functional.log_softmax(sh_preds, dim=1) 
                * torch.ones(online_batch_size, cfg.MODEL.NUM_CLASSES, device=shuffled_inputs.device) 
                / cfg.MODEL.NUM_CLASSES, dim=1))

            # Compute the loss.
            loss = loss_fun(p_, labels) + order_loss + debias_loss + flow_loss
            return loss, p_


        if num_iters <= 1:
            optimizer.zero_grad()
            with autocast(cfg.ERM_TRAIN.FP16):
                loss, preds = closure()

            # Perform the backward pass.
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            if cur_iter == 0:
                optimizer.zero_grad()
            
            if (cur_iter + 1) % num_iters != 0:
                if cur_iter < num_iters:
                    logger.info(f"{cur_iter + 1}/{data_size}. No Synced forward")
                with model.no_sync():
                    with torch.cuda.amp.autocast(enabled=cfg.ERM_TRAIN.FP16):
                        loss, preds = closure()
                    # no synchronization, accumulate grads
                    if cfg.ERM_TRAIN.FP16:
                        scaler.scale(loss).backward()

                    else:
                        loss.backward()
            
            if (cur_iter + 1) % num_iters == 0:
                if cur_iter < num_iters:
                    logger.info(f"{cur_iter + 1}/{data_size}. Synced forward")
                with torch.cuda.amp.autocast(enabled=cfg.ERM_TRAIN.FP16):
                    loss, preds = closure()
                # synchronize grads
                if cfg.ERM_TRAIN.FP16:
                    scaler.scale(loss).backward()

                else:
                    loss.backward()

                # unscale gradients if mixed precision
                if cfg.ERM_TRAIN.FP16:
                    scaler.unscale_(optimizer)

                # scale gradients so that correct lr@GLOBAL_BATCH_SIZE is applied.
                with torch.no_grad():
                    for name, param in model.named_parameters():
                        if param.grad is None:
                            if cur_iter < num_iters:
                                logger.info(f"Skipping a param {name}")
                        else:
                            # if cur_iter < num_iters:
                            #     logger.info(f"Scaling a param {name} by 1/{num_iters}")
                            param.grad /= num_iters

                if cfg.ERM_TRAIN.FP16:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                
                optimizer.zero_grad()

        top1_err, top5_err = None, None

        # Compute the errors.
        num_topks_correct = metrics.topks_correct(preds, labels, (1, 5))
        top1_err, top5_err = [
            (1.0 - x / preds.size(0)) * 100.0 for x in num_topks_correct
        ]
        # Gather all the predictions across all the devices.
        if cfg.NUM_GPUS > 1:
            loss, top1_err, top5_err = du.all_reduce(
                [loss, top1_err, top5_err]
            )

        # Copy the stats from GPU to CPU (sync point).
        loss, top1_err, top5_err = (
            loss.item(),
            top1_err.item(),
            top5_err.item(),
        )

        # write to tensorboard format if available.
        if writer is not None:
            writer.add_scalars(
                {
                    "Train/loss": loss,
                    "Train/lr": lr,
                    "Train/Top1_err": top1_err,
                    "Train/Top5_err": top5_err,
                },
                global_step=data_size * cur_epoch + cur_iter,
            )

        train_meter.iter_toc()  # measure allreduce for this meter

        # Update and log stats.
        train_meter.update_stats(
            top1_err,
            top5_err,
            loss,
            lr,
            online_batch_size
            * max(
                cfg.NUM_GPUS, 1
            ),  # If running  on CPU (cfg.NUM_GPUS == 1), use 1 to represent 1 CPU.
        )

        train_meter.log_iter_stats(cur_epoch, cur_iter)
        train_meter.iter_tic()

    # Log epoch stats.
    train_meter.log_epoch_stats(cur_epoch)
    train_meter.reset()


@torch.no_grad()
def eval_epoch(val_loader, model, val_meter, cur_epoch, cfg, writer=None):
    """
    Evaluate the model on the val set.
    Args:
        val_loader (loader): data loader to provide validation data.
        model (model): model to evaluate the performance.
        val_meter (ValMeter): meter instance to record and calculate the metrics.
        cur_epoch (int): number of the current epoch of training.
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
        writer (TensorboardWriter, optional): TensorboardWriter object
            to writer Tensorboard log.
    """

    # Evaluation mode enabled. The running stats would not be updated.
    model.eval()
    val_meter.iter_tic()
    
    for cur_iter, (inputs, labels, meta) in enumerate(val_loader):
        if cfg.NUM_GPUS:
            if isinstance(inputs, (list,)):
                for i in range(len(inputs)):
                    inputs[i] = inputs[i].cuda(non_blocking=True)
            else:
                inputs = inputs.cuda(non_blocking=True)

            labels = labels.cuda()

        online_batch_size = inputs[0].size(0)
        val_meter.data_toc()
        
        with autocast(cfg.ERM_TRAIN.FP16):
            preds = model(inputs)
        
        # Compute the errors.
        num_topks_correct = metrics.topks_correct(preds, labels, (1, 5))

        # Combine the errors across the GPUs.
        top1_err, top5_err = [
            (1.0 - x / preds.size(0)) * 100.0 for x in num_topks_correct
        ]
        if cfg.NUM_GPUS > 1:
            top1_err, top5_err = du.all_reduce([top1_err, top5_err])

        # Copy the errors from GPU to CPU (sync point).
        top1_err, top5_err = top1_err.item(), top5_err.item()

        
        # Update and log stats.
        val_meter.update_stats(
            top1_err,
            top5_err,
            online_batch_size
            * max(
                cfg.NUM_GPUS, 1
            ),  # If running  on CPU (cfg.NUM_GPUS == 1), use 1 to represent 1 CPU.
        )
        # write to tensorboard format if available.
        if writer is not None:
            writer.add_scalars(
                {"Val/Top1_err": top1_err, "Val/Top5_err": top5_err},
                global_step=len(val_loader) * cur_epoch + cur_iter,
            )

        val_meter.update_predictions(preds, labels)

        val_meter.iter_toc()
        val_meter.log_iter_stats(cur_epoch, cur_iter)
        val_meter.iter_tic()

    # Log epoch stats.
    val_meter.log_epoch_stats(cur_epoch)
    # write to tensorboard format if available.
    if writer is not None:
        all_preds = [pred.clone().detach() for pred in val_meter.all_preds]
        all_labels = [
            label.clone().detach() for label in val_meter.all_labels
        ]
        if cfg.NUM_GPUS:
            all_preds = [pred.cpu() for pred in all_preds]
            all_labels = [label.cpu() for label in all_labels]
        writer.plot_eval(
            preds=all_preds, labels=all_labels, global_step=cur_epoch
        )

    val_meter.reset()