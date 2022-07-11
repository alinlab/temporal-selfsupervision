# Modified by Sukmin Yun (sukmin.yun@kaist.ac.kr)
#
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Train a video classification model."""
import pprint
import torch
import pickle
import os.path as osp
from fvcore.nn.precise_bn import get_bn_modules, update_bn_stats
from torch.cuda.amp import autocast, GradScaler
from fvcore.common.file_io import PathManager

import research_platform.utils.losses as losses
import research_platform.utils.optimizer as optim
import research_platform.utils.loader as loader
import research_platform.utils.checkpoint as cu
import research_platform.utils.distributed as du
import research_platform.utils.logging as logging
import research_platform.utils.misc as misc
import research_platform.visualization.tensorboard_vis as tb

from research_platform.models import build_model
from research_platform.utils.meters import TestMeter
import numpy as np

logger = logging.get_logger(__name__)

@torch.no_grad()
def perform_test(test_loader, model, test_meter, cfg, shuffle=False, writer=None):
    """
    For classification:
    Perform mutli-view testing that uniformly samples N clips from a video along
    its temporal axis. For each clip, it takes 3 crops to cover the spatial
    dimension, followed by averaging the softmax scores across all Nx3 views to
    form a video-level prediction. All video predictions are compared to
    ground-truth labels and the final testing performance is logged.
    For detection:
    Perform fully-convolutional testing on the full frames without crop.
    Args:
        test_loader (loader): video testing loader.
        model (model): the pretrained video model to test.
        test_meter (TestMeter): testing meters to log and ensemble the testing
            results.
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
        writer (TensorboardWriter object, optional): TensorboardWriter object
            to writer Tensorboard log.
    """
    # Enable eval mode.
    model.eval()
    test_meter.iter_tic()


    for cur_iter, (inputs, labels, meta) in enumerate(test_loader):
        # Transfer the data to the current GPU device.        
        if cfg.NUM_GPUS:
            if isinstance(inputs, (list,)):
                for i in range(len(inputs)):
                    inputs[i] = inputs[i].cuda(non_blocking=True)
            else:
                inputs = inputs.cuda(non_blocking=True)

            labels = labels.cuda()

        if shuffle:
            B, T, C, H, W = inputs.shape
            shuffled_indices = np.random.permutation(T)
            inputs = inputs[:, shuffled_indices]

        online_batch_size = inputs[0].size(0)
        test_meter.data_toc()

        # Perform the forward pass.
        preds = model(inputs)

        video_index = meta['video_index'].cuda(non_blocking=True)
        # Gather all the predictions across all the devices to perform ensemble.
        if cfg.NUM_GPUS > 1:
            preds, labels, video_index = du.all_gather(
                [preds, labels, video_index]
            )
        if cfg.NUM_GPUS:
            preds = preds.cpu()
            labels = labels.cpu()
            video_index = video_index.cpu()

        test_meter.iter_toc()
        # Update and log stats.
        test_meter.update_stats(
            preds.detach(), labels.detach(), video_index.detach()
        )
        test_meter.log_iter_stats(cur_iter)

        test_meter.iter_tic()

    # Log epoch stats and print the final testing results.
    all_preds = test_meter.video_preds.clone().detach()
    all_labels = test_meter.video_labels
    if cfg.NUM_GPUS:
        all_preds = all_preds.cpu()
        all_labels = all_labels.cpu()
    if writer is not None:
        writer.plot_eval(preds=all_preds, labels=all_labels)

    if cfg.ERM_TEST.SAVE_RESULTS_PATH != "":
        save_path = osp.join(cfg.OUTPUT_DIR, cfg.ERM_TEST.SAVE_RESULTS_PATH)

        with PathManager.open(save_path, "wb") as f:
            pickle.dump([all_labels, all_labels], f)

        logger.info(
            "Successfully saved prediction results to {}".format(save_path)
        )

    test_meter.finalize_metrics()
    return test_meter


def test(cfg):
    """
    Perform multi-view testing on the pretrained video model.
    Args:
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
    """
    # Set up environment.
    du.init_distributed_training(cfg)
    # Set random seed from configs.
    misc.set_random_seed(cfg.RNG_SEED)

    # Setup logging format.
    logging.setup_logging(cfg.OUTPUT_DIR)

    # Print config.
    logger.info("Test with config:")
    logger.info(cfg)

    # Build the video model and print model statistics.
    model = build_model(cfg)
    if du.is_master_proc() and cfg.LOG_MODEL_INFO:
        misc.log_model_info(model, cfg, use_train_input=False)

    cu.load_test_checkpoint(cfg, model)

    # Create video testing loaders.
    test_loader = loader.construct_loader(cfg, "test")
    logger.info("Testing model for {} iterations".format(len(test_loader)))

    assert (
        len(test_loader.dataset)
        % (cfg.ERM_TEST.NUM_ENSEMBLE_VIEWS * cfg.ERM_TEST.NUM_SPATIAL_CROPS)
        == 0
    )
    # Create meters for multi-view testing.
    test_meter = TestMeter(
        len(test_loader.dataset)
        // (cfg.ERM_TEST.NUM_ENSEMBLE_VIEWS * cfg.ERM_TEST.NUM_SPATIAL_CROPS),
        cfg.ERM_TEST.NUM_ENSEMBLE_VIEWS * cfg.ERM_TEST.NUM_SPATIAL_CROPS,
        cfg.MODEL.NUM_CLASSES,
        len(test_loader),
        ensemble_method=cfg.DATA.ENSEMBLE_METHOD,
    )

    # Set up writer for logging to Tensorboard format.
    if cfg.TENSORBOARD.ENABLE and du.is_master_proc(
        cfg.NUM_GPUS * cfg.NUM_SHARDS
    ):
        writer = tb.TensorboardWriter(cfg)
    else:
        writer = None

    # # Perform multi-view test on the entire dataset.
    test_meter = perform_test(test_loader, model, test_meter, cfg, writer)

    if writer is not None:
        writer.close()