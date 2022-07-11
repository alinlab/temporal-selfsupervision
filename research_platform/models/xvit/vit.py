#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Video models."""

from functools import partial

import torch
import torch.nn as nn
from torch.nn.init import constant_, normal_
from timm.models import create_model

from . import head_helper
from .temporal_shift import ConsensusModule, make_temporal_shift
from ..build import MODEL_REGISTRY


class GN(nn.GroupNorm):
    def __init__(
        self, num_channels: int, num_groups: int, eps: float, affine: bool
    ) -> None:
        super(GN, self).__init__(
            num_groups, num_channels, eps=eps, affine=affine
        )


class XVIT(nn.Module):
    def __init__(self, cfg):
        super(XVIT, self).__init__()
        self.cfg = cfg
        self.reshape = True

        if not cfg.XVIT.BEFORE_SOFTMAX and cfg.XVIT.CONSENSUS_TYPE != "avg":
            raise ValueError("Only avg consensus can be used after Softmax")

        self.new_length = 1

        self._prepare_base_model(self.cfg.XVIT.BASE_MODEL)

        # feature_dim = self._prepare_classifier(cfg.MODEL.NUM_CLASSES)

        if cfg.XVIT.CONSENSUS_TYPE == "vit":
            self.consensus = head_helper.VitHead(cfg.TEMPORAL_HEAD.HIDDEN_DIM, cfg)
        else:
            self.consensus = ConsensusModule(cfg.XVIT.CONSENSUS_TYPE)
        self.temp_head = nn.Linear(cfg.TEMPORAL_HEAD.HIDDEN_DIM, cfg.DATA.NUM_FRAMES)

        self.mlp_q = nn.Linear(cfg.TEMPORAL_HEAD.HIDDEN_DIM, cfg.TEMPORAL_HEAD.HIDDEN_DIM, bias=True)
        self.mlp_k = nn.Linear(cfg.TEMPORAL_HEAD.HIDDEN_DIM, cfg.TEMPORAL_HEAD.HIDDEN_DIM, bias=True)
        self.mlp_v = nn.Linear(cfg.TEMPORAL_HEAD.HIDDEN_DIM, cfg.TEMPORAL_HEAD.HIDDEN_DIM, bias=True)
        self.mlp_proj = nn.Linear(cfg.TEMPORAL_HEAD.HIDDEN_DIM, cfg.TEMPORAL_HEAD.HIDDEN_DIM)
        self.mlp_norm = partial(nn.LayerNorm, eps=1e-6)(cfg.TEMPORAL_HEAD.HIDDEN_DIM)
        self.flow_head = nn.Linear(cfg.TEMPORAL_HEAD.HIDDEN_DIM, 8+1)

        if not cfg.XVIT.BEFORE_SOFTMAX:
            self.softmax = nn.Softmax()

    def _prepare_classifier(self, num_class):
        feature_dim = getattr(
            self.base_model, self.base_model.last_layer_name
        ).in_features
        if self.cfg.XVIT.DROPOUT_RATE == 0:
            setattr(
                self.base_model,
                self.base_model.last_layer_name,
                nn.Linear(feature_dim, num_class),
            )
            self.new_fc = None
        else:
            setattr(
                self.base_model,
                self.base_model.last_layer_name,
                nn.Dropout(p=self.cfg.XVIT.DROPOUT_RATE),
            )
            #self.new_fc = nn.Linear(feature_dim, num_class)
            self.new_fc = None

        std = 0.001
        if self.new_fc is None:
            normal_(
                getattr(
                    self.base_model, self.base_model.last_layer_name
                ).weight,
                0,
                std,
            )
            constant_(
                getattr(self.base_model, self.base_model.last_layer_name).bias,
                0,
            )
        else:
            if hasattr(self.new_fc, "weight"):
                normal_(self.new_fc.weight, 0, std)
                constant_(self.new_fc.bias, 0)
        return feature_dim

    def _prepare_base_model(self, base_model):
        print("=> base model: {}".format(base_model))
        if "deit" in base_model or "vit" in base_model:

            if self.cfg.XVIT.BACKBONE.NORM_LAYER == "LN":
                norm_layer = partial(nn.LayerNorm, eps=1e-6)
            elif self.cfg.XVIT.BACKBONE.NORM_LAYER == "GN":
                norm_layer = partial(GN, num_groups=32, affine=True, eps=1e-6)
            elif self.cfg.XVIT.BACKBONE.NORM_LAYER == "IN":
                norm_layer = partial(nn.InstanceNorm1d, eps=1e-6)
            elif self.cfg.XVIT.BACKBONE.NORM_LAYER == "BN":
                norm_layer = partial(nn.BatchNorm1d, eps=1e-6)

            self.base_model = create_model(
                self.cfg.XVIT.BASE_MODEL,
                pretrained=self.cfg.XVIT.PRETRAIN,
                img_size=self.cfg.DATA.TEST_CROP_SIZE,
                num_classes=1000,
                drop_rate=0,
                drop_connect_rate=None,  # DEPRECATED, use drop_path
                drop_path_rate=self.cfg.XVIT.BACKBONE.DROP_PATH_RATE,
                attn_drop_rate=self.cfg.XVIT.BACKBONE.DROP_ATTN_RATE,
                drop_block_rate=None,
                norm_layer=norm_layer,
                global_pool=None,
            )

            if self.cfg.XVIT.USE_XVIT:
                make_temporal_shift(
                    self.base_model,
                    self.cfg.XVIT.NUM_SEGMENTS,
                    n_div=self.cfg.XVIT.SHIFT_DIV,
                    locations_list=self.cfg.XVIT.LOCATIONS_LIST,
                )

            self.base_model.last_layer_name = "head"
        else:
            raise ValueError("Unknown base model: {}".format(base_model))

    def train(self, mode=True):
        """
        Override the default train() to freeze the BN parameters
        :return:
        """
        super(XVIT, self).train(mode)

    def forward(self, input, head_only=False, no_reshape=False):
        with torch.set_grad_enabled(True):
            if isinstance(input, list):
                if self.cfg.XVIT.CONSENSUS_TYPE == "vit":
                    if len(input) == 1:
                        input = input[0]
                    positional_index = input[1]
                    input = input[0]
                else:
                    input = input[0]
                    positional_index = None
            if isinstance(input, list):
                input = input[0]
            input = input.permute(0, 2, 1, 3, 4) # original input B T C H W
            B, C, F, W, H = input.size()
            if not no_reshape:
                sample_len = 3 * self.new_length
                input = torch.transpose(input, 1, 2).contiguous()
                input = input.view((-1, sample_len) + input.size()[-2:])

            base_out, x_t = self.base_model.forward_features(input)

            if isinstance(base_out, tuple):
                base_out = base_out[0]

        if (
            self.cfg.XVIT.DROPOUT_RATE > 0
            and not self.cfg.XVIT.CONSENSUS_TYPE == "vit"
        ):
            base_out = self.new_fc(base_out)

        if (
            not self.cfg.XVIT.BEFORE_SOFTMAX
            and not self.cfg.XVIT.CONSENSUS_TYPE == "vit"
        ):
            base_out = self.softmax(base_out)

        if self.cfg.XVIT.CONSENSUS_TYPE == "vit":
            base_out = base_out.view(B, F, -1)
            output, x = self.consensus(base_out)

            if not self.training:
                return output
            else:
                if head_only:
                    return output

                B, T, N, C = x_t.shape
                x_predict = x_t[:,:-1]
                x_target = x_t[:,1:]
 
                q = self.mlp_q(x_target)
                k = self.mlp_k(x_predict)
                attn = (q @ k.transpose(-2, -1)) * (C ** -0.5)
                v = self.mlp_v(x_predict)
                attn = attn.softmax(dim=-1)
 
                x_predict = self.mlp_proj(attn @ v)
                x_predict = self.mlp_norm(x_predict)

                return output, self.temp_head(x_t.mean(2)), self.flow_head(x_predict)

            if not self.training:
                return output
            else:
                return output, self.temp_head(base_out)

        # return base_out
        if self.reshape:
            base_out = base_out.view(
                (-1, self.cfg.XVIT.NUM_SEGMENTS) + base_out.size()[1:]
            )
            output = self.consensus(base_out)
            return output.squeeze(1)


@MODEL_REGISTRY.register()
def xvit_vit_base_patch16_224(cfg):
    """ ViT-Base (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights fine-tuned from in21k @ 224x224, source https://github.com/google-research/vision_transformer.
    """

    assert cfg.XVIT.BASE_MODEL == 'vit_base_patch16_224'

    model = XVIT(cfg)
    return model
