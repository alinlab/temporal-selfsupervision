#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# Copyright 2020 Ross Wightman
# Modified Model definition

"""Video models."""

from collections import OrderedDict
import math
import torch
import torch.nn as nn
from functools import partial

from .vit_layers import trunc_normal_
from . import vit_helper
from ..build import MODEL_REGISTRY
from einops import rearrange

class MotionFormer(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """

    def __init__(self, cfg):
        super().__init__()
        self.img_size = cfg.DATA.TRAIN_CROP_SIZE
        self.patch_size = cfg.VIT.PATCH_SIZE
        self.in_chans = cfg.VIT.CHANNELS
        self.num_classes = cfg.MODEL.NUM_CLASSES
        self.embed_dim = cfg.VIT.EMBED_DIM
        self.depth = cfg.VIT.DEPTH
        self.num_heads = cfg.VIT.NUM_HEADS
        self.mlp_ratio = cfg.VIT.MLP_RATIO
        self.qkv_bias = cfg.VIT.QKV_BIAS
        self.drop_rate = cfg.VIT.DROP
        self.drop_path_rate = cfg.VIT.DROP_PATH
        self.head_dropout = cfg.VIT.HEAD_DROPOUT
        self.video_input = cfg.VIT.VIDEO_INPUT
        self.temporal_resolution = cfg.VIT.TEMPORAL_RESOLUTION
        self.use_mlp = cfg.VIT.USE_MLP
        self.num_features = self.embed_dim
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.attn_drop_rate = cfg.VIT.ATTN_DROPOUT
        self.head_act = cfg.VIT.HEAD_ACT
        self.cfg = cfg

        if self.cfg.VIT.PATCH_SIZE_TEMP > 1:
            # 3D Patch Embedding
            self.patch_embed_3d = vit_helper.PatchEmbed3D(
                img_size=self.img_size,
                temporal_resolution=self.temporal_resolution,
                patch_size=self.patch_size,
                in_chans=self.in_chans,
                embed_dim=self.embed_dim,
                z_block_size=self.cfg.VIT.PATCH_SIZE_TEMP
            )
            self.patch_embed_3d.proj.weight.data = torch.zeros_like(
                self.patch_embed_3d.proj.weight.data)

        # Patch Embedding
        self.patch_embed = vit_helper.PatchEmbed(
            img_size=224,
            patch_size=self.patch_size,
            in_chans=self.in_chans,
            embed_dim=self.embed_dim
        )

        # Number of patches
        if self.video_input:
            num_patches = self.patch_embed.num_patches * self.temporal_resolution
        else:
            num_patches = self.patch_embed.num_patches
        self.patch_embed_num_patches = self.patch_embed.num_patches
        self.num_patches = num_patches

        # CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        trunc_normal_(self.cls_token, std=.02)

        # Positional embedding
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.patch_embed.num_patches + 1, self.embed_dim))
        self.pos_drop = nn.Dropout(p=cfg.VIT.POS_DROPOUT)
        trunc_normal_(self.pos_embed, std=.02)

        if self.cfg.VIT.POS_EMBED == "joint":
            self.st_embed = nn.Parameter(
                torch.zeros(1, num_patches + 1, self.embed_dim))
            trunc_normal_(self.st_embed, std=.02)
        elif self.cfg.VIT.POS_EMBED == "separate":
            self.temp_embed = nn.Parameter(
                torch.zeros(1, self.temporal_resolution, self.embed_dim))

        # Layer Blocks
        dpr = [x.item() for x in torch.linspace(
            0, self.drop_path_rate, self.depth)]
        if self.cfg.VIT.ATTN_LAYER == "divided":
            self.blocks = nn.ModuleList([
                vit_helper.DividedSpaceTimeBlock(
                    attn_type=cfg.VIT.ATTN_LAYER,
                    dim=self.embed_dim,
                    num_heads=self.num_heads,
                    mlp_ratio=self.mlp_ratio,
                    qkv_bias=self.qkv_bias,
                    drop=self.drop_rate,
                    attn_drop=self.attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                )
                for i in range(self.depth)
            ])
        else:
            self.blocks = nn.ModuleList([
                vit_helper.Block(
                    attn_type=cfg.VIT.ATTN_LAYER,
                    dim=self.embed_dim,
                    num_heads=self.num_heads,
                    mlp_ratio=self.mlp_ratio,
                    qkv_bias=self.qkv_bias,
                    drop=self.drop_rate,
                    attn_drop=self.attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                    use_original_code=self.cfg.VIT.USE_ORIGINAL_TRAJ_ATTN_CODE
                )
                for i in range(self.depth)
            ])
        self.norm = norm_layer(self.embed_dim)

        # MLP head
        if self.use_mlp:
            hidden_dim = self.embed_dim
            if self.head_act == 'tanh':
                print("Using TanH activation in MLP")
                act = nn.Tanh()
            elif self.head_act == 'gelu':
                print("Using GELU activation in MLP")
                act = nn.GELU()
            else:
                print("Using ReLU activation in MLP")
                act = nn.ReLU()
            self.pre_logits = nn.Sequential(OrderedDict([
                ('fc', nn.Linear(self.embed_dim, hidden_dim)),
                ('act', act),
            ]))
        else:
            self.pre_logits = nn.Identity()

        # Classifier Head
        self.head_drop = nn.Dropout(p=self.head_dropout)
        if isinstance(self.num_classes, (list,)) and len(self.num_classes) > 1:
            for a, i in enumerate(range(len(self.num_classes))):
                setattr(self, "head%d" % a, nn.Linear(self.embed_dim, self.num_classes[i]))
        else:
            self.head = (nn.Linear(self.embed_dim, self.num_classes)
                         if self.num_classes > 0 else nn.Identity())

        self.temp_head = nn.Linear(self.embed_dim, self.temporal_resolution)

        self.mlp_q = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        self.mlp_k = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        self.mlp_v = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        self.mlp_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.mlp_norm = norm_layer(self.embed_dim)
        self.flow_head = nn.Linear(self.embed_dim, 8+1)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        if self.cfg.VIT.POS_EMBED == "joint":
            return {'pos_embed', 'cls_token', 'st_embed'}
        else:
            return {'pos_embed', 'cls_token', 'temp_embed'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = (nn.Linear(self.embed_dim, num_classes) if num_classes > 0
                     else nn.Identity())

    def forward_features(self, x):
        # if self.video_input:
        #     x = x[0]

        x = x.permute(0, 2, 1, 3, 4)
        B = x.shape[0]

        # Tokenize input
        if self.cfg.VIT.PATCH_SIZE_TEMP > 1:
            x = self.patch_embed_3d(x)
        else:
            # 2D tokenization
            if self.video_input:
                x = x.permute(0, 2, 1, 3, 4)
                (B, T, C, H, W) = x.shape
                x = x.reshape(B * T, C, H, W)

            x = self.patch_embed(x)

            if self.video_input:
                (B2, T2, D2) = x.shape
                x = x.reshape(B, T * T2, D2)

        # Append CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # Interpolate positinoal embeddings
        if self.cfg.DATA.TRAIN_CROP_SIZE != 224:
            pos_embed = self.pos_embed
            N = pos_embed.shape[1] - 1
            npatch = int((x.size(1) - 1) / self.temporal_resolution)
            class_emb = pos_embed[:, 0]
            pos_embed = pos_embed[:, 1:]
            dim = x.shape[-1]
            pos_embed = torch.nn.functional.interpolate(
                pos_embed.reshape(
                    1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(
                    0, 3, 1, 2),
                scale_factor=math.sqrt(npatch / N),
                mode='bicubic',
            )
            pos_embed = pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
            new_pos_embed = torch.cat((class_emb.unsqueeze(0), pos_embed), dim=1)
        else:
            new_pos_embed = self.pos_embed
            npatch = self.patch_embed_num_patches

        # Add positional embeddings to input
        if self.video_input:
            if self.cfg.VIT.POS_EMBED == "separate":
                cls_embed = self.pos_embed[:, 0, :].unsqueeze(1)
                tile_pos_embed = new_pos_embed[:, 1:, :].repeat(
                    1, self.temporal_resolution, 1)
                tile_temporal_embed = self.temp_embed.repeat_interleave(
                    npatch, 1)
                total_pos_embed = tile_pos_embed + tile_temporal_embed
                total_pos_embed = torch.cat([cls_embed, total_pos_embed], dim=1)
                x = x + total_pos_embed
            elif self.cfg.VIT.POS_EMBED == "joint":
                x = x + self.st_embed
        else:
            # image input
            x = x + new_pos_embed

        # Apply positional dropout
        x = self.pos_drop(x)

        #  Encoding using transformer layers
        for i, blk in enumerate(self.blocks):
            x = blk(
                x,
                seq_len=npatch,
                num_frames=self.temporal_resolution,
                approx=self.cfg.VIT.APPROX_ATTN_TYPE,
                num_landmarks=self.cfg.VIT.APPROX_ATTN_DIM
            )

        x = self.norm(x)
        x = self.pre_logits(x)

        return x[:,0], rearrange(x[:,1:], 'b (t n) m -> b t n m',b=B,t=self.temporal_resolution)

    def forward(self, x, head_only=False):
        x, x_t = self.forward_features(x)
        x = self.head_drop(x)
        if not self.training:
            return self.head(x)
        else:
            if head_only:
                return self.head(x)

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

            return self.head(x), self.temp_head(x_t.mean(2)), self.flow_head(x_predict)

@MODEL_REGISTRY.register()
def motionformer_vit_base_patch16_224(cfg):
    """ ViT-Base (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights fine-tuned from in21k @ 224x224, source https://github.com/google-research/vision_transformer.
    """
    model = MotionFormer(cfg)

    vit_helper.load_pretrained(
        model, cfg=cfg, num_classes=cfg.MODEL.NUM_CLASSES,
        in_chans=cfg.VIT.CHANNELS, filter_fn=vit_helper._conv_filter,
        strict=False
    )
    if hasattr(model, 'st_embed'):
        model.st_embed.data[:, 1:, :] = model.pos_embed.data[:, 1:, :].repeat(
            1, cfg.VIT.TEMPORAL_RESOLUTION, 1)
        model.st_embed.data[:, 0, :] = model.pos_embed.data[:, 0, :]
    if hasattr(model, 'patch_embed_3d'):
        model.patch_embed_3d.proj.weight.data = torch.zeros_like(
            model.patch_embed_3d.proj.weight.data)
        n = math.floor(model.patch_embed_3d.proj.weight.shape[2] / 2)
        model.patch_embed_3d.proj.weight.data[:, :, n, :, :] = model.patch_embed.proj.weight.data
        model.patch_embed_3d.proj.bias.data = model.patch_embed.proj.bias.data

    if cfg.VIT.PATCH_SIZE_TEMP > 1:
        del(model.patch_embed)

    return model
