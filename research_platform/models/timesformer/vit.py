# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# Copyright 2020 Ross Wightman
# Modified Model definition

from collections import OrderedDict
from functools import partial
import math
import warnings
import logging

import torch
from torch import _assert
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch import einsum
from einops import rearrange, reduce, repeat

from .helpers import build_model_with_cfg, named_apply, adapt_input_conv
from .vit_utils import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
from .vit_layers import PositionalEncoding, PatchEmbed, Mlp, DropPath, trunc_normal_, lecun_normal_
from ..build import MODEL_REGISTRY

_logger = logging.getLogger(__name__)

def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic', 'fixed_input_size': True,
        'mean': IMAGENET_INCEPTION_MEAN, 'std': IMAGENET_INCEPTION_STD,
        'first_conv': 'patch_embed.proj', 'classifier': 'head',
        **kwargs
    }

default_cfgs = {
    # patch models (weights from official Google JAX impl)
    'vit_base_patch16_224': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p16_224-80ecf9dd.pth')
}

class DotProduct(nn.Module):
    """ Explicit dot product layer for pretty flops count printing.
    """
    def __init__(self, scale=None):
        super().__init__()
        self.scale = scale
    
    def forward(self, x, y):
        if self.scale is not None:
            x = x * self.scale
        out = x @ y

        return out

    def extra_repr(self) -> str:
        return 'scale={}'.format(
            self.scale
        )

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        # self.attn_drop = nn.Dropout(attn_drop)

        ##
        self.scaled_dot_product = DotProduct(scale=head_dim ** -0.5)
        self.softmax = nn.Softmax(dim=-1)
        self.attn_drop = nn.Dropout(attn_drop)
        self.dot_product = DotProduct()
        ##

        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)

        # attn = (q @ k.transpose(-2, -1)) * self.scale
        # attn = attn.softmax(dim=-1)
        # attn = self.attn_drop(attn)

        # x = (attn @ v).transpose(1, 2).reshape(B, N, C)

        attn = self.scaled_dot_product(q, k.transpose(-2, -1))
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)
        
        x = self.dot_product(attn, v).transpose(1, 2).reshape(B, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)

        return x, attn

class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0.1, act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
        self.temporal_norm1 = norm_layer(dim)
        self.temporal_attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.temporal_fc = nn.Linear(dim, dim)
        
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, T, N, return_attention=False):
        """
        x (Tensor): shape (B, (1 + T N), C)
        T (int): input time length
        N (int): input num patches
        """
        init_cls_tokens, x = torch.split(x, [1, T*N], dim=1)

        # Temporal attention
        xt = rearrange(x, 'b (t n) c -> (b n) t c', t=T, n=N)
        xt, time_attn = self.temporal_attn(self.temporal_norm1(xt))
        xt = self.drop_path(xt)
        xt = rearrange(self.temporal_fc(xt), '(b n) t c -> b (t n) c', t=T, n=N)

        x = x + xt

        # Spatial attention
        cls_token = init_cls_tokens.expand(-1, T, -1) # expand cls_token over time dimension
        cls_token = rearrange(cls_token, 'b t c -> (b t) () c')
        xs = rearrange(x, 'b (t n) c -> (b t) n c', t=T, n=N)

        xs = torch.cat([cls_token, xs], dim=1)
        xs, space_attn = self.attn(self.norm1(xs))

        if return_attention:
            return (time_attn, space_attn)

        xs = self.drop_path(xs)

        cls_token, xs = torch.split(xs, [1, N], dim=1)
        cls_token = reduce(cls_token, '(b t) () c -> b () c', 'mean', t=T) # average cls tkn over time dimension
        xs = rearrange(xs, '(b t) n c -> b (t n) c', t=T, n=N)

        x = torch.cat([init_cls_tokens, x], dim=1) + torch.cat([cls_token, xs], dim=1)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        
        return x

class TimeSformer(nn.Module):
    """ TimeSformer (based on ViT from "pytorch-image-models" by Ross Wightman)
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, time_length=8, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0.1, embed_layer=PatchEmbed, norm_layer=None, act_layer=None, weight_init=''):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            num_classes (int): number of classes for classification head
            embed_dim (int): embedding dimension
            time_length (int): time embedding length
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            embed_layer (nn.Module): patch embedding layer
            norm_layer: (nn.Module): normalization layer
            weight_init: (str): weight init scheme
        """
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_tokens = 1
        self.patch_size = patch_size

        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.patch_embed = embed_layer(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        self.num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dist_token = None # originally defined for distilation in Pytorch vision models; not used here.
        
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + self.num_tokens, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        self.time_embed = nn.Parameter(torch.zeros(1, time_length, embed_dim))
        self.time_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        self.pre_logits = nn.Identity() # originally defined for distilation in Pytorch vision models; not used here

        # Classifier head(s)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        self.temp_head = nn.Linear(self.num_features, time_length)

        self.mlp_q = nn.Linear(self.num_features, self.num_features, bias=True)
        self.mlp_k = nn.Linear(self.num_features, self.num_features, bias=True)
        self.mlp_v = nn.Linear(self.num_features, self.num_features, bias=True)
        self.mlp_proj = nn.Linear(self.num_features, self.num_features)
        self.mlp_norm = norm_layer(self.num_features)
        self.flow_head = nn.Linear(self.num_features, 8+1)

        self.init_weights(weight_init)

        i = 0
        for m in self.blocks.modules():
            m_str = str(m)
            if 'Block' in m_str:
                if i > 0:
                    nn.init.constant_(m.temporal_fc.weight, 0)
                    nn.init.constant_(m.temporal_fc.bias, 0)
                i += 1

    def init_weights(self, mode=''):
        assert mode in ('jax', 'jax_nlhb', 'nlhb', '')
        head_bias = -math.log(self.num_classes) if 'nlhb' in mode else 0.
        trunc_normal_(self.pos_embed, std=.02)

        if self.dist_token is not None:
            trunc_normal_(self.dist_token, std=.02)

        if mode.startswith('jax'):
            # leave cls token as zeros to match jax impl
            named_apply(partial(_init_vit_weights, head_bias=head_bias, jax_impl=True), self)
        else:
            trunc_normal_(self.cls_token, std=.02)
            self.apply(_init_vit_weights)

    def _init_weights(self, m):
        # this fn left here for compat with downstream users
        _init_vit_weights(m)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'time_embed', 'cls_token', 'dist_token'}

    def get_classifier(self):
        if self.dist_token is None:
            return self.head
        else:
            return self.head, self.head_dist

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        if self.num_tokens == 2:
            self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()

    def prepare_tokens(self, x, space_only=False):
        _assert(x.dim() == 5, f"Input dimension size should be 5. Received shape {x.shape}")
        B, T = x.size(0), x.size(1) # assume x.shape: B x T x C x H x W

        x = rearrange(x, 'b t c h w -> (b t) c h w')
        x = self.patch_embed(x)
        N = x.size(1) # assume x.shape: (B T) x N x C'

        cls_token = self.cls_token.expand(B*T, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_token, x), dim=1) # for vectorized addtion of self.pos_embed

        x = x + self.pos_embed
        x = self.pos_drop(x)

        if not space_only:
            # Time Embeddings
            cls_tokens, x = torch.split(x, [1, N], dim=1)
            cls_tokens = cls_tokens[:B] # thorow-away overly expanded cls_tokens in the above lines 
            x = rearrange(x, '(b t) n c -> (b n) t c', t=T, n=N)
            
            # Resizing time embeddings in case they don't match
            if T != self.time_embed.size(1):
                # _logger.warning(f"Input length {T} is different to pre-defined length {self.time_embed.size(1)}.")
                time_embed = rearrange(self.time_embed, '() t c -> () c t') # rearrange to match F.interpolate's input spec.
                new_time_embed = F.interpolate(time_embed, size=(T), mode='nearest')
                new_time_embed = rearrange(new_time_embed, '() c t -> () t c')
                x = x + new_time_embed
            else:
                x = x + self.time_embed
            x = self.time_drop(x)
            x = rearrange(x, '(b n) t c -> b (t n) c', t=T, n=N)
            
            x = torch.cat([cls_tokens, x], dim=1)
        else:
            T = 1

        return x, T, N

    def forward_features(self, x):
        x, T, N = self.prepare_tokens(x)

        for blk in self.blocks:
            x = blk(x, T, N)

        x = self.norm(x)
        x = self.pre_logits(x)
        return x[:, 0], rearrange(x[:,1:], 'b (t n) m -> b t n m',t=T)

    def forward(self, x, head_only=False):
        x, x_t = self.forward_features(x)

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

def _init_vit_weights(module: nn.Module, name: str = '', head_bias: float = 0., jax_impl: bool = False):
    """ ViT weight initialization
    * When called without n, head_bias, jax_impl args it will behave exactly the same
      as my original init for compatibility with prev hparam / downstream use cases (ie DeiT).
    * When called w/ valid n (module name) and jax_impl=True, will (hopefully) match JAX impl
    """
    if isinstance(module, nn.Linear):
        if name.startswith('head'):
            nn.init.zeros_(module.weight)
            nn.init.constant_(module.bias, head_bias)
        elif name.startswith('pre_logits'):
            lecun_normal_(module.weight)
            nn.init.zeros_(module.bias)
        else:
            if jax_impl:
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    if 'mlp' in name:
                        nn.init.normal_(module.bias, std=1e-6)
                    else:
                        nn.init.zeros_(module.bias)
            else:
                trunc_normal_(module.weight, std=.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    elif jax_impl and isinstance(module, nn.Conv2d):
        # NOTE conv was left to pytorch default in my original init
        lecun_normal_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
        nn.init.zeros_(module.bias)
        nn.init.ones_(module.weight)

def resize_pos_embed(posemb, posemb_new, num_tokens=1, gs_new=()):
    # Rescale the grid of position embeddings when loading from state_dict. Adapted from
    # https://github.com/google-research/vision_transformer/blob/00883dd691c63a6830751563748663526e811cee/vit_jax/checkpoint.py#L224
    _logger.info('Resized position embedding: %s to %s', posemb.shape, posemb_new.shape)
    ntok_new = posemb_new.shape[1]
    if num_tokens:
        posemb_tok, posemb_grid = posemb[:, :num_tokens], posemb[0, num_tokens:]
        ntok_new -= num_tokens
    else:
        posemb_tok, posemb_grid = posemb[:, :0], posemb[0]
    gs_old = int(math.sqrt(len(posemb_grid)))
    if not len(gs_new):  # backwards compatibility
        gs_new = [int(math.sqrt(ntok_new))] * 2
    assert len(gs_new) >= 2
    _logger.info('Position embedding grid-size from %s to %s', [gs_old, gs_old], gs_new)
    posemb_grid = posemb_grid.reshape(1, gs_old, gs_old, -1).permute(0, 3, 1, 2)
    posemb_grid = F.interpolate(posemb_grid, size=gs_new, mode='bicubic', align_corners=False)
    posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, gs_new[0] * gs_new[1], -1)
    posemb = torch.cat([posemb_tok, posemb_grid], dim=1)
    return posemb

def resize_time_embed(posemb, posemb_new):
    # Rescale the grid of position embeddings when loading from state_dict. Adapted from
    # https://github.com/google-research/vision_transformer/blob/00883dd691c63a6830751563748663526e811cee/vit_jax/checkpoint.py#L224
    _logger.info('Resized position embedding: %s to %s', posemb.shape, posemb_new.shape)
    ntok_new = posemb_new.shape[1]

    posemb_tok, posemb_grid = posemb[:, :0], posemb[0]
    gs_old = posemb_grid
    gs_new = ntok_new.shape[1]
    _logger.info('Position embedding grid-size from %s to %s', gs_old, gs_new)
    
    posemb_grid = posemb_grid.permute(0, 2, 1)
    posemb_grid = F.interpolate(posemb_grid, size=gs_new, mode='bicubic', align_corners=False)
    posemb_grid = posemb_grid.permute(0, 2, 1)
    posemb = torch.cat([posemb_tok, posemb_grid], dim=1)
    return posemb

def checkpoint_filter_fn(state_dict, model):
    """ convert patch embedding weight from manual patchify + linear proj to conv"""
    out_dict = {}
    if 'model' in state_dict:
        # For deit models
        state_dict = state_dict['model']
    
    for k, v in state_dict.items():
        if 'patch_embed.proj.weight' in k and len(v.shape) < 4:
            # For old models that I trained prior to conv based patchification
            O, I, H, W = model.patch_embed.proj.weight.shape
            v = v.reshape(O, -1, H, W)
        elif k == 'pos_embed' and v.shape != model.pos_embed.shape:
            # To resize pos embedding when using model at different size from pretrained weights
            v = resize_pos_embed(
                v, model.pos_embed, getattr(model, 'num_tokens', 1), model.patch_embed.grid_size)

        elif k == 'time_embed' and v.shape != model.time_embed.shape:
            v = resize_time_embed(
                v, model.time_embed)
        
        elif 'blocks' in k and 'attn' in k:
            new_key = k.replace('attn','temporal_attn')
            if not new_key in state_dict:
                out_dict[new_key] = v

        elif 'blocks' in k and 'norm1' in k:
            new_key = k.replace('norm1','temporal_norm1')
            if not new_key in state_dict:
                out_dict[new_key] = v

        out_dict[k] = v

    return out_dict


def _create_vision_transformer(variant, pretrained=False, default_cfg=None, **kwargs):
    default_cfg = default_cfg or default_cfgs[variant]
    if kwargs.get('features_only', None):
        raise RuntimeError('features_only not implemented for Vision Transformer models.')

    model = build_model_with_cfg(
        TimeSformer, variant, pretrained,
        default_cfg=default_cfg,
        pretrained_filter_fn=checkpoint_filter_fn,
        **kwargs)
    return model

@MODEL_REGISTRY.register()
def timesformer_vit_base_patch16_224(cfg):
    """ ViT-Base (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights fine-tuned from in21k @ 224x224, source https://github.com/google-research/vision_transformer.
    """
    num_classes = cfg.MODEL.NUM_CLASSES
    in_chans = cfg.DATA.INPUT_CHANNEL_NUM[0]
    time_length = cfg.DATA.NUM_FRAMES

    model_kwargs = dict(patch_size=16, embed_dim=768, in_chans=in_chans, num_classes=num_classes, time_length=time_length, depth=12, num_heads=12)
    model = _create_vision_transformer('vit_base_patch16_224', pretrained=True, **model_kwargs)
    return model