# --------------------------------------------------------
# Copyright (c) 2023 Microsoft
# Licensed under The MIT License
# --------------------------------------------------------
import einops
import numpy as np
from torch import nn, Tensor
import math
import torch
from torch.nn import functional as F
from typing import Optional, Dict, Tuple, Union, Sequence
import typing
from typing import Any, List
from einops.layers.torch import Rearrange
import math
import torch
import torch.fft
import torch.nn as nn
import torch.nn.functional as F
import time


class LayerNorm2D_NCHW(nn.GroupNorm):
    def __init__(
        self,
        num_features: int,
        eps: Optional[float] = 1e-5,
        elementwise_affine: Optional[bool] = True,
        *args,
        **kwargs
    ) -> None:
        super().__init__(
            num_channels=num_features, eps=eps, affine=elementwise_affine, num_groups=1
        )
        self.num_channels = num_features

    def __repr__(self):
        return "{}(num_channels={}, eps={}, affine={})".format(
            self.__class__.__name__, self.num_channels, self.eps, self.affine
        )


def make_divisible(
    v: Union[float, int],
    divisor: Optional[int] = 8,
    min_value: Optional[Union[float, int]] = None,
) -> Union[float, int]:

    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class InvertedResidual(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int,
        expand_ratio: Union[int, float],
        dilation: int = 1,
        skip_connection: Optional[bool] = True,
        *args,
        **kwargs
    ) -> None:
        assert stride in [1, 2]
        hidden_dim = make_divisible(int(round(in_channels * expand_ratio)), 8)

        super().__init__()

        block = nn.Sequential()
        if expand_ratio != 1:
            block.add_module(
                name="exp_1x1",
                module=ConvLayer(
                    c1=in_channels,
                    c2=hidden_dim,
                    k=1,
                    act=True,
                ),
            )

        block.add_module(
            name="conv_3x3",
            module=ConvLayer(
                c1=hidden_dim,
                c2=hidden_dim,
                s=stride,
                k=3,
                g=hidden_dim,
                act=True,
            ),
        )

        block.add_module(
            name="red_1x1",
            module=ConvLayer(
                c1=hidden_dim,
                c2=out_channels,
                k=1,
                act=False,
            ),
        )

        self.block = block
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.exp = expand_ratio
        self.dilation = dilation
        self.stride = stride
        self.use_res_connect = (
            self.stride == 1 and in_channels == out_channels and skip_connection
        )

    def forward(self, x: Tensor, *args, **kwargs) -> Tensor:
        if self.use_res_connect:
            return x + self.block(x)
        else:
            return self.block(x)

    def __repr__(self) -> str:
        return "{}(in_channels={}, out_channels={}, stride={}, exp={}, dilation={}, skip_conn={})".format(
            self.__class__.__name__,
            self.in_channels,
            self.out_channels,
            self.stride,
            self.exp,
            self.dilation,
            self.use_res_connect,
        )


class ConvLayer(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, k // 2, groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.ReLU(inplace=True) if act is True else (
            act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

    def extra_repr(self):
        return f'drop_prob={round(self.drop_prob, 3):0.3f}'


class AFNO2D_channelfirst(nn.Module):
    """
    hidden_size: channel dimension size
    num_blocks: how many blocks to use in the block diagonal weight matrices (higher => less complexity but less parameters)
    sparsity_threshold: lambda for softshrink
    hard_thresholding_fraction: how many frequencies you want to completely mask out (lower => hard_thresholding_fraction^2 less FLOPs)
    input shape [B N C]
    """

    def __init__(self, hidden_size, num_blocks=8, sparsity_threshold=0.01, hard_thresholding_fraction=1,
                 hidden_size_factor=1):
        super().__init__()
        assert hidden_size % num_blocks == 0, f"hidden_size {hidden_size} should be divisble by num_blocks {num_blocks}"

        self.hidden_size = hidden_size
        self.sparsity_threshold = 0.01
        self.num_blocks = num_blocks
        self.block_size = self.hidden_size // self.num_blocks
        self.hard_thresholding_fraction = hard_thresholding_fraction
        self.hidden_size_factor = hidden_size_factor
        self.scale = 0.02

        self.w1 = nn.Parameter(
            self.scale * torch.randn(2, self.num_blocks, self.block_size, self.block_size * self.hidden_size_factor))
        self.b1 = nn.Parameter(self.scale * torch.randn(2, self.num_blocks,
                               self.block_size * self.hidden_size_factor))
        self.w2 = nn.Parameter(
            self.scale * torch.randn(2, self.num_blocks, self.block_size * self.hidden_size_factor, self.block_size))
        self.b2 = nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size))
        self.act = nn.ReLU(inplace=True)
        self.act2 = nn.ReLU(inplace=True)

    @torch.cuda.amp.autocast(enabled=False)
    def forward(self, x, spatial_size=None):
        bias = x

        dtype = x.dtype
        x = x.float()
        B, C, H, W = x.shape
        # x = self.fu(x)

        x = torch.fft.rfft2(x, dim=(2, 3), norm="ortho")
        origin_ffted = x
        x = x.reshape(B, self.num_blocks, self.block_size, x.shape[2], x.shape[3])

        o1_real = self.act(
            torch.einsum('bkihw,kio->bkohw', x.real, self.w1[0]) -
            torch.einsum('bkihw,kio->bkohw', x.imag, self.w1[1]) +
            self.b1[0, :, :, None, None]
        )

        o1_imag = self.act2(
            torch.einsum('bkihw,kio->bkohw', x.imag, self.w1[0]) +
            torch.einsum('bkihw,kio->bkohw', x.real, self.w1[1]) +
            self.b1[1, :, :, None, None]
        )

        o2_real = (
            torch.einsum('bkihw,kio->bkohw', o1_real, self.w2[0]) -
            torch.einsum('bkihw,kio->bkohw', o1_imag, self.w2[1]) +
            self.b2[0, :, :, None, None]
        )

        o2_imag = (
            torch.einsum('bkihw,kio->bkohw', o1_imag, self.w2[0]) +
            torch.einsum('bkihw,kio->bkohw', o1_real, self.w2[1]) +
            self.b2[1, :, :, None, None]
        )

        x = torch.stack([o2_real, o2_imag], dim=-1)
        x = F.softshrink(x, lambd=self.sparsity_threshold)
        x = torch.view_as_complex(x)
        x = x.reshape(B, C, x.shape[3], x.shape[4])

        x = x * origin_ffted
        x = torch.fft.irfft2(x, s=(H, W), dim=(2, 3), norm="ortho")
        x = x.type(dtype)

        return x + bias


class Block(nn.Module):
    def __init__(self, dim, hidden_size, num_blocks, double_skip, mlp_ratio=4., drop_path=0.):
        # input shape [B C H W]
        super().__init__()
        self.norm1 = LayerNorm2D_NCHW(dim)
        self.filter = AFNO2D_channelfirst(hidden_size=hidden_size, num_blocks=num_blocks, sparsity_threshold=0.01,
                                          hard_thresholding_fraction=1, hidden_size_factor=1)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = LayerNorm2D_NCHW(dim)
        self.mlp = InvertedResidual(
            in_channels=dim,
            out_channels=dim,
            stride=1,
            expand_ratio=mlp_ratio,
        )
        self.double_skip = double_skip

    def forward(self, x):
        residual = x
        x = self.norm1(x)
        # x = self.filter(x)
        x = self.mlp(x)

        if self.double_skip:
            x = x + residual
            residual = x

        x = self.norm2(x)
        # x = self.mlp(x)
        x = self.filter(x)
        x = self.drop_path(x)
        x = x + residual
        return x


class AFFBlock(nn.Module):

    def __init__(
            self,
            in_channels: int,
            transformer_dim: int,
            ffn_dim: int,
            n_transformer_blocks: Optional[int] = 1,
            head_dim: Optional[int] = 32,

            attn_dropout: Optional[float] = 0.0,
            dropout: Optional[int] = 0.01,
            ffn_dropout: Optional[int] = 0.0,
            patch_h: Optional[int] = 2,
            patch_w: Optional[int] = 2,
            attn_norm_layer: Optional[str] = "layer_norm_2d",
            conv_ksize: Optional[int] = 3,
            dilation: Optional[int] = 1,
            no_fusion: Optional[bool] = True,
            *args,
            **kwargs
    ) -> None:

        conv_1x1_out = ConvLayer(
            c1=transformer_dim,
            c2=in_channels,
            k=1,
            s=1,
            act=False,
        )
        conv_3x3_out = None
        if not no_fusion:
            conv_3x3_out = ConvLayer(
                c1=2 * in_channels,
                c2=in_channels,
                k=1,  # conv_ksize -> 1
                s=1,
                act=True,
            )
        super().__init__()

        assert transformer_dim % head_dim == 0
        num_heads = transformer_dim // head_dim
        self.enable_coreml_compatible_fn = False

        global_rep = [
            # TODO: to check the double skip
            Block(
                dim=transformer_dim,
                hidden_size=transformer_dim,
                num_blocks=8,
                double_skip=False,
                mlp_ratio=ffn_dim / transformer_dim,
            )
            for _ in range(n_transformer_blocks)
        ]
        global_rep.append(
            LayerNorm2D_NCHW(transformer_dim)
        )
        self.global_rep = nn.Sequential(*global_rep)

        self.conv_proj = conv_1x1_out

        self.fusion = conv_3x3_out

        self.patch_h = patch_h
        self.patch_w = patch_w
        self.patch_area = self.patch_w * self.patch_h

        self.cnn_in_dim = in_channels
        self.cnn_out_dim = transformer_dim
        self.n_heads = num_heads
        self.ffn_dim = ffn_dim
        self.dropout = dropout
        self.attn_dropout = attn_dropout
        self.ffn_dropout = ffn_dropout
        self.dilation = dilation
        self.n_blocks = n_transformer_blocks
        self.conv_ksize = conv_ksize

    def __repr__(self) -> str:
        repr_str = "{}(".format(self.__class__.__name__)

        repr_str += "\n\t Global representations with patch size of {}x{}".format(
            self.patch_h, self.patch_w
        )
        if isinstance(self.global_rep, nn.Sequential):
            for m in self.global_rep:
                repr_str += "\n\t\t {}".format(m)
        else:
            repr_str += "\n\t\t {}".format(self.global_rep)

        if isinstance(self.conv_proj, nn.Sequential):
            for m in self.conv_proj:
                repr_str += "\n\t\t {}".format(m)
        else:
            repr_str += "\n\t\t {}".format(self.conv_proj)

        if self.fusion is not None:
            repr_str += "\n\t Feature fusion"
            if isinstance(self.fusion, nn.Sequential):
                for m in self.fusion:
                    repr_str += "\n\t\t {}".format(m)
            else:
                repr_str += "\n\t\t {}".format(self.fusion)

        repr_str += "\n)"
        return repr_str

    def forward_spatial(self, x: Tensor) -> Tensor:
        res = x

        # fm = self.local_rep(x)
        patches = x

        # b, c, h, w = fm.size()
        # patches = einops.rearrange(fm, 'b c h w -> b (h w) c')

        # learn global representations
        for transformer_layer in self.global_rep:
            patches = transformer_layer(patches)

        # fm = einops.rearrange(patches, 'b (h w) c -> b c h w', h=h, w=w)

        fm = self.conv_proj(patches)

        if self.fusion is not None:
            fm = self.fusion(torch.cat((res, fm), dim=1))
        return fm

    def forward(
            self, x: Union[Tensor, Tuple[Tensor]], *args, **kwargs
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        if isinstance(x, Tuple) and len(x) == 2:
            # for spatio-temporal
            return self.forward_temporal(x=x[0], x_prev=x[1])
        elif isinstance(x, Tensor):
            # For image data
            return self.forward_spatial(x)
        else:
            raise NotImplementedError


if __name__ == '__main__':
    im = torch.randn(2, 64, 32, 32)
    model = AFFBlock(64, 64, 128)
    y = model(im)
    print(y.shape)
