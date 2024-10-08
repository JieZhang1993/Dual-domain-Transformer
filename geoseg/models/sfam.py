import torch
import torch.nn as nn
import torch.nn.functional as F


def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob


    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class Mlp(nn.Module):
    """
    MLP as used in Vision Transformer, MLP-Mixer and related networks
    """


    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU,
                 drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)


    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self,
                 dim1,  # 输入token的dim
                 dim2,  # 输入token的dim
                 hidden_dim,
                 num_heads=8,
                 qkv_bias=True,
                 qk_scale=None,
                 attn_drop_ratio=0.,
                 proj_drop_ratio=0.):
        super(Attention, self).__init__()
        num_heads = num_heads or hidden_dim // 32
        self.num_heads = num_heads
        head_dim = hidden_dim // num_heads
        self.hidden_dim = hidden_dim
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim1, hidden_dim, bias=qkv_bias)
        self.kv = nn.Linear(dim2, hidden_dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_ratio)
        self.proj = nn.Linear(hidden_dim, dim2)
        self.proj_drop = nn.Dropout(proj_drop_ratio)


    def forward(self, x):
        # [batch_size, num_patches + 1, total_embed_dim]
        x_high, x_low = x
        B, N, _ = x_high.shape
        C = self.hidden_dim

        # qkv(): -> [batch_size, num_patches , 3 * total_embed_dim]
        # reshape: -> [batch_size, num_patches, 3, num_heads, embed_dim_per_head]
        # permute: -> [3, batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        q = self.q(x_high).reshape(B, N, 1, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1,
                                                                                         4)

        kv = self.kv(x_low).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3,
                                                                                          1,
                                                                                          4)
        # [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        q, k, v = q[0], kv[0], kv[1]  # make torchscript happy (cannot use tensor as tuple)

        # transpose: -> [batch_size, num_heads, embed_dim_per_head, num_patches + 1]
        # @: multiply -> [batch_size, num_heads, num_patches + 1, num_patches + 1]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # @: multiply -> [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        # transpose: -> [batch_size, num_patches + 1, num_heads, embed_dim_per_head]
        # reshape: -> [batch_size, num_patches + 1, total_embed_dim]
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class SFAM(nn.Module):
    def __init__(self,
                 dim1,
                 dim2,
                 hidden_dim,
                 num_heads=None,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop_ratio=0.,
                 attn_drop_ratio=0.,
                 drop_path_ratio=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm):
        super(SFAM, self).__init__()
        self.norm1_1 = norm_layer(dim1)
        self.norm1_2 = norm_layer(dim2)
        self.attn = Attention(dim1, dim2, hidden_dim, num_heads=num_heads, qkv_bias=qkv_bias,
                              qk_scale=qk_scale,
                              attn_drop_ratio=attn_drop_ratio, proj_drop_ratio=drop_ratio)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path_ratio) if drop_path_ratio > 0. else nn.Identity()
        self.norm2 = norm_layer(dim2)
        mlp_hidden_dim = int(dim2 * mlp_ratio)
        self.mlp = Mlp(in_features=dim2, hidden_features=mlp_hidden_dim, act_layer=act_layer,
                       drop=drop_ratio)


    def forward(self, x_high, x_low):
        # [high, low]
        # x_high, x_low = x
        b, c, h, w = x_low.shape
        x_high = F.interpolate(x_high, size=(h, w), mode='bilinear', align_corners=False)

        x_high = self.norm1_1(x_high.flatten(2).permute(0, 2, 1).contiguous())
        x_low = self.norm1_2(x_low.flatten(2).permute(0, 2, 1).contiguous())

        x = x_low + self.drop_path(self.attn([x_high, x_low]))
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        x = x.reshape(b, h, w, c).permute(0, 3, 1, 2).contiguous()
        return x


def _init_vit_weights(m):
    """
    ViT weight initialization
    :param m: module
    """
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=.01)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out")
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight)


if __name__ == '__main__':
    im1 = torch.randn(2, 128, 64, 64)
    im2 = torch.randn(2, 256, 32, 32)
    m = SFAM(128, 256, 128)
    y = m(im1, im2)
    print(y.shape)
