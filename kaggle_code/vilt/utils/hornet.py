import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath
import torch.fft


class GlobalLocalFilter(nn.Module):
    def __init__(self, dim, h=14, w=8):
        super().__init__()
        self.dw = nn.Conv2d(dim // 2+1, dim // 2+1, kernel_size=3, padding=1, bias=False, groups=dim // 2+1)
        self.complex_weight = nn.Parameter(torch.randn(dim // 2, h, w, 2, dtype=torch.float32) * 0.02)  # 最后一个维度表示实部和虚部
        trunc_normal_(self.complex_weight, std=.02)  # 截断正太分布 将正态分布的变量范围限制在3sigma内
        self.pre_norm = LayerNorm(dim, eps=1e-6, data_format='channels_first')
        self.post_norm = LayerNorm(dim, eps=1e-6, data_format='channels_first')

    def forward(self, x):
        x = self.pre_norm(x)
        x1, x2 = torch.chunk(x, 2, dim=1)  # 将通道C平分成两份
        x1 = self.dw(x1)

        x2 = x2.to(torch.float32)
        B, C, a, b = x2.shape
        x2 = torch.fft.rfft2(x2, dim=(2, 3), norm='ortho')  # 对最后两个维度进行2维fft变换 只保留实部(如果是fft2则shape相同但数据类型是complex而不是实数)

        weight = self.complex_weight
        if not weight.shape[1:3] == x2.shape[2:4]:
            weight = F.interpolate(weight.permute(3, 0, 1, 2), size=x2.shape[2:4], mode='bilinear',
                                   align_corners=True).permute(1, 2, 3, 0)  # 上采样至1,2维度(h,w)与x2的2,3维度(a,b)相同

        weight = torch.view_as_complex(weight.contiguous())  # 将实部和虚部转成复数格式 (dim//2, h, w)

        x2 = x2 * weight
        x2 = torch.fft.irfft2(x2, s=(a, b), dim=(2, 3), norm='ortho')  # 对最后两个维度进行2维反fft变换

        x = torch.cat([x1[:, :-1].unsqueeze(2), x2.unsqueeze(2)], dim=2).reshape(B, 2 * C, a, b)
        x = torch.cat([x, x1[:, -1].unsqueeze(1)], dim=1)
        x = self.post_norm(x)
        return x


class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class Gnconv(nn.Module):
    def __init__(self, dim, order=5, gflayer=None, h=14, w=8, s=1.0, in_dim=1):
        super().__init__()
        self.order = order
        self.scale = s
        self.dims = [dim // 2 ** i for i in range(order)]
        self.gnconv_in = nn.Conv2d(in_dim, dim, 1)
        self.proj_in = nn.Conv2d(dim, dim * 2, 1)

        if gflayer is None:
            self.dwconv = self.get_dwconv(sum(self.dims), 7, True)
        else:
            self.dwconv = gflayer(sum(self.dims), h=h, w=w)

        self.pws = nn.ModuleList([
            nn.Conv2d(self.dims[i], self.dims[i+1], 1) for i in range(order - 1)
        ])

    def get_dwconv(self, dim, kernel, bias):  # 不变性卷积
        return nn.Conv2d(dim, dim, kernel_size=kernel, padding=(kernel - 1) // 2, bias=bias, groups=dim)

    def forward(self, x):
        x = self.gnconv_in(x)
        x = self.proj_in(x)

        pwa, abc = torch.split(x, [self.dims[-1], sum(self.dims)], dim=1)
        dw_abc = self.dwconv(abc) * self.scale
        dw_list = torch.split(dw_abc, self.dims, dim=1)
        x = dw_list[0]
        for i in range(self.order - 1):
            x = self.pws[i](x) * dw_list[i + 1]
        x = pwa * x

        return x


class MatchingAttention(nn.Module):

    def __init__(self, batch_size, seq_length, dim=64, order=7, use_gflayer=False, h=32, s=1.0):
        super(MatchingAttention, self).__init__()
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.h = h
        if use_gflayer:
            self.transform = Gnconv(dim=dim, order=order, gflayer=GlobalLocalFilter, h=h, w=seq_length, s=s)
        else:
            self.transform = Gnconv(dim=dim, order=order, gflayer=None, h=h, w=seq_length, s=s)

    def forward(self, features):
        features = torch.reshape(features, (self.batch_size, 1, 1, self.seq_length)). \
            repeat(1, 1, self.h, 1)  # (batch, 1, 32, seqlen)
        att_feats = self.transform(features)  # (batch, 1, 32, seqlen)
        att_feats = torch.mean(att_feats, dim=2).view(-1, 1)  # (batch*seqlen, 1)

        return att_feats
