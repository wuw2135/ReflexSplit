import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import to_2tuple, trunc_normal_
import math
from collections import OrderedDict

def lambda_init_fn(depth):
    return 0.8 - 0.6 * math.exp(-0.3 * depth)

class Conv1x1Fp32(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=1)

    def forward(self, x):
        dtype = x.dtype
        with torch.cuda.amp.autocast(enabled=False):
            y = self.conv(x.float())
        return y.to(dtype)

class LayerNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps
        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_tensors
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)
        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return gx, (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0), grad_output.sum(dim=3).sum(dim=2).sum(dim=0), None

class LayerNorm2d(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)

class CABlock(nn.Module):
    def __init__(self, channels):
        super(CABlock, self).__init__()
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels, 1)
        )

    def forward(self, x):
        return x * self.ca(x)

class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2

class SinBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.block1 = nn.Sequential(
            LayerNorm2d(c),
            nn.Conv2d(c, c * 2, 1),
            nn.Conv2d(c * 2, c * 2, 3, padding=1, groups=c * 2),
            SimpleGate(),
            CABlock(c),
            nn.Conv2d(c, c, 1)
        )

        self.block2 = nn.Sequential(
            LayerNorm2d(c),
            nn.Conv2d(c, c * 2, 1),
            SimpleGate(),
            nn.Conv2d(c, c, 1)
        )

        self.a = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.b = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):
        x = self.block1(inp)
        x_skip = inp + x * self.a
        x = self.block2(x_skip)
        out = x_skip + x * self.b
        return out

class DualStreamGate(nn.Module):
    def forward(self, x, y):
        x1, x2 = x.chunk(2, dim=1)
        y1, y2 = y.chunk(2, dim=1)
        return x1 * y2, y1 * x2

class DualStreamSeq(nn.Sequential):
    def forward(self, x, y=None):
        y = y if y is not None else x
        for module in self:
            x, y = module(x, y)
        return x, y

class DualStreamBlock(nn.Module):
    def __init__(self, *args):
        super(DualStreamBlock, self).__init__()
        self.seq = nn.Sequential()
        if len(args) == 1 and isinstance(args[0], OrderedDict):
            for key, module in args[0].items():
                self.seq.add_module(key, module)
        else:
            for idx, module in enumerate(args):
                self.seq.add_module(str(idx), module)

    def forward(self, x, y):
        return self.seq(x), self.seq(y)

def window_partition(x, window_size):
    if not isinstance(window_size, tuple):
        window_size = (window_size, window_size)
    B, H, W, C = x.shape
    x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)
    return windows

def window_reverse(windows, window_size, H, W):
    if not isinstance(window_size, tuple):
        window_size = (window_size, window_size)
    B = int(windows.shape[0] / (H * W / window_size[0] / window_size[1]))
    x = windows.view(B, H // window_size[0], W // window_size[1], window_size[0], window_size[1], -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x



class DifferentialDualStreamAttention(nn.Module):
    """融合 Differential 機制的雙流雙維度注意力"""
    def __init__(self, dim, window_size, num_heads, depth=1, qkv_bias=True, qk_scale=None, 
                 attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = qk_scale or self.head_dim ** -0.5

        # Differential parameters
        self.lambda_init = lambda_init_fn(depth)
        self.lambda_sa = nn.Parameter(torch.tensor(0.1))
        self.lambda_ca = nn.Parameter(torch.tensor(0.1))

        # Self-Attention 分支
        self.sa_qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        
        # ===== 新增：跨流交互投影 (SA) =====
        self.sa_cross_trans_proj = nn.Linear(dim, dim, bias=False)
        self.sa_cross_refl_proj = nn.Linear(dim, dim, bias=False)
        self.sa_enhance_weight = nn.Parameter(torch.tensor(0.1))      
        #self.sa_enhance_weight = nn.Parameter(torch.tensor(0.2), requires_grad=False)

        
        # Cross-Attention 分支
        self.ca_q = nn.Linear(dim, dim, bias=qkv_bias)
        self.ca_kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        
        # ===== 新增：跨流交互投影 (CA) =====
        self.ca_cross_trans_proj = nn.Linear(dim, dim, bias=False)
        self.ca_cross_refl_proj = nn.Linear(dim, dim, bias=False)
        self.ca_enhance_weight = nn.Parameter(torch.tensor(0.1))
        #self.ca_enhance_weight = nn.Parameter(torch.tensor(0.2), requires_grad=False)


        # 相對位置編碼
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))

        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_sa = nn.Linear(dim, dim)
        self.proj_ca = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward_sa(self, x_concat):
        """
        Self-Attention: 在 batch 維度 concat，加入跨流交互
        x_concat: [2*B, N, C] - torch.cat([x_windows, y_windows], dim=0)
        返回: [2*B, N, C]
        """
        B2, N, C = x_concat.shape
        B = B2 // 2
        
        # 分離 trans 和 refl
        x_trans = x_concat[:B]
        x_refl = x_concat[B:]
        
        # ===== 跨流特徵增強 =====
        trans_enhanced = x_trans + self.sa_enhance_weight * self.sa_cross_refl_proj(x_refl)
        refl_enhanced = x_refl + self.sa_enhance_weight * self.sa_cross_trans_proj(x_trans)
        
        # 計算各自的 QKV（使用增強後的特徵）
        qkv_trans = self.sa_qkv(trans_enhanced).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        qkv_refl = self.sa_qkv(refl_enhanced).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        
        q_trans, k_trans, v_trans = qkv_trans[0], qkv_trans[1], qkv_trans[2]
        q_refl, k_refl, v_refl = qkv_refl[0], qkv_refl[1], qkv_refl[2]

        # 計算注意力
        attn_trans = (q_trans * self.scale) @ k_trans.transpose(-2, -1)
        attn_refl = (q_refl * self.scale) @ k_refl.transpose(-2, -1)

        # 添加位置編碼
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(N, N, -1).permute(2, 0, 1).contiguous()
        
        attn_trans = attn_trans + relative_position_bias.unsqueeze(0)
        attn_refl = attn_refl + relative_position_bias.unsqueeze(0)

        attn_trans = self.softmax(attn_trans)
        attn_refl = self.softmax(attn_refl)

        # Differential Attention
        #lambda_sa = torch.sigmoid(self.lambda_sa)
        lambda_sa = torch.clamp(torch.sigmoid(self.lambda_sa), 0.01, 0.99)

        diff_attn_trans = attn_trans - lambda_sa * attn_refl
        diff_attn_refl = attn_refl - lambda_sa * attn_trans

        diff_attn_trans = self.attn_drop(diff_attn_trans)
        diff_attn_refl = self.attn_drop(diff_attn_refl)

        out_trans = (diff_attn_trans @ v_trans).transpose(1, 2).reshape(B, N, C)
        out_refl = (diff_attn_refl @ v_refl).transpose(1, 2).reshape(B, N, C)

        out_trans = self.proj_sa(out_trans)
        out_refl = self.proj_sa(out_refl)
        out_trans = self.proj_drop(out_trans)
        out_refl = self.proj_drop(out_refl)

        # 重新 concat 回去
        return torch.cat([out_trans, out_refl], dim=0)

    def forward_ca(self, x_concat):
        """
        Cross-Attention: 在 sequence 維度 concat，加入跨流交互
        x_concat: [B, 2*N, C] - torch.cat([x_windows, y_windows], dim=-2)
        返回: [B, 2*N, C]
        """
        B, N2, C = x_concat.shape
        N = N2 // 2
        
        # 分離 trans 和 refl
        x_trans = x_concat[:, :N, :]
        x_refl = x_concat[:, N:, :]
        
        # ===== 跨流特徵增強 =====
        trans_enhanced = x_trans + self.ca_enhance_weight * self.ca_cross_refl_proj(x_refl)
        refl_enhanced = x_refl + self.ca_enhance_weight * self.ca_cross_trans_proj(x_trans)
        
        # Query 從增強後的特徵計算
        q_trans = self.ca_q(trans_enhanced).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        q_refl = self.ca_q(refl_enhanced).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        
        # KV 從增強後的 concat 特徵計算
        x_concat_enhanced = torch.cat([trans_enhanced, refl_enhanced], dim=-2)
        kv = self.ca_kv(x_concat_enhanced).reshape(B, N2, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        
        k_trans, k_refl = k[:, :, :N, :], k[:, :, N:, :]
        v_trans, v_refl = v[:, :, :N, :], v[:, :, N:, :]

        # 計算注意力
        attn_trans = (q_trans * self.scale) @ k_trans.transpose(-2, -1)
        attn_refl = (q_refl * self.scale) @ k_refl.transpose(-2, -1)

        # 添加位置編碼
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(N, N, -1).permute(2, 0, 1).contiguous()
        
        attn_trans = attn_trans + relative_position_bias.unsqueeze(0)
        attn_refl = attn_refl + relative_position_bias.unsqueeze(0)

        attn_trans = self.softmax(attn_trans)
        attn_refl = self.softmax(attn_refl)

        # Differential Cross-Attention
        lambda_ca = torch.clamp(torch.sigmoid(self.lambda_ca), 0.01, 0.99)
        diff_attn_trans = attn_trans - lambda_ca * attn_refl
        diff_attn_refl = attn_refl - lambda_ca * attn_trans

        diff_attn_trans = self.attn_drop(diff_attn_trans)
        diff_attn_refl = self.attn_drop(diff_attn_refl)

        out_trans = (diff_attn_trans @ v_trans).transpose(1, 2).reshape(B, N, C)
        out_refl = (diff_attn_refl @ v_refl).transpose(1, 2).reshape(B, N, C)

        out_trans = self.proj_ca(out_trans)
        out_refl = self.proj_ca(out_refl)
        out_trans = self.proj_drop(out_trans)
        out_refl = self.proj_drop(out_refl)

        # 重新 concat 回去
        return torch.cat([out_trans, out_refl], dim=-2)

class DifferentialDualAttentionInteractiveBlock(nn.Module):
    def __init__(self, dim, input_resolution, num_heads, window_size=12, depth=1, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        
        if min(self.input_resolution) <= self.window_size:
            self.window_size = min(self.input_resolution)
        
        self.norm1 = DualStreamBlock(norm_layer(dim))
        
        # Differential 注意力
        self.diff_attention = DifferentialDualStreamAttention(
            dim, to_2tuple(self.window_size), num_heads, depth
        )
        
        self.feedforward = DualStreamSeq(
            DualStreamBlock(
                LayerNorm2d(dim),
                nn.Conv2d(dim, dim * 2, 1),
                nn.Conv2d(dim * 2, dim * 2, 3, padding=1, groups=dim * 2)
            ),
            DualStreamGate(),
            DualStreamBlock(CABlock(dim)),
            DualStreamBlock(nn.Conv2d(dim, dim, 1))
        )
        
        self.a = nn.Parameter(torch.zeros((1, 1, dim)), requires_grad=True)
        self.b = nn.Parameter(torch.zeros((1, dim, 1, 1)), requires_grad=True)
    
    def forward(self, x, y):
        B, C, H, W = x.shape
        
        x_seq = x.permute(0, 2, 3, 1).contiguous().view(B, H * W, C)
        y_seq = y.permute(0, 2, 3, 1).contiguous().view(B, H * W, C)
        
        x_skip, y_skip = x_seq, y_seq
        x_norm, y_norm = self.norm1(x_seq, y_seq)
        
        x_win = x_norm.view(B, H, W, C)
        y_win = y_norm.view(B, H, W, C)
        
        if not self.training:
            pad_l = pad_t = 0
            pad_r = (self.window_size - W % self.window_size) % self.window_size
            pad_b = (self.window_size - H % self.window_size) % self.window_size
            x_win = F.pad(x_win, (0, 0, pad_l, pad_r, pad_t, pad_b))
            y_win = F.pad(y_win, (0, 0, pad_l, pad_r, pad_t, pad_b))
            _, Hp, Wp, _ = x_win.shape
        else:
            Hp, Wp = H, W
            pad_r, pad_b = 0, 0

        # 窗口分割
        x_windows = window_partition(x_win, self.window_size).view(-1, self.window_size * self.window_size, C)
        y_windows = window_partition(y_win, self.window_size).view(-1, self.window_size * self.window_size, C)
        
        # ===== 保持原始的 concat + chunk 模式 =====
        # Self-Attention: batch 維度 concat
        xx_windows, yy_windows = self.diff_attention.forward_sa(
            torch.cat([x_windows, y_windows], dim=0)
        ).chunk(2, dim=0)
        
        # Cross-Attention: sequence 維度 concat
        xy_windows, yx_windows = self.diff_attention.forward_ca(
            torch.cat([x_windows, y_windows], dim=-2)
        ).chunk(2, dim=-2)
    
        # 加權融合
        x_windows = (xx_windows + xy_windows).view(-1, self.window_size, self.window_size, C)
        y_windows = (yy_windows + yx_windows).view(-1, self.window_size, self.window_size, C)
        
        # 窗口反轉
        x = window_reverse(x_windows, self.window_size, Hp, Wp)
        y = window_reverse(y_windows, self.window_size, Hp, Wp)
        
        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()
            y = y[:, :H, :W, :].contiguous()
        
        # 殘差連接
        x = x_skip + x.view(B, H * W, C) * self.a
        y = y_skip + y.view(B, H * W, C) * self.a
        
        # 前饋網路
        x_skip = x.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        y_skip = y.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        
        x, y = self.feedforward(x_skip, y_skip)
        x, y = x_skip + x * self.b, y_skip + y * self.b
        
        return x, y

class MuGIBlock(nn.Module):
    def __init__(self, c, shared_b=True):
        super().__init__()
        self.block1 = DualStreamSeq(
            DualStreamBlock(
                LayerNorm2d(c),
                nn.Conv2d(c, c * 2, 1),
                nn.Conv2d(c * 2, c * 2, 3, padding=1, groups=c * 2)
            ),
            DualStreamGate(),
            DualStreamBlock(CABlock(c)),
            DualStreamBlock(nn.Conv2d(c, c, 1))
        )

        self.a_l = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.a_r = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

        self.block2 = DualStreamSeq(
            DualStreamBlock(
                LayerNorm2d(c),
                nn.Conv2d(c, c * 2, 1)
            ),
            DualStreamGate(),
            DualStreamBlock(
                nn.Conv2d(c, c, 1)
            )

        )

        self.shared_b = shared_b
        if shared_b:
            self.b = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        else:
            self.b_l = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
            self.b_r = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp_l, inp_r):
        x, y = self.block1(inp_l, inp_r)
        x_skip, y_skip = inp_l + x * self.a_l, inp_r + y * self.a_r
        x, y = self.block2(x_skip, y_skip)
        if self.shared_b:
            out_l, out_r = x_skip + x * self.b, y_skip + y * self.b
        else:
            out_l, out_r = x_skip + x * self.b_l, y_skip + y * self.b_r
        return out_l, out_r

    
class LocalFeatureExtractor(nn.Module):
    """使用 MuGI 的局部特徵提取器"""
    def __init__(self, dims, enc_blk_nums=[]):
        super(LocalFeatureExtractor, self).__init__()
        self.dims = dims
        
        c = dims
        self.stem = DualStreamBlock(nn.Conv2d(3, c, 3, padding=1))
        self.block1 = DualStreamSeq(
            *[MuGIBlock(c) for _ in range(enc_blk_nums[0])],
            DualStreamBlock(nn.Conv2d(c, c * 2, 2, 2))
        )
        c *= 2
        self.block2 = DualStreamSeq(
            *[MuGIBlock(c) for _ in range(enc_blk_nums[1])],
            DualStreamBlock(nn.Conv2d(c, c * 2, 2, 2))
        )
        c *= 2
        self.block3 = DualStreamSeq(
            *[MuGIBlock(c) for _ in range(enc_blk_nums[2])],
            DualStreamBlock(nn.Conv2d(c, c * 2, 2, 2))
        )
        c *= 2
        self.block4 = DualStreamSeq(
            *[MuGIBlock(c) for _ in range(enc_blk_nums[3])],
            DualStreamBlock(nn.Conv2d(c, c * 2, 2, 2))
        )
        c *= 2
        self.block5 = DualStreamSeq(
            *[MuGIBlock(c) for _ in range(enc_blk_nums[4])],
            DualStreamBlock(nn.Conv2d(c, c * 2, 2, 2))
        )
    
    def forward(self, x):
        x0, y0 = self.stem(x, x)
        x1, y1 = self.block1(x0, y0)
        x2, y2 = self.block2(x1, y1)
        x3, y3 = self.block3(x2, y2)
        x4, y4 = self.block4(x3, y3)
        x5, y5 = self.block5(x4, y4)
        return (x0, y0), (x1, y1), (x2, y2), (x3, y3), (x4, y4), (x5, y5)

class GrayWorldRetinex(nn.Module):
    def __init__(self, eps=1e-6):
        super(GrayWorldRetinex, self).__init__()
        self.eps = eps

    def forward(self, x):
        B, C, H, W = x.shape
        mean = x.mean(dim=(2, 3), keepdim=True)  # [B, C, 1, 1]
        gray_mean = mean.mean(dim=1, keepdim=True)  # [B, 1, 1, 1]
        gain = gray_mean / (mean + self.eps)
        x = x * gain  # white balance
        x_log = torch.log(x + self.eps)
        x_log = x_log - x_log.mean(dim=(2, 3), keepdim=True)
        x_out = torch.exp(x_log)
        x_min = x_out.amin(dim=(-2, -1), keepdim=True)
        x_max = x_out.amax(dim=(-2, -1), keepdim=True)
        x_out = (x_out - x_min) / (x_max - x_min + self.eps)
        return x_out


def _rgb2luma_tensor(x):
    """x: [B,3,H,W]"""
    return 0.2126*x[:,0:1,:,:] + 0.7152*x[:,1:2,:,:] + 0.0722*x[:,2:3,:,:]



class MultiScaleFusion(nn.Module):
    """多尺度特徵融合模組"""
    def __init__(self, channels):
        super().__init__()
        self.conv_current = nn.Conv2d(channels, channels, 1)
        self.conv_prev = nn.Conv2d(channels, channels, 1)
        self.weight = nn.Parameter(torch.ones(2))
        
    def forward(self, x_current, x_prev):
        """
        x_current: 當前層特徵 [B, C, H, W]
        x_prev: 上一層特徵 [B, C, H, W]
        """
        # 確保尺寸一致
        if x_prev.shape[-2:] != x_current.shape[-2:]:
            x_prev = F.interpolate(x_prev, size=x_current.shape[-2:], 
                                  mode='bilinear', align_corners=False)
        
        # 計算融合權重
        w = F.softmax(self.weight, dim=0)
        
        # 加權融合
        out = w[0] * self.conv_current(x_current) + \
              w[1] * self.conv_prev(x_prev)
        
        return out


class ReflexSplit(nn.Module):
    """融入 Differential Transformer 的反射移除模型"""
    def __init__(self, args, input_resolution=(384, 384), window_size=12, 
                 enc_blk_nums=[12, 8, 4, 2, 2], dec_blk_nums=[2, 2, 2, 2, 2], 
                 use_white_balance=False, use_axs=False):
        super().__init__()
        
        # 保持原有的先驗網路
        from models.arch.swin_det import swin_large_384_det
        self.swin_prior = swin_large_384_det(args.backbone_weight_path)
        self.swin_prior.eval()
        self.swin_prior.requires_grad_(False)
        
        # 使用 Differential MuGI 的局部特徵提取器
        self.conv_prior = LocalFeatureExtractor(48, [2, 2, 2, 2, 2])
        
        self.input_resolution = input_resolution
        self.window_size = window_size
        H, W = input_resolution
        
        # 使用 Differential 注意力交互塊
        self.aib5 = DualStreamSeq(
            DualStreamBlock(nn.PixelShuffle(2)),
            DifferentialDualAttentionInteractiveBlock(384, (H // 16, W // 16), 8, window_size, depth=5),
            DualStreamBlock(nn.Conv2d(in_channels=384, out_channels=768, kernel_size=1)),
        )
        
        self.aib4 = DualStreamSeq(
            DifferentialDualAttentionInteractiveBlock(768, (H // 16, W // 16), 8, window_size, depth=4)
        )
        
        self.lib4 = DualStreamSeq(
            *[DifferentialDualAttentionInteractiveBlock(768, (H // 16, W // 16), 8, window_size, depth=4)
              for _ in range(enc_blk_nums[0])],
            DualStreamBlock(nn.PixelShuffle(2)),
            *[MuGIBlock(192) for _ in range(dec_blk_nums[0])],
            DualStreamBlock(Conv1x1Fp32(192, 384))
        )
        
        self.aib3 = DualStreamSeq(
            DifferentialDualAttentionInteractiveBlock(384, (H // 8, W // 8), 8, window_size, depth=3),
        )
        
        self.lib3 = DualStreamSeq(
            *[DifferentialDualAttentionInteractiveBlock(384, (H // 8, W // 8), 8, window_size, depth=3)
              for _ in range(enc_blk_nums[1])],
            DualStreamBlock(nn.PixelShuffle(2)),
            *[MuGIBlock(96) for _ in range(dec_blk_nums[1])],
            DualStreamBlock(Conv1x1Fp32(96, 192))
        )
        
        self.aib2 = DualStreamSeq(
            DifferentialDualAttentionInteractiveBlock(192, (H // 4, W // 4), 4, window_size, depth=2),
        )
        
        self.lib2 = DualStreamSeq(
            *[DifferentialDualAttentionInteractiveBlock(192, (H // 4, W // 4), 4, window_size, depth=2)
              for _ in range(enc_blk_nums[2])],
            DualStreamBlock(nn.PixelShuffle(2)),
            *[MuGIBlock(48) for _ in range(dec_blk_nums[2])],
            DualStreamBlock(Conv1x1Fp32(48, 96))
        )
        
        self.lib1 = DualStreamSeq(
            *[DifferentialDualAttentionInteractiveBlock(96, (H // 2, W // 2), 2, window_size, depth=1)
              for _ in range(enc_blk_nums[3])],
            DualStreamBlock(nn.PixelShuffle(2)),
            *[MuGIBlock(24) for _ in range(dec_blk_nums[3])],
            DualStreamBlock(Conv1x1Fp32(24, 48))
        )
        
        self.lib0 = DualStreamSeq(
            *[DifferentialDualAttentionInteractiveBlock(48, (H, W), 2, window_size, depth=0)
              for _ in range(enc_blk_nums[4])],
            *[MuGIBlock(48) for _ in range(dec_blk_nums[4])],
        )
        
        # ===== 新增：Multi-Scale Fusion 模組 =====
        self.fusion3 = MultiScaleFusion(384)
        self.fusion2 = MultiScaleFusion(192)
        self.fusion1 = MultiScaleFusion(96)
        
        # 輸出層
        self.out = DualStreamBlock(nn.Conv2d(in_channels=48, out_channels=3, kernel_size=3, padding=1))
        self.lrm = nn.Sequential(
            SinBlock(48),
            nn.Conv2d(48, 3, 3, padding=1),
            nn.Tanh()
        )
    
    def train(self, mode=True):
        super().train(mode)
    
    def forward(self, data):
        if isinstance(data, tuple):
            inp, fn = data
        else:
            inp = data
            fn = None
            
        B, _, H, W = inp.shape

        # 獲取先驗特徵
        sp2, sp3, sp4, sp5 = self.swin_prior(inp)
        
        # 獲取卷積先驗
        (cp0_l, cp0_r), (cp1_l, cp1_r), (cp2_l, cp2_r), (cp3_l, cp3_r), (cp4_l, cp4_r), (
            cp5_l, cp5_r) = self.conv_prior(inp)

        # Level 5
        scp5_l, csp5_l = self.aib5(sp5, cp5_l)
        scp5_r, csp5_r = self.aib5(sp5, cp5_r)
        del sp5, cp5_l, cp5_r

        # Level 4
        scp4_l, csp4_l = self.aib4(sp4, cp4_l)
        scp4_r, csp4_r = self.aib4(sp4, cp4_r)
        del sp4, cp4_l, cp4_r

        # Lib4
        f4_l, f4_r = self.lib4(scp5_l + csp5_l + scp4_l + csp4_l, scp5_r + csp5_r + scp4_r + csp4_r)
        del scp5_l, csp5_l, scp4_l, csp4_l, scp5_r, csp5_r, scp4_r, csp4_r

        # Level 3 + Fusion
        scp3_l, csp3_l = self.aib3(sp3, cp3_l)
        scp3_r, csp3_r = self.aib3(sp3, cp3_r)
        del sp3, cp3_l, cp3_r
        
        # ===== 新增：融合上一層特徵 =====
        f3_raw_l = f4_l + scp3_l + csp3_l
        f3_raw_r = f4_r + scp3_r + csp3_r
        f3_fused_l = self.fusion3(f3_raw_l, f4_l)
        f3_fused_r = self.fusion3(f3_raw_r, f4_r)
        
        # Lib3
        f3_l, f3_r = self.lib3(f3_fused_l, f3_fused_r)
        del scp3_l, csp3_l, scp3_r, csp3_r

        # Level 2 + Fusion
        scp2_l, csp2_l = self.aib2(sp2, cp2_l)
        scp2_r, csp2_r = self.aib2(sp2, cp2_r)
        del sp2, cp2_l, cp2_r
        
        # ===== 新增：融合上一層特徵 =====
        f2_raw_l = f3_l + scp2_l + csp2_l
        f2_raw_r = f3_r + scp2_r + csp2_r
        f2_fused_l = self.fusion2(f2_raw_l, f3_l)
        f2_fused_r = self.fusion2(f2_raw_r, f3_r)
        
        # Lib2
        f2_l, f2_r = self.lib2(f2_fused_l, f2_fused_r)
        del scp2_l, csp2_l, scp2_r, csp2_r

        # Level 1 + Fusion
        # ===== 新增：融合上一層特徵 =====
        f1_raw_l = f2_l + cp1_l
        f1_raw_r = f2_r + cp1_r
        f1_fused_l = self.fusion1(f1_raw_l, f2_l)
        f1_fused_r = self.fusion1(f1_raw_r, f2_r)
        
        # Lib1
        f1_l, f1_r = self.lib1(f1_fused_l, f1_fused_r)
        del cp1_l, cp1_r

        # Level 0
        f0_l, f0_r = self.lib0(f1_l + cp0_l, f1_r + cp0_r)
        del f1_l, f1_r, cp0_l, cp0_r

        # 輸出
        out_l, out_r = self.out(f0_l, f0_r)
        out_rr = self.lrm(f0_l + f0_r)
        
        return out_l, out_r, out_rr


if __name__ == '__main__':
    inp = torch.randn(1, 3, 384, 384).cuda()
    model = DSIT(enc_blk_nums=[12, 8, 4, 2, 2], dec_blk_nums=[2, 2, 2, 2, 2]).cuda()
    print(model(inp)[0].shape)
