"""2D Haar 离散小波变换（DWT/IDWT），depthwise 卷积实现，无额外依赖。

动机见 计划.md §2.3：WaveMamba 用 DWT 把每个模态特征拆成 1 个低频子带（LL，
承载全局结构/低频几何信息）+ 3 个高频子带（LH/HL/HH，承载边缘/纹理细节），
对 depth（低频主导）和 RGB（高频细节更丰富）这类模态差异场景分别处理。

Haar 是最简单的正交小波（2x2 均值/差分），滤波器是固定常数，不需要学习，
用 stride=2、groups=channels 的卷积一次完成"降采样+分解"，四个子带的
kernel 权重共享同一套 2x2 pattern，只是符号不同：
    LL: [[1,1],[1,1]]/2   HL: [[1,-1],[1,-1]]/2
    LH: [[1,1],[-1,-1]]/2 HH: [[1,-1],[-1,1]]/2
（这里用的是归一化正交版本，缩放因子 1/2 保证 IDWT 是精确逆变换。）
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

_HAAR_LL = torch.tensor([[1.0, 1.0], [1.0, 1.0]]) * 0.5
_HAAR_LH = torch.tensor([[1.0, 1.0], [-1.0, -1.0]]) * 0.5
_HAAR_HL = torch.tensor([[1.0, -1.0], [1.0, -1.0]]) * 0.5
_HAAR_HH = torch.tensor([[1.0, -1.0], [-1.0, 1.0]]) * 0.5


class HaarDWT2D(nn.Module):
    """输入 B,C,H,W -> 输出 4 个子带，各 B,C,H/2,W/2（LL, LH, HL, HH）。

    奇数 H/W 会先在右/下边缘做一次复制 padding 补到偶数，IDWT 端需要外部
    记住原始 H,W 再裁剪回去（本模块只做正向变换，裁剪逻辑放在调用侧，
    因为 OPF 融合场景下低频/高频分别处理后才合并，裁剪时机由调用者决定）。
    """

    def __init__(self, channels: int):
        super().__init__()
        self.channels = channels
        filt = torch.stack([_HAAR_LL, _HAAR_LH, _HAAR_HL, _HAAR_HH], dim=0)  # 4,2,2
        # depthwise 卷积权重: (4*C, 1, 2, 2)，按子带分组排列方便后续 reshape 取出
        weight = filt.unsqueeze(1).repeat(channels, 1, 1, 1)  # (4*C,1,2,2)，先按C重复再按子带交织见下
        # 上面的 repeat 顺序是 [LL,LH,HL,HH] 各重复 C 次，reshape 成 (4,C,1,2,2) 再 permute
        # 更直接：直接按 (C,4,1,2,2) 顺序构造，forward 里再 reshape 回 (B,4,C,H',W')
        weight = filt.repeat(channels, 1, 1).view(channels, 4, 2, 2).reshape(channels * 4, 1, 2, 2)
        self.register_buffer("weight", weight, persistent=False)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        b, c, h, w = x.shape
        assert c == self.channels
        if h % 2 == 1:
            x = F.pad(x, (0, 0, 0, 1), mode="replicate")
        if w % 2 == 1:
            x = F.pad(x, (0, 1, 0, 0), mode="replicate")

        # groups=C，每组输入 1 通道、输出 4 通道（该通道的 LL/LH/HL/HH）
        out = F.conv2d(x, self.weight, stride=2, groups=c)  # B, C*4, H/2, W/2
        out = out.view(b, c, 4, out.shape[-2], out.shape[-1])
        ll, lh, hl, hh = out[:, :, 0], out[:, :, 1], out[:, :, 2], out[:, :, 3]
        return ll, lh, hl, hh


class HaarIDWT2D(nn.Module):
    """HaarDWT2D 的逆变换：4 个子带 -> B,C,H,W（H,W 需外部传入以裁剪 padding）。"""

    def __init__(self, channels: int):
        super().__init__()
        self.channels = channels
        filt = torch.stack([_HAAR_LL, _HAAR_LH, _HAAR_HL, _HAAR_HH], dim=0)  # 4,2,2
        weight = filt.repeat(channels, 1, 1).view(channels, 4, 2, 2).reshape(channels * 4, 1, 2, 2)
        self.register_buffer("weight", weight, persistent=False)

    def forward(
        self,
        ll: torch.Tensor,
        lh: torch.Tensor,
        hl: torch.Tensor,
        hh: torch.Tensor,
        out_hw: tuple[int, int] | None = None,
    ) -> torch.Tensor:
        b, c, hh_, ww_ = ll.shape
        assert c == self.channels
        x = torch.stack([ll, lh, hl, hh], dim=2).reshape(b, c * 4, hh_, ww_)
        out = F.conv_transpose2d(x, self.weight, stride=2, groups=c)  # B,C,2H',2W'

        if out_hw is not None:
            th, tw = out_hw
            out = out[:, :, :th, :tw]
        return out
