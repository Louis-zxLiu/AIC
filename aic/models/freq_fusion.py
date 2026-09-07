"""Pure-PyTorch FreqFusion for the top-down D-FINE feature pyramid.

The implementation follows the public FreqFusion design (ALPF, AHPF and
local-similarity-guided resampling) without requiring mmcv's CARAFE op.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from third_party_dfine import HybridEncoder


def _hamming2d(size: int) -> torch.Tensor:
    window = torch.hamming_window(size, periodic=False)
    return window[:, None] * window[None, :]


def _carafe(x: torch.Tensor, mask: torch.Tensor, kernel_size: int, up: int) -> torch.Tensor:
    """Content-aware reassembly for group=1, specialized for upsampling by 2."""
    if up not in (1, 2):
        raise ValueError(f"FreqFusion only supports up=1 or up=2, got {up}")
    b, c, h, w = x.shape
    expected_h, expected_w = h * up, w * up
    if mask.shape[-2:] != (expected_h, expected_w):
        raise ValueError(
            f"mask spatial size {tuple(mask.shape[-2:])} does not match "
            f"input {(expected_h, expected_w)}"
        )
    k2 = kernel_size * kernel_size
    if mask.shape[1] != k2:
        raise ValueError(f"expected {k2} mask channels, got {mask.shape[1]}")

    pad = kernel_size // 2
    padded = F.pad(x, (pad, pad, pad, pad), mode="reflect")
    patches = F.unfold(padded, kernel_size=kernel_size).view(b, c, k2, h, w)
    if up == 1:
        return (patches * mask.view(b, 1, k2, h, w)).sum(dim=2)

    # Pixel-unshuffle keeps only the four sub-pixel masks per source location,
    # avoiding the 4x intermediate tensor used by the reference fallback.
    subpixel_mask = F.pixel_unshuffle(mask, 2).view(b, k2, 4, h, w)
    out = torch.einsum("bckhw,bkshw->bcshw", patches, subpixel_mask)
    return F.pixel_shuffle(out.reshape(b, c * 4, h, w), 2)


def _normal_init(module: nn.Module, std: float = 1.0) -> None:
    nn.init.normal_(module.weight, std=std)
    if module.bias is not None:
        nn.init.constant_(module.bias, 0.0)


def _constant_init(module: nn.Module, value: float) -> None:
    nn.init.constant_(module.weight, value)
    if module.bias is not None:
        nn.init.constant_(module.bias, 0.0)


def compute_similarity(x: torch.Tensor, kernel_size: int = 3, dilation: int = 1) -> torch.Tensor:
    b, c, h, w = x.shape
    patches = F.unfold(
        x,
        kernel_size=kernel_size,
        padding=(kernel_size // 2) * dilation,
        dilation=dilation,
    ).view(b, c, kernel_size * kernel_size, h, w)
    center = patches[:, :, kernel_size * kernel_size // 2 : kernel_size * kernel_size // 2 + 1]
    similarity = F.cosine_similarity(center, patches, dim=1)
    center_idx = kernel_size * kernel_size // 2
    return torch.cat([similarity[:, :center_idx], similarity[:, center_idx + 1 :]], dim=1)


class LocalSimGuidedSampler(nn.Module):
    """Offset generator from FreqFusion, with four offset groups by default."""

    def __init__(
        self,
        in_channels: int,
        groups: int = 4,
        kernel_size: int = 3,
        use_direct_scale: bool = True,
        norm: bool = True,
        corrected: bool = False,
    ) -> None:
        super().__init__()
        if in_channels % groups != 0 or in_channels < groups:
            raise ValueError("in_channels must be divisible by the number of offset groups")
        self.scale = 2
        self.groups = groups
        feature_channels = in_channels + 8
        self.offset = nn.Conv2d(feature_channels, 2 * groups * 4, kernel_size, padding=kernel_size // 2)
        self.hr_offset = nn.Conv2d(feature_channels, 2 * groups, kernel_size, padding=kernel_size // 2)
        _normal_init(self.offset, std=0.001)
        _normal_init(self.hr_offset, std=0.001)
        self.direct_scale = None
        self.hr_direct_scale = None
        if use_direct_scale:
            self.direct_scale = nn.Conv2d(feature_channels, 2 * groups * 4, kernel_size, padding=kernel_size // 2)
            self.hr_direct_scale = nn.Conv2d(feature_channels, 2 * groups, kernel_size, padding=kernel_size // 2)
            _constant_init(self.direct_scale, 0.0)
            _constant_init(self.hr_direct_scale, 0.0)
        self.norm_hr = nn.GroupNorm(in_channels // 8, in_channels) if norm else nn.Identity()
        self.norm_lr = nn.GroupNorm(in_channels // 8, in_channels) if norm else nn.Identity()
        base = torch.arange(-0.25, 0.26, 0.5)
        init_pos = torch.stack(torch.meshgrid(base, base, indexing="ij"))
        # The official implementation transposes the two spatial axes.  The
        # legacy branch preserves the first experiment's incorrect layout so
        # its checkpoint remains reproducible.
        init_pos = init_pos.transpose(1, 2) if corrected else init_pos.transpose(0, 1)
        self.register_buffer("init_pos", init_pos)
        self.init_pos = self.init_pos.repeat(1, groups, 1).reshape(1, -1, 1, 1)

    def sample(self, x: torch.Tensor, offset: torch.Tensor) -> torch.Tensor:
        b, _, h, w = offset.shape
        offset = offset.view(b, 2, -1, h, w)
        coords_h = torch.arange(h, device=x.device, dtype=x.dtype) + 0.5
        coords_w = torch.arange(w, device=x.device, dtype=x.dtype) + 0.5
        coords = torch.stack(torch.meshgrid(coords_w, coords_h, indexing="ij"), dim=0)
        coords = coords.transpose(1, 2).unsqueeze(0).unsqueeze(2)
        coords = 2 * (coords + offset) / torch.tensor([w, h], device=x.device, dtype=x.dtype).view(1, 2, 1, 1, 1) - 1
        coords = F.pixel_shuffle(coords.reshape(b, -1, h, w), 2)
        coords = coords.view(b, 2, -1, h * 2, w * 2).permute(0, 2, 3, 4, 1)
        coords = coords.reshape(b * self.groups, h * 2, w * 2, 2)
        sampled = F.grid_sample(
            x.reshape(b * self.groups, -1, x.shape[-2], x.shape[-1]),
            coords,
            mode="bilinear",
            align_corners=False,
            padding_mode="border",
        )
        return sampled.view(b, -1, h * 2, w * 2)

    def forward(self, hr_x: torch.Tensor, lr_x: torch.Tensor, feat2sample: torch.Tensor) -> torch.Tensor:
        hr_x = self.norm_hr(hr_x)
        lr_x = self.norm_lr(lr_x)
        hr_sim = torch.cat([hr_x, compute_similarity(hr_x, 3, dilation=2)], dim=1)
        lr_sim = torch.cat([lr_x, compute_similarity(lr_x, 3, dilation=2)], dim=1)
        offset = self.offset(lr_sim) + F.pixel_unshuffle(self.hr_offset(hr_sim), 2)
        if self.direct_scale is not None:
            scale = self.direct_scale(lr_sim) + F.pixel_unshuffle(self.hr_direct_scale(hr_sim), 2)
            offset = offset * scale.sigmoid()
        return self.sample(feat2sample, offset + self.init_pos.to(offset))


class FreqFusion(nn.Module):
    """Frequency-aware fusion of a high-resolution and a 2x lower-resolution feature."""

    def __init__(
        self,
        hr_channels: int,
        lr_channels: int,
        lowpass_kernel: int = 5,
        highpass_kernel: int = 3,
        compressed_channels: int = 64,
        feature_resample: bool = True,
        feature_resample_group: int = 4,
        use_high_pass: bool = True,
        use_low_pass: bool = True,
        hr_residual: bool = True,
        hamming_window: bool = True,
        use_checkpoint: bool = False,
        corrected: bool = False,
    ) -> None:
        super().__init__()
        self.lowpass_kernel = lowpass_kernel
        self.highpass_kernel = highpass_kernel
        self.compressed_channels = compressed_channels
        self.feature_resample = feature_resample
        self.use_high_pass = use_high_pass
        self.use_low_pass = use_low_pass
        self.hr_residual = hr_residual
        self.use_checkpoint = use_checkpoint
        self.corrected = corrected
        self.hr_channel_compressor = nn.Conv2d(hr_channels, compressed_channels, 1)
        self.lr_channel_compressor = nn.Conv2d(lr_channels, compressed_channels, 1)
        self.content_encoder = nn.Conv2d(compressed_channels, lowpass_kernel**2, 3, padding=1)
        self.content_encoder2 = nn.Conv2d(compressed_channels, highpass_kernel**2, 3, padding=1) if use_high_pass else None
        self.dysampler = (
            LocalSimGuidedSampler(
                compressed_channels,
                groups=feature_resample_group,
                corrected=corrected,
            )
            if feature_resample
            else None
        )
        lowpass_window = _hamming2d(lowpass_kernel) if hamming_window else torch.ones(lowpass_kernel, lowpass_kernel)
        highpass_window = _hamming2d(highpass_kernel) if hamming_window else torch.ones(highpass_kernel, highpass_kernel)
        self.register_buffer("hamming_lowpass", lowpass_window.view(1, 1, lowpass_kernel, lowpass_kernel))
        self.register_buffer("hamming_highpass", highpass_window.view(1, 1, highpass_kernel, highpass_kernel))
        for module in (self.hr_channel_compressor, self.lr_channel_compressor, self.content_encoder, self.content_encoder2):
            if module is not None:
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
        _normal_init(self.content_encoder, std=0.001)
        if self.content_encoder2 is not None:
            _normal_init(self.content_encoder2, std=0.001)

    def kernel_normalizer(self, mask: torch.Tensor, kernel: int, hamming: torch.Tensor) -> torch.Tensor:
        b, channels, h, w = mask.shape
        mask = mask.view(b, channels // (kernel * kernel), kernel * kernel, h, w)
        mask = F.softmax(mask.float(), dim=2).to(mask.dtype)
        mask = mask.view(b, -1, kernel, kernel, h, w).permute(0, 1, 4, 5, 2, 3).reshape(-1, kernel, kernel)
        mask = mask * hamming.to(mask)
        mask = mask / mask.sum(dim=(-1, -2), keepdim=True).clamp_min(1e-6)
        return mask.reshape(b, -1, h, w)

    def forward(self, hr_feat: torch.Tensor, lr_feat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.use_checkpoint and self.training:
            return checkpoint(self._forward, hr_feat, lr_feat, use_reentrant=False)
        return self._forward(hr_feat, lr_feat)

    def _forward(self, hr_feat: torch.Tensor, lr_feat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        compressed_hr = self.hr_channel_compressor(hr_feat)
        compressed_lr = self.lr_channel_compressor(lr_feat)
        mask_hr_hr = None
        if self.use_high_pass:
            mask_hr_hr = self.content_encoder2(compressed_hr)
            hr_init = self.kernel_normalizer(mask_hr_hr, self.highpass_kernel, self.hamming_highpass)
            compressed_hr = compressed_hr + compressed_hr - _carafe(compressed_hr, hr_init, self.highpass_kernel, 1)
        mask_lr_hr = self.content_encoder(compressed_hr)
        mask_lr_lr = self.content_encoder(compressed_lr)
        mask_lr_init = self.kernel_normalizer(mask_lr_hr, self.lowpass_kernel, self.hamming_lowpass)
        mask_lr_lr = F.interpolate(
            _carafe(mask_lr_lr, mask_lr_init, self.lowpass_kernel, 2),
            size=compressed_hr.shape[-2:],
            mode="nearest",
        )
        mask_lr_logits = mask_lr_hr + mask_lr_lr
        mask_lr = self.kernel_normalizer(mask_lr_logits, self.lowpass_kernel, self.hamming_lowpass)
        upsampled_lr = _carafe(lr_feat, mask_lr, self.lowpass_kernel, 2) if self.use_low_pass else F.interpolate(lr_feat, scale_factor=2, mode="nearest")
        if self.use_high_pass:
            # The official semi-convolution path reassembles the low-resolution
            # high-pass logits with the *initial* low-pass kernel.  Reusing the
            # final fused low-pass kernel couples ALPF and AHPF incorrectly.
            mask_for_hr = mask_lr_init if self.corrected else self.kernel_normalizer(
                mask_lr_logits,
                self.lowpass_kernel,
                self.hamming_lowpass,
            )
            mask_hr_lr = F.interpolate(
                _carafe(self.content_encoder2(compressed_lr), mask_for_hr, self.lowpass_kernel, 2),
                size=compressed_hr.shape[-2:],
                mode="nearest",
            )
            mask_hr_base = mask_hr_hr if self.corrected else self.content_encoder2(compressed_hr)
            mask_hr = self.kernel_normalizer(mask_hr_base + mask_hr_lr, self.highpass_kernel, self.hamming_highpass)
            hr_high = hr_feat - _carafe(hr_feat, mask_hr, self.highpass_kernel, 1)
            hr_feat = hr_feat + hr_high if self.hr_residual else hr_high
        if self.dysampler is not None:
            upsampled_lr = self.dysampler(compressed_hr, compressed_lr, upsampled_lr)
        return mask_lr, hr_feat, upsampled_lr


class FreqFusionHybridEncoder(HybridEncoder):
    """Official HybridEncoder with FreqFusion replacing top-down nearest upsampling."""

    def __init__(self, freqfusion: dict | None = None, **kwargs) -> None:
        super().__init__(**kwargs)
        cfg = freqfusion or {}
        self.freqfusion_use_checkpoint = bool(cfg.get("use_checkpoint", False))
        self.freqfusion_corrected = bool(cfg.get("corrected", False))
        self.freqfusion_active_stages = {
            int(stage) for stage in cfg.get("active_stages", range(len(self.in_channels) - 1))
        }
        invalid_stages = self.freqfusion_active_stages - set(range(len(self.in_channels) - 1))
        if invalid_stages:
            raise ValueError(f"invalid FreqFusion stages: {sorted(invalid_stages)}")
        gate_init = cfg.get("gate_init")
        if gate_init is None:
            self.register_parameter("freqfusion_gates", None)
        else:
            gate_init = float(gate_init)
            if not 0.0 < gate_init < 1.0:
                raise ValueError("freqfusion.gate_init must be between 0 and 1")
            gate_logit = torch.logit(torch.tensor(gate_init, dtype=torch.float32))
            self.freqfusion_gates = nn.Parameter(gate_logit.repeat(len(self.in_channels) - 1))
        self.freq_fusions = nn.ModuleList(
            [
                FreqFusion(
                    hr_channels=self.hidden_dim,
                    lr_channels=self.hidden_dim,
                    lowpass_kernel=int(cfg.get("lowpass_kernel", 5)),
                    highpass_kernel=int(cfg.get("highpass_kernel", 3)),
                    compressed_channels=int(cfg.get("compressed_channels", 64)),
                    feature_resample=bool(cfg.get("feature_resample", True)),
                    feature_resample_group=int(cfg.get("feature_resample_group", 4)),
                    use_checkpoint=self.freqfusion_use_checkpoint,
                    corrected=self.freqfusion_corrected,
                )
                for _ in range(len(self.in_channels) - 1)
            ]
        )

    def forward(self, feats):
        assert len(feats) == len(self.in_channels)
        proj_feats = [self.input_proj[i](feat) for i, feat in enumerate(feats)]
        if self.num_encoder_layers > 0:
            for i, enc_ind in enumerate(self.use_encoder_idx):
                h, w = proj_feats[enc_ind].shape[2:]
                src_flatten = proj_feats[enc_ind].flatten(2).permute(0, 2, 1)
                if self.training or self.eval_spatial_size is None:
                    pos_embed = self.build_2d_sincos_position_embedding(w, h, self.hidden_dim, self.pe_temperature).to(src_flatten.device)
                else:
                    pos_embed = getattr(self, f"pos_embed{enc_ind}").to(src_flatten.device)
                memory = self.encoder[i](src_flatten, pos_embed=pos_embed)
                proj_feats[enc_ind] = memory.permute(0, 2, 1).reshape(-1, self.hidden_dim, h, w).contiguous()

        inner_outs = [proj_feats[-1]]
        for idx in range(len(self.in_channels) - 1, 0, -1):
            stage = len(self.in_channels) - 1 - idx
            low_resolution = self.lateral_convs[stage](inner_outs[0])
            high_resolution = proj_feats[idx - 1]
            upsampled = F.interpolate(low_resolution, scale_factor=2.0, mode="nearest")
            if stage in self.freqfusion_active_stages:
                _, enhanced_high, freq_upsampled = self.freq_fusions[stage](
                    hr_feat=high_resolution,
                    lr_feat=low_resolution,
                )
                if self.freqfusion_gates is None:
                    high_resolution, upsampled = enhanced_high, freq_upsampled
                else:
                    gate = self.freqfusion_gates[stage].sigmoid().to(dtype=high_resolution.dtype)
                    high_resolution = high_resolution + gate * (enhanced_high - high_resolution)
                    upsampled = upsampled + gate * (freq_upsampled - upsampled)
            inner_outs.insert(
                0,
                self.fpn_blocks[stage](torch.cat([upsampled, high_resolution], dim=1)),
            )

        outs = [inner_outs[0]]
        for idx in range(len(self.in_channels) - 1):
            feat_low = outs[-1]
            feat_high = inner_outs[idx + 1]
            downsampled = self.downsample_convs[idx](feat_low)
            outs.append(self.pan_blocks[idx](torch.cat([downsampled, feat_high], dim=1)))
        return outs
