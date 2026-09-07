"""Identity-initialized, RGB-guided alignment for IR fusion features."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class IRAlignmentLevel(nn.Module):
    """Predict a bounded dense IR warp and a confidence-controlled residual gate."""

    def __init__(
        self,
        channels: int,
        stride: int,
        hidden_channels: int = 16,
        max_offset_pixels: float = 32.0,
        gate_bias: float = -2.0,
    ) -> None:
        super().__init__()
        if hidden_channels < 4:
            raise ValueError("hidden_channels must be at least 4")
        if max_offset_pixels <= 0:
            raise ValueError("max_offset_pixels must be positive")
        groups = 4 if hidden_channels % 4 == 0 else 1
        self.stride = int(stride)
        self.max_offset_cells = float(max_offset_pixels) / self.stride
        self.rgb_projection = nn.Sequential(
            nn.Conv2d(channels, hidden_channels, 1, bias=False),
            nn.GroupNorm(groups, hidden_channels),
            nn.SiLU(inplace=True),
        )
        self.ir_projection = nn.Sequential(
            nn.Conv2d(channels, hidden_channels, 1, bias=False),
            nn.GroupNorm(groups, hidden_channels),
            nn.SiLU(inplace=True),
        )
        self.context = nn.Sequential(
            nn.Conv2d(hidden_channels * 3, hidden_channels, 3, padding=1, bias=False),
            nn.GroupNorm(groups, hidden_channels),
            nn.SiLU(inplace=True),
        )
        self.head = nn.Conv2d(hidden_channels, 3, 3, padding=1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        with torch.no_grad():
            self.head.bias[2] = float(gate_bias)

        self._last_metrics: dict[str, torch.Tensor] = {}

    def forward(self, rgb: torch.Tensor, ir: torch.Tensor) -> torch.Tensor:
        if rgb.shape != ir.shape:
            raise ValueError(f"RGB/IR feature shape mismatch: {rgb.shape} vs {ir.shape}")
        rgb_context = self.rgb_projection(rgb)
        ir_context = self.ir_projection(ir)
        context = self.context(
            torch.cat([rgb_context, ir_context, rgb_context - ir_context], dim=1)
        )
        prediction = self.head(context)
        offsets = torch.tanh(prediction[:, :2]) * self.max_offset_cells
        gate = torch.sigmoid(prediction[:, 2:3])

        height, width = ir.shape[-2:]
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1.0, 1.0, height, device=ir.device, dtype=torch.float32),
            torch.linspace(-1.0, 1.0, width, device=ir.device, dtype=torch.float32),
            indexing="ij",
        )
        base_grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)
        offset_grid = torch.stack(
            [
                2.0 * offsets[:, 0] / max(width - 1, 1),
                2.0 * offsets[:, 1] / max(height - 1, 1),
            ],
            dim=-1,
        ).float()
        # CUDA bfloat16 grid_sample introduces visible interpolation error even
        # for a zero offset. Sampling in FP32 preserves the identity start while
        # keeping the surrounding backbone and fusion under autocast.
        with torch.autocast(device_type=ir.device.type, enabled=False):
            warped = F.grid_sample(
                ir.float(),
                base_grid + offset_grid,
                mode="bilinear",
                padding_mode="border",
                align_corners=True,
            ).to(dtype=ir.dtype)
        aligned = ir + gate * (warped - ir)

        offset_pixels = offsets.detach() * self.stride
        self._last_metrics = {
            "gate": gate.detach().mean(),
            "offset": torch.linalg.vector_norm(offset_pixels, dim=1).mean(),
        }
        return aligned

    def last_metrics(self) -> dict[str, torch.Tensor]:
        return self._last_metrics


class MultiScaleIRAlignment(nn.Module):
    """Apply independent bounded alignment at D-FINE strides 8, 16 and 32."""

    def __init__(
        self,
        channels_per_scale: list[int],
        strides: list[int] | tuple[int, ...],
        hidden_channels: int = 16,
        max_offset_pixels: float = 32.0,
        gate_bias: float = -2.0,
    ) -> None:
        super().__init__()
        if len(channels_per_scale) != len(strides):
            raise ValueError("alignment channels and strides must have the same length")
        self.levels = nn.ModuleList(
            IRAlignmentLevel(
                channels,
                stride,
                hidden_channels=hidden_channels,
                max_offset_pixels=max_offset_pixels,
                gate_bias=gate_bias,
            )
            for channels, stride in zip(channels_per_scale, strides)
        )

    def forward(
        self,
        rgb_features: list[torch.Tensor],
        ir_features: list[torch.Tensor],
    ) -> list[torch.Tensor]:
        if len(rgb_features) != len(self.levels) or len(ir_features) != len(self.levels):
            raise ValueError("feature levels do not match the configured IR aligner")
        return [
            level(rgb, ir)
            for level, rgb, ir in zip(self.levels, rgb_features, ir_features)
        ]

    def last_metrics(self) -> dict[str, torch.Tensor]:
        metrics: dict[str, torch.Tensor] = {}
        for index, level in enumerate(self.levels):
            for name, value in level.last_metrics().items():
                metrics[f"align_{name}_l{index}"] = value
        return metrics
