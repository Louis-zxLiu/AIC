"""Small identity, gradient and bfloat16 smoke test for IR feature alignment."""

from __future__ import annotations

import torch

from aic.models.ir_alignment import MultiScaleIRAlignment


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    module = MultiScaleIRAlignment(
        channels_per_scale=[16, 32, 64],
        strides=[8, 16, 32],
        hidden_channels=8,
        max_offset_pixels=32.0,
        gate_bias=-2.0,
    ).to(device=device, dtype=dtype)
    rgb = [
        torch.randn(2, 16, 16, 24, device=device, dtype=dtype),
        torch.randn(2, 32, 8, 12, device=device, dtype=dtype),
        torch.randn(2, 64, 4, 6, device=device, dtype=dtype),
    ]
    ir = [tensor.detach().clone().requires_grad_(True) for tensor in rgb]
    output = module(rgb, ir)
    max_identity_error = max(
        float((aligned - source).abs().max().detach())
        for aligned, source in zip(output, ir)
    )
    if max_identity_error > 1e-3:
        raise RuntimeError(f"identity initialization error is too large: {max_identity_error}")

    # A detection-style downstream objective must reach the zero-initialized head.
    loss = sum((aligned * torch.randn_like(aligned)).mean() for aligned in output)
    loss.backward()
    head_gradient = sum(
        float(level.head.weight.grad.abs().sum())
        for level in module.levels
        if level.head.weight.grad is not None
    )
    if not torch.isfinite(torch.tensor(head_gradient)) or head_gradient <= 0:
        raise RuntimeError(f"alignment head received invalid gradient: {head_gradient}")
    print(
        "ir-alignment-smoke-ok",
        {"dtype": str(dtype), "identity_error": max_identity_error, "head_gradient": head_gradient},
    )


if __name__ == "__main__":
    main()
