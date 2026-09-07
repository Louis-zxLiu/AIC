"""Run one end-to-end D-FINE-X forward/backward through the IR aligner."""

from __future__ import annotations

import argparse

import torch
import yaml

from aic.models.tristream_dfine import build_tristream_dfine


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    with open(args.config, "r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)

    device = torch.device("cuda")
    torch.manual_seed(int(cfg["train"].get("seed", 42)))
    model = build_tristream_dfine(cfg).to(device).eval()
    if model.ir_alignment is None:
        raise RuntimeError("smoke config did not enable IR alignment")

    height, width = 256, 448
    rgb = torch.randn(1, 3, height, width, device=device)
    ir = torch.randn(1, 1, height, width, device=device)
    depth = torch.randn(1, 5, height, width, device=device)
    depth[:, 4:5] = 1.0
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        output = model(rgb, ir, depth)
        loss = output["pred_logits"].float().mean() + output["pred_boxes"].float().mean()
    loss.backward()
    gradient = sum(
        float(level.head.weight.grad.abs().sum())
        for level in model.ir_alignment.levels
        if level.head.weight.grad is not None
    )
    if not torch.isfinite(loss) or gradient <= 0:
        raise RuntimeError(f"invalid full-model smoke result: loss={loss}, gradient={gradient}")
    state_keys = model.state_dict().keys()
    if not any(key.startswith("ir_alignment.levels.") for key in state_keys):
        raise RuntimeError("alignment parameters are missing from model.state_dict()")
    print(
        "ir-alignment-full-model-smoke-ok",
        {"loss": float(loss.detach()), "head_gradient": gradient},
    )


if __name__ == "__main__":
    main()
