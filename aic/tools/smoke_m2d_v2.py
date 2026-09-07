from __future__ import annotations

import os
import tempfile

import torch

from aic.models.backbone import TriStreamHGNetv2
from aic.models.distill import M2DDistillLoss


def checkpoint_from_backbone(path: str, backbone: TriStreamHGNetv2) -> None:
    state = {f"backbone.{key}": value.detach().cpu() for key, value in backbone.state_dict().items()}
    torch.save({"ema": state}, path)


def targets(device: torch.device, ir_valid: float, depth_valid: float) -> list[dict]:
    return [
        {
            "boxes": torch.tensor([[0.5, 0.5, 0.3, 0.3]], device=device),
            "labels": torch.tensor([0], device=device),
            "ir_distill_valid": torch.tensor(ir_valid, device=device),
            "depth_distill_valid": torch.tensor(depth_valid, device=device),
        }
    ]


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = {
        "name": "B0",
        "use_lab": False,
        "return_idx": [1, 2, 3],
        "share_stages": True,
        "depth_in_chs": 5,
    }
    with tempfile.TemporaryDirectory(dir="/home/u703/AIC_Project_sy/AIC/tmp") as temp_dir:
        source = TriStreamHGNetv2(name="B0", pretrained=False, depth_in_chs=5)
        ir_checkpoint = os.path.join(temp_dir, "ir.pt")
        depth_checkpoint = os.path.join(temp_dir, "depth.pt")
        checkpoint_from_backbone(ir_checkpoint, source)
        checkpoint_from_backbone(depth_checkpoint, source)

        loss_fn = M2DDistillLoss(
            bb_cfg=cfg,
            ir_teacher_ckpt=ir_checkpoint,
            depth_teacher_ckpt=depth_checkpoint,
            use_teacher_ema=True,
            foreground_weight=2.0,
            level_weights=[2.0, 1.0, 1.0],
            use_depth_valid_mask=True,
            cross_weight=1.0,
        ).to(device)
        student = TriStreamHGNetv2(name="B0", pretrained=False, depth_in_chs=5).to(device)
        rgb = torch.rand(1, 3, 64, 96, device=device)
        ir = torch.rand(1, 1, 64, 96, device=device)
        depth = torch.rand(1, 5, 64, 96, device=device)
        depth[:, 1:4] = 1.0
        depth[:, 4:5] = torch.rand_like(depth[:, 4:5])
        _, ir_features, depth_features = student(rgb, ir, depth)

        total, metrics = loss_fn(
            ir,
            depth,
            ir_features,
            depth_features,
            targets(device, 1.0, 1.0),
        )
        if not torch.isfinite(total):
            raise RuntimeError(f"non-finite M2D loss: {total}")
        total.backward()
        adapter_grad = sum(
            float(parameter.grad.abs().sum())
            for parameter in loss_fn.parameters()
            if parameter.requires_grad and parameter.grad is not None
        )
        if adapter_grad <= 0:
            raise RuntimeError("M2D adapters received no gradient")

        _, no_ir = loss_fn(ir * 0, depth, ir_features, depth_features, targets(device, 0.0, 1.0))
        _, no_depth = loss_fn(ir, depth * 0, ir_features, depth_features, targets(device, 1.0, 0.0))
        assert no_ir["loss_ir"].item() == 0.0 and no_ir["loss_cross"].item() == 0.0
        assert no_depth["loss_depth"].item() == 0.0 and no_depth["loss_cross"].item() == 0.0

        loss_fn.cross_weight = 0.0
        loss_fn.ir_weight = 1.0
        loss_fn.depth_weight = 0.0
        ir_only_total, ir_only = loss_fn(
            ir, depth, ir_features, depth_features, targets(device, 1.0, 1.0)
        )
        assert ir_only["loss_ir"].item() > 0.0
        assert ir_only["loss_depth"].item() == 0.0
        assert torch.equal(ir_only_total.detach(), ir_only["loss_ir"])

        loss_fn.ir_weight = 0.0
        loss_fn.depth_weight = 1.0
        depth_only_total, depth_only = loss_fn(
            ir, depth, ir_features, depth_features, targets(device, 1.0, 1.0)
        )
        assert depth_only["loss_ir"].item() == 0.0
        assert depth_only["loss_depth"].item() > 0.0
        assert torch.equal(depth_only_total.detach(), depth_only["loss_depth"])
        print(
            "smoke-ok",
            {key: round(float(value), 6) for key, value in metrics.items()},
            "adapter_grad",
            adapter_grad,
        )


if __name__ == "__main__":
    main()
