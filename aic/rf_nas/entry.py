"""Bind the shared trainer to elastic RF-DETR implementations."""
import sys
import torch
import yaml


def main():
    mode = sys.argv.pop(1)
    cfg = yaml.safe_load(open(sys.argv[sys.argv.index("--config") + 1]))
    from aic.tools import train
    from .model import RFNASDetector, NASCriterion, param_groups
    from . import distill
    from aic.rf_native.assignment import install_assignment_fallback
    install_assignment_fallback()
    distill.CONFIG = cfg
    # RFDistill reads CONFIG from its defining module.
    import aic.rf_native.distill as native_distill
    native_distill.CONFIG = cfg
    train.build_single_modality_dfine = RFNASDetector
    train.build_tristream_dfine = RFNASDetector
    train.DFINECriterion = NASCriterion
    train.M2DDistillLoss = distill.NASRFDistill
    train.build_param_groups = param_groups
    original = train.compute_class_balanced_weights
    train.compute_class_balanced_weights = lambda *a, **k: [min(v, 4.0) for v in original(*a, **k)]

    def ema_update(self, model):
        inner = model.module if hasattr(model, "module") else model
        self.updates += 1
        decay = self.decay * (1 - torch.exp(torch.tensor(-self.updates / self.warmups))).item() if self.warmups else self.decay
        with torch.no_grad():
            for name, value in inner.state_dict().items():
                if value.is_floating_point():
                    old, cur = self.shadow[name], value.detach().float()
                    self.shadow[name] = torch.where(torch.isfinite(old) & torch.isfinite(cur), old * decay + cur * (1 - decay), cur)
                else:
                    self.shadow[name] = value.detach().clone()
    train.ModelEMA.update = ema_update
    if mode == "train":
        train.main()
    elif mode == "infer":
        from aic.tools import infer
        infer.build_tristream_dfine = RFNASDetector
        infer.main()
    else:
        raise ValueError(mode)


if __name__ == "__main__":
    main()
