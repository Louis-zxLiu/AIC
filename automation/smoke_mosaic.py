"""One-sample remote smoke test for synchronized Mosaic."""

import yaml

from aic.data.dataset import MultiModalDetDataset, kfold_split, scan_samples
from aic.data.transforms import SyncMultiModalTransform


def main():
    with open("configs/dfine_tristream_x_m2d_deim_mal_mosaic_fold0_64.yml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    samples = scan_samples(cfg["train"]["data_root"], require_labels=True)
    train, _ = kfold_split(samples, n_folds=5, fold=0)
    ds = MultiModalDetDataset(
        train,
        SyncMultiModalTransform(cfg["image_size"], train=True),
        disp8_mode="use",
        mosaic_prob=1.0,
        mosaic_start_epoch=0,
        mosaic_stop_epoch=2,
    )
    ds.set_epoch(0)
    rgb, ir, depth, target = ds[0]
    assert rgb.shape[-2:] == ir.shape[-2:] == depth.shape[-2:]
    assert depth.shape[0] == 5
    assert len(target["boxes"]) == len(target["labels"])
    assert float(target["boxes"].min()) >= 0.0 and float(target["boxes"].max()) <= 1.0
    print("mosaic_smoke_ok", tuple(rgb.shape), tuple(ir.shape), tuple(depth.shape), len(target["boxes"]))


if __name__ == "__main__":
    main()
