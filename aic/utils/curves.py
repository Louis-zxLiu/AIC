"""训练曲线记录与出图。不依赖 TensorBoard/端口转发——远端训练时直接把
PNG 落盘到 save_dir，用户用 scp 或 AutoDL 文件浏览器下载查看即可。
"""

from __future__ import annotations

import json
import os


class CurveLogger:
    """累积训练过程中的标量指标，定期画图存 PNG。

    两条时间轴分开累积：step 粒度的训练 loss/lr，epoch 粒度的验证 mAP。
    同时把原始数据落一份 json，方便后续用其它工具重新画图或核对数值。
    """

    def __init__(self, save_dir: str):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        self.train_steps: list[int] = []
        self.train_losses: dict[str, list[float]] = {}
        self.train_lrs: list[float] = []
        self.eval_epochs: list[int] = []
        self.eval_metrics: dict[str, list[float]] = {}
        self._load_existing()

    def _load_existing(self) -> None:
        """resume 训练时（换了 save_dir 或原地续跑）接上已有曲线数据，
        避免重新画图把前面几十个 epoch 的历史丢掉。"""
        path = os.path.join(self.save_dir, "curves.json")
        if not os.path.exists(path):
            return
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        self.train_steps = data.get("train_steps", [])
        self.train_losses = data.get("train_losses", {})
        self.train_lrs = data.get("train_lrs", [])
        self.eval_epochs = data.get("eval_epochs", [])
        self.eval_metrics = data.get("eval_metrics", {})

    def log_train_step(self, step: int, loss_dict: dict[str, float], lr: float) -> None:
        self.train_steps.append(step)
        self.train_lrs.append(lr)
        for k, v in loss_dict.items():
            self.train_losses.setdefault(k, []).append(v)

    def truncate_after(self, epoch: int, steps_per_epoch: int) -> None:
        keep = sum(ep <= epoch for ep in self.eval_epochs)
        self.eval_epochs = self.eval_epochs[:keep]
        self.eval_metrics = {k:v[:keep] for k,v in self.eval_metrics.items()}
        keep = sum(step < (epoch+1)*steps_per_epoch for step in self.train_steps)
        self.train_steps = self.train_steps[:keep]
        self.train_lrs = self.train_lrs[:keep]
        self.train_losses = {k:v[:keep] for k,v in self.train_losses.items()}

    def log_eval(self, epoch: int, metrics: dict[str, float]) -> None:
        self.eval_epochs.append(epoch)
        for k, v in metrics.items():
            self.eval_metrics.setdefault(k, []).append(v)

    def save_json(self) -> None:
        data = {
            "train_steps": self.train_steps,
            "train_losses": self.train_losses,
            "train_lrs": self.train_lrs,
            "eval_epochs": self.eval_epochs,
            "eval_metrics": self.eval_metrics,
        }
        path = os.path.join(self.save_dir, "curves.json")
        temporary = path + ".tmp"
        with open(temporary, "w", encoding="utf-8") as f:
            json.dump(data, f)
        os.replace(temporary, path)

    # train_losses 里以此为前缀的 key 归入"辅助层损失"子图；"total" 单独一张图
    # （量级远高于其它 loss，混在同图里会把主头 5 项的刻度压扁）；剩下的（主头
    # 5 项）归入"主头损失"子图——分组含义见 train.py 里 group_sums 的注释。
    _AUX_GROUP_KEYS = ("grp_aux", "grp_enc", "grp_dn", "grp_pre")
    _TOTAL_KEY = "total"

    @staticmethod
    def _smooth(values: list[float], steps: list[int]) -> tuple[list[int], list[float]] | tuple[None, None]:
        """滑动平均去噪，窗口按数据量自适应（sqrt 增长）：数据点太少时窗口至少
        为 2 才有意义，数据点很多（训练拉满全程）时窗口跟着变大，不然长训练曲线
        还是会被逐 step 的噪声淹没。窗口太大会吃掉真实的阶段性变化，sqrt 增长比
        固定窗口或线性增长更折中。
        """
        n = len(values)
        window = max(3, int(n**0.5))
        if n < window * 2:
            return None, None
        import numpy as np

        kernel = np.ones(window) / window
        smoothed = np.convolve(values, kernel, mode="valid")
        offset = (window - 1) // 2
        smoothed_steps = steps[offset : offset + len(smoothed)]
        return smoothed_steps, smoothed.tolist()

    def plot(self, filename: str = "curves.png") -> str:
        """五联图：主头训练 loss、总 loss（原始+滑动平均拟合）、辅助层训练 loss
        （深监督/去噪/encoder 辅助头，按组求和）、学习率、验证 mAP。数据不足时
        对应子图留空而不是报错，训练早期也能正常出图。
        """
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 5, figsize=(30, 5))

        ax = axes[0]
        for k, v in self.train_losses.items():
            if k in self._AUX_GROUP_KEYS or k == self._TOTAL_KEY:
                continue
            steps_k = self.train_steps[: len(v)]
            sm_steps, sm_v = self._smooth(v, steps_k)
            if sm_v is not None:
                (line,) = ax.plot(steps_k, v, alpha=0.25, linewidth=1)
                ax.plot(sm_steps, sm_v, color=line.get_color(), linewidth=2, label=k)
            else:
                ax.plot(steps_k, v, label=k, linewidth=1)
        ax.set_xlabel("step")
        ax.set_ylabel("loss")
        ax.set_title("main head loss")
        if self.train_losses:
            ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

        ax = axes[1]
        total_v = self.train_losses.get(self._TOTAL_KEY)
        if total_v:
            total_steps = self.train_steps[: len(total_v)]
            ax.plot(total_steps, total_v, color="tab:purple", alpha=0.3, linewidth=1, label="total (raw)")
            sm_steps, sm_v = self._smooth(total_v, total_steps)
            if sm_v is not None:
                ax.plot(sm_steps, sm_v, color="tab:purple", linewidth=2, label="total (smoothed)")
            ax.legend(fontsize=8)
        ax.set_xlabel("step")
        ax.set_ylabel("loss")
        ax.set_title("total loss")
        ax.grid(alpha=0.3)

        ax = axes[2]
        has_aux = False
        for k in self._AUX_GROUP_KEYS:
            v = self.train_losses.get(k)
            if not v:
                continue
            has_aux = True
            ax.plot(self.train_steps[: len(v)], v, label=k, linewidth=1)
        ax.set_xlabel("step")
        ax.set_ylabel("loss")
        ax.set_title("auxiliary head loss (grouped sum)")
        if has_aux:
            ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

        ax = axes[3]
        if self.train_lrs:
            ax.plot(self.train_steps, self.train_lrs, color="tab:orange", linewidth=1)
        ax.set_xlabel("step")
        ax.set_ylabel("lr")
        ax.set_title("learning rate")
        ax.grid(alpha=0.3)

        ax = axes[4]
        for k, v in self.eval_metrics.items():
            if k.startswith('raw_'):
                continue
            ax.plot(self.eval_epochs[: len(v)], v, marker="o", label=k, linewidth=1)
        ax.set_xlabel("epoch")
        ax.set_ylabel("mAP")
        ax.set_title("validation mAP")
        if self.eval_metrics:
            ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

        fig.tight_layout()
        out_path = os.path.join(self.save_dir, filename)
        fig.savefig(out_path, dpi=120)
        plt.close(fig)
        return out_path
