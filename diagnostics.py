
import csv
import math
import os
from dataclasses import dataclass, field
from typing import List, Tuple, Dict

import torch

# ---------- Gradient anomaly logging ----------

def log_anomaly_row(csv_path: str, row: Dict):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    header = ["epoch","phase","batch_idx","loss","grad_norm","lr","note"]
    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        if write_header:
            w.writeheader()
        w.writerow({k: row.get(k) for k in header})


def check_and_log_anomaly(epoch: int, phase: str, batch_idx: int, loss: float, grad_norm: float, lr: float, csv_path: str):
    note = []
    if not math.isfinite(loss):
        note.append("non_finite_loss")
    if not math.isfinite(grad_norm):
        note.append("non_finite_grad_norm")
    if grad_norm > 1e3:
        note.append("large_grad_norm")
    if note:
        log_anomaly_row(csv_path, {
            "epoch": epoch,
            "phase": phase,
            "batch_idx": batch_idx,
            "loss": float(loss),
            "grad_norm": float(grad_norm),
            "lr": float(lr),
            "note": "|".join(note)
        })


# ---------- Spline outside-portion hooks ----------

@dataclass
class OutsideStats:
    layer_names: List[str] = field(default_factory=list)
    counts: List[int] = field(default_factory=list)
    sums: List[float] = field(default_factory=list)

    def ensure(self, n: int):
        while len(self.counts) < n:
            self.counts.append(0)
            self.sums.append(0.0)

    def add(self, idx: int, value: float):
        self.ensure(idx + 1)
        self.counts[idx] += 1
        self.sums[idx] += float(value)

    def means(self) -> List[float]:
        return [ (s / c if c>0 else 0.0) for s, c in zip(self.sums, self.counts) ]

    def dump_csv(self, csv_path: str, epoch: int):
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        write_header = not os.path.exists(csv_path)
        header = ["epoch","layer_idx","layer_name","outside_fraction_mean"]
        with open(csv_path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=header)
            if write_header:
                w.writeheader()
            for i,(name,mean_val) in enumerate(zip(self.layer_names, self.means())):
                w.writerow({
                    "epoch": epoch,
                    "layer_idx": i,
                    "layer_name": name,
                    "outside_fraction_mean": mean_val
                })


def register_spline_outside_hooks(model) -> Tuple[List[torch.utils.hooks.RemovableHandle], OutsideStats]:
    """
    Registers forward hooks on modules whose class name contains 'spline' (case-insensitive).
    Each hook measures the fraction of pre-activations outside [-x_limit, x_limit].
    Returns (handles, stats) so callers can remove handles and dump stats.
    """
    stats = OutsideStats()
    handles = []

    def make_hook(idx: int, module):
        x_limit = getattr(module, "x_limit", None)
        if x_limit is None:
            # fallback to 2 if not found
            x_limit = 2

        def hook(mod, inp, out):
            # inp is a tuple (x,), use the tensor in 0
            x = inp[0]
            with torch.no_grad():
                outside = (x.abs() > x_limit).float().mean().item()
            stats.add(idx, outside)

        return hook

    for idx, m in enumerate(model.modules()):
        name = m.__class__.__name__.lower()
        if "spline" in name:
            stats.layer_names.append(m.__class__.__name__)
            handles.append(m.register_forward_hook(make_hook(len(stats.layer_names)-1, m)))

    return handles, stats
