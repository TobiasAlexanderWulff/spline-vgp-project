
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import argparse
from pathlib import Path

# keep user's non-interactive backend
matplotlib.use("Agg")

LOG_PATH = Path("logs/")
RESULTS_PATH = Path("results/")

def extract_layer_number(param_name: str) -> int:
    parts = param_name.split(".")
    try:
        return int(parts[1])
    except (IndexError, ValueError):
        return 0

# ---------- NEW: diagnostics helpers ----------

def load_anomaly_epochs(exp_dir: Path) -> list[int]:
    csv_path = exp_dir / "diag" / "anomalies.csv"
    if not csv_path.exists():
        return []
    df = pd.read_csv(csv_path)
    if df.empty:
        return []
    # unique epochs with at least one anomaly row
    return sorted(int(e) for e in pd.unique(df["epoch"]))


def plot_outside_heatmap(name: str, exp_dir: Path, result_path: Path):
    csv_path = exp_dir / "diag" / "outside.csv"
    if not csv_path.exists():
        return False

    df = pd.read_csv(csv_path)
    if df.empty:
        return False

    # pivot to layer x epoch
    p = df.pivot(index="layer_idx", columns="epoch", values="outside_fraction_mean").sort_index()
    fig = plt.figure(figsize=(18, max(6, p.shape[0] * 0.4)))
    ax = fig.add_subplot(1,1,1)
    im = ax.imshow(p.values, aspect="auto", origin="upper")
    ax.set_title(f"{name.replace('_',' ').title()} - Spline Outside Fraction (|x|>x_limit)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Layer")
    ax.set_yticks(np.arange(p.shape[0]))
    ax.set_yticklabels(p.index.tolist())
    ax.set_xticks(np.arange(p.shape[1])[::max(1, p.shape[1]//16)])
    ax.set_xticklabels(list(p.columns.tolist())[::max(1, p.shape[1]//16)], rotation=0)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()

    save_path = result_path / "plots" / "diagnostics" / f"{name}_outside_heatmap.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f'✅ Plot "Outside Heatmap" of {name} saved to {save_path}')
    return True


def plot_anomaly_overview(name: str, exp_dir: Path, result_path: Path):
    csv_path = exp_dir / "diag" / "anomalies.csv"
    if not csv_path.exists():
        return False

    df = pd.read_csv(csv_path)
    if df.empty:
        return False

    grp = df.groupby("epoch").agg(
        count=("batch_idx","count"),
        max_gnorm=("grad_norm","max"),
        max_loss=("loss","max"),
    ).reset_index()

    fig = plt.figure(figsize=(12,5))
    ax = fig.add_subplot(1,1,1)
    ax.bar(grp["epoch"], grp["count"], width=0.8, label="# anomalies (batches)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Count")
    ax2 = ax.twinx()
    ax2.plot(grp["epoch"], grp["max_gnorm"], marker="o", label="Max grad-norm")
    ax2.set_ylabel("Max grad-norm")
    ax.set_title(f"{name.replace('_',' ').title()} - Anomaly Overview")
    # combine legends
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1+h2, l1+l2, loc="upper right")
    ax.grid(True, axis="y", linestyle=":")
    fig.tight_layout()

    save_path = result_path / "plots" / "diagnostics" / f"{name}_anomalies.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f'✅ Plot "Anomalies" of {name} saved to {save_path}')
    return True

# ---------- Original plots with vertical anomaly markers ----------

def plot_heatmap_with_markers(name: str, df: pd.DataFrame, evs: dict, result_path: Path, anomaly_epochs: list[int]):
    # separate weights/bias
    df_w = df[df["parameter"].str.contains("weight")].copy()
    df_b = df[df["parameter"].str.contains("bias")].copy()

    pivot_w_mean_abs = df_w.pivot(index="parameter", columns="epoch", values="mean_abs_gradient")
    pivot_b_mean_abs = df_b.pivot(index="parameter", columns="epoch", values="mean_abs_gradient")
    pivot_w_norm = df_w.pivot(index="parameter", columns="epoch", values="norm_gradient")
    pivot_b_norm = df_b.pivot(index="parameter", columns="epoch", values="norm_gradient")

    sorted_w = sorted(pivot_w_mean_abs.index, key=extract_layer_number)
    sorted_b = sorted(pivot_b_mean_abs.index, key=extract_layer_number)
    pivot_w_mean_abs = pivot_w_mean_abs.loc[sorted_w]
    pivot_b_mean_abs = pivot_b_mean_abs.loc[sorted_b]
    pivot_w_norm = pivot_w_norm.loc[sorted_w]
    pivot_b_norm = pivot_b_norm.loc[sorted_b]

    data_w_mean_abs = np.log10(pivot_w_mean_abs.values)
    data_b_mean_abs = np.log10(pivot_b_mean_abs.values)
    data_w_norm = np.log10(pivot_w_norm.values)
    data_b_norm = np.log10(pivot_b_norm.values)

    fig, axs = plt.subplots(2, 2, figsize=(18, max(8, len(pivot_w_mean_abs) * 0.4)), sharey=True)
    title = name.replace("_", " ").title()
    labels_layers = list(map(lambda n: extract_layer_number(n) // 2, pivot_w_mean_abs.index))

    # helper to draw a heatmap and anomaly lines
    def draw(ax, data, vmin, vmax, epoch_indexer):
        im = ax.imshow(data, aspect="auto", origin="upper", vmin=vmin, vmax=vmax)
        # vertical lines at anomaly epochs
        for e in anomaly_epochs:
            if e in epoch_indexer:
                ax.axvline(epoch_indexer[e]-0.5, linewidth=0.8, linestyle="--")
        return im

    epoch_indexer = {int(e):i for i,e in enumerate(pivot_w_mean_abs.columns.tolist())}

    im = draw(axs[0,0], data_w_mean_abs, evs["mean_min"], evs["mean_max"], epoch_indexer)
    axs[0,0].set_title(f"{title} - Weight Log10 |Gradient| Mean"); axs[0,0].set_xlabel("Epoch"); axs[0,0].set_ylabel("Layer")
    axs[0,0].set_yticks(np.arange(len(labels_layers))); axs[0,0].set_yticklabels(labels_layers)

    im = draw(axs[0,1], data_b_mean_abs, evs["mean_min"], evs["mean_max"], epoch_indexer)
    axs[0,1].set_title(f"{title} - Bias Log10 |Gradient| Mean"); axs[0,1].set_xlabel("Epoch"); axs[0,1].set_ylabel("Layer")
    axs[0,1].set_yticks(np.arange(len(labels_layers))); axs[0,1].set_yticklabels(labels_layers)
    fig.colorbar(im, ax=axs[0,1])

    im = draw(axs[1,0], data_w_norm, evs["norm_min"], evs["norm_max"], epoch_indexer)
    axs[1,0].set_title(f"{title} - Weight Gradient Norm"); axs[1,0].set_xlabel("Epoch"); axs[1,0].set_ylabel("Layer")
    axs[1,0].set_yticks(np.arange(len(labels_layers))); axs[1,0].set_yticklabels(labels_layers)

    im = draw(axs[1,1], data_b_norm, evs["norm_min"], evs["norm_max"], epoch_indexer)
    axs[1,1].set_title(f"{title} - Bias Gradient Norm"); axs[1,1].set_xlabel("Epoch"); axs[1,1].set_ylabel("Layer")
    axs[1,1].set_yticks(np.arange(len(labels_layers))); axs[1,1].set_yticklabels(labels_layers)
    fig.colorbar(im, ax=axs[1,1])

    fig.tight_layout()
    save_path = result_path / "plots" / "gradient_heatmaps" / f"{name}_gradient_heatmap_diag.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f'✅ Plot "Gradient Heatmap (diag)" of {name} saved to {save_path}')


def plot_acc_and_loss_with_markers(name: str, df: pd.DataFrame, evs: dict, result_path: Path, anomaly_epochs: list[int]):
    df = df.drop_duplicates(subset="epoch", keep="first", ignore_index=True)
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    title = name.replace("_", " ").title()

    axs[0].set_title(f"{title} - Loss")
    axs[0].plot(df["epoch"], np.log10(df["loss_train"]), label="Log10 Train Loss")
    axs[0].plot(df["epoch"], np.log10(df["loss_val"]), label="Log10 Val Loss")
    for e in anomaly_epochs:
        axs[0].axvline(e, linestyle="--", linewidth=0.8)
    axs[0].set_xlabel("Epoch"); axs[0].set_ylabel("Loss"); axs[0].set_ylim(evs["loss_min"], evs["loss_max"]); axs[0].legend(); axs[0].grid(True)

    axs[1].set_title(f"{title} - Accuracy")
    axs[1].plot(df["epoch"], df["acc_train"], label="Train Accuracy")
    axs[1].plot(df["epoch"], df["acc_val"], label="Val Accuracy")
    for e in anomaly_epochs:
        axs[1].axvline(e, linestyle="--", linewidth=0.8)
    axs[1].set_xlabel("Epoch"); axs[1].set_ylabel("Accuracy"); axs[1].set_ylim(evs["acc_min"], evs["acc_max"]); axs[1].legend(); axs[1].grid(True)

    fig.tight_layout()
    save_path = result_path / "plots" / "loss_acc" / f"{name}_loss_acc_diag.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f'✅ Plot "Loss/Acc (diag)" of {name} saved to {save_path}')


def create_ev_dict(dfs: dict) -> dict:
    evs = {}
    for prefix, (cols, transform) in {
        "mean": ("mean_abs_gradient", np.log10),
        "norm": ("norm_gradient", np.log10),
        "loss": (["loss_train", "loss_val"], np.log10),
        "acc": (["acc_train", "acc_val"], None)
    }.items():
        all_vals = [df[cols].values for df in dfs.values()]
        evs[f"{prefix}_min"] = np.min([np.min(arr) for arr in all_vals])
        evs[f"{prefix}_max"] = np.max([np.max(arr) for arr in all_vals])
        if transform:
            evs[f"{prefix}_min"] = transform(evs[f"{prefix}_min"])
            evs[f"{prefix}_max"] = transform(evs[f"{prefix}_max"])
    return evs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", type=str, default=None, help="Name of a single experiment dir under logs/ to plot")
    args = parser.parse_args()

    if args.exp is None:
        exps = [d for d in LOG_PATH.iterdir() if d.is_dir()]
    else:
        exps = [LOG_PATH/args.exp]
    dfs = {d.name: pd.read_csv(d/"metrics.csv") for d in exps}
    evs = create_ev_dict(dfs)

    for d in exps:
        name = d.name
        df = dfs[name]
        anomaly_epochs = load_anomaly_epochs(d)
        # overlay markers
        plot_acc_and_loss_with_markers(name, df, evs, RESULTS_PATH, anomaly_epochs)
        plot_heatmap_with_markers(name, df, evs, RESULTS_PATH, anomaly_epochs)
        # diagnostics-only visuals
        _ = plot_outside_heatmap(name, d, RESULTS_PATH)
        _ = plot_anomaly_overview(name, d, RESULTS_PATH)

    plt.close("all")


if __name__ == "__main__":
    main()
