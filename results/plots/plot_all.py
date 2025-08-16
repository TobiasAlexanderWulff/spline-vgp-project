import seaborn as sns
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import argparse
from pathlib import Path

matplotlib.use("Agg")

LOG_PATH = Path("logs/")
SMOKE_LOG_PATH = Path("logs_smoke/")
RESULTS_PATH = Path("results/")
SMOKE_RESULTS_PATH = Path("results/smoke/")


def extract_layer_number(param_name: str) -> int:
    """
    Extract the layer number of a parameter name like `model.2.weight`.
    If the format is incorrect `0` is returned.
    """
    parts = param_name.split(".")
    try:
        return int(parts[1])
    except (IndexError, ValueError):
        return 0


def plot_heatmap(name: str, df: pd.DataFrame, evs: dict, result_path: Path):
    """
    Plot heatmaps for `mean_abs_gradient` and `norm_gradient` values. *Weight* and *Bias* data is plotted on different subplots.
    """
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
    
    epsilon = 1e-12
    data_w_mean_abs = np.log10(np.maximum(pivot_w_mean_abs.values, epsilon))
    data_b_mean_abs = np.log10(np.maximum(pivot_b_mean_abs.values, epsilon))
    data_w_norm = pivot_w_norm.values
    data_b_norm = pivot_b_norm.values
    
    fig, axs = plt.subplots(
        2, 2,
        figsize=(18, max(8, len(pivot_w_mean_abs) * 0.4)),
        sharey=True
    )
    
    title = name.replace("_", " ").title()
    labels_layers = list(map(lambda name: extract_layer_number(name) // 2, pivot_w_mean_abs.index))
        
    sns.heatmap(
        data_w_mean_abs,
        cmap="Spectral",
        ax=axs[0, 0],
        vmin=evs["mean_min"],
        vmax=evs["mean_max"],
        cbar=None
    )
    axs[0, 0].set_title(f"{title} - Weight Log10-|Gradient| Mean")
    axs[0, 0].set_xlabel("Epoch")
    axs[0, 0].set_ylabel("Layer")
    axs[0, 0].set_yticklabels(labels_layers, rotation=0)
    
    sns.heatmap(
        data_b_mean_abs,
        cmap="Spectral",
        ax=axs[0, 1],
        vmin=evs["mean_min"],
        vmax=evs["mean_max"]
    )
    axs[0, 1].set_title(f"{title} - Bias Log10-|Gradient| Mean")
    axs[0, 1].set_xlabel("Epoch")
    axs[0, 1].set_ylabel("Layer")
    axs[0, 1].set_yticklabels(labels_layers, rotation=0)
    
    sns.heatmap(
        data_w_norm,
        cmap="Spectral",
        ax=axs[1, 0],
        vmin=evs["norm_min"],
        vmax=evs["norm_max"],
        cbar=None
    )
    axs[1, 0].set_title(f"{title} - Weight Gradient Norm")
    axs[1, 0].set_xlabel("Epoch")
    axs[1, 0].set_ylabel("Layer")
    axs[1, 0].set_yticklabels(labels_layers, rotation=0)
    
    sns.heatmap(
        data_b_norm,
        cmap="Spectral",
        ax=axs[1, 1],
        vmin=evs["norm_min"],
        vmax=evs["norm_max"]
    )
    axs[1, 1].set_title(f"{title} - Bias Gradient Norm")
    axs[1, 1].set_xlabel("Epoch")
    axs[1, 1].set_ylabel("Layer")
    axs[1, 1].set_yticklabels(labels_layers, rotation=0)
    
    fig.tight_layout()
    
    save_path = result_path / "plots" / "gradient_heatmaps" / f"{name}_gradient_heatmap.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=300)
    plt.close(fig)
    
    print(f'✅ Plot "Gradient Heatmap" of {name} was saved successfully.')



def plot_acc_and_loss(name: str, df: pd.DataFrame, evs: dict, result_path: Path):
    df = df.drop_duplicates(subset="epoch", keep="first", ignore_index=True)
        
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    
    title = name.replace("_", " ").title()
    
    axs[0].set_title(f"{title} - Loss")
    axs[0].plot(df["epoch"], df["loss_train"], label="Train Loss")
    axs[0].plot(df["epoch"], df["loss_val"], label="Val Loss")
    axs[0].set_xlabel("Epoch")
    axs[0].set_ylabel("Loss")
    axs[0].set_ylim(evs["loss_min"], evs["loss_max"])
    axs[0].legend()
    axs[0].grid(True)
    
    axs[1].set_title(f"{title} - Accuracy")
    axs[1].plot(df["epoch"], df["acc_train"], label="Train Accuracy")
    axs[1].plot(df["epoch"], df["acc_val"], label="Val Accuracy")
    axs[1].set_xlabel("Epoch")
    axs[1].set_ylabel("Accuracy")
    axs[1].set_ylim(evs["acc_min"], evs["acc_max"])
    axs[1].legend()
    axs[1].grid(True)
    
    fig.tight_layout()
    
    save_path = result_path / "plots" / "loss_acc" / f"{name}_loss_acc.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=300)
    plt.close(fig)
    
    print(f'✅ Plot "Loss and Accuracy" of {name} was saved successfully.')


def plot_train_times(dfs: dict, result_path: Path):
    df_train_times = pd.DataFrame(
        [(name.split("_")[1], 
          name.split("_")[0], 
          df.iloc[-1]["train_time"] # last entry of "train_time" column
        ) for name, df in sorted(dfs.items())], 
        columns=["dataset", "activation", "train_time"]
    )

    fig = plt.figure(figsize=(8, 5))
    
    plt.title("Train Times")
    sns.barplot(
        x="dataset",
        y="train_time",
        hue="activation",
        data=df_train_times
    )
    plt.xlabel(None)
    plt.ylabel("Time (s)", rotation=0)
    
    fig.tight_layout()
    
    save_path = result_path / "plots" / "train_times.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=300)
    plt.close(fig)
    
    print(f'✅ Plot "Train Times" was saved successfully.')

def create_ev_dict(dfs: dict) -> dict:
    """
    Create a dictionary of all **extrem values (evs)** over every dataframe.
    Contains overall min- and max-values.
    """
    evs = {} # Extrem Values (evs)
    
    calc_config = {
        "mean": ("mean_abs_gradient", np.log10),
        "norm": ("norm_gradient", None),
        "loss": (["loss_train", "loss_val"], None),
        "acc": (["acc_train", "acc_val"], None)
    }
    
    for prefix, (cols, transform) in calc_config.items():
        # extract all values of cols into lists
        all_vals = [df[cols].values for df in dfs.values()]
        
        # save all min and max values into dict
        evs[f"{prefix}_min"] = np.min([np.min(arr) for arr in all_vals])
        evs[f"{prefix}_max"] = np.max([np.max(arr) for arr in all_vals])
        
        # apply transforms
        if transform:
            evs[f"{prefix}_min"] = transform(evs[f"{prefix}_min"])
            evs[f"{prefix}_max"] = transform(evs[f"{prefix}_max"])

    vals = np.concatenate([df["mean_abs_gradient"].to_numpy() for df in dfs.values()])
    vals = np.clip(vals, 1e-12, None)
    evs["mean_min"] = np.log10(vals.min())
    evs["mean_max"] = np.log10(vals.max())

    return evs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="Set flag to plot smoke runs instead")
    args = parser.parse_args()
    
    if args.smoke:
        dfs = {dir.name: pd.read_csv(dir / "metrics.csv") for dir in SMOKE_LOG_PATH.iterdir()}
        result_path = SMOKE_RESULTS_PATH
    else:
        dfs = {dir.name: pd.read_csv(dir / "metrics.csv") for dir in LOG_PATH.iterdir()}
        result_path = RESULTS_PATH
        
    evs = create_ev_dict(dfs) # Extrem Values (evs)
    
    for name, df in dfs.items():
        plot_acc_and_loss(name, df, evs, result_path)
        plot_heatmap(name, df, evs, result_path)
    plot_train_times(dfs, result_path)
    
    plt.close("all")
    

if __name__ == "__main__":
    main()
