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


def _parse_name_to_dataset_activation(name: str) -> tuple[str, str]:
    """
    Returns (dataset, activation) from a log directory name.
    Expected patterns like 'relu_cifar10', 'spline_fashionmnist', 'sigmoid_tinyimagenet'.
    Falls back to 'unknown' if parsing fails.
    """
    parts = name.lower().split("_")
    if len(parts) >= 2:
        activation, dataset = parts[0], parts[1]
    else:
        activation, dataset = "unknown", "unknown"
    return dataset, activation


def _epoch_view(df: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse the metrics dataframe to one row per epoch so that duplicated rows
    (caused by parameter-wise logging) do not distort the curves.
    """
    cols = ["loss_train", "loss_val", "acc_train", "acc_val"]
    g = df.groupby("epoch", as_index=False)[cols].first()
    return g.sort_values("epoch")


def _pretty_dataset_name(ds: str) -> str:
    ds = ds.lower()
    if ds == "fashionmnist":
        return "FashionMNIST"
    if ds == "cifar10":
        return "CIFAR-10"
    if ds in ("tinyimagenet", "tinyimagenet-200", "tiny_image_net", "tiny-image-net"):
        return "Tiny ImageNet-200"
    return ds


def _pretty_act_name(act: str) -> str:
    a = act.lower()
    if a == "relu":
        return "ReLU"
    if a == "spline":
        return "Spline"
    if a == "sigmoid":
        return "Sigmoid"
    return act


def plot_combined_for_all_datasets(dfs: dict[str, pd.DataFrame], evs: dict, result_path: Path) -> None:
    """
    Create for each dataset two figures:
      1) Accuracy (train = solid, val = dashed) for all activations
      2) Log10-Loss (train = solid, val = dashed) for all activations
    """
    # group by dataset
    grouped: dict[str, dict[str, pd.DataFrame]] = {}
    for name, df in dfs.items():
        dataset, activation = _parse_name_to_dataset_activation(name)
        grouped.setdefault(dataset, {})[activation] = _epoch_view(df)

    color_map = {"relu": "C0", "spline": "C1", "sigmoid": "C2"}
    order = ["relu", "spline", "sigmoid"]

    outdir = result_path / "plots" / "combined"
    outdir.mkdir(parents=True, exist_ok=True)

    for dataset, act_dict in grouped.items():
        # ACCURACY
        fig_acc, ax_acc = plt.subplots(figsize=(10, 5), constrained_layout=True)
        title = f"{_pretty_dataset_name(dataset)} – Accuracy (alle Aktivierungen)"
        ax_acc.set_title(title)
        for act in [a for a in order if a in act_dict]:
            dfe = act_dict[act]
            label_base = _pretty_act_name(act)
            color = color_map.get(act, None)
            ax_acc.plot(dfe["epoch"], dfe["acc_train"], label=f"{label_base} Train", linestyle="-", color=color)
            ax_acc.plot(dfe["epoch"], dfe["acc_val"],   label=f"{label_base} Val",   linestyle="--", color=color)
        ax_acc.set_xlabel("Epoch")
        ax_acc.set_ylabel("Accuracy")
        if "acc_min" in evs and "acc_max" in evs:
            ax_acc.set_ylim(evs["acc_min"], evs["acc_max"])
        ax_acc.grid(True, alpha=0.3)
        ax_acc.legend(ncols=3, frameon=True)
        fig_acc.savefig(outdir / f"{dataset}_combined_accuracy.png", dpi=300)
        plt.close(fig_acc)

        # LOG10 LOSS
        fig_loss, ax_loss = plt.subplots(figsize=(10, 5), constrained_layout=True)
        title = f"{_pretty_dataset_name(dataset)} – Log10-Loss (alle Aktivierungen)"
        ax_loss.set_title(title)
        for act in [a for a in order if a in act_dict]:
            dfe = act_dict[act]
            label_base = _pretty_act_name(act)
            color = color_map.get(act, None)
            tr = np.log10(dfe["loss_train"].to_numpy())
            va = np.log10(dfe["loss_val"].to_numpy())
            ax_loss.plot(dfe["epoch"], tr, label=f"{label_base} Train", linestyle="-",  color=color)
            ax_loss.plot(dfe["epoch"], va, label=f"{label_base} Val",   linestyle="--", color=color)
        ax_loss.set_xlabel("Epoch")
        ax_loss.set_ylabel("Log10-Loss")
        if "loss_min" in evs and "loss_max" in evs:
            ax_loss.set_ylim(evs["loss_min"], evs["loss_max"])
        ax_loss.grid(True, alpha=0.3)
        ax_loss.legend(ncols=3, frameon=True)
        fig_loss.savefig(outdir / f"{dataset}_combined_log10loss.png", dpi=300)
        plt.close(fig_loss)


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


def plot_overall_heatmap(name: str, df: pd.DataFrame,evs: dict, result_path: Path):
    """
    Zeichnet eine Heatmap pro Experiment, die die Gradienten-Verläufe pro Epoche und Schicht
    zusammenfasst. Weight und Bias werden dabei je Schicht/ Epoche gemeinsam (Mittelwert)
    aggregiert. Es wird nur 'mean_abs_gradient' abgebildet (kein Norm-Plot).

    Args:
        name (str): Experimentname (= Log-Verzeichnisname).
        df (pd.DataFrame): Metrics-DataFrame des Experiments (eingelesen aus metrics.csv).
        result_path (Path): Basis-Ausgabepfad (z. B. RESULTS_PATH oder SMOKE_RESULTS_PATH).
    """
    df_grad = df[df["parameter"].str.contains("weight|bias", regex=True)].copy()

    # Layer-Nummer aus Parameternamen (z.B. 'model.2.weight') bestimmen und auf Linears abbilden
    # In diesem Projekt sind Linears bei Indizes 0,2,4,... -> Layer = index // 2
    df_grad["layer"] = df_grad["parameter"].map(lambda p: extract_layer_number(p) // 2)

    # Mittelwert über Weight & Bias je (layer, epoch)
    df_layer_epoch = (
        df_grad.groupby(["layer", "epoch"], as_index=False)["mean_abs_gradient"]
        .mean()
        .sort_values(["layer", "epoch"])
    )

    # Pivot: Zeilen = Layer, Spalten = Epoche
    pivot = df_layer_epoch.pivot(index="layer", columns="epoch", values="mean_abs_gradient")
    pivot = pivot.sort_index(axis=0)  # Layer aufsteigend
    pivot = pivot.reindex(sorted(pivot.columns), axis=1)  # Epochen aufsteigend

    #epsilon = 1e-24
    #data = np.log10(np.maximum(pivot.values, epsilon))
    data = np.log10(pivot.values)

    # Plot
    fig_height = max(4, pivot.shape[0] * 0.45)
    fig, ax = plt.subplots(figsize=(12, fig_height))

    title = name.replace("_", " ").title()
    ax = sns.heatmap(
        data,
        cmap="Spectral",
        ax=ax,
        vmin=evs["overall_min"],
        vmax=evs["overall_max"],
        cbar_kws={"label": r"log$_{10}$ mean |grad|"},
        xticklabels=10
    )

    ax.set_title(f"{title} – Overall Gradient Heatmap (W + B, abs-mean per layer/epoch)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Layer")
    #ax.set_yticks(np.arange(pivot.shape[0]) + 0.5)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    #ax.set_xticks(np.arange(pivot.shape[1]) + 0.5)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)

    fig.tight_layout()

    # Speichern & schließen 
    save_path = result_path / "plots" / "overall_heatmaps" / f"{name}_overall_gradient_heatmap.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=300)
    plt.close(fig)

    print(f'[SUCCESS] Plot "Overall Gradient Heatmap" of {name} was saved successfully.')



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
    
    #epsilon = 1e-24
    #data_w_mean_abs = np.log10(np.maximum(pivot_w_mean_abs.values, epsilon))
    #data_b_mean_abs = np.log10(np.maximum(pivot_b_mean_abs.values, epsilon))
    #data_w_norm = np.log10(np.maximum(pivot_w_norm.values, epsilon))
    #data_b_norm = np.log10(np.maximum(pivot_b_norm.values, epsilon))
    data_w_mean_abs = np.log10(pivot_w_mean_abs.values)
    data_b_mean_abs = np.log10(pivot_b_mean_abs.values)
    data_w_norm = np.log10(pivot_w_norm.values)
    data_b_norm = np.log10(pivot_b_norm.values)
    
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
    axs[0, 0].set_title(f"{title} - Weight Log10 |Gradient| Mean")
    axs[0, 0].set_xlabel("Epoch")
    axs[0, 0].set_ylabel("Layer")
    axs[0, 0].set_yticklabels(labels_layers, rotation=0)
    
    sns.heatmap(
        data_b_mean_abs,
        cmap="Spectral",
        ax=axs[0, 1],
        vmin=evs["mean_min"],
        vmax=evs["mean_max"],
        cbar_kws={"label": r"log$_{10}$ mean |grad|"}
    )
    axs[0, 1].set_title(f"{title} - Bias Log10 |Gradient| Mean")
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
    axs[1, 0].set_title(f"{title} - Weight Log10 Gradient Norm")
    axs[1, 0].set_xlabel("Epoch")
    axs[1, 0].set_ylabel("Layer")
    axs[1, 0].set_yticklabels(labels_layers, rotation=0)
    
    sns.heatmap(
        data_b_norm,
        cmap="Spectral",
        ax=axs[1, 1],
        vmin=evs["norm_min"],
        vmax=evs["norm_max"],
        cbar_kws={"label": r"log$_{10}$ Norm"}
    )
    axs[1, 1].set_title(f"{title} - Bias Log10 Gradient Norm")
    axs[1, 1].set_xlabel("Epoch")
    axs[1, 1].set_ylabel("Layer")
    axs[1, 1].set_yticklabels(labels_layers, rotation=0)
    
    fig.tight_layout()
    
    save_path = result_path / "plots" / "gradient_heatmaps" / f"{name}_gradient_heatmap.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=300)
    plt.close(fig)
    
    print(f'[SUCCESS] Plot "Gradient Heatmap" of {name} was saved successfully.')



def plot_acc_and_loss(name: str, df: pd.DataFrame, evs: dict, result_path: Path):
    df = df.drop_duplicates(subset="epoch", keep="first", ignore_index=True)
        
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    
    title = name.replace("_", " ").title()
    
    axs[0].set_title(f"{title} - Log10-Loss")
    axs[0].plot(df["epoch"], np.log10(df["loss_train"]), label="Train Loss")
    axs[0].plot(df["epoch"], np.log10(df["loss_val"]), label="Val Loss")
    axs[0].set_xlabel("Epoch")
    axs[0].set_ylabel("Log10-Loss")
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
    
    print(f'[SUCCESS] Plot "Loss and Accuracy" of {name} was saved successfully.')


def plot_train_times(dfs: dict, result_path: Path, smoke: bool):
    if smoke:
        df_train_times = pd.DataFrame(
            [(name.split("_")[1], 
              name.split("_")[0],
              df.iloc[-1]["train_time"], # last entry of "train_time" column
              int(name.split("_")[-1]),
            ) for name, df in sorted(dfs.items())], 
            columns=["dataset", "activation", "train_time", "batch_size"]
    )
    else:
        df_train_times = pd.DataFrame(
            [(name.split("_")[1], 
              name.split("_")[0], 
              df.iloc[-1]["train_time"] # last entry of "train_time" column
            ) for name, df in sorted(dfs.items())], 
            columns=["dataset", "activation", "train_time"]
        )

    fig = plt.figure(figsize=(8, 5))
    
    plt.title("Train Times")
    
    if smoke:
        sns.barplot(
            x="batch_size",
            y="train_time",
            hue="activation",
            data=df_train_times
        )
    else:
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
    
    print(f'[SUCCESS] Plot "Train Times" was saved successfully.')


def create_ev_dict(dfs: dict) -> dict:
    """
    Create a dictionary of all **extrem values (evs)** over every dataframe.
    Contains overall min- and max-values.
    """
    evs = {} # Extrem Values (evs)
    
    calc_config = {
        "mean": ("mean_abs_gradient", np.log10),
        "norm": ("norm_gradient", np.log10),
        "overall": ("mean_abs_gradient", np.log10),
        "loss": (["loss_train", "loss_val"], np.log10),
        "acc": (["acc_train", "acc_val"], None)
    }
    
    for prefix, (cols, transform) in calc_config.items():
        # extract all values of cols into lists
        if prefix == "overall":
            all_vals = []
            for df in dfs.values():
                # caluculate bias-weight mean values
                df["layer"] = df["parameter"].map(lambda p: extract_layer_number(p) // 2)
                df = df.groupby(["layer", "epoch"], as_index=False)["mean_abs_gradient"].mean()
                all_vals.append(df.pivot(index="layer", columns="epoch", values="mean_abs_gradient").values)
        else:
            all_vals = [df[cols].values for df in dfs.values()]
        
        # save all min and max values into dict
        evs[f"{prefix}_min"] = np.min([np.min(arr) for arr in all_vals])
        evs[f"{prefix}_max"] = np.max([np.max(arr) for arr in all_vals])
        
        # apply transforms
        if transform:
            evs[f"{prefix}_min"] = transform(evs[f"{prefix}_min"])
            evs[f"{prefix}_max"] = transform(evs[f"{prefix}_max"])

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
        plot_overall_heatmap(name, df, evs, result_path)
    plot_train_times(dfs, result_path, args.smoke)
    
    plot_combined_for_all_datasets(dfs, evs, result_path)
    
    plt.close("all")
    

if __name__ == "__main__":
    main()
