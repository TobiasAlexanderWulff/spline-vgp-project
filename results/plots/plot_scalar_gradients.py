import argparse
from pathlib import Path
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import numpy as np


def extract_scalar_gradient(log_dir):
    ea = EventAccumulator(log_dir)
    ea.Reload()

    scalar_tags = [tag for tag in ea.Tags()["scalars"] if tag.startswith("gradients/norm_")]
    gradients = {"weight": {}, "bias": {}}

    for tag in scalar_tags:
        events = ea.Scalars(tag)
        steps = [e.step for e in events]
        values = [e.value for e in events]
        name = tag.split("/")[-1]
        key = "weight" if "weight" in name else "bias"
        gradients[key][name] = (steps, values)

    return gradients


def plot_gradients(gradient_data, experiment_name, save_dir=None, mark_outliers=True, outlier_threshold=1.0):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    fig.suptitle(f"Mean |Gradient| (log) – {experiment_name}")

    for key, ax in zip(("weight", "bias"), (ax1, ax2)):
        for name, (steps, values) in gradient_data[key].items():
            steps = np.array(steps)
            values = np.array(values)

            values = np.clip(values, a_min=1e-10, a_max=None)  # für log
            ax.plot(steps, values, label=name)

            if mark_outliers:
                outlier_mask = values > outlier_threshold
                ax.scatter(steps[outlier_mask], values[outlier_mask], color="red", marker="x", s=15)

        ax.set_yscale("log")
        ax.set_ylabel(f"{key.capitalize()} Gradients (log)")
        ax.grid(True)
        ax.legend(fontsize=8, ncol=2)

    ax2.set_xlabel("Epoch")

    if save_dir:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        out_path = Path(save_dir) / f"{experiment_name}_scalar_gradients.png"
        plt.savefig(out_path, dpi=300)
        print(f"[INFO] Plot gespeichert unter {out_path}")

    plt.tight_layout()
    #plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", type=str, required=True, help="Pfad zum TensorBoard-Logordner")
    parser.add_argument("--save_dir", type=str, default=None, help="Speicherort für PNG")
    parser.add_argument("--no_outliers", action="store_true", help="Keine Marker für Ausreißer anzeigen")
    parser.add_argument("--outlier_threshold", type=float, default=1.0, help="Grenze für Ausreißer-Marker")
    args = parser.parse_args()

    experiment_name = Path(args.log_dir).name
    gradients = extract_scalar_gradient(args.log_dir)
    plot_gradients(
        gradients,
        experiment_name,
        save_dir=args.save_dir,
        mark_outliers=not args.no_outliers,
        outlier_threshold=args.outlier_threshold
    )
