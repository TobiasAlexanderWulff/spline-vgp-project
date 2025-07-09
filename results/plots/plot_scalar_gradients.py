import argparse
from pathlib import Path
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def extract_scalar_gradients(log_dir):
    ea = EventAccumulator(log_dir)
    ea.Reload()

    scalar_tags = [tag for tag in ea.Tags()["scalars"] if tag.startswith("grad_mean/")]

    if not scalar_tags:
        print(f"[WARN] Keine scalar-Gradienten-Tags in: {log_dir}")
        return {}

    gradients = {}
    for tag in scalar_tags:
        events = ea.Scalars(tag)
        steps = [e.step for e in events]
        values = [e.value for e in events]
        gradients[tag] = (steps, values)

    return gradients

def plot_gradients(gradient_curves, experiment_name, save_dir=None):
    if not gradient_curves:
        print(f"[ERROR] Keine Gradientenwerte zum Plotten für {experiment_name}.")
        return

    plt.figure(figsize=(10, 6))
    for tag, (steps, values) in gradient_curves.items():
        layer = tag.replace("grad_mean/", "")
        plt.plot(steps, values, label=layer)

    plt.title(f"Mean |Gradient| (scalar) – {experiment_name}")
    plt.xlabel("Epoch")
    plt.ylabel("Mean |Gradient|")
    plt.grid(True)
    plt.legend(loc="best", fontsize=7)

    if save_dir:
        save_path = Path(save_dir) / f"{experiment_name}_scalar_gradients.png"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300)
        print(f"[SUCCESS] Plot gespeichert unter: {save_path}")
    #plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", type=str, required=True)
    parser.add_argument("--save_dir", type=str, default=None)
    args = parser.parse_args()

    experiment = Path(args.log_dir).name
    curves = extract_scalar_gradients(args.log_dir)
    plot_gradients(curves, experiment, args.save_dir)

