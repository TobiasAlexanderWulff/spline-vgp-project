import argparse
from pathlib import Path
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import numpy as np
import os

def extract_gradient_curves(log_dir):
    """Lädt alle Gradientenkurven aus einem TensorBoard-Log-Verzeichnis."""
    ea = EventAccumulator(log_dir)
    ea.Reload()
    
    gradient_tags = [tag for tag in ea.Tags()["histograms"] if tag.startswith("gradients/")]
    
    if not gradient_tags:
        print(f"[WARN]  Keine Gradienten-Tags im TensorBoard-Log gefunden: {log_dir}")
        print("[INFO]  Stelle sicher, dass in trainer.py folgendes aktiv ist:\n"
              "    writer.add_histogram(f\"gradients/{{name}}\", param.grad, epoch)")
        return {}
    
    gradient_curves = {}
    for tag in gradient_tags:
        events = ea.Histograms(tag)
       
        steps = []
        means = []
        for event in events:
            step = event.step
            bucket = event.histogram_value
            
            # Wenn Buckets leer sind, NaN einfügen
            if len(bucket.bucket_limit) == 0 or len(bucket.bucket) == 0:
                mean_val = float("nan")
            else:
                total_weight = sum(bucket.bucket)
                if total_weight == 0:
                    mean_val = float("nan")
                else:
                    weighted = [
                        abs(val) * count
                        for val, count in zip(bucket.bucket_limit, bucket.bucket)
                    ]
                    mean_val = sum(weighted) / total_weight
                    
            steps.append(step)
            means.append(mean_val)
       
        gradient_curves[tag] = (steps, means)
    
    return gradient_curves

def plot_gradient_curves(gradient_curves, experiment_name, save_dir=None):
    """Erstellt einen Plot aller Gradientenverläufe."""
    if not gradient_curves:
        print(f"[ERROR] Keine Gradienten zum Plotten für {experiment_name}.")
        return
    
    plt.figure(figsize=(10, 6))
    for tag, (steps, values) in gradient_curves.items():
        layer_name = tag.replace("gradients/", "")
        plt.plot(steps, values, label=layer_name)
    
    plt.title(f"Gradientenverlauf - {experiment_name}")
    plt.xlabel("Epoch")
    plt.ylabel("Mean |Gradient|")
    plt.grid(True)
    plt.legend(loc="best", fontsize=8)
    
    if save_dir:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        out_path = Path(save_dir) / f"{experiment_name}_gradients.png"
        plt.savefig(out_path, dpi=300)
        print(f"[INFO] Plot gespeichert unter {out_path}")
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", type=str, required=True, help="Pfad zum TensorBoard-Logordner (z. B. logs/tensorboard/spline_cifar10)")
    parser.add_argument("--save_dir", type=str, default=None, help="Optionaler Speicherort für PNG")
    args = parser.parse_args()
    
    experiment_name = Path(args.log_dir).name
    gradient_curves = extract_gradient_curves(args.log_dir)
    plot_gradient_curves(gradient_curves, experiment_name, args.save_dir)
