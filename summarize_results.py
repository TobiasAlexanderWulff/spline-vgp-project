import pandas as pd
from pathlib import Path
import yaml

CONFIG_DIR = Path("experiments/configs")
LOG_ROOT = Path("logs/tensorboard")
OUTPUT_CSV = Path("results/summary/experiment_results.csv")
OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

rows = []

for config_path in sorted(CONFIG_DIR.glob("*.yaml")):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    name = config["experiment_name"]
    dataset = config["dataset"]
    activation = config["activation"]

    log_dir = LOG_ROOT / name
    csv_file = log_dir / "metrics.csv"
    time_file = log_dir / "training_duration.txt"

    if not csv_file.exists():
        print(f"⚠️ Überspringe {name} – keine metrics.csv gefunden")
        continue

    df = pd.read_csv(csv_file)
    if len(df) == 0:
        continue
    last_row = df.iloc[-1]

    duration_str = ""
    if time_file.exists():
        duration_str = time_file.read_text().strip().replace(" seconds", "")
        if duration_str.isdigit():
            total_sec = int(duration_str)
            duration_str = f"{total_sec // 60:02}:{total_sec % 60:02}"

    row = {
        "experiment": name,
        "dataset": dataset,
        "activation": activation,
        "train_acc": round(last_row["train_accuracy"], 4),
        "val_acc": round(last_row.get("val_accuracy", 0.0), 4),
        "train_loss": round(last_row["train_loss"], 4),
        "val_loss": round(last_row.get("val_loss", 0.0), 4),
        "train_time": duration_str
    }

    rows.append(row)

# In Tabelle schreiben
df_out = pd.DataFrame(rows)
df_out.to_csv(OUTPUT_CSV, index=False)
print(f"\n📄 Ergebnisse gespeichert unter: {OUTPUT_CSV}")