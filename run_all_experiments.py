import subprocess
from pathlib import Path
import time

CONFIG_DIR = Path("experiments/configs")
PLOT_DIR = Path("results/plots")
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

EXPERIMENT_TIMEOUT = 2 * 60 * 60    # 2 Stunden


def run_experiment(config_path):
    return True
    experiment_name = config_path.stem
    log_file = LOG_DIR / f"{experiment_name}.log"
    print(f"\n🔁 Starte Experiment: {experiment_name} → logge nach {log_file}")
    
    try:
        with open(log_file, "w") as logfile:
            result = subprocess.run(
                ["python", "experiments/run_experiment.py", "--config", str(config_path)],
                stdout=logfile,
                stderr=None,
                timeout=EXPERIMENT_TIMEOUT
            )

        if result.returncode != 0:
            print(f"❌ Fehler: {experiment_name} endete mit Exit-Code {result.returncode}")
            return False

        return True

    except subprocess.TimeoutExpired:
        print(f"⏰ Timeout: {experiment_name} nach 2 Stunden abgebrochen")
        return False
    except Exception as e:
        print(f"❌ Ausnahme bei {experiment_name}: {e}")
        return False


def plot_metrics(csv_path):
    subprocess.run([
        "python", "results/plots/plot_metrics.py",
        "--csv", str(csv_path),
        "--save_dir", str(PLOT_DIR)
    ], check=True)


def plot_gradients(log_dir):
    subprocess.run([
        "python", "results/plots/plot_scalar_gradients.py",
        "--log_dir", str(log_dir),
        "--save_dir", str(PLOT_DIR)
    ], check=True)




def main():
    config_files = sorted(CONFIG_DIR.glob("*.yaml"))
    
    for config_path in config_files:
        experiment_name = config_path.stem
        
        success = run_experiment(config_path)
        if not success:
            continue
        
        print(f"📊 Generiere alle Plots für {experiment_name}")
        subprocess.run([
            "python", "results/plots/plot_all.py",
            "--name", experiment_name,
            "--save_dir", "results/plots/"
        ])

    print("\n✅ Alle Experimente abgeschlossen – erstelle Zusammenfassungstabelle ...")
    subprocess.run(["python", "summarize_results.py"], check=True)


if __name__ == "__main__":
    main()
