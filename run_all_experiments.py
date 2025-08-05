import subprocess
import os
import shutil
from pathlib import Path


CONFIG_DIR = Path("experiments/configs")
PLOT_DIR = Path("results/plots")
LOG_DIR = Path("logs/")

EXPERIMENT_TIMEOUT = 3 * 60 * 60    # 3 Stunden


def _clear_log_dir(log_dir: str):
    """Clears the spezified log_dir."""
    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    os.makedirs(log_dir, exist_ok=True)


def run_experiment(config_path: str) -> bool:
    """Runs an experiment. Needs its config path."""
    experiment_name = config_path.stem
    experiment_log_dir = LOG_DIR / experiment_name
    _clear_log_dir(experiment_log_dir)
    experiment_log_dir.mkdir(parents=True, exist_ok=True)
    log_file = experiment_log_dir / "training.log"
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
        print(
            f"⏰ Timeout: {experiment_name} nach {EXPERIMENT_TIMEOUT // 3600} Stunden abgebrochen"
        )
        return False
    except Exception as e:
        print(f"❌ Ausnahme bei {experiment_name}: {e}")
        return False


def main():
    config_files = sorted(CONFIG_DIR.glob("*.yaml"))
    
    for config_path in config_files:        
        success = run_experiment(config_path)
        if not success:
            continue

    print("\n✅ Alle Experimente abgeschlossen")
    print(f"📊 Generiere alle Plots ...")
    subprocess.run([
        "python", "results/plots/plot_all.py",
    ])
    print(f"\n✅ Generierung aller Plots abgeschlossen")


if __name__ == "__main__":
    main()
