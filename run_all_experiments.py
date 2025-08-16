import subprocess
import os
import shutil
import argparse
from pathlib import Path


CONFIG_DIR = Path("experiments/configs/")
SMOKE_CONFIG_DIR = Path("experiments/configs/smoke/")
LOG_DIR = Path("logs/")
SMOKE_LOG_DIR = Path("logs_smoke/")

EXPERIMENT_TIMEOUT = 3 * 60 * 60    # 3 Stunden


def _clear_log_dir(log_dir: str):
    """Clears the spezified log_dir."""
    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    os.makedirs(log_dir, exist_ok=True)


def run_experiment(config_path: str, log_dir: Path) -> bool:
    """Runs an experiment. Needs its config path."""
    experiment_name = config_path.stem
    experiment_log_dir = log_dir / experiment_name
    
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
    except subprocess.TimeoutExpired:
        print(
            f"⏰ Timeout: {experiment_name} nach {EXPERIMENT_TIMEOUT // 3600} Stunden abgebrochen"
        )
    except Exception as e:
        print(f"❌ Ausnahme bei {experiment_name}: {e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="Set flag to run smoke experiments instead")
    args = parser.parse_args()
    
    if args.smoke:
        config_dir = SMOKE_CONFIG_DIR
        log_dir = SMOKE_LOG_DIR
    else:
        config_dir = CONFIG_DIR
        log_dir = LOG_DIR
    
    config_files = sorted(config_dir.glob("*.yaml"))
    
    for config_path in config_files:        
        run_experiment(config_path, log_dir)

    print("\n✅ Alle Experimente abgeschlossen")
    print("📊 Generiere alle Plots ...")
    if args.smoke:
        subprocess.run([
            "python", "results/plots/plot_all.py", "--smoke",
        ])
    else:
        subprocess.run([
            "python", "results/plots/plot_all.py",
        ])
    print("\n✅ Generierung aller Plots abgeschlossen")


if __name__ == "__main__":
    main()
