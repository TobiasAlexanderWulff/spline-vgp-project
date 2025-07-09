# Spline-VGP-Project

Untersuchung splinebasierter Aktivierungsfunktionen in tiefen Feedforward-Netzen zur Milderung des Vanishing-Gradient-Problems.

---

## 📦 Setup

```sh
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Für CUDA 12.6 Unterstützung:

``` sh
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```

---

## 🚀 Einzelnes Experiment starten

``` sh
python experiments/run_experiment.py --config experiments/configs/spline_cifar10.yaml
```
- Logs: `logs/tensorboard/<experiment_name>/`
- Metriken: `metrics.csv`
- Dauer: `training_duration.txt`

---

### 📊 Metriken plotten

``` sh
python results/plots/plot_metrics.py \
  --csv logs/tensorboard/spline_cifar10/metrics.csv \
  --save_dir results/plots/
```

---

### 📈 Gradientenverlauf (Skalare)

```bash
python results/plots/plot_scalar_gradients.py \
  --log_dir logs/tensorboard/spline_cifar10 \
  --save_dir results/plots/
```

---

## 🔁 Alle Experimente automatisch ausführen

``` sh
python run_all_experiments.py
```
- Läuft alle `.yaml` in `experiments/configs/` durch
- Führt Training, CSV-Logging und Plotting durch

---

## 📂 Projektstruktur

``` sh
spline-vgp-project/
├── data/
├── experiments/
│   ├── configs/            # YAML-Experimente
│   └── run_experiment.py   # Einzellauf
├── logs/
│   └── tensorboard/        # Trainingsergebnisse
├── models/                 # Feedforward-Netz & Aktivierungen
├── results/
│   └── plots/              # Diagramme
├── training/
│   ├── trainer.py
│   └── utils.py
├── run_all_experiments.py  # Automatisierter Vergleich
├── requirements.txt
├── README.md
└── .gitignore
```
