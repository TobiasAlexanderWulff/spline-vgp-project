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

### TinyImageNet Download

:)

---

## 🚀 Einzelnes Experiment starten

``` sh
python experiments/run_experiment.py --config experiments/configs/spline_cifar10.yaml
```
- Logs: `logs/<experiment_name>/`
- Metriken: `metrics.csv`
- Trainings Log: `training.log`

---

## 📊 Alle vorhandenen csv-Logs plotten

``` sh
python results/plots/plot_all.py
```
- Durchsucht alle vorhandenen log-Unterverzeichnisse nach `metrics.csv` Dateien
- Plottet Gradient Heatmaps, sowie Loss und Accuracies entsprechend
- Plots werden unter `results/plots/gradient_heatmaps/` und `results/plots/loss_acc/` entsprechend gespeichert

---

## 🔁 Alle Experimente automatisch ausführen

``` sh
python run_all_experiments.py
```
- Läuft alle `.yaml` in `experiments/configs/` durch
- Führt Training, CSV-Logging und Plotting entsprechend durch

---

## 📂 Projektstruktur

``` sh
spline-vgp-project/
├── data/
├── experiments/
│   ├── configs/            # YAML-Experimente
│   └── run_experiment.py   # Einzellauf
├── logs/
├── models/  
│   ├── activations.py      # Stellt Aktivierungen bereit
│   ├── feedforward.py      # FFN-Implementierung
│   └── sigmoid_spline_activation.py  # Eigene optimierte Sigmoid Implementierung mit Hilfe von Splines
├── results/
│   └── plots/              
│       ├── gradient_heatmaps         # Heatmaps von Gradientenverläufen (norm und mean_abs)
│       └── loss_acc                  # Graphen für Loss- und Accuracyverläufe (train und val)
├── training/
│   ├── trainer.py          # Train- und Validation-Implementation
│   └── utils.py            # Utils wie dataloader-Bereitstellung, seed-setting oder Gewichtsinitialisierungs
├── run_all_experiments.py  # Automatisierter Durchlauf aller Experimente + Plotting
├── requirements.txt
├── README.md
└── .gitignore
```
