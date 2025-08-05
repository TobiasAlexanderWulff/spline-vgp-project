# Spline-VGP-Projekt

Dieses Projekt untersucht eine splinebasierte Variante der Sigmoid-Aktivierungsfunktion, um das **Vanishing-Gradient-Problem (VGP)** zu verringern.

---

## 📈 Aktivierungen

Um die neue Spline-Aktivierung zu bewerten, vergleiche ich ihre Ergebnisse mit denen der ReLU- und der klassischen Sigmoidfunktion. Sigmoid ist in tiefen Netzen besonders anfällig für das VGP, während ReLU als robuster Industriestandard gilt.

Aktivierungen:

- ReLU
- Sigmoid
- Spline (eigene Variante der Sigmoidfunktion)

> Im Folgenden wird die splinebasierte Aktivierungsfunktion der Einfachheit halber nur **Spline** genannt.

### 📈 Eigene Spline-Aktivierung

Die Funktion basiert auf der Sigmoidfunktion. Ein kubischer Spline approximiert den Bereich `[-2, 2]`; außerhalb verläuft die Funktion linear weiter.

![sigmoid_vs_spline.png](spline_vs_sigmoid.png)

Wie im rechten Plot zu sehen, bleibt die Ableitung außerhalb des Bereichs `[-2, 2]` steiler und sollte damit dem VGP entgegenwirken.

---

## 📦 Datasets

Die folgenden drei Datasets decken unterschiedliche Schwierigkeitsgrade ab und liefern damit diversere Ergebnisse.

1. **FashionMNIST** – 70.000 Graustufenbilder (28×28) mit 10 Klassen.
2. **CIFAR10** – 60.000 Farbbilder (32×32) mit 10 Klassen.
3. **TinyImageNet‑200** – 200 Klassen mit insgesamt 120.000 Farbbildern (64×64; Train/Val/Test 100k/10k/10k).

Für FashionMNIST und CIFAR10 dient das jeweilige Testset als Validation-Set. Eine abschließende Testphase entfällt, da der Fokus auf den Gradientenverläufen während des Trainings liegt. Verlust und Genauigkeit dienen lediglich als zusätzliche Indikatoren.

---

## 🛠️ Setup

Projekt Setup:

```sh
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Für CUDA 12.6 Unterstützung:

```sh
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```

TinyImageNet‑200 herunterladen von [Kaggle](https://www.kaggle.com/datasets/nikhilshingadiya/tinyimagenet200) oder via:

```sh
wget http://cs231n.stanford.edu/tiny-imagenet-200.zip
```

Anschließend den entpackten Ordner unter `data/` ablegen.

> Alle anderen Datasets werden beim ersten Ausführen automatisch heruntergeladen und gespeichert.

---

## 🚀 Einzelnes Experiment starten

```sh
python experiments/run_experiment.py --config experiments/configs/spline_cifar10.yaml
```
- Logs: `logs/<experiment_name>/`
- Metriken: `metrics.csv`
- Trainingslog: `training.log`

---

## 📊 Alle vorhandenen CSV-Logs plotten

```sh
python results/plots/plot_all.py
```
- durchsucht alle Log-Unterverzeichnisse nach `metrics.csv`
- erstellt Gradient-Heatmaps sowie Loss- und Accuracy-Verläufe
- speichert Plots unter `results/plots/gradient_heatmaps/` bzw. `results/plots/loss_acc/`

---

## 🔁 Alle Experimente automatisch ausführen

```sh
python run_all_experiments.py
```
- iteriert über alle `.yaml`-Dateien in `experiments/configs/`
- führt Training, Logging und Plotting durch

---

## 📂 Projektstruktur

```sh
spline-vgp-project/
├── data/
├── experiments/
│   ├── configs/            # YAML-Experimente
│   └── run_experiment.py   # Einzellauf
├── logs/
├── models/
│   ├── activations.py      # Stellt Aktivierungen bereit
│   ├── feedforward.py      # FFN-Implementierung
│   └── sigmoid_spline_activation.py  # Spline-basierte Sigmoid-Implementierung
├── results/
│   └── plots/
│       ├── gradient_heatmaps         # Heatmaps der Gradienten
│       └── loss_acc                  # Loss- und Accuracy-Verläufe
├── training/
│   ├── trainer.py          # Training und Validierung
│   └── utils.py            # Dataloader, Seed-Setting, Gewichtsinitialisierung
├── run_all_experiments.py  # Automatisierter Durchlauf aller Experimente
├── requirements.txt
├── README.md
└── .gitignore
```
