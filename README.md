# Spline-VGP-Project

In diesem Projekt will ich versuchen, mit Hilfe einer *splinebasierten Abwandlung der Sigmoidfunktion*, das **Vanishing-Gradient-Problem (VGP)** zu vermindern.

---

## 📈 Aktivierungen

Um die neue ***splinebasierte Sigmoidfunktion*** auf ihre Nutzen zu überprüfen, vergleiche ich sämtliche Ergebnisse mit denen der ReLU- und Sigmoidfunktion. Die ***Sigmoidfunktion*** ist bekannt dafür in tiefen Netzen mit vielen Schichten besonders anfällig für das VGP zu sein. Die ***ReLU-Aktivierungsfunktion*** hingegen soll im Vergleich als Wegweiser gelten, da sie vom Design her nicht anfällig für das VGP ist und außerdem als Industriestandard für die meisten Neuronalen Netze gilt.

Aktivierungen:

- ReLU

- Sigmoid

- Spline (eigene splinebasierte Abwandlung der Sigmoidfunktion)

> Die ***splinebasierte Abwandlung der Sigmoid-Aktivierungsfunktion*** wird in jeglichen folgenden Abbildungen und Erwähnungen, der Einfachheit halber, nur als ***spline*** bezeichnet.

### 📈 Eigene Spline Aktivierungs

Meine eigene splinebasierte Aktivierungsfunktion basiert auf der *Sigmoidfunktion* und modifiziert diese. Unter der verwendung eines *kubischen Splines* wird die Sigmoidfunktion im Bereich `[-2, 2]` nachgebildet und verläuft außerhalb dieser grenzen linear weiter.

![sigmoid_vs_spline.png](/spline_vs_sigmoid.png)

Wie auf dem rechten Plot der Abbildung zu sehen ist, verläuft die Ableitung der Splinevariante nicht mehr flach außerhalb von `[-2, 2]`, was in der Theorie bereits dem VGP entgegenwirken sollte.

---

## 📦 Datasets

Ich habe mich für die folgenden drei Datasets entschieden, da sie das Netz unterschiedlich stark fordern und somit möglichst diversifizierte Ergebnisse liefern. ***FashionMnist*** ist hier das simpelste Dataset. Es beruht auf 70.000 (train/test 60.000/10.000) 28x28 Bildern, die nur in Grautönen abgebildet sind. Jedes Bild wird hier einem von 10 Bezeichnungen (Klassen) zugeordnet, was in diesem Fall Zalando-Artikel sind. ***Cifar10*** soll hier als erste Steigerung in der Schwierigkeit gelten. Es besteht aus 60.000 (train/test 50.000/10.000) 32x32 farbigen Bildern, die ebenfalls je einer von 10 Klassen zugeordnet werden. Als höchste Steigerung für dieses Projekt soll das Dataset ***TinyImagenet-200*** als Skalierungstest dienen. Mit 100.000 (train/test/val 100.000/10.000/10.000) 64x64 Bildern in Farbe und 200 Klassen ist es das größte und komplexeste Dataset im Vergleich. Wichtig ist zu erwähnen, dass ich bei den Datasets *FashionMnist* und *Cifar10* das vorgeschriebene Test-Dataset als Validation-Dataset zweckentfremdet habe. Das hat den Grund, dass ich für diesen Versuchsaufbau auf eine Testphase nach dem Trainieren der Modelle verzichtet habe, da für mich die *Gradientenveränderungen* während des Trainings im Vordergrund standen. *Verlust* und *Genauigkeit* habe ich hier eher als zusätzliche Indikatoren gesehen, um etwas diverser auf die Nützlichkeit meiner splinebasierten Aktivierung zu schließen.

Datasets:

1. FashionMnist

2. Cifar10

3. TinyImagenet-200

---

## 🛠️ Setup

Projekt Setup:

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

TinyImagenet-200 herunterladen von [kaggle](https://www.kaggle.com/datasets/nikhilshingadiya/tinyimagenet200) oder mit:

``` sh
wget http://cs231n.stanford.edu/tiny-imagenet-200.zip
```

Danach **entpackt** im Projektordner unter `data/` ablegen.

> Alle anderen Datasets werden automatisch mit ausführen des Codes heruntergeladen und gespeichert.

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
