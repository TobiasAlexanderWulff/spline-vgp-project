import torch
import torch.nn as nn
from models.feedforward import FNN
from torchviz import make_dot

# Beispielmodell (FashionMNIST: 28x28=784 Input, 10 Klassen)
model = FNN(
    input_dim=28*28,
    hidden_dim=512,
    depth=16,
    activation_fn=nn.ReLU(),
    output_dim=10
)

# Dummy-Input
x = torch.randn(1, 28*28)

# Forward Pass
y = model(x)

# Nur den Modellgraph zeichnen, Parameter werden NICHT alle aufgelistet
dot = make_dot(y, params=dict(list(model.named_parameters())[:2]))  # nur 1–2 Parameter für Übersicht

# Format einstellen und speichern
dot.format = "svg"   # oder "png"
dot.render("ffn_architecture_schematic")

