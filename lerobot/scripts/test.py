import torch
import torch.nn as nn

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

model = nn.Sequential(
    nn.Linear(1024, 2048),
    nn.ReLU(),
    nn.Linear(2048, 10)
).to(device)

x = torch.randn(32, 1024).to(device)
for i in range(100):
    out = model(x)