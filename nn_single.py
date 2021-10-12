import json
from pathlib import Path

import numpy as np
import torch

from CustomScaler import CustomScaler
from network import Network
from simulation_list import SimulationList

resolution = 100

with open("pytorch_model.json") as f:
    data = json.load(f)
    scaler = CustomScaler()
    scaler.means = np.array(data["means"])
    scaler.stds = np.array(data["stds"])

model = Network()
model.load_state_dict(torch.load("pytorch_model.zip"))

ang = 30
v = 2
m = 1e24
gamma = 0.6
wp = wt = 1e-4
print(model(torch.Tensor(list(scaler.transform_parameters([ang, v, m, gamma, wt, wp])))))
