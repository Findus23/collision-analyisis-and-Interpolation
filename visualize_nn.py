import json

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import from_numpy, Tensor

from CustomScaler import CustomScaler
from network import Network

resolution = 300


def main():
    mcode, gamma, wt, wp = [10 ** 24, 0.4, 1e-5, 1e-5]
    with open("pytorch_model.json") as f:
        data = json.load(f)
        scaler = CustomScaler()
        scaler.means = np.array(data["means"])
        scaler.stds = np.array(data["stds"])

    model = Network()
    model.load_state_dict(torch.load("pytorch_model.zip"))

    alpharange = np.linspace(0, 60, resolution)
    vrange = np.linspace(0.5, 5.5, resolution)
    grid_alpha, grid_v = np.meshgrid(alpharange, vrange)
    mcode = 1e24
    wpcode = 1e-5

    wtcode = 1e-5
    gammacode = 0.6
    testinput = np.array([[np.nan, np.nan, mcode, gammacode, wtcode, wpcode]] * resolution * resolution)
    testinput[::, 0] = grid_alpha.flatten()
    testinput[::, 1] = grid_v.flatten()
    testinput = scaler.transform_data(testinput)

    print(testinput)
    print(testinput.shape)
    network = Network()
    network.load_state_dict(torch.load("pytorch_model.zip"))

    print(testinput)
    testoutput: Tensor = network(from_numpy(testinput).to(torch.float))
    data = testoutput.detach().numpy()
    grid_result = np.reshape(data[::, 0], (300, 300))
    print("minmax")
    print(np.nanmin(grid_result), np.nanmax(grid_result))
    cmap = "Blues"
    plt.figure()
    plt.title(
        "m={:3.0e}, gamma={:3.1f}, wt={:2.0e}%, wp={:2.0e}%\n".format(mcode, gammacode, wtcode, wpcode))
    plt.imshow(grid_result, interpolation='none', cmap=cmap, aspect="auto", origin="lower", vmin=0, vmax=1,
               extent=[grid_alpha.min(), grid_alpha.max(), grid_v.min(), grid_v.max()])

    plt.colorbar().set_label("water retention fraction")
    plt.xlabel("impact angle $\\alpha$ [$^{\circ}$]")
    plt.ylabel("velocity $v$ [$v_{esc}$]")
    plt.tight_layout()
    plt.savefig("/home/lukas/tmp/nn.pdf", transparent=True)
    plt.show()

if __name__ == '__main__':
    main()
