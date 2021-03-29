import json
import random
from pathlib import Path

import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from torch import nn, optim, from_numpy, Tensor
from torch.utils.data import DataLoader, TensorDataset

from CustomScaler import CustomScaler
from network import Network
from simulation_list import SimulationList


def train():
    filename = "rsmc_dataset"

    simulations = SimulationList.jsonlines_load(Path(f"{filename}.jsonl"))

    # random.seed(1)
    test_data = random.sample(simulations.simlist, int(len(simulations.simlist) * 0.2))
    test_set = set(test_data)  # use a set for faster *in* computation
    train_data = [s for s in simulations.simlist if s not in test_set]
    print(len(train_data), len(test_data))

    X = np.array(
        [[s.alpha, s.v, s.projectile_mass, s.gamma, s.target_water_fraction, s.projectile_water_fraction] for s in
         train_data])
    scaler = CustomScaler()
    scaler.fit(X)
    x = scaler.transform_data(X)
    print(x.shape)
    Y = np.array([[
        s.water_retention_both, s.mantle_retention_both, s.core_retention_both,
        s.output_mass_fraction
    ] for s in train_data])

    X_test = np.array(
        [[s.alpha, s.v, s.projectile_mass, s.gamma, s.target_water_fraction, s.projectile_water_fraction] for s in
         test_data])
    Y_test = np.array([[
        s.water_retention_both, s.mantle_retention_both, s.core_retention_both,
        s.output_mass_fraction
    ] for s in test_data])
    x_test = scaler.transform_data(X_test)
    random.seed()

    dataset = TensorDataset(from_numpy(x).to(torch.float), from_numpy(Y).to(torch.float))
    train_dataset = TensorDataset(from_numpy(x_test).to(torch.float), from_numpy(Y_test).to(torch.float))
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    validation_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=False)

    network = Network()

    loss_fn = nn.MSELoss()

    optimizer = optim.Adam(network.parameters())

    loss_train = []
    loss_vali = []

    max_epochs = 120
    epochs = 0

    fig: Figure = plt.figure()
    ax: Axes = fig.gca()
    x_axis = np.arange(epochs)
    loss_plot: Line2D = ax.plot(x_axis, loss_train, label="loss_train")[0]
    vali_plot: Line2D = ax.plot(x_axis, loss_vali, label="loss_validation")[0]
    ax.legend()
    plt.ion()
    plt.pause(0.01)
    plt.show()

    for e in range(max_epochs):
        print(f"Epoch: {e}")
        total_loss = 0
        network.train()
        for xs, ys in dataloader:
            # Training pass
            optimizer.zero_grad()

            output = network(xs)
            loss = loss_fn(output, ys)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        loss_train.append(float(total_loss / len(dataloader)))
        print(f"Training loss: {total_loss / len(dataloader)}")

        # validation:
        network.eval()
        total_loss_val = 0
        for xs, ys in validation_dataloader:
            output = network(xs)
            total_loss_val += loss_fn(output, ys).item()
        loss_vali.append(float(total_loss_val / len(validation_dataloader)))
        print(f"Validation loss: {total_loss_val / len(validation_dataloader)}")
        epochs += 1

        x_axis = np.arange(epochs)
        loss_plot.set_xdata(x_axis)
        vali_plot.set_xdata(x_axis)
        loss_plot.set_ydata(loss_train)
        vali_plot.set_ydata(loss_vali)
        ax.relim()
        ax.autoscale_view(True, True, True)
        plt.pause(0.01)
        # plt.draw()
        # if epochs > 6:
        #     a = np.sum(np.array(loss_vali[-3:]))
        #     b = np.sum(np.array(loss_vali[-6:-3]))
        #     if a > b:  # overfitting on training data, stop training
        #         print("early stopping")
        #         break
    plt.ioff()
    torch.save(network.state_dict(), "pytorch_model.zip")
    with open("pytorch_model.json", "w") as f:
        export_dict = {}
        value_tensor: Tensor
        for key, value_tensor in network.state_dict().items():
            export_dict[key] = value_tensor.detach().tolist()
        export_dict["means"] = scaler.means.tolist()
        export_dict["stds"] = scaler.stds.tolist()
        json.dump(export_dict, f)

    xrange = np.linspace(-0.5, 60.5, 300)
    yrange = np.linspace(0.5, 5.5, 300)
    xgrid, ygrid = np.meshgrid(xrange, yrange)
    mcode = 1e24
    wpcode = 1e-4

    wtcode = 1e-4
    gammacode = 0.6
    testinput = np.array([[np.nan, np.nan, mcode, gammacode, wtcode, wpcode]] * 300 * 300)
    testinput[::, 0] = xgrid.flatten()
    testinput[::, 1] = ygrid.flatten()
    testinput = scaler.transform_data(testinput)

    print(testinput)
    print(testinput.shape)
    testoutput: Tensor = network(from_numpy(testinput).to(torch.float))
    data = testoutput.detach().numpy()
    outgrid = np.reshape(data[::, 0], (300, 300))
    print("minmax")
    print(np.nanmin(outgrid), np.nanmax(outgrid))
    cmap = "Blues"
    plt.title(
        "m={:3.0e}, gamma={:3.1f}, wt={:2.0f}%, wp={:2.0f}%\n".format(mcode, gammacode, wtcode * 100, wpcode * 100))
    plt.imshow(outgrid, interpolation='none', cmap=cmap, aspect="auto", origin="lower", vmin=0, vmax=1,
               extent=[xgrid.min(), xgrid.max(), ygrid.min(), ygrid.max()])

    plt.colorbar().set_label("water retention fraction")
    plt.xlabel("impact angle $\\alpha$ [$^{\circ}$]")
    plt.ylabel("velocity $v$ [$v_{esc}$]")
    plt.tight_layout()
    # plt.savefig("/home/lukas/tmp/nn.svg", transparent=True)
    plt.show()


if __name__ == '__main__':
    train()
