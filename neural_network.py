import json
import random
from pathlib import Path
from typing import List

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
from simulation import Simulation
from simulation_list import SimulationList


def x_array(s: Simulation) -> List[float]:
    return [s.alpha, s.v, s.projectile_mass, s.gamma,
            s.target_water_fraction, s.projectile_water_fraction]


def y_array(s: Simulation) -> List[float]:
    return [
        s.water_retention_both, s.mantle_retention_both,
        s.core_retention_both, s.output_mass_fraction
    ]


def train():
    filename = "rsmc_dataset"

    simulations = SimulationList.jsonlines_load(Path(f"{filename}.jsonl"))

    random.seed(1)
    random.shuffle(simulations.simlist)
    num_test = int(len(simulations.simlist) * 0.2)
    test_data = simulations.simlist[:num_test]
    train_data = simulations.simlist[num_test:]
    print(len(train_data), len(test_data))
    a = set(s.runid for s in train_data)
    b = set(s.runid for s in test_data)
    assert len(a & b) == 0, "no overlap between test data and training data"

    X = np.array([x_array(s) for s in train_data])
    scaler = CustomScaler()
    scaler.fit(X)
    x = scaler.transform_data(X)
    del X
    print(x.shape)
    Y = np.array([y_array(s) for s in train_data])

    X_test = np.array([x_array(s) for s in test_data])
    Y_test = np.array([y_array(s) for s in test_data])
    x_test = scaler.transform_data(X_test)
    del X_test
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

    max_epochs = 200
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
    np.savetxt("loss.txt", np.array([x_axis, loss_train, loss_vali]).T)
    torch.save(network.state_dict(), "pytorch_model.zip")
    with open("pytorch_model.json", "w") as f:
        export_dict = {}
        value_tensor: Tensor
        for key, value_tensor in network.state_dict().items():
            export_dict[key] = value_tensor.detach().tolist()
        export_dict["means"] = scaler.means.tolist()
        export_dict["stds"] = scaler.stds.tolist()
        json.dump(export_dict, f)
    plt.ioff()
    model_test_y = []
    for x in x_test:
        result = network(from_numpy(np.array(x)).to(torch.float))
        y = result.detach().numpy()
        model_test_y.append(y)
    model_test_y = np.asarray(model_test_y)
    plt.figure()
    plt.xlabel("model output")
    plt.ylabel("real data")
    for i, name in enumerate(["shell", "mantle", "core", "mass fraction"]):
        plt.scatter(model_test_y[::, i], Y_test[::, i], s=0.2, label=name)
    plt.legend()
    plt.show()


if __name__ == '__main__':
    train()
