import numpy as np
from matplotlib import pyplot as plt

from CustomScaler import CustomScaler
from interpolators.griddata import GriddataInterpolator
from simulation_list import SimulationList


def main():
    mcode, gamma, wt, wp = [10 ** 23, 1, 10.0, 10.0]
    simlist = SimulationList.jsonlines_load()

    data = simlist.X
    values = simlist.Y

    scaler = CustomScaler()
    scaler.fit(data)
    scaled_data = scaler.transform_data(data)
    interpolator = GriddataInterpolator(scaled_data, values)

    alpharange = np.linspace(-0.5, 60.5, 100)
    vrange = np.linspace(0.5, 5.5, 100)
    grid_alpha, grid_v = np.meshgrid(alpharange, vrange)

    parameters = [grid_alpha, grid_v, mcode, gamma, wt, wp]
    scaled_parameters = list(scaler.transform_parameters(parameters))

    grid_result = interpolator.interpolate(*scaled_parameters)

    plt.title("m={:3.0f}, gamma={:3.1f}, wt={:2.0f}, wp={:2.0f}\n".format(mcode, gamma, wt, wp))

    # plt.contourf(grid_x, grid_y, grid_c, N, cmap="Blues", vmin=0, vmax=1)
    plt.pcolormesh(grid_alpha, grid_v, grid_result, cmap="Blues", vmin=0, vmax=1)
    plt.colorbar().set_label("water retention")
    # plt.scatter(data[:, 0], data[:, 1], c=values, cmap="Blues")
    plt.xlabel("impact angle $\\alpha$")
    plt.ylabel("velocity $v$")
    plt.tight_layout()
    # plt.savefig("vis.png", transparent=True)
    plt.show()


if __name__ == '__main__':
    main()
