import numpy as np
from matplotlib import pyplot as plt, cm

from CustomScaler import CustomScaler
from config import water_fraction
from interpolators.rbf import RbfInterpolator
from simulation_list import SimulationList

plt.style.use('dark_background')


def main():
    mcode, gamma, wt, wp = [10 ** 22, 0.6, 15 / 100, 15 / 100]
    simlist = SimulationList.jsonlines_load()
    # for s in simlist.simlist:
    #     if s.type!="original":
    #         continue
    #     print(s.wpcode,s.projectile_water_fraction)
    # exit()
    data = simlist.X
    print("-----")
    print(len(data))
    # print(data[0])
    # exit()
    values = simlist.Y

    scaler = CustomScaler()
    scaler.fit(data)
    scaled_data = scaler.transform_data(data)
    interpolator = RbfInterpolator(scaled_data, values)
    # interpolator = GriddataInterpolator(scaled_data, values)

    alpharange = np.linspace(-0.5, 60.5, 300)
    vrange = np.linspace(0.5, 5.5, 300)
    grid_alpha, grid_v = np.meshgrid(alpharange, vrange)

    parameters = [grid_alpha, grid_v, mcode, gamma, wt, wp]
    scaled_parameters = list(scaler.transform_parameters(parameters))

    grid_result = interpolator.interpolate(*scaled_parameters)
    print("minmax")
    print(np.nanmin(grid_result), np.nanmax(grid_result))

    plt.title("m={:3.0e}, gamma={:3.1f}, wt={:2.0f}%, wp={:2.0f}%\n".format(mcode, gamma, wt*100, wp*100))
    cmap = cm.Blues if water_fraction else cm.Oranges
    cmap.set_bad('white', 1.)  # show nan white
    # plt.contourf(grid_alpha, grid_v, grid_result, 100, cmap="Blues", vmin=0, vmax=1)
    # plt.pcolormesh(grid_alpha, grid_v, grid_result, cmap="Blues", vmin=0, vmax=1)
    plt.imshow(grid_result, interpolation='none', cmap=cmap, aspect="auto", origin="lower", vmin=0, vmax=1,
               extent=[grid_alpha.min(), grid_alpha.max(), grid_v.min(), grid_v.max()])
    plt.colorbar().set_label("water retention fraction" if water_fraction else "core mass retention fraction")
    # plt.scatter(data[:, 0], data[:, 1], c=values, cmap="Blues")
    plt.xlabel("impact angle $\\alpha$ [$^{\circ}$]")
    plt.ylabel("velocity $v$ [$v_{esc}$]")
    plt.tight_layout()
    # plt.savefig("vis.png", transparent=True)
    plt.savefig("/home/lukas/tmp/test.svg", transparent=True)
    plt.show()


if __name__ == '__main__':
    main()
