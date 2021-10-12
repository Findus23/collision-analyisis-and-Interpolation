from copy import copy
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt, cm

from CustomScaler import CustomScaler
from config import water_fraction
from interpolators.griddata import GriddataInterpolator
from interpolators.rbf import RbfInterpolator
from simulation import Simulation
from simulation_list import SimulationList

plt.style.use('dark_background')


def main():
    mcode, gamma, wt, wp = [10 ** 22, 0.6, 1e-5, 1e-5]
    simlist = SimulationList.jsonlines_load(Path("rsmc_dataset.jsonl"))
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
    values = simlist.Y_mass_fraction

    scaler = CustomScaler()
    scaler.fit(data)
    scaled_data = scaler.transform_data(data)
    interpolator = RbfInterpolator(scaled_data, values)
    # interpolator = GriddataInterpolator(scaled_data, values)

    alpharange = np.linspace(0, 60, 300)
    vrange = np.linspace(0.5, 5.5, 300)
    grid_alpha, grid_v = np.meshgrid(alpharange, vrange)

    parameters = [grid_alpha, grid_v, mcode, gamma, wt, wp]
    scaled_parameters = list(scaler.transform_parameters(parameters))

    grid_result = interpolator.interpolate(*scaled_parameters)
    print("minmax")
    print(np.nanmin(grid_result), np.nanmax(grid_result))

    plt.title("m={:3.0e}, gamma={:3.1f}, wt={:2.0e}, wp={:2.0e}\n".format(mcode, gamma, wt, wp))
    cmap = cm.get_cmap("Blues") if water_fraction else cm.get_cmap("Oranges")
    cmap = copy(cmap)
    cmap.set_bad('white', 1.)  # show nan white
    # plt.contourf(grid_alpha, grid_v, grid_result, 100, cmap="Blues", vmin=0, vmax=1)
    # plt.pcolormesh(grid_alpha, grid_v, grid_result, cmap="Blues", vmin=0, vmax=1)
    plt.imshow(grid_result, interpolation='none', cmap=cmap, aspect="auto", origin="lower", vmin=0, vmax=1,
               extent=[grid_alpha.min(), grid_alpha.max(), grid_v.min(), grid_v.max()])
    # plt.scatter(data[:, 0], data[:, 1], c=values, cmap="Blues")
    plt.xlabel("impact angle $\\alpha$ [$^{\circ}$]")
    plt.ylabel("velocity $v$ [$v_{esc}$]")
    s: Simulation
    xs = []
    ys = []
    zs = []
    for s in simlist.simlist:
        # if not (0.4 < s.gamma < 0.6) or not (1e23 < s.total_mass < 5e24):
        #     continue
        # if s.alpha < 60 or s.v > 5 or s.v < 2:
        #     continue
        z = s.output_mass_fraction
        zs.append(z)
        xs.append(s.alpha)
        ys.append(s.v)
        print(z, s.runid)
    plt.scatter(xs, ys, c=zs, cmap=cmap, vmin=0, vmax=1)
    plt.colorbar().set_label("stone retention fraction" if water_fraction else "core mass retention fraction")
    plt.tight_layout()
    # plt.savefig("vis.png", transparent=True)
    plt.savefig("/home/lukas/tmp/test.pdf", transparent=True)
    plt.show()


if __name__ == '__main__':
    main()
