import sys
from math import pi, sqrt

from scipy.constants import G, astronomical_unit

from CustomScaler import CustomScaler
from interpolators.rbf import RbfInterpolator
from simulation_list import SimulationList


def clamp(n, smallest, largest):
    assert smallest < largest
    return max(smallest, min(n, largest))


if len(sys.argv) < 2:
    print("specify filename")
    exit(1)
if sys.argv[1] == "-h":
    print("alpha\t\t\tthe impact angle \t[degrees]")
    print("velocity\t\tthe impact velocity \t[AU/58d]")
    print("projectile-mass\t\tmass of the projectile \t[M_⊙]")
    print("target-mass\t\tmass of the projectile \t[M_⊙]")
    exit()

with open(sys.argv[1]) as f:
    entries = f.readline().split()
    if len(entries) != 4:
        print("file must contain 4 parameters")
    argalpha, argvelocity, argmp, argmt = map(float, entries)

solar_mass = 1.98847542e+30  # kg
ice_density = 0.917 / 1000 * 100 ** 3  # kg/m^3
basalt_density = 2.7 / 1000 * 100 ** 3  # kg/m^3
water_fraction = 0.15

alpha = argalpha

target_water_fraction = water_fraction
projectile_water_fraction = water_fraction

projectile_mass_sm = argmp
target_mass_sm = argmt
projectile_mass = projectile_mass_sm * solar_mass
target_mass = target_mass_sm * solar_mass


def core_radius(total_mass, water_fraction, density):
    core_mass = total_mass * (1 - water_fraction)
    return (core_mass / density * 3 / 4 / pi) ** (1 / 3)


def total_radius(total_mass, water_fraction, density, inner_radius):
    mantle_mass = total_mass * water_fraction
    return (mantle_mass / density * 3 / 4 / pi + inner_radius ** 3) ** (1 / 3)


target_core_radius = core_radius(target_mass, target_water_fraction, basalt_density)
target_radius = total_radius(target_mass, target_water_fraction, ice_density, target_core_radius)
projectile_core_radius = core_radius(projectile_mass, projectile_water_fraction, basalt_density)
projectile_radius = total_radius(projectile_mass, projectile_water_fraction, ice_density, projectile_core_radius)

escape_velocity = sqrt(2 * G * (target_mass + projectile_mass) / (target_radius + projectile_radius))
velocity_original = argvelocity

const = 365.256 / (2 * pi)  # ~58.13

velocity_si = velocity_original * astronomical_unit / const / (60 * 60 * 24)
velocity = velocity_si / escape_velocity
gamma = projectile_mass_sm / target_mass_sm

if alpha > 90:
    alpha = 180 - alpha
if gamma > 1:
    gamma = 1 / gamma
alpha = clamp(alpha, 0, 60)
velocity = clamp(velocity, 1, 5)

m_ceres = 9.393e+20
m_earth = 5.9722e+24
projectile_mass = clamp(projectile_mass, 2 * m_ceres, 2 * m_earth)
gamma = clamp(gamma, 1 / 10, 1)
simulations = SimulationList.jsonlines_load()

scaler = CustomScaler()
scaler.fit(simulations.X)

scaled_data = scaler.transform_data(simulations.X)
water_interpolator = RbfInterpolator(scaled_data, simulations.Y_water)
mass_interpolator = RbfInterpolator(scaled_data, simulations.Y_mantle)

testinput = [32, 1, 7.6e22, 0.16, 0.15, 0.15]


scaled_input = list(scaler.transform_parameters(testinput))
water_retention = water_interpolator.interpolate(*scaled_input)
mass_retention = mass_interpolator.interpolate(*scaled_input)

water_retention = clamp(water_retention, 0, 1)
mass_retention = clamp(mass_retention, 0, 1)

print(water_retention)
print(mass_retention)
