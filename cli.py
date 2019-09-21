import argparse

from CustomScaler import CustomScaler
from interpolators.rbf import RbfInterpolator
from simulation_list import SimulationList

parser = argparse.ArgumentParser(description="interpolate water retention rate using RBF",
                                 epilog="returns water retention fraction and mass_retention fraction seperated by a newline")
requiredNamed = parser.add_argument_group('required named arguments')
requiredNamed.add_argument("-a", "--alpha", type=float, required=True, help="the impact angle")
requiredNamed.add_argument("-v", "--velocity", type=float, required=True,
                           help="the impact velocity in units of mutual escape velocity")
requiredNamed.add_argument("-mp", "--projectile-mass", type=float, required=True, help="mass of the projectile [kg]")
mass_or_gamma = requiredNamed.add_mutually_exclusive_group(required=True)
mass_or_gamma.add_argument("-mt", "--target-mass", type=float, help="mass of the projectile [kg]")
mass_or_gamma.add_argument("-g", "--gamma", type=float, help="fraction between projectile mass and target mass")
requiredNamed.add_argument("-wfp", "--projectile-water-fraction", type=float, required=True,
                           help="water fraction of projectile")
requiredNamed.add_argument("-wft", "--target-water-fraction", type=float, required=True,
                           help="water fraction of target")
args = parser.parse_args()
print(args)

simulations = SimulationList.jsonlines_load()

scaler = CustomScaler()
scaler.fit(simulations.X)

scaled_data = scaler.transform_data(simulations.X)
water_interpolator = RbfInterpolator(scaled_data, simulations.Y_water)
mass_interpolator = RbfInterpolator(scaled_data, simulations.Y_mass)

if args.gamma:
    gamma = args.gamma
else:
    gamma = args.projectile_mass / args.target_mass

testinput = [args.alpha, args.velocity, args.projectile_mass, gamma,
             args.target_water_fraction, args.projectile_water_fraction]
scaled_input = list(scaler.transform_parameters(testinput))
water_retention = water_interpolator.interpolate(*scaled_input)
mass_retention = mass_interpolator.interpolate(*scaled_input)

print(water_retention)
print(mass_retention)
