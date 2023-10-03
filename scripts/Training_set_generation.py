import json
import numpy as np
import os
import sys
from QuantumATK import *


def checkAlloy(dir_name, id_):
    present = False

    with open(dir_name + "configurations.dat", "a+") as f:
        for line in f:
            if id_ in line:
                present = True
                break
    return present


# Create unique sequence for each configuration
def conf2Sequence(configuration):
    sequence = ""
    for el in configuration.elements():
        if el.name() == "Antimony":
            sequence += "S"
        elif el.name() == "Arsenic":
            sequence += "A"
    return sequence


# Interpolation functions of alpha and lattice constant
def linearInterpolations(a_antimonide, a_arsenide, alpha_antimonide, alpha_arsenide):

    lin_lattice_const = np.poly1d(np.polyfit([0., 100.], [a_arsenide, a_antimonide], 1))
    lin_alpha = np.poly1d(np.polyfit([0., 100.], [alpha_arsenide, alpha_antimonide], 1))

    return lin_lattice_const, lin_alpha


# Create constituents
def createConstituents(lattice_constant, cation):
    lattice = FaceCenteredCubic(lattice_constant * Angstrom)
    fractional_coordinates = [[0., 0., 0.],
                              [0.25, 0.25, 0.25]]

    if str(cation).strip() == 'al':
        arsenide = BulkConfiguration(
            bravais_lattice=lattice,
            elements=[Aluminium, Arsenic],
            fractional_coordinates=fractional_coordinates
        )

        antimonide = BulkConfiguration(
            bravais_lattice=lattice,
            elements=[Aluminium, Antimony],
            fractional_coordinates=fractional_coordinates
        )

    elif str(cation).strip() == 'ga':
        arsenide = BulkConfiguration(
            bravais_lattice=lattice,
            elements=[Gallium, Arsenic],
            fractional_coordinates=fractional_coordinates
        )

        antimonide = BulkConfiguration(
            bravais_lattice=lattice,
            elements=[Gallium, Antimony],
            fractional_coordinates=fractional_coordinates
        )

    elif str(cation).strip() == 'in':
        arsenide = BulkConfiguration(
            bravais_lattice=lattice,
            elements=[Indium, Arsenic],
            fractional_coordinates=fractional_coordinates
        )

        antimonide = BulkConfiguration(
            bravais_lattice=lattice,
            elements=[Indium, Antimony],
            fractional_coordinates=fractional_coordinates
        )
    else:
        raise ValueError("\nInvalid option for cation choice. Possible arguments are:\n- al\n- ga\n- in")

    return arsenide, antimonide


# Function for simulations at a given Sb% (x)
def run_ensemble(x, alpha, n_ens, arsenide, antimonide, dir_name, relax=True):
    max_tries = 50
    n = 0
    t = 0

    if isMainProcess():
        print("\n| INFORMATIONS ABOUT ENSEMBLE RUN:")
        print(f"|\tAlloy of: {[el.name() for el in arsenide.uniqueElements()]}, "
              f"{[el.name() for el in antimonide.uniqueElements()]}")
        print("|\tPercentage of Sb: ", x)
        print("|\tNr. of realizations: ", n_ens)
        print("|\tMaximum nr. of tries: ", max_tries)
        print("|\tRelaxation of structures: ", relax)
        print("\n")

    while n < n_ens and t < max_tries:
        # Generate alloy
        alloy = genericAlloy(
            configurations=[arsenide, antimonide],
            percentages=[100. - x, x],
            algorithm=FixedFraction,
            repetitions=(3, 3, 3)
        )
        id_ = conf2Sequence(alloy)
        # Check for uniqueness of alloy in Training Set and save elements configuration
        if checkAlloy(dir_name, id_):
            t += 1
            continue
        else:
            t += 1

        # Relaxation step
        if relax:
            potentialSet = Tersoff_Powell_2007()
            calculator = TremoloXCalculator(parameters=potentialSet)
            alloy.setCalculator(calculator)
            alloy.update()

            optimized_configuration = OptimizeGeometry(
                configuration=alloy,
                constraints=[
                    BravaisLatticeConstraint()
                ],
            optimize_cell=False
            )
        else:
            optimized_configuration = alloy

        # DFT simulation
        exchange_correlation = HSECustomExchangeCorrelation(
            screening_length=0.11 * 1 / Bohr,
            exx_fraction=alpha,
            number_of_spins=4,
            spin_orbit=True)

        k_point_sampling = KpointDensity(
            density_a=4.0 * Angstrom,
            density_b=4.0 * Angstrom,
            density_c=4.0 * Angstrom
        )

        numerical_accuracy_parameters = NumericalAccuracyParameters(
            k_point_sampling=k_point_sampling
        )

        checkpoint_handler = NoCheckpointHandler

        calculator_1 = LCAOCalculator(
            exchange_correlation=exchange_correlation,
            numerical_accuracy_parameters=numerical_accuracy_parameters,
            checkpoint_handler=checkpoint_handler
        )

        optimized_configuration.setCalculator(calculator_1)
        optimized_configuration.update()

        # Continue only if SCF is converged
        if optimized_configuration.calculator().isConverged():
            n += 1
            conf_file_path = dir_name + "{:}_{:}_conf_{:}.hdf5".format(int(x), n-1, id_)
            band_file_path = dir_name + "{:}_{:}_band_{:}.hdf5".format(int(x), n-1, id_)

            if isMainProcess():
                with open(dir_name + "configurations.dat", "a") as f:
                    f.write(id_ + "\n")

            nlsave(conf_file_path, optimized_configuration)

            # Bandstructure
            bandstructure = Bandstructure(
                configuration=optimized_configuration,
                route=['L', 'G', 'X', 'U', 'G'],
                bands_above_fermi_level=10
            )
            # Save bandstructure
            nlsave(band_file_path, bandstructure)


def main():

    try:
        percentage = float(sys.argv[1])
    except IndexError:
        raise ValueError("\nPlease enter the desired Sb percentage as an argument for the script")

    if percentage < 0. or percentage > 100.:
        raise ValueError("\nPlease enter the Sb content (%).\nIt must be a number between 0 and 100")

    with open("DSgen_options.json", ) as f:
        options = json.load(f)

    a_antimonide = options['a_antimonide']
    a_arsenide = options['a_arsenide']
    alpha_antimonide = options['alpha_antimonide']
    alpha_arsenide = options['alpha_arsenide']
    n_ens = options['n_ens']  # alloy realizations for each percentage

    dir_name = "{:d}/".format(int(percentage))
    if isMainProcess():
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)

    lin_lattice_const, lin_alpha = linearInterpolations(a_antimonide, a_arsenide, alpha_antimonide, alpha_arsenide)
    a = lin_lattice_const(percentage)
    alpha = lin_alpha(percentage)

    cation = options['cation']
    arsenide, antimonide = createConstituents(a, cation)

    relax = options['relax']
    run_ensemble(percentage, alpha, n_ens, arsenide, antimonide, dir_name, relax=relax)


if __name__ == "__main__":
    main()
