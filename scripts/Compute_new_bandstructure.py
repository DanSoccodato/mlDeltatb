import sys
import tensorflow as tf
import pickle
import numpy

from mlDeltatb.Common.Model import TBNN
from mlDeltatb.Descriptors.Descriptor import MTPDescriptor
from mlDeltatb.Common.ATKCorrection import correctedSemiEmpiricalCalculator


from QuantumATK import *


def buildModel(n_features, saved_weights_path, saved_weights_epoch):

    model = TBNN(4, [15, 10, 7])

    model.build((None, n_features))
    print(f"  Loading saved_weights_{saved_weights_epoch}.pickle ...")
    with open(saved_weights_path + f"/saved_weights_{saved_weights_epoch}.pickle", "rb") as f:
        initial_weights = pickle.load(f)

    model.set_weights(initial_weights)
    print(f"Done.")
    return model


def main(input_path, configuration_name, saved_weights_epoch, fermi_level_guess):

    conf = nlread(input_path + "/" + configuration_name, BulkConfiguration)[0]
    fermi_level_guess = float(fermi_level_guess)

    with open(input_path + "/rcut.pickle", "rb") as f:
        r_cut = pickle.load(f)

    with open(input_path + "/mean_x.pickle", "rb") as f:
        mean = pickle.load(f)

    with open(input_path + "/var_x.pickle", "rb") as f:
        variance = pickle.load(f)

    print("Computing the descriptor...")
    input_descriptors = MTPDescriptor(r_cut=r_cut, n_basis=20).computeFeatures(conf)
    input_descriptors = numpy.array(input_descriptors)
    input_descriptors = (input_descriptors - mean) / numpy.sqrt(variance)
    input_descriptors = tf.convert_to_tensor(input_descriptors)

    # Set the variables used in the script
    n_features = len(input_descriptors[0])

    # Initialize model
    print("\nInitializing the model...")
    model = buildModel(n_features, input_path, saved_weights_epoch)

    print(f"\nComputing the corrections on configuration {configuration_name}...")
    corrections = model(input_descriptors, training=False).numpy()
    print("Done.")

    corrections = corrections * eV
    corrections = numpy.reshape(corrections.inUnitsOf(Hartree), -1)

    basis_set_path = input_path + "/start_basis_set.pickle"
    calculator = correctedSemiEmpiricalCalculator(corrections, basis_set_path, correction=True)
    conf.setCalculator(calculator)

    bandstructure = Bandstructure(
        configuration=conf,
        route=['L', 'G', 'X'],
        diagonalization_method=IterativeDiagonalizationSolver(
            bands_around_fermi_level=500,
            target_fermi_level=fermi_level_guess * eV,
            subspace_dimension_factor=1.0,
            interior_eigenvalue_strategy=ShiftAndInvert
        )
    )
    # Save bandstructure
    nlsave(input_path + "/MLDeltaTB-prediction-" + configuration_name, bandstructure)


if __name__ == "__main__":

    try:
        args = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]

    except IndexError:
        raise ValueError("Please enter the following arguments to the script:\n"
                         "1. input_path\n"
                         "2. name of BulkConfiguration input file\n"
                         "3. epoch of the converged saved ML model (in the filename saved_weights_{epoch}.pickle)\n"
                         "4. the guess for the fermi level [eV]")

    main(*args)
