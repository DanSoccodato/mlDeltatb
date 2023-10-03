import pickle
import os
import json
import numpy
import tensorflow as tf
import sys

from NL.CommonConcepts import PhysicalQuantity as Units

from mlDeltatb.Descriptors.Descriptor import MTPDescriptor
from mlDeltatb.Common.Model import TBNN
from mlDeltatb.Common.DatasetProcessing import importDataset, processDataset, normalizeDataset
from mlDeltatb.BackpropOptimization.BandsOps import computeBandstructure, alignBands, shiftAndCropBands, getBandsAttributes
from mlDeltatb.BackpropOptimization.Corrections import OnsiteSubshellCorrection, OnsiteOrbitalCorrection


def loadGlobalVariables(input_path, output_path):

    if not os.path.exists(output_path + "/Results_test"):
        os.makedirs(output_path + "/Results_test")
        os.makedirs(output_path + "/Results_test/bandstructures")
    results_folder = output_path + "/Results_test"

    with open(input_path + "/config.json", ) as f:
        infos = json.load(f)

    infos['input_path'] = input_path
    infos['results_folder'] = results_folder

    return infos


def printInfo(infos, x_train, x_test):
    total_atoms = 0
    for batch_train in x_train:
        total_atoms += len(batch_train)
    for batch_test in x_test:
        total_atoms += len(batch_test)

    print(f"  Dimension of atomistic descriptors: {infos['n_features']}")
    print(f"  Number of atoms per structure: {infos['n_atoms']}")
    print(f"  Total number of structures: {infos['n_structures']}")
    print(f"  Total number of atoms: {total_atoms}")
    print(f"  Energy range for bandstructure fitting: {infos['emin_emax']} eV")
    print(f"  Validation is being performed on {int(infos['n_kpoints']/infos['k_downsampling'])} kpoints, "
          f"on a down-sampled bandstructure (factor: 1/{infos['k_downsampling']}x).")
    print(f"  Saving results in: {infos['results_folder']}\n")


def predictBandstructure(corrections, target_bandstructure, n_atoms, n_kpoints, k_downsampling, emin_emax, infos):
    """
    Computes a bandstructure, shifts it and aligns it with the target.
    Useful for validating results. Does not work for batches.

    :param corrections:
        The output of the neural network. It is a collection of vectors of 4 entries, one for each orbital.
        The input shape is (n_atoms, 4)
    :type corrections: tf.Tensor

    :param target_bandstructure:
        The target band structure object used for comparing the result of the ML correction.
    :type target_bandstructure: Bandstructure

    :param n_atoms:
        The number of atoms in a single structure.
    :type n_atoms: int

    :param n_kpoints:
        The number of kpoints to consider when comparing the band structures. The range considered will be [0, n_kpoints]
    :type n_kpoints: int

    :param k_downsampling:
        Parameter to determine how many k-points to consider. Only one every `k_downsampling` points wil be considered.
        Useful to perform a downsampling of the ML- and target band structures
    :type k_downsampling: int

    :param emin_emax:
        Bounds of the energy window for Band structure fitting
    :type emin_emax: tuple of floats

    :return computed_bands, target_tensor:
        Two sets of tensors, containing the bands of the ML model and of the reference system
    :rtype:  tuple of tf.Tensor
    """

    configuration = target_bandstructure._configuration()

    correction_type = infos['model']['correction_type']
    basis_set_path = infos['input_path'] + "/" + infos['start_basis_set_name']
    if correction_type == 'onsite_subshell':
        correction_model = OnsiteSubshellCorrection([configuration], basis_set_path)
        diagonal_correction = correction_model.createDiagonalCorrection(corrections, n_atoms)[0]

    elif correction_type == 'onsite_orbital':
        correction_model = OnsiteOrbitalCorrection([configuration], basis_set_path)
        diagonal_correction = correction_model.createDiagonalCorrection(corrections, n_atoms)[0]

    else:
        raise NotImplementedError("Please define an existing correction_type in config.json."
                                  "\nPossible options are:\n"
                                  "- onsite_subshell")

    kpoints = target_bandstructure.kpoints()
    emin = emin_emax[0]
    emax = emin_emax[1] + target_bandstructure.indirectBandGap().inUnitsOf(Units.eV)

    computed_bands = computeBandstructure(0, kpoints[:n_kpoints+1:k_downsampling], diagonal_correction,
                                          correction_model)

    computed_valence_edge, computed_occupied_bands = getBandsAttributes(configuration, computed_bands)

    computed_bands = shiftAndCropBands(computed_bands,
                                       computed_valence_edge,
                                       emin, emax)

    target_tensor = tf.convert_to_tensor(target_bandstructure
                                         .evaluate().inUnitsOf(Units.eV))[:n_kpoints+1:k_downsampling, :]
    target_tensor = shiftAndCropBands(target_tensor,
                                      target_bandstructure.valenceBandEdge().inUnitsOf(Units.eV),
                                      emin, emax)

    computed_bands, target_tensor = alignBands(computed_bands, computed_occupied_bands,
                                               target_tensor, target_bandstructure._numberOfOccupiedBands()[0])

    return computed_bands, target_tensor


def evaluateAndPlot(corrections, target_bandstructure, n_atoms, n_kpoints, k_downsampling, emin_emax, infos, out_path):
    import matplotlib.pyplot as plt

    ml_bands, reference = predictBandstructure(corrections, target_bandstructure, n_atoms,
                                               n_kpoints, k_downsampling, emin_emax, infos)

    plt.plot(reference[:, :-1], 'b-')
    plt.plot(ml_bands[:, :-1], 'r--')
    plt.plot(reference[:, -1], 'b-', label='DFT')
    plt.plot(ml_bands[:, -1], 'r--', label='ML$\Delta$TB')
    plt.ylabel("Energy [eV]")
    plt.title('Bandstructure fit')
    plt.legend()
    plt.savefig(out_path, dpi=200)
    plt.close()
    loss = tf.reduce_mean(tf.math.square(ml_bands - reference))

    return loss, ml_bands, reference


def evaluateSet(epoch, model, x_set, y_set, infos, test_set=True, zero_correction=False):

    n_atoms = infos['n_atoms']
    n_kpoints = infos['n_kpoints']
    k_downsampling = infos['k_downsampling']
    emin_emax = tuple(infos['emin_emax'])
    results_folder = infos['results_folder']

    if test_set:
        loss_filename = f"/loss_test.dat"
    else:
        loss_filename = f"/loss_train.dat"
    set_loss = []

    for i, (x, y) in enumerate(zip(x_set, y_set)):
        print(f"  Evaluating model on bandstructure {i}...")

        if zero_correction:
            corrections = tf.zeros([len(x), 4])
        else:
            corrections = model(x, training=False)

        if test_set:
            img_filename = f"/eval_{i}"
            band_filename = f"/bandstructures/test_bandstructure_{i}_mlDeltatb.pickle"
            dft_filename = f"/bandstructures/test_bandstructure_{i}_dft.pickle"
        else:
            img_filename = f"/fit_after_{epoch}_epochs_{i}"
            band_filename = f"/bandstructures/train_bandstructure_{i}_mlDeltatb.pickle"
            dft_filename = f"/bandstructures/train_bandstructure_{i}_dft.pickle"

        target_bands = y[0]
        loss, bandstructure, reference = evaluateAndPlot(corrections, target_bands, n_atoms, n_kpoints, k_downsampling
                               , emin_emax, infos, results_folder + img_filename + ".png")
        set_loss.append(loss)

        with open(results_folder + loss_filename, "a+") as f:
            line = f"{i})\t{loss}\n"
            f.write(line)

        with open(results_folder + band_filename, "wb") as f:
            pickle.dump(bandstructure.numpy(), f)

        with open(results_folder + dft_filename, "wb") as f:
            pickle.dump(reference.numpy(), f)

    with open(results_folder + loss_filename, "a+") as f:
        line = f"\nMean value: {numpy.mean(set_loss)}"
        f.write(line)


def main(dataset_path=".", input_path=".", output_path="."):
    # Import raw dataset
    infos = loadGlobalVariables(input_path, output_path)
    raw_dataset, rng = importDataset(dataset_path, infos)

    with open(input_path + "/rcut.pickle", "rb") as f:
        r_cut = pickle.load(f)

    n_basis = 20

    descriptor = MTPDescriptor(r_cut, n_basis)

    # Process raw dataset
    x_train, y_train, x_test, y_test, = processDataset(raw_dataset, rng, descriptor, infos,
                                                       shuffle=False, normalize=False, save_moments=False)

    print("\nNormalizing the dataset using saved moments in: " + input_path)
    with open(input_path + "/mean_x.pickle", "rb") as f:
        mean = pickle.load(f)

    with open(input_path + "/var_x.pickle", "rb") as f:
        variance = pickle.load(f)

    # Normalize with moments from Training.py
    n_train = len(x_train)
    n_test = len(x_test)
    n_features = len(x_train[0][0])
    x_train, x_test = normalizeDataset(x_train, x_test, n_train, n_test, n_features,
                                       infos, save_moments=False, mean_x=mean, var_x=variance)

    x_train = tf.convert_to_tensor(x_train)
    x_test = tf.convert_to_tensor(x_test)

    # Configure I/O and options
    n_features = infos['n_features'] = len(x_train[0, 0, :])
    infos["n_structures"] = len(y_train) + len(y_test)
    correction_type = infos['model']['correction_type']

    # Initialize model
    print("\nInitializing the model...")
    if correction_type == 'onsite_subshell':
        output_nodes = 4

    elif correction_type == 'onsite_orbital':
        output_nodes = 10

    else:
        raise NotImplementedError("Please define an existing correction_type in config.json."
                                  "\nPossible options are:\n"
                                  "- onsite_subshell\n"
                                  "- onsite_orbital")

    layer_nodes = infos['model']['layer_nodes']
    model = TBNN(output_nodes, layer_nodes)
    model.build((None, n_features))

    printInfo(infos, x_train, x_test)
    epoch_min = infos["validation"]["epoch_min"]
    epoch_max = infos["validation"]["epoch_max"]
    epoch_step = infos["validation"]["epoch_step"]

    for epoch in range(epoch_min, epoch_max + epoch_step, epoch_step):
        print(f"\nElaborating results of epoch {epoch}")

        print(f"  Loading saved weights...")
        with open(input_path + f"/saved_weights_{epoch}.pickle", "rb") as f:
            initial_weights = pickle.load(f)
        model.set_weights(initial_weights)
        print(f"  Done.")

        print("\n  Evaluation in Training set:")
        evaluateSet(epoch, model, x_train, y_train, infos, test_set=False)
        print("  Done.")

        print("\n  Evaluation in Test set:")
        evaluateSet(epoch, model, x_test, y_test, infos, test_set=True)
        print("  Done.")


if __name__ == "__main__":
    args = sys.argv
    if len(args) == 1:
        main()

    elif len(args) == 4:
        main(args[1], args[2], args[3])

    else:
        raise ValueError("\nPossible input arguments:\n"
                         "1. dataset path;\n"
                         "2. input path;\n"
                         "3. output path.\n"
                         "Please provide all arguments or none (default paths).")
