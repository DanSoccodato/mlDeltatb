import tensorflow as tf
# import numpy
import sys
import os
import pickle
import json
import time
import datetime

from QuantumATK import Angstrom

from mlDeltatb.Common.Model import TBNN
from mlDeltatb.Common.DatasetProcessing import importDataset, processDataset
from mlDeltatb.Descriptors.Descriptor import MTPDescriptor
from mlDeltatb.BackpropOptimization.Loss import bandstructureLoss
from mlDeltatb.BackpropOptimization.Corrections import OnsiteSubshellCorrection, OnsiteOrbitalCorrection

NUM_THREADS = 16
print(f"\nSetting threading variables to {NUM_THREADS}.")
tf.config.threading.set_inter_op_parallelism_threads(NUM_THREADS)
tf.config.threading.set_intra_op_parallelism_threads(NUM_THREADS)


def loadGlobalVariables(input_path, output_path):

    if not os.path.exists(output_path + "/Results_train"):
        os.makedirs(output_path + "/Results_train")
    results_folder = output_path + "/Results_train"

    with open(input_path + "/config.json", ) as f:
        infos = json.load(f)

    infos['results_folder'] = results_folder
    infos['input_path'] = input_path

    return infos


def printInfo(infos, batched_x):
    print(f"  Dimension of atomistic descriptors: {infos['n_features']}")
    print(f"  Number of atoms per structure: {infos['n_atoms']}")
    print(f"  Batch size: {infos['batch_size']}")
    print(f"  Total number of structures: {infos['n_structures']}")
    print(f"  Total number of atoms: {sum([len(batch) for batch in batched_x])}")
    print(f"  Energy range for bandstructure fitting: {infos['emin_emax']} eV")
    print(f"  Fitting is being performed on {int(infos['n_kpoints']/infos['k_downsampling'])} kpoints, "
          f"on a down-sampled bandstructure (factor: 1/{infos['k_downsampling']}x)")
    print(f"  Saving results in: {infos['results_folder']}\n")


def main(dataset_path=".", input_path=".", output_path="."):

    # Configure I/O and options
    infos = loadGlobalVariables(input_path, output_path)

    # Import raw dataset
    raw_dataset, rng = importDataset(dataset_path, infos)

    # Use one structure in raw dataset to compute r_cut and initialize the descriptor
    # 1st neighbour r_cut
    # y_train = raw_dataset[1]
    # configuration = y_train[0][0]._configuration()
    # a = configuration.bravaisLattice().a() / 3.
    #
    # d_n1 = numpy.sqrt(3.0) / 4.0 * a
    # d_n2 = numpy.sqrt(2.0) / 2.0 * a
    # r_cut = (d_n1 + d_n2) / 2.

    # Hard-coded, 3rd neighbour r_cut (used in the paper)
    r_cut = 4.6 * Angstrom

    n_basis = 20

    results_folder = infos['results_folder']
    with open(results_folder + "/rcut.pickle", "wb") as f:
        pickle.dump(r_cut, f)

    descriptor = MTPDescriptor(r_cut, n_basis)

    # Process raw dataset
    x_train, y_train, x_test, y_test, = processDataset(raw_dataset, rng, descriptor, infos,
                                                       shuffle=True, normalize=True, save_moments=True)
    x_train = tf.convert_to_tensor(x_train)
    x_test = tf.convert_to_tensor(x_test)

    # Set the variables used in the script
    n_features = infos["n_features"] = len(x_train[0, 0, :])
    n_structures = infos["n_structures"] = len(y_train)
    n_atoms = infos['n_atoms']
    emin_emax = tuple(infos['emin_emax'])
    n_kpoints = infos['n_kpoints']
    k_downsampling = infos['k_downsampling']
    batch_size = infos['batch_size']
    epochs = infos['training']['epochs']
    checkpoint_epochs = infos['training']['checkpoint_epochs']
    correction_type = infos['model']['correction_type']

    # Reshape x to disassemble structures
    x_train = tf.reshape(x_train, [-1, n_features])

    # Batch the dataset
    batched_x = [x_train[i*n_atoms:(i + batch_size)*n_atoms] for i in range(0, n_structures, batch_size)]
    batched_y = [y_train[i:i + batch_size] for i in range(0, n_structures, batch_size)]

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

    resume_training = infos["training"]["resume_training"]
    resume_epoch = infos["training"]["resume_epoch"]

    model.build((None, n_features))
    if resume_training:
        # Use previously calculated weights to resume the training
        print(f"  Loading saved_weights_{resume_epoch}.pickle ...")
        with open(input_path + f"/saved_weights_{resume_epoch}.pickle", "rb") as f:
            initial_weights = pickle.load(f)
        model.set_weights(initial_weights)
        print(f"  Done.")
    else:
        # Use pre-trained weights to initialize the model variables
        print(f"  Loading pre-trained weights...")
        with open(input_path + "/pre_trained_weights.pickle", "rb") as f:
            initial_weights = pickle.load(f)
        model.set_weights(initial_weights)
        print(f"  Done.")

    printInfo(infos, batched_x)

    def train_step(descriptors, target_objects, infos):

        with tf.GradientTape() as tape:
            output = model(descriptors, training=True)

            # Collect configurations
            configurations = []
            for bands, loss_weight in target_objects:
                configurations.append(bands._configuration())

            # Define type of correction
            basis_set_path = infos["input_path"] + "/" + infos["start_basis_set_name"]
            if correction_type == "onsite_subshell":
                correction_model = OnsiteSubshellCorrection(configurations, basis_set_path)

            elif correction_type == "onsite_orbital":
                correction_model = OnsiteOrbitalCorrection(configurations, basis_set_path)

            loss = bandstructureLoss(output, correction_model, target_objects,
                                     n_kpoints, k_downsampling, emin_emax, n_atoms)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        return loss

    # Define tensorflow optimizer
    optimizer = tf.keras.optimizers.Adam()

    # Train
    print(f"Starting training ({epochs} epochs). Results will be saved every {checkpoint_epochs} epochs.")
    if resume_training:
        start_epoch = resume_epoch + 1
        print(f"Resuming training from epoch {start_epoch}")
    else:
        start_epoch = 0
    st = time.time()
    for epoch in range(start_epoch, epochs):
        epoch_loss = 0
        for descriptors, target_objects in zip(batched_x, batched_y):
            epoch_loss += train_step(descriptors, target_objects, infos)

        if (epoch + 1) % checkpoint_epochs == 0:
            print(
                f'Epoch {epoch + 1}, '
                f'Loss: {epoch_loss} '
            )
            en = time.time()
            print(f"  Time elapsed: {str(datetime.timedelta(seconds=en - st))}")

            with open(results_folder + f"/saved_weights_{epoch + 1}.pickle", "wb") as f:
                pickle.dump(model.get_weights(), f)

            with open(results_folder + "/loss_history.dat", "a+") as f:
                line = f"Loss over epoch {epoch+1}/{epochs}:\t{epoch_loss}." \
                       f"\tTime elapsed: {str(datetime.timedelta(seconds=en - st))}\n"
                f.write(line)
            st = time.time()

    print(f"  Training done.\n")


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
