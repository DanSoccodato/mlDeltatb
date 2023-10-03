import sys
import tensorflow as tf
import numpy
import pickle
import json

from mltb.Common.Model import TBNN
from mltb.Common.DatasetProcessing import importDataset, processDataset
from mltb.Descriptors.Descriptor import MTPDescriptor


def loadGlobalVariables(input_path, output_path):

    with open(input_path + "/config.json", ) as f:
        infos = json.load(f)

    infos["input_path"] = input_path
    infos["results_folder"] = output_path

    return infos


def main(dataset_path, input_path, output_path):
    # Import raw dataset
    infos = loadGlobalVariables(input_path, output_path)

    raw_dataset, rng = importDataset(dataset_path, infos)

    # Use one structure in raw dataset to compute r_cut and initialize the descriptor
    y_train = raw_dataset[1]
    configuration = y_train[0][0]._configuration()
    a = configuration.bravaisLattice().a() / 3.

    # 1st neighbour r_cut
    # Use one structure in raw dataset to compute r_cut and initialize the descriptor
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

    descriptor = MTPDescriptor(r_cut, n_basis)

    # Process raw dataset
    x_train, y_train, x_test, y_test, = processDataset(raw_dataset, rng, descriptor, infos,
                                                       shuffle=True, normalize=True, save_moments=False)
    x_train = tf.convert_to_tensor(x_train)
    x_test = tf.convert_to_tensor(x_test)

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

    # Create dummy targets to force network output to be small
    y_zeros = [numpy.zeros((1, output_nodes)) for _ in range(len(x_train))]
    y_zeros = tf.convert_to_tensor(y_zeros)

    layer_nodes = infos['model']['layer_nodes']
    model = TBNN(output_nodes, layer_nodes)

    # Define tensorflow optimizer and metrics
    optimizer = tf.keras.optimizers.Adam()
    loss_fn = tf.keras.losses.MeanSquaredError()

    train_loss = tf.keras.metrics.Mean(name='train_loss')

    def train_step(in_vars, out_vars):

        with tf.GradientTape() as tape:
            predictions = model(in_vars, training=True)
            loss = loss_fn(out_vars, predictions)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        train_loss(loss)

    # Pre-train the model
    epochs = 3000
    for epoch in range(epochs):
        # Reset the metrics at the start of the next epoch
        train_loss.reset_states()

        train_step(x_train, y_zeros)

        if (epoch + 1) % 50 == 0:
            print(
                f'Epoch {epoch + 1}, '
                f'Loss: {train_loss.result()} '
            )

    print(f"Initial Corrections for all descriptors:\n")
    print(model(x_train))

    with open(output_path + "/pre_trained_weights.pickle", "wb") as f:
        pickle.dump(model.get_weights(), f)


if __name__ == "__main__":
    dataset_path = str(sys.argv[1])
    input_path = str(sys.argv[2])
    output_path = str(sys.argv[3])
    main(dataset_path, input_path, output_path)
