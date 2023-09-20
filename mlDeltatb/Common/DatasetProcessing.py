import pickle

import numpy
import fnmatch
import os

from QuantumATK import Bandstructure
from QuantumATK import nlread


def readBandstructureFiles(directory):
    bandstructure_list = []

    for file in os.listdir(directory):
        if fnmatch.fnmatch(file, "*_band_*.hdf5"):
            bandstructure_list.append(nlread(directory + file, Bandstructure)[0])

    return bandstructure_list


def importDataset(input_path, infos):
    """
    :param input_path:
        Path where the ab-initio dataset is stored
    :type input_path:  str

    :param infos:
        The dictionary with all the options. Used to get the input concentrations
    :type infos:  dict

    :returns dataset, rng:
        The first argument is a tuple containing (x_train, y_train, x_test, y_test); where x_train and x_test contain
        only the BulkConfiguration objects (raw data). The second argument is a random number generator, used in this
        function to take a random structure for each Sb percentage and in the `processDataset` function to shuffle the
        dataset.
    :rtype  tuple, numpy.random._generator.Generator
    """

    concentrations = infos["dataset"]["concentrations"]
    repeats_per_concentration = infos["dataset"]["train_structures_per_concentration"]
    loss_weights = infos["dataset"]["loss_weights"]

    x_train = []
    y_train = []
    x_test = []
    y_test = []

    # Random (but reproducible) selection for the training set
    rng = numpy.random.default_rng(42)
    train_indices = []

    print("\nReading Dataset from files: ")
    for k, concentration in enumerate(concentrations):
        directory = input_path + "/" + concentration + "/"
        bandstructures = readBandstructureFiles(directory)
        configurations = [bs._configuration() for bs in bandstructures]

        partial_x = []
        partial_y = []
        weight = loss_weights[k]
        repeat = repeats_per_concentration[k]

        for i in range(len(configurations)):
            conf = configurations[i]
            bands = bandstructures[i]

            partial_x.append(conf)
            partial_y.append(bands)

        for r in range(repeat):
            index = rng.integers(0, len(partial_x))
            x_train.append(partial_x.pop(index))
            y_train.append((partial_y.pop(index), weight))
            train_indices.append(index)

        for x, y in zip(partial_x, partial_y):
            x_test.append(x)
            y_test.append((y, weight))

    print("  Training set is composed by: ", train_indices)

    print("Done.")
    return (x_train, y_train, x_test, y_test), rng


def processDataset(dataset, rng, descriptor, infos, shuffle=True, normalize=True, save_moments=True):
    """
    :param dataset:
        The raw dataset coming from the `importDataset` function.
        Needs to be a tuple of the kind: (x_train, y_train, x_test, y_test)
    :type dataset:  tuple

    :param rng:
        The random number generator coming from the `importDataset` function. Needed to perform consistent shuffling.
    :type rng: numpy.random._generator.Generator

    :param descriptor:
        The generic descriptor class initialized outside of this function. It will be used to compute the
        features vector through the method chosen by the user.
    :type descriptor: Descriptor

    :param infos:
        The dictionary with all the options. Used to get the results folder path
    :type infos:  dict

    :param shuffle:
        Optional flag that determines if training and test set will be shuffled
    :type shuffle:  Bool

    :param normalize:
        Optional flag that determines if training and test set will be normalized
    :type normalize:  Bool

    :param save_moments:
        Optional flag that determines whether the mean and variance used to normalize the dataset need to be saved
    :type save_moments:  Bool

    :return: x_train, y_train, x_test, y_test
        The dataset already split in train and test sets.
        The output shapes for x_train and x_test are: (n_train, n_atoms, n_features) and (n_test, n_atoms, n_features)
    """

    raw_x_train, y_train, raw_x_test, y_test = dataset

    # Random (but reproducible) selection for the training set
    x_train = []
    x_test = []
    print("\nCreating Descriptors: ")
    for conf in raw_x_train:
        configuration_descriptors = descriptor.computeFeatures(conf)
        x_train.append(numpy.array(configuration_descriptors))

    for conf in raw_x_test:
        configuration_descriptors = descriptor.computeFeatures(conf)
        x_test.append(numpy.array(configuration_descriptors))

    if shuffle:
        print("\nShuffling the dataset...")
        x_train, y_train = shuffleDataset((x_train, y_train), rng)
        x_test, y_test = shuffleDataset((x_test, y_test), rng)

    else:
        x_train = numpy.array(x_train)
        y_train = numpy.array(y_train)

        x_test = numpy.array(x_test)
        y_test = numpy.array(y_test)

    if normalize:
        print("\nNormalizing the dataset...")
        n_train = len(x_train)
        n_test = len(x_test)
        n_features = len(x_train[0][0])

        x_train, x_test = normalizeDataset(x_train, x_test, n_train, n_test, n_features,
                                           infos, save_moments)
    else:
        results_folder = infos["results_folder"]
        mean = numpy.zeros((x_train.shape[-1]))
        variance = numpy.ones((x_train.shape[-1]))

        if save_moments:
            with open(results_folder + "/mean_x.pickle", "wb") as f:
                pickle.dump(mean, f)

            with open(results_folder + "/var_x.pickle", "wb") as f:
                pickle.dump(variance, f)

    return x_train, y_train, x_test, y_test


def shuffleDataset(dataset, generator):
    x, y = dataset
    shuffle_indices = list(range(len(x)))
    generator.shuffle(shuffle_indices)

    x = numpy.array(x)[shuffle_indices, ]
    y = numpy.array(y)[shuffle_indices, ]

    return x, y


def normalizeDataset(x_train, x_test, n_train, n_test, n_features, infos, save_moments=True,
                     mean_x=None, var_x=None):
    """
    Normalize the dataset. Computes the mean and the variance vectors across all atoms in the training set.
    In particular, one mean value and one variance value are computed for each of the `n_features` elements
    in the descriptor. This mean (and variance) vector is used to normalize both the training and the test set.

    :param x_train:
        Training set. It will determine the mean and variance vectors to be used for normalization.
        Input shape is: (n_train, n_atoms, n_features)
    :type x_train  numpy.ndarray

    :param x_test:
        Test set. It will be normalized using mean and variance vectors computed from `x_train`.
        Input shape is: (n_test, n_atoms, n_features)
    :type x_test  numpy.ndarray

    :param n_train:
        Number of structures present in the training set.
    :type n_train  int

    :param n_test:
        Number of structures present in the test set.
    :type n_test:  int

    :param n_features:
        Dimension of the descriptor
    :type n_features  int

    :param infos
        The dictionary with all the options.
    :type infos:  dict

    :param save_moments:
        Optional flag that determines whether the mean and variance used to normalize the dataset need to be saved
    :type save_moments:  Bool

    :param mean_x:
        Optional array that contains the externally-specified mean to normalize the dataset. If None, mean is calculated
        using x_train.
    :type mean_x  numpy.ndarray or None

    :param var_x:
        Optional array that contains the externally-specified variance to normalize the dataset. If None, variance is
        calculated using x_train.
    :type var_x  numpy.ndarray or None

    :return:
    The normalized x_train and x_test. Shapes are respectively:
    (n_train, n_atoms, n_features) and (n_test, n_atoms, n_features); where n_atoms is the inferred number of atoms
    present in each structure
    """

    x_train = x_train.reshape((-1, n_features))
    x_test = x_test.reshape((-1, n_features))

    if mean_x is None and var_x is None:
        mean = numpy.mean(x_train, axis=0)
        variance = numpy.var(x_train, axis=0)
    else:
        mean = mean_x
        variance = var_x

    normalized_train = (x_train - mean) / numpy.sqrt(variance)
    normalized_test = (x_test - mean) / numpy.sqrt(variance)

    normalized_train = normalized_train.reshape((n_train, -1, n_features))
    normalized_test = normalized_test.reshape((n_test, -1, n_features))

    if save_moments:
        results_folder = infos["results_folder"]
        with open(results_folder + "/mean_x.pickle", "wb") as f:
            pickle.dump(mean, f)

        with open(results_folder + "/var_x.pickle", "wb") as f:
            pickle.dump(variance, f)

    return normalized_train, normalized_test
