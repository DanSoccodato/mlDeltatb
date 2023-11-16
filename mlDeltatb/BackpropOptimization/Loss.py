import tensorflow as tf

from .BandsOps import computeBandstructure
from .BandsOps import alignBands, shiftAndCropBands, getBandsAttributes

from NL.CommonConcepts import PhysicalQuantity as Units


def bandstructureLoss(network_output, correction_model, target_objects, n_kpoints, k_downsampling, emin_emax, n_atoms):
    """
    Defines a custom Mean-Squared Error Loss function, that compares the band structures.

    :param network_output:
        The corrections coming from the output of the network. The shape is (batch_size*n_atoms, 4)
    :type network_output: tf.Tensor

    :param correction_model:
        The class defining the type of correction. Needed to create the correction and to determine
        the proper way of adding it to the Hamiltonian.
    :type correction_model:  MLCorrection

    :param target_objects:
        In the first entry, the target band structure objects used for comparing the result of the ML correction. 
        In the second entry, the weight to associate to that target.
        The shape is (batch_size, 2)
    :type target_objects: list of [Bandstructure, float]

    :param n_kpoints:
        The number of kpoints to consider when fitting the band structures. The range considered will be [0, n_kpoints]
    :type n_kpoints: int

    :param k_downsampling:
        Parameter to determine how many k-points to consider. Only one every `k_downsampling` points wil be considered.
        Useful to perform a downsampling of the ML and target band structures
    :type k_downsampling: int

    :param emin_emax:
        Bounds of the energy window for Band structure fitting
    :type emin_emax: tuple of floats

    :param n_atoms:
        Number of atoms in each of the structures in the training set
    :type n_atoms:  int


    :return loss:
        The loss as an average over all the batch computations.
    :rtype:  float
    """
    batch_loss = []
    batched_corrections = correction_model.createDiagonalCorrection(network_output, n_atoms)

    # Expecting batch: compute loss for each structure in batch.
    for i, (corrections, target_object) in enumerate(zip(batched_corrections, target_objects)):

        target_bandstructure = target_object[0]
        weight = target_object[1]

        configuration = target_bandstructure._configuration()
        kpoints = target_bandstructure.kpoints()

        emin = emin_emax[0]
        emax = emin_emax[1] + target_bandstructure.indirectBandGap().inUnitsOf(Units.eV)

        # Compute bands using TensorFlow and the ML-corrected Hamiltonian
        computed_bands = computeBandstructure(i, kpoints[:n_kpoints:k_downsampling],
                                              corrections, correction_model)

        # Find necessary band quantities
        computed_valence_edge, computed_occupied_bands = getBandsAttributes(configuration, computed_bands)

        # Shift computed bands and apply energy window
        computed_bands = shiftAndCropBands(computed_bands,
                                           computed_valence_edge,
                                           emin, emax)

        # Convert ATK Bandstructure to tensor
        target_tensor = tf.convert_to_tensor(target_bandstructure.evaluate()
                                             .inUnitsOf(Units.eV))[:n_kpoints:k_downsampling, :]
        # Shift target bands and apply energy window
        target_tensor = shiftAndCropBands(target_tensor,
                                          target_bandstructure.valenceBandEdge().inUnitsOf(Units.eV),
                                          emin, emax)

        # Align computed bands with target
        computed_bands, target_tensor = alignBands(computed_bands, computed_occupied_bands,
                                                   target_tensor, target_bandstructure._numberOfOccupiedBands()[0])

        # Compute the loss function
        loss = tf.math.square(computed_bands - target_tensor)
        batch_loss.append(weight * loss)

    return tf.reduce_mean(tf.concat(batch_loss, axis=-1))

