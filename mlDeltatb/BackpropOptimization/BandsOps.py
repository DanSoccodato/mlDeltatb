import tensorflow as tf

import NLEngine
from NL.Analysis.AnalysisUtilities import numberOfElectrons


def computeBandstructure(i, kpoints, corrections, correction_model):
    """
    Uses the TensorFlow built-in eigensolver to compute the band structure. It is necessary in order to use
    the automatic derivatives and the back-propagation methods.
    """

    corrected_matrices = correction_model.correctHamiltonians(kpoints, corrections, i)
    computed_bands, _ = tf.linalg.eigh(corrected_matrices)

    return tf.cast(computed_bands, tf.float64)


def getBandsAttributes(configuration, computed_bands):
    number_of_electrons = numberOfElectrons(configuration)
    number_of_occupied_bands = int(round(number_of_electrons)) // 2

    spin_multiplier = 2 if configuration.calculator()._spinType() >= NLEngine.UNPOLARIZED else 1
    number_of_occupied_bands *= spin_multiplier  # accounts for spin

    # Find bandstructure characteristics
    valence_edge_index = number_of_occupied_bands - 1
    valence_edge = tf.reduce_max(computed_bands[:, valence_edge_index])

    return valence_edge, number_of_occupied_bands


def shiftAndCropBands(bands_tensor, valence_edge, emin, emax):
    # Shift bands
    bands_tensor -= valence_edge
    # Apply energy window
    bands_tensor = tf.where(bands_tensor > emin, bands_tensor, emin)
    bands_tensor = tf.where(bands_tensor < emax, bands_tensor, emax)

    return bands_tensor


def alignBands(computed_bands, computed_occupied_bands, target_tensor, target_occupied_bands):
    offset_target = max(0, target_occupied_bands - computed_occupied_bands)
    offset_computed = max(0, computed_occupied_bands - target_occupied_bands)

    nband = min(
        target_tensor.shape[1] - offset_target,
        computed_bands.shape[1] - offset_computed)

    target_tensor = target_tensor[
                    :,
                    offset_target:offset_target + nband]

    computed_bands = computed_bands[
                     :,
                     offset_computed:offset_computed + nband]

    return computed_bands, target_tensor


