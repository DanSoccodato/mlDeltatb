import abc
import itertools
import os
import numpy
import scipy
import h5py as h5

from NL.CommonConcepts.BasicValuesChecks import NUMERIC_PY_TYPES
from NL.Dynamics.Utilities import checkAndSetPhysicalQuantity
from QuantumATK import *

from scaitools import moment_tensor_potentials as mtp
from scaitools.moment_tensor_potentials.basis import AbstractBasis
from scaitools.moment_tensor_potentials.basis import PredefinedBasis
from scaitools.moment_tensor_potentials.training import _BASES_DIRECTORY
from scaitools.moment_tensor_potentials.util import level


def atomicNumber(x):
    return x.atomicNumber()


class Descriptor:
    def __init__(self,
                 r_cut=None):
        """Base class for computing the local environment of an atom

        :param r_cut: The maximum neighbor distance, from the central atom, to include in the neighborhood.
                      |DEFAULT| 6.0 * Angstrom
        :type r_cut:  |PhysicalQuantity| of type float
        """

        r_cut = checkAndSetPhysicalQuantity(
            r_cut,
            default=3.5 * Angstrom,
            name='r_cut',
            min_value=0.1 * Angstrom,
        )

        self._r_cut = r_cut

    def getNeighbours(self, configuration):
        """Find the neighbors of every atom in the configuration, based on the cut radius r_cut.

        :param configuration:
            The atomic configuration.
        :type configuration:  |BULK_CONFIGURATION|

        :returns reduced_neighbors :
            A list containing the indices of the neighbors for any atom in the configuration
        :rtype:  list of lists
        """
        # Repeat configuration since kd-tree does not support periodic boundary conditions.
        repeats = [3, 3, 3]
        repeated_configuration = configuration.repeat(*repeats)

        # Extract out the indices of the central cell.
        central_cell_min_index = len(configuration) * (numpy.prod(repeats) // 2)
        central_cell_max_index = central_cell_min_index + len(configuration)
        central_cell_indices = numpy.arange(central_cell_min_index, central_cell_max_index)

        coordinates = repeated_configuration.cartesianCoordinates().inUnitsOf(Angstrom)

        # Setup kd-tree to find neighbors.
        kd_tree = scipy.spatial.cKDTree(coordinates)

        # Find neighbors. Note, this includes the index of the central atom.
        neighbors = kd_tree.query_ball_point(
            coordinates[central_cell_indices],
            self._r_cut.inUnitsOf(Angstrom),
        )

        # If only one atom is queried then fix up the neighbours list to be a list of lists.
        if not isinstance(neighbors[0], list):
            neighbors = [neighbors]

        reduced_neighbours = []
        for i, central_cell_index in enumerate(central_cell_indices):
            # Remove the central atom from the neighbor list.
            local_neighbors = [n for n in neighbors[i] if n != central_cell_index]
            # Wrap the indices back to the central cell
            wrapped_neighbors = [(n - central_cell_min_index) % len(configuration) for n in local_neighbors]
            reduced_neighbours.append(wrapped_neighbors)

        return reduced_neighbours

    @abc.abstractmethod
    def computeFeatures(self, configuration):
        pass


class CountSpeciesDescriptor(Descriptor):
    """Simple descriptor that counts the atomic species in the neighbourhood."""

    def __init__(self, r_cut=None):
        super(CountSpeciesDescriptor, self).__init__(r_cut)

    def computeFeatures(self, configuration):
        """
        :param configuration:
            The atomic configuration.
        :type configuration:  |BULK_CONFIGURATION|

        :returns descriptors:
            A list of descriptors, one for each atom.  Each descriptor is the count of elements present
            in the neighbourhood (including the central atom)
        :rtype:  list of lists
        """

        neighbours = self.getNeighbours(configuration)

        unique_elements = sorted(configuration.uniqueElements(), key=atomicNumber)
        atom_list = configuration.elements()

        descriptors = []
        for i in range(len(configuration)):
            descriptor = []
            for element in unique_elements:
                central_atom = atom_list[i]
                count = 0
                # Count the central atom
                if central_atom is element:
                    count += 1

                # Count the neighbours
                for neighbour_index in neighbours[i]:
                    atom = atom_list[neighbour_index]
                    if atom is element:
                        count += 1
                descriptor.append(float(count))

            descriptors.append(numpy.array(descriptor))

        return descriptors


class CoulombMatrixDescriptor(Descriptor):
    """Descriptor based on the Coulomb Matrix"""
    def __init__(self, r_cut=None):
        super(CoulombMatrixDescriptor, self).__init__(r_cut)

    def computeFeatures(self, configuration, alpha=4.0):
        """
        :param configuration:
            The atomic configuration.
        :type configuration:  |BULK_CONFIGURATION|

        :param alpha:
        This decay parameter controls the weight of atoms further away from the central atom. A
        large value of alpha reduces the importantance of atoms that are further away.
        |DEFAULT| 4.0
        :type alpha:  float

        :returns descriptors:
            Two-dimensional array of descriptors, one line for each atom. Each descriptor is a "reduced Coulomb matrix",
            as described in: https://arxiv.org/pdf/1611.05126.pdf#page=5
        :rtype:  numpy.ndarray
        """

        if isinstance(alpha, NUMERIC_PY_TYPES) and not isinstance(alpha, bool):
            if alpha < 0.0:
                raise TypeError('The parameter, alpha, must be positive.')
        else:
            raise TypeError('The parameter, alpha, must be a positive float.')

        neighbours = self.getNeighbours(configuration)

        atomic_numbers = numpy.array(configuration.atomicNumbers())
        coordinates = configuration.cartesianCoordinates().inUnitsOf(Angstrom)

        max_neighbors = max(len(n) for n in neighbours)
        descriptors = numpy.zeros((len(configuration), 2 * max_neighbors + 1))

        # Loop over each atom and build the descriptor.
        for i in range(len(configuration)):
            # Get a reference to the descriptor for this atom.
            descriptor = descriptors[i]
            local_neighbours = neighbours[i]

            central_position = coordinates[i]

            # Calculate the distance from the central atom to its neighbors.
            distances = numpy.zeros(len(local_neighbours))
            for j, n in enumerate(local_neighbours):
                distances[j] = numpy.linalg.norm(central_position - coordinates[n])

            # The remaining elements of the descriptor are calculated sorted by distance from the
            # central atom.
            sorted_indices = numpy.argsort(distances)
            sorted_distances = distances[sorted_indices]
            sorted_atomic_numbers = atomic_numbers[local_neighbours][sorted_indices]

            # Calculate M_11.
            zi = atomic_numbers[i]
            descriptor[0] = 0.5 * zi ** 2.4

            for j, (distance, zj) in enumerate(zip(sorted_distances, sorted_atomic_numbers)):
                # Calculate M_1j.
                descriptor[j + 1] = zi * zj / ((2 * distance) ** alpha)
                # Calculate M_jj.
                descriptor[j + 1 + max_neighbors] = zj ** 2 / ((2 * distance) ** alpha)

        return descriptors


class MTPDescriptor(Descriptor):
    """Descriptor based on the Moment Tensor Potentials. See https://arxiv.org/pdf/1512.06054v1.pdf
        for more information."""

    def __init__(self, r_cut, n_basis=20):
        """
        :param n_basis:
            Dimensionality of the resulting atomistic descriptor. The MTPs are systematically improvable, so
            in principle increasing `n_basis` means improving the local description
        :type n_basis:  int
        """
        super(MTPDescriptor, self).__init__(r_cut)
        self._n_basis = n_basis

    def computeFeatures(self, configuration):
        """
        :param configuration:
            The atomic configuration.
        :type configuration:  |BULK_CONFIGURATION|

        :returns descriptors:
            A list of descriptors, one for each atom.  Each element in the descriptor is the contraction of various
            moment tensors as defined in the article linked above
        :rtype:  list of numpy.ndarray
         """

        n_basis = self._n_basis
        unique_elements = sorted(configuration.uniqueElements(), key=atomicNumber)

        cutoff_radii = {}

        for (e1, e2) in itertools.combinations_with_replacement(unique_elements, 2):
            cutoff_radii[(e1, e2)] = [1.0, 1.1, self._r_cut.inUnitsOf(Angstrom)] * Angstrom

        with h5.File(os.path.join(_BASES_DIRECTORY, PredefinedBasis.BIG.value), 'r') as f:
            big_basis = AbstractBasis.load_from_hdf5(f['basis'], cutoff_radii)

        alphas = big_basis.alphas
        alphas = sorted(alphas, key=level)
        basis = mtp.AbstractBasis(alphas[:n_basis], cutoff_radii)

        coeff_array = None

        descriptors = basis.evaluate_raw_fingerprints(configuration, coeff_array)
        descriptors = numpy.array(descriptors)

        # Extend the descriptor to take into account atomic number of central atom
        extended_descriptors = numpy.zeros((descriptors.shape[0], descriptors.shape[1] + 1))

        for i, (desc, el) in enumerate(zip(descriptors, configuration.elements())):
            extended_descriptors[i, 0] = float(el.atomicNumber())
            extended_descriptors[i, 1:] = desc

        return extended_descriptors
