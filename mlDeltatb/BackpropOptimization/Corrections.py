import tensorflow as tf
import numpy
import abc

from .ATKUtils import attachCalculator

import NLEngine

from QuantumATK import calculateHamiltonianAndOverlap
from QuantumATK import orbitalInfo
from QuantumATK import setVerbosity

from NL.CommonConcepts import PhysicalQuantity as Units
from NL.IO.LogRegionVerbosity import SilentLog


class MLCorrection:
    """
    Base class for defining the type of correction to the Hamiltonian
    """
    def __init__(self, configurations, basis_set_path):
        setVerbosity(SilentLog)

        self._configurations = []
        for conf in configurations:
            conf = attachCalculator(conf, basis_set_path)
            conf.update()
            self._configurations.append(conf)

        self._orbital_infos = [numpy.array(orbitalInfo(conf)) for conf in self._configurations]

    def getOrbitalInfos(self):
        return self._orbital_infos

    def getConfigurations(self):
        return self._configurations

    @abc.abstractmethod
    def createDiagonalCorrection(self, orbital_correction, n_atoms):
        pass

    @abc.abstractmethod
    def correctHamiltonians(self, kpoints, corrections, i):
        pass


class OnsiteSubshellCorrection(MLCorrection):
    """
    Correction on the s, p, d, s* orbitals, shell resolved. The output of the network has dimension 4.
    This is the type of correction defined in the paper.
    """

    def __init__(self, configurations, basis_set_patch):
        super(OnsiteSubshellCorrection, self).__init__(configurations, basis_set_patch)

    def createRepetitions(self, n_atoms):
        """
        Creates the vector of repetitions for each atom. Defines the remapping of the 4-dimensional output to the
        diagonal of the onsite block.

        :param n_atoms:
            The number of atoms composing each structure in the batch.
        :type n_atoms:  int

        :return repetitions:
            The vector containing how many times each of the 4 outputs needs to be repeated, for all the atoms in the
            batch. The output shape is: (batch_size*n_atoms, )
        :rtype tf.Tensor
        """
        configurations = self.getConfigurations()
        orbital_infos = self.getOrbitalInfos()

        repetitions_per_structure = []

        for configuration, orbital_info in zip(configurations, orbital_infos):
            spin_multiplier = 2 if configuration.calculator()._spinType() >= NLEngine.UNPOLARIZED else 1

            for atom in range(n_atoms):
                atomic_orbitals = orbital_info[orbital_info[:, 0] == atom]
                atomic_orbitals = atomic_orbitals[:, 1]

                count = 1
                for i in range(len(atomic_orbitals) - 1):
                    if atomic_orbitals[i + 1] == atomic_orbitals[i]:
                        count += 1
                    else:
                        repetitions_per_structure.append(spin_multiplier * count)
                        count = 1
                repetitions_per_structure.append(spin_multiplier * count)

        repetitions = tf.reshape(tf.convert_to_tensor(repetitions_per_structure), [-1])
        return repetitions

    def createDiagonalCorrection(self, orbital_correction, n_atoms):
        """
        Creates the vectors to be added on the diagonal of the target Hamiltonians. The 4-orbital corrections are mapped
        to 20 numbers for each atom.

        :param orbital_correction:
            The output of the neural network. It is a collection of vectors of 4 entries, one for each orbital.
            The input shape is (batch_size*n_atoms, 4)
        :type orbital_correction: tf.Tensor

        :param n_atoms:
            The number of atoms in a single structure. It is used to compute how many structures in a batch, and therefore
            to determine the output shape
        :type n_atoms:  int

        :return diagonal_correction:
            The vectors containing the diagonal correction for the Hamiltonians.
            The output shape is (batch_size, n_atoms*20) i.e. one diagonal per each structure in the batch
        :rtype:  tf.Tensor
        """

        # Necessary for output shape
        batch_size = len(orbital_correction) // n_atoms

        # Flatten tensor
        orbital_correction = tf.reshape(orbital_correction, [-1])
        # Define repetitions for the orbitals.
        repeats = self.createRepetitions(n_atoms)

        # Repeat corrections and stack them together
        diagonal_correction = tf.concat([tf.tile([o], [r]) for o, r in zip(orbital_correction, repeats)], axis=0)
        # Reshape to account for batches
        return tf.reshape(diagonal_correction, [batch_size, -1])

    def correctHamiltonians(self, kpoints, corrections, i):
        """
        Calls QuantumATK routine to compute Hamiltonians for each kpoint, and then sums the ML correction
        on the diagonal.
        """

        setVerbosity(SilentLog)
        corrected_hamiltonians = []

        configuration = self._configurations[i]

        for index, kpoint in enumerate(kpoints):
            hamiltonian, _ = calculateHamiltonianAndOverlap(configuration, kpoint=kpoint)
            hamiltonian = tf.convert_to_tensor(hamiltonian.inUnitsOf(Units.eV))

            # Correct the diagonal
            correction_matrix = tf.cast(tf.linalg.diag(corrections), hamiltonian.dtype)
            hamiltonian += correction_matrix

            corrected_hamiltonians.append(hamiltonian)

        return tf.convert_to_tensor(corrected_hamiltonians)


class OnsiteOrbitalCorrection(MLCorrection):
    """
       Experimental correction on the s, p, d, s* orbitals, orbital resolved.
       The output of the network has dimension 10.
    """

    def __init__(self, configurations, basis_set_patch):
        super(OnsiteOrbitalCorrection, self).__init__(configurations, basis_set_patch)

    def createRepetitions(self, n_atoms):
        configurations = self.getConfigurations()
        orbital_infos = self.getOrbitalInfos()

        repetitions_per_structure = []

        for configuration, orbital_info in zip(configurations, orbital_infos):
            spin_multiplier = 2 if configuration.calculator()._spinType() >= NLEngine.UNPOLARIZED else 1

            for atom in range(n_atoms):
                atomic_orbitals = orbital_info[orbital_info[:, 0] == atom]
                atomic_orbitals = atomic_orbitals[:, 1]

                repetitions_per_structure.append(len(atomic_orbitals)*[spin_multiplier])

        repetitions = tf.reshape(tf.convert_to_tensor(repetitions_per_structure), [-1])
        return repetitions

    def createDiagonalCorrection(self, orbital_correction, n_atoms):
        """
        Creates the vectors to be added on the diagonal of the target Hamiltonians. The 4-orbital corrections are mapped
        to 20 numbers for each atom.

        :param orbital_correction:
            The output of the neural network. It is a collection of vectors of 4 entries, one for each orbital.
            The input shape is (batch_size*n_atoms, 4)
        :type orbital_correction: tf.Tensor

        :param n_atoms:
            The number of atoms in a single structure. It is used to compute how many structures in a batch, and therefore
            to determine the output shape
        :type n_atoms:  int

        :return diagonal_correction:
            The vectors containing the diagonal correction for the Hamiltonians.
            The output shape is (batch_size, n_atoms*20) i.e. one diagonal per each structure in the batch
        :rtype:  tf.Tensor
        """

        # Necessary for output shape
        batch_size = len(orbital_correction) // n_atoms

        # Flatten tensor
        orbital_correction = tf.reshape(orbital_correction, [-1])
        # Define repetitions for the orbitals.
        repeats = self.createRepetitions(n_atoms)

        # Repeat corrections and stack them together
        diagonal_correction = tf.concat([tf.tile([o], [r]) for o, r in zip(orbital_correction, repeats)], axis=0)
        # Reshape to account for batches
        return tf.reshape(diagonal_correction, [batch_size, -1])

    def correctHamiltonians(self, kpoints, corrections, i):
        """
        Calls QuantumATK routine to compute Hamiltonians for each kpoint, and then sums the ML correction on the diagonal.
        """

        setVerbosity(SilentLog)
        corrected_hamiltonians = []

        configuration = self._configurations[i]

        for index, kpoint in enumerate(kpoints):
            hamiltonian, _ = calculateHamiltonianAndOverlap(configuration, kpoint=kpoint)
            hamiltonian = tf.convert_to_tensor(hamiltonian.inUnitsOf(Units.eV))

            # Correct the diagonal
            correction_matrix = tf.cast(tf.linalg.diag(corrections), hamiltonian.dtype)
            hamiltonian += correction_matrix

            corrected_hamiltonians.append(hamiltonian)

        return tf.convert_to_tensor(corrected_hamiltonians)
