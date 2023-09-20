"""
Module for the wrapper class around Slater Koster-type basis sets.
"""
import numpy

import NLEngine

from QuantumATK import Angstrom
from QuantumATK import KpointDensity
from QuantumATK import NumericalAccuracyParameters
from QuantumATK import SemiEmpiricalCalculator
from QuantumATK import SlaterKosterHamiltonianParametrization
from QuantumATK import SpinOrbit


from NL.Calculators.SemiEmpirical.Builders.AbstractSemiEmpiricalBuilder import loadAndCheckBasis
from NL.ComputerScienceUtilities.Functions import numpyToRealVector

from mltb.Common.ETBStartingPoint import basisSetFromFile


def correctedSemiEmpiricalCalculator(orbital_corrections, basis_set_path, correction=True):
    """Function that generates an ML-corrected HamiltonianParametrization and creates a SemiEmpiricalCalculator

    :param orbital_corrections:
        For each atom, the corrections to the 4 atomic orbitals s, p, d, s*.
    :type orbital_corrections:  numpy.ndarray

    :param basis_set_path:
        The user-defined path to the starting ETB parametrization
    :type basis_set_path:  str

    :param correction:
        Flag to specify if the correction is needed.
    :type correction: bool

    :return SemiEmpiricalCalculator:
        The ML-corrected calculator ready to be attached to a BulkConfiguration object
    """

    basis_set = basisSetFromFile(basis_set_path)
    k_point_sampling = KpointDensity(
        density_a=4.0 * Angstrom,
        density_b=4.0 * Angstrom,
        density_c=4.0 * Angstrom
    )

    numerical_accuracy_parameters = NumericalAccuracyParameters(
        k_point_sampling=k_point_sampling
    )

    if correction:
        corrections = createDiagonalCorrection(orbital_corrections)
        hamiltonian_parametrization = MLDiagonalCorrectionHamiltonianParametrization(basis_set, corrections)
    else:
        hamiltonian_parametrization = SlaterKosterHamiltonianParametrization(
            basis_set)
    return SemiEmpiricalCalculator(
        hamiltonian_parametrization=hamiltonian_parametrization,
        numerical_accuracy_parameters=numerical_accuracy_parameters,
        spin_polarization=SpinOrbit)


def createDiagonalCorrection(orbital_corrections):
    # There are 4 slots per atom but we need 10
    diagonal_correction = numpy.zeros(int(2.5 * len(orbital_corrections)))

    # Spread out the 4 corrections on the 10 entries for each atom
    for i in range(0, len(orbital_corrections), 4):
        s_0 = orbital_corrections[i]
        p_0 = orbital_corrections[i + 1]
        d_0 = orbital_corrections[i + 2]
        s_1 = orbital_corrections[i + 3]

        # 1 s-orbital
        diagonal_correction[int(2.5 * i)] = s_0
        # 3 p-orbitals
        diagonal_correction[int(2.5 * i) + 1:int(2.5 * i) + 4] = p_0
        # 5 d-orbitals
        diagonal_correction[int(2.5 * i) + 4:int(2.5 * i) + 9] = d_0
        # 1 s*-orbital
        diagonal_correction[int(2.5 * i) + 9] = s_1

    return diagonal_correction


class MLDiagonalCorrectionHamiltonianParametrization(SlaterKosterHamiltonianParametrization):

    def __init__(self, basis_set, onsite_correction=None):
        """
        An hamiltonian parametrization object which allows to introduce a diagonal Machine-Learned
        correction via a shift in the onsite energies of all the atoms in the structure.


        :param basis_set:
            An object describing the basis set used for the Slater-Koster calculation.
        :type basis_set:  :class:`~.SlaterKosterTable` |
                          :class:`~.DFTBDirectory` |
                          :class:`~.HotbitDirectory`

        :param onsite_correction:
            The array of diagonal corrections for the Hamiltonian
        :type onsite_correction:  numpy.ndarray

        """
        super().__init__(basis_set)
        self.__configuration = None
        self.__onsite_correction = onsite_correction

    def onsiteCorrection(self):
        """
        :return:
            The array of diagonal corrections for the Hamiltonian
        :rtype:  numpy.ndarray

        """
        return self.__onsite_correction

    def _backEngine(
            self,
            configuration,
            density_cutoff,
            interaction_maximum_range,
            number_of_reciprocal_points,
            reciprocal_energy_cutoff,
            enable_spin_orbit_split):
        """
        Create the underlying C++ matrix element calculator object.

        :param configuration:
            The configuration for which the matrix calculator is created.
        :type configuration:    |ALL_CONFIGURATIONS|

        :param density_cutoff:
            Not used.
        :type density_cutoff:   float

        :param interaction_maximum_range:
            The maximum allowed interaction distance between two orbitals.
        :type interaction_maximum_range:  |PHYSICALQUANTITY| of type length

        :param number_of_reciprocal_points:
            Not used.
        :type number_of_reciprocal_points:  int

        :param reciprocal_energy_cutoff:
            Not used.
        :type reciprocal_energy_cutoff:  |PHYSICALQUANTITY| of type energy

        :param enable_spinorbit_split:
            Flag to determine whether the spin-orbit splitting is included.
        :type enable_spinorbit_split:   bool

        :returns:   The C++ Huckel matrix calculator.
        :rype:      ``NLEngine.SlaterKosterCalculator``
        """
        onsite_correction_tags = [
            t for t in configuration.tags() if t in self.onsiteCorrection().keys()]
        self.__onsite_correction_identifier = list(configuration.elements())
        for tag in onsite_correction_tags:
            for i in configuration.indicesFromTags([tag]):
                self.__onsite_correction_identifier[i] = tag

        self.__configuration = configuration

        matrix_calculator = super()._backEngine(
            configuration,
            density_cutoff,
            interaction_maximum_range,
            number_of_reciprocal_points,
            reciprocal_energy_cutoff,
            enable_spin_orbit_split)

        return matrix_calculator

    def _applyCustomHamiltonianCorrection(self, density_matrix_calculator):
        """
        Add the onsite shift correction to a given density matrix calculator.
        """

        correction = self.GetAndCheckOnsiteCorrection(self.__onsite_correction, density_matrix_calculator)
        correction = numpyToRealVector(correction)
        overlap = density_matrix_calculator.overlap()
        hamiltonian_container = density_matrix_calculator.h0()

        NLEngine.shiftSparseMatrix_distributed(
            hamiltonian_container.hamiltonian().getSparseCSR(NLEngine.UPUP),
            overlap.getSparseCSR(NLEngine.UPUP),
            correction)

        if density_matrix_calculator.spinType() > NLEngine.UNPOLARIZED:
            NLEngine.shiftSparseMatrix_distributed(
                hamiltonian_container.hamiltonian().getSparseCSR(NLEngine.DOWN),
                overlap.getSparseCSR(NLEngine.DOWN),
                correction)

    def _createReducedParametrization(self, configuration):
        """
        Method for creating a reduced parametrization matching
        the elements in the given configuration. The method is needed
        for compatibility, but it returns always the same parameterization as
        we have no way to strip out data from the parameter dictionary.

        :param configuration:
            The configuration, the elements of which the reduced
            basis set corresponds to.
        :type configuration:    |ALL_CONFIGURATIONS|

        :returns:   The reduced Hamiltonian parametrization.
        :rtype:     :class:`~.SlaterKosterHamiltonianParametrization`
        """
        reduced_basis_set = loadAndCheckBasis(configuration, self.basisSet())
        hamiltonian_parametrization = MLDiagonalCorrectionHamiltonianParametrization(
            reduced_basis_set, self.onsiteCorrection())
        return hamiltonian_parametrization


    def GetAndCheckOnsiteCorrection(self, onsite_correction, density_matrix_calculator):
        """
        Utility to check that the onsite shift parameters are well defined.
        """
        if onsite_correction is None:
            return None

        # Check that it is an array of length: n_atoms*n_orbitals_per_atom
        orbital_map = density_matrix_calculator.orbitalMap()
        n_orbitals = orbital_map.numberOfOrbitals()
        if n_orbitals != len(onsite_correction):
            raise ValueError("onsite_correction, of length {}, does not match number of orbitals "
                                          "in the structure (n_orbitals = {})"
                                          .format(len(onsite_correction), n_orbitals))
        return onsite_correction
