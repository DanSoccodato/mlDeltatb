from QuantumATK import Angstrom
from QuantumATK import KpointDensity
from QuantumATK import NumericalAccuracyParameters
from QuantumATK import SemiEmpiricalCalculator
from QuantumATK import SlaterKosterHamiltonianParametrization
from QuantumATK import SpinOrbit

from mltb.Common.ETBStartingPoint import basisSetFromFile


def attachCalculator(configuration, basis_set_path):
    """
    Creates a calculator and attaches it to the input configuration taken from the target band structure.
    The basis set for the calculator is the starting parametrization to improve with the ML method.

    :param configuration:
        The atomistic configuration, to which the calculator needs to be attached
    :type configuration:  BulkConfiguration

    :param basis_set_path:
        The user-defined path to the starting ETB parametrization
    :type basis_set_path:  str

    :return configuration:
        Configuration with the SemiEmpiricalCalculator attached
    :rtype:  BulkConfiguration
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

    hamiltonian_parametrization = SlaterKosterHamiltonianParametrization(
        basis_set)

    calculator = SemiEmpiricalCalculator(
        hamiltonian_parametrization=hamiltonian_parametrization,
        numerical_accuracy_parameters=numerical_accuracy_parameters,
        spin_polarization=SpinOrbit)

    configuration.setCalculator(calculator)

    return configuration
