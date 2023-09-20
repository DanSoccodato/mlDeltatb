import os.path
import sys
import numpy

from QuantumATK import *

from mltb.TightBindingFitting.EmpiricalSlaterKosterUtilities import parameterDictionaryToHamiltonianParametrization
from mltb.TightBindingFitting.Fitter import ReferenceBandstructureFromMemory, FreeParameter, BoundParameter, TargetParameters
from mltb.TightBindingFitting.Fitter import SemiEmpiricalFitter

from NL.ComputerScienceUtilities.Exceptions import NLValueError
from NL.Calculators.SemiEmpirical.SemiEmpiricalCalculator import SemiEmpiricalCalculator
from NL.CommonConcepts import PhysicalQuantity as Units
from NL.CommonConcepts.PeriodicTable import Gallium, Aluminium, Antimony, Arsenic

from QuantumATK import nlread


def fitGaSb(filepath, output_path, optimizer, optimizer_kwargs):

    target_bands = nlread(filepath, Bandstructure)[0]
    reference = ReferenceBandstructureFromMemory(
        target_bands,
        object_id='GaSb',
        emin_emax=(-9.0, 6.0) * Units.eV
    )

    # Create and initialize the fitting parameters.
    params = TargetParameters()
    params['elements'] = [Gallium, Antimony]
    # Couple of atoms
    a = 6.0583
    d_n1 = numpy.sqrt(3.0) / 4.0 * a
    d_n2 = numpy.sqrt(2.0) / 2.0 * a
    params['Gallium', 'Antimony', 'nearest_neighbor_distance'] = d_n1
    params['Gallium', 'Antimony', 'second_neighbor_distance'] = d_n2
    params['Antimony', 'Gallium', 'nearest_neighbor_distance'] = d_n1
    params['Antimony', 'Gallium', 'second_neighbor_distance'] = d_n2

    # On site terms
    params['Gallium', 'orbitals'] = ['s0', 'p0', 'd0', 's1']
    params['Antimony', 'orbitals'] = ['s0', 'p0', 'd0', 's1']

    params['Gallium', 'number_of_valence_electrons'] = 3
    params['Antimony', 'number_of_valence_electrons'] = 5

    params['Gallium', 'ionization_potential', 's0'] = FreeParameter(-0.4003)
    params['Gallium', 'ionization_potential', 'p0'] = FreeParameter(6.3801)
    params['Gallium', 'ionization_potential', 'd0'] = FreeParameter(11.5944)
    params['Gallium', 'ionization_potential', 's1'] = FreeParameter(16.6388)
    params['Antimony', 'ionization_potential', 's0'] = FreeParameter(-4.9586)
    params['Antimony', 'ionization_potential', 'p0'] = FreeParameter(4.0735)
    params['Antimony', 'ionization_potential', 'd0'] = BoundParameter(('Gallium', 'ionization_potential', 'd0'))
    params['Antimony', 'ionization_potential', 's1'] = BoundParameter(('Gallium', 'ionization_potential', 's1'))

    params['Gallium', 'onsite_spin_orbit_split'] = [0., 2*0.0432, 0., 0.]
    params['Antimony', 'onsite_spin_orbit_split'] = [0., 2*0.4552, 0., 0.]

    params['Gallium', 'onsite_hartree_shift'] = ATK_U(Gallium, ['3p', '3p', '3p', '3p']).inUnitsOf(Units.eV)
    params['Antimony', 'onsite_hartree_shift'] = ATK_U(Antimony, ['3p', '3p', '3p', '3p']).inUnitsOf(Units.eV)

    params['Gallium', 'onsite_spin_split'] = ATK_W(Gallium, ['3p', '3p', '3p', '3p']).inUnitsOf(Units.eV)
    params['Antimony', 'onsite_spin_split'] = ATK_W(Antimony, ['3p', '3p', '3p', '3p']).inUnitsOf(Units.eV)

    # Off-site parameters
    params['Gallium', 'Antimony', 'hamiltonian_matrix_element', 's0s0s'] = FreeParameter(-1.3671)
    params['Gallium', 'Antimony', 'hamiltonian_matrix_element', 's0p0s'] = FreeParameter(2.7093)
    params['Antimony', 'Gallium', 'hamiltonian_matrix_element', 's0p0s'] = FreeParameter(2.5624)
    params['Gallium', 'Antimony', 'hamiltonian_matrix_element', 's0d0s'] = FreeParameter(-2.4274)
    params['Antimony', 'Gallium', 'hamiltonian_matrix_element', 's0d0s'] = FreeParameter(-2.6143)
    params['Antimony', 'Gallium', 'hamiltonian_matrix_element', 's1s0s'] = FreeParameter(-1.9813)
    params['Antimony', 'Gallium', 'hamiltonian_matrix_element', 's0s1s'] = FreeParameter(-1.6622)
    params['Gallium', 'Antimony', 'hamiltonian_matrix_element', 's1p0s'] = FreeParameter(2.4596)
    params['Antimony', 'Gallium', 'hamiltonian_matrix_element', 's1p0s'] = FreeParameter(3.0164)
    params['Gallium', 'Antimony', 'hamiltonian_matrix_element', 's1d0s'] = FreeParameter(-0.8007)
    params['Antimony', 'Gallium', 'hamiltonian_matrix_element', 's1d0s'] = FreeParameter(-0.8557)
    params['Gallium', 'Antimony', 'hamiltonian_matrix_element', 's1s1s'] = FreeParameter(-3.2355)
    params['Gallium', 'Antimony', 'hamiltonian_matrix_element', 'p0p0p'] = FreeParameter(-1.6809)
    params['Gallium', 'Antimony', 'hamiltonian_matrix_element', 'p0p0s'] = FreeParameter(4.4500)
    params['Gallium', 'Antimony', 'hamiltonian_matrix_element', 'p0d0p'] = FreeParameter(1.8670)
    params['Gallium', 'Antimony', 'hamiltonian_matrix_element', 'p0d0s'] = FreeParameter(-2.2429)
    params['Antimony', 'Gallium', 'hamiltonian_matrix_element', 'p0d0s'] = FreeParameter(-2.0377)
    params['Antimony', 'Gallium', 'hamiltonian_matrix_element', 'p0d0p'] = FreeParameter(1.9790)
    params['Gallium', 'Antimony', 'hamiltonian_matrix_element', 'd0d0s'] = FreeParameter(-1.2492)
    params['Gallium', 'Antimony', 'hamiltonian_matrix_element', 'd0d0p'] = FreeParameter(2.1970)
    params['Gallium', 'Antimony', 'hamiltonian_matrix_element', 'd0d0d'] = FreeParameter(-1.7451)

    # Build a dummy initial parameters dictionary.
    initial_parameters = params.asPlainDictionary()
    hamiltonian_parametrization = parameterDictionaryToHamiltonianParametrization(initial_parameters)
    # K-point sampling same as DFT calculation:
    k_point_sampling = KpointDensity(
        density_a=4.0 * Units.Angstrom,
        density_b=4.0 * Units.Angstrom,
        density_c=4.0 * Units.Angstrom
    )
    numerical_accuracy_parameters = NumericalAccuracyParameters(k_point_sampling=k_point_sampling)
    calculator = SemiEmpiricalCalculator(
        hamiltonian_parametrization=hamiltonian_parametrization,
        numerical_accuracy_parameters=numerical_accuracy_parameters,
        spin_polarization=SpinOrbit
    )

    # Fit.
    fitter = SemiEmpiricalFitter(
        targets=[
            reference],
        semi_empirical_calculator=calculator,
        parameters=params,
        optimizer=optimizer,
        optimizer_kwargs=optimizer_kwargs,
        dictionary_to_parametrization_method=parameterDictionaryToHamiltonianParametrization,
        filename_prefix=output_path + '/fitting_results/start_basis_set'
        )
    parameters = fitter.update()
    hamiltonian_parametrization = parameterDictionaryToHamiltonianParametrization(parameters)

    k_point_sampling = KpointDensity(
        density_a=4.0 * Units.Angstrom,
        density_b=4.0 * Units.Angstrom,
        density_c=4.0 * Units.Angstrom
    )
    numerical_accuracy_parameters = NumericalAccuracyParameters(k_point_sampling=k_point_sampling)

    calculator = SemiEmpiricalCalculator(
        hamiltonian_parametrization=hamiltonian_parametrization,
        numerical_accuracy_parameters=numerical_accuracy_parameters,
        spin_polarization=SpinOrbit)
    reference.plotComparison(calculator, output_path + '/fitting_results/Gasb_fit.png')

def fitGaAs(filepath, output_path, optimizer, optimizer_kwargs):

    target_bands = nlread(filepath, Bandstructure)[0]
    reference = ReferenceBandstructureFromMemory(
        target_bands,
        object_id='GaAs',
        emin_emax=(-9.0, 6.0) * Units.eV
    )

    # Create and initialize the fitting parameters.
    params = TargetParameters()
    params['elements'] = [Gallium, Arsenic]
    # Couple of atoms
    a = 5.65359
    d_n1 = numpy.sqrt(3.0) / 4.0 * a
    d_n2 = numpy.sqrt(2.0) / 2.0 * a
    params['Gallium', 'Arsenic', 'nearest_neighbor_distance'] = d_n1
    params['Gallium', 'Arsenic', 'second_neighbor_distance'] = d_n2
    params['Arsenic', 'Gallium', 'nearest_neighbor_distance'] = d_n1
    params['Arsenic', 'Gallium', 'second_neighbor_distance'] = d_n2

    # On site terms
    params['Gallium', 'orbitals'] = ['s0', 'p0', 'd0', 's1']
    params['Arsenic', 'orbitals'] = ['s0', 'p0', 'd0', 's1']

    params['Gallium', 'number_of_valence_electrons'] = 3
    params['Arsenic', 'number_of_valence_electrons'] = 5

    params['Gallium', 'ionization_potential', 's0'] = FreeParameter(-0.4028)
    params['Gallium', 'ionization_potential', 'p0'] = FreeParameter(6.3853)
    params['Gallium', 'ionization_potential', 'd0'] = FreeParameter(13.1023)
    params['Gallium', 'ionization_potential', 's1'] = FreeParameter(19.4220)
    params['Arsenic', 'ionization_potential', 's0'] = FreeParameter(-5.9819)
    params['Arsenic', 'ionization_potential', 'p0'] = FreeParameter(3.5820)
    params['Arsenic', 'ionization_potential', 'd0'] = BoundParameter(('Gallium', 'ionization_potential', 'd0'))
    params['Arsenic', 'ionization_potential', 's1'] = BoundParameter(('Gallium', 'ionization_potential', 's1'))

    params['Arsenic', 'onsite_spin_orbit_split'] = [0., 2*0.1824, 0., 0.]
    params['Gallium', 'onsite_spin_orbit_split'] = [0., 2*0.0408, 0., 0.]

    params['Gallium', 'onsite_hartree_shift'] = ATK_U(Gallium, ['3p', '3p', '3p', '3p']).inUnitsOf(Units.eV)
    params['Arsenic', 'onsite_hartree_shift'] = ATK_U(Arsenic, ['3p', '3p', '3p', '3p']).inUnitsOf(Units.eV)

    params['Gallium', 'onsite_spin_split'] = ATK_W(Gallium, ['3p', '3p', '3p', '3p']).inUnitsOf(Units.eV)
    params['Arsenic', 'onsite_spin_split'] = ATK_W(Arsenic, ['3p', '3p', '3p', '3p']).inUnitsOf(Units.eV)

    # Off-site parameters
    params['Gallium', 'Arsenic', 'hamiltonian_matrix_element', 's0s0s'] = FreeParameter(-1.6187)
    params['Gallium', 'Arsenic', 'hamiltonian_matrix_element', 's0p0s'] = FreeParameter(2.9382)
    params['Arsenic', 'Gallium', 'hamiltonian_matrix_element', 's0p0s'] = FreeParameter(2.4912)
    params['Gallium', 'Arsenic', 'hamiltonian_matrix_element', 's0d0s'] = FreeParameter(-2.4095)
    params['Arsenic', 'Gallium', 'hamiltonian_matrix_element', 's0d0s'] = FreeParameter(-2.7333)
    params['Arsenic', 'Gallium', 'hamiltonian_matrix_element', 's1s0s'] = FreeParameter(-1.9927)
    params['Arsenic', 'Gallium', 'hamiltonian_matrix_element', 's0s1s'] = FreeParameter(-1.5648)
    params['Gallium', 'Arsenic', 'hamiltonian_matrix_element', 's1p0s'] = FreeParameter(2.2086)
    params['Arsenic', 'Gallium', 'hamiltonian_matrix_element', 's1p0s'] = FreeParameter(2.1835)
    params['Gallium', 'Arsenic', 'hamiltonian_matrix_element', 's1d0s'] = FreeParameter(-0.6486)
    params['Arsenic', 'Gallium', 'hamiltonian_matrix_element', 's1d0s'] = FreeParameter(-0.6906)
    params['Gallium', 'Arsenic', 'hamiltonian_matrix_element', 's1s1s'] = FreeParameter(-3.6761)
    params['Gallium', 'Arsenic', 'hamiltonian_matrix_element', 'p0p0p'] = FreeParameter(-1.4572)
    params['Gallium', 'Arsenic', 'hamiltonian_matrix_element', 'p0p0s'] = FreeParameter(4.4094)
    params['Gallium', 'Arsenic', 'hamiltonian_matrix_element', 'p0d0p'] = FreeParameter(2.079)
    params['Gallium', 'Arsenic', 'hamiltonian_matrix_element', 'p0d0s'] = FreeParameter(-1.8002)
    params['Arsenic', 'Gallium', 'hamiltonian_matrix_element', 'p0d0s'] = FreeParameter(-1.7811)
    params['Arsenic', 'Gallium', 'hamiltonian_matrix_element', 'p0d0p'] = FreeParameter(1.7821)
    params['Gallium', 'Arsenic', 'hamiltonian_matrix_element', 'd0d0s'] = FreeParameter(-1.1409)
    params['Gallium', 'Arsenic', 'hamiltonian_matrix_element', 'd0d0p'] = FreeParameter(2.2030)
    params['Gallium', 'Arsenic', 'hamiltonian_matrix_element', 'd0d0d'] = FreeParameter(-1.9770)

    # Build a dummy initial parameters dictionary.
    initial_parameters = params.asPlainDictionary()
    hamiltonian_parametrization = parameterDictionaryToHamiltonianParametrization(initial_parameters)
    # K-point sampling same as DFT calculation:
    k_point_sampling = KpointDensity(
        density_a=4.0 * Units.Angstrom,
        density_b=4.0 * Units.Angstrom,
        density_c=4.0 * Units.Angstrom
    )
    numerical_accuracy_parameters = NumericalAccuracyParameters(k_point_sampling=k_point_sampling)
    calculator = SemiEmpiricalCalculator(
        hamiltonian_parametrization=hamiltonian_parametrization,
        numerical_accuracy_parameters=numerical_accuracy_parameters,
        spin_polarization=SpinOrbit
    )

    # Fit.
    fitter = SemiEmpiricalFitter(
        targets=[
            reference],
        semi_empirical_calculator=calculator,
        parameters=params,
        optimizer=optimizer,
        optimizer_kwargs=optimizer_kwargs,
        dictionary_to_parametrization_method=parameterDictionaryToHamiltonianParametrization,
        filename_prefix=output_path + '/fitting_results/start_basis_set'
        )
    parameters = fitter.update()
    hamiltonian_parametrization = parameterDictionaryToHamiltonianParametrization(parameters)

    k_point_sampling = KpointDensity(
        density_a=4.0 * Units.Angstrom,
        density_b=4.0 * Units.Angstrom,
        density_c=4.0 * Units.Angstrom
    )
    numerical_accuracy_parameters = NumericalAccuracyParameters(k_point_sampling=k_point_sampling)

    calculator = SemiEmpiricalCalculator(
        hamiltonian_parametrization=hamiltonian_parametrization,
        numerical_accuracy_parameters=numerical_accuracy_parameters,
        spin_polarization=SpinOrbit)
    reference.plotComparison(calculator, output_path + '/fitting_results/GaAs_fit.png')


def fitGaAs_and_GaSb(filepath_gaas, filepath_gasb, output_path, optimizer, optimizer_kwargs):

    target_bands_gaas = nlread(filepath_gaas, Bandstructure)[0]
    reference_gaas = ReferenceBandstructureFromMemory(
        target_bands_gaas,
        object_id='GaAs',
        emin_emax=(-9.0, 6.0) * Units.eV
    )

    target_bands_gasb = nlread(filepath_gasb, Bandstructure)[0]
    reference_gasb = ReferenceBandstructureFromMemory(
        target_bands_gasb,
        object_id='GaSb',
        emin_emax=(-9.0, 6.0) * Units.eV
    )

    # Create and initialize the fitting parameters.
    params = TargetParameters()
    params['elements'] = [Gallium, Arsenic, Antimony]

    # Couple of atoms
    a_sb = 6.0583
    d_n1_sb = numpy.sqrt(3.0) / 4.0 * a_sb
    d_n2_sb = numpy.sqrt(2.0) / 2.0 * a_sb
    a_as = 5.65359
    d_n1_as = numpy.sqrt(3.0) / 4.0 * a_as
    d_n2_as = numpy.sqrt(2.0) / 2.0 * a_as

    params['Gallium', 'Antimony', 'nearest_neighbor_distance'] = d_n1_sb
    params['Gallium', 'Antimony', 'second_neighbor_distance'] = d_n2_sb
    params['Antimony', 'Gallium', 'nearest_neighbor_distance'] = d_n1_sb
    params['Antimony', 'Gallium', 'second_neighbor_distance'] = d_n2_sb

    params['Gallium', 'Arsenic', 'nearest_neighbor_distance'] = d_n1_as
    params['Gallium', 'Arsenic', 'second_neighbor_distance'] = d_n2_as
    params['Arsenic', 'Gallium', 'nearest_neighbor_distance'] = d_n1_as
    params['Arsenic', 'Gallium', 'second_neighbor_distance'] = d_n2_as

    # On site terms
    params['Antimony', 'orbitals'] = ['s0', 'p0', 'd0', 's1']
    params['Gallium', 'orbitals'] = ['s0', 'p0', 'd0', 's1']
    params['Arsenic', 'orbitals'] = ['s0', 'p0', 'd0', 's1']

    params['Gallium', 'number_of_valence_electrons'] = 3
    params['Antimony', 'number_of_valence_electrons'] = 5
    params['Arsenic', 'number_of_valence_electrons'] = 5

    params['Gallium', 'ionization_potential', 's0'] = FreeParameter(-0.4028)
    params['Gallium', 'ionization_potential', 'p0'] = FreeParameter(6.3853)
    params['Gallium', 'ionization_potential', 'd0'] = FreeParameter(13.1023)
    params['Gallium', 'ionization_potential', 's1'] = FreeParameter(19.4220)
    params['Antimony', 'ionization_potential', 's0'] = FreeParameter(-4.9586)
    params['Antimony', 'ionization_potential', 'p0'] = FreeParameter(4.0735)
    params['Antimony', 'ionization_potential', 'd0'] = BoundParameter(('Gallium', 'ionization_potential', 'd0'))
    params['Antimony', 'ionization_potential', 's1'] = BoundParameter(('Gallium', 'ionization_potential', 's1'))
    params['Arsenic', 'ionization_potential', 's0'] = FreeParameter(-5.9819)
    params['Arsenic', 'ionization_potential', 'p0'] = FreeParameter(3.5820)
    params['Arsenic', 'ionization_potential', 'd0'] = BoundParameter(('Gallium', 'ionization_potential', 'd0'))
    params['Arsenic', 'ionization_potential', 's1'] = BoundParameter(('Gallium', 'ionization_potential', 's1'))

    params['Gallium', 'onsite_spin_orbit_split'] = [0., 2 * 0.0432, 0., 0.]
    params['Antimony', 'onsite_spin_orbit_split'] = [0., 2 * 0.4552, 0., 0.]
    params['Arsenic', 'onsite_spin_orbit_split'] = [0., 2 * 0.1824, 0., 0.]

    params['Gallium', 'onsite_hartree_shift'] = ATK_U(Gallium, ['3p', '3p', '3p', '3p']).inUnitsOf(Units.eV)
    params['Antimony', 'onsite_hartree_shift'] = ATK_U(Antimony, ['3p', '3p', '3p', '3p']).inUnitsOf(Units.eV)
    params['Arsenic', 'onsite_hartree_shift'] = ATK_U(Arsenic, ['3p', '3p', '3p', '3p']).inUnitsOf(Units.eV)

    params['Gallium', 'onsite_spin_split'] = ATK_W(Gallium, ['3p', '3p', '3p', '3p']).inUnitsOf(Units.eV)
    params['Antimony', 'onsite_spin_split'] = ATK_W(Antimony, ['3p', '3p', '3p', '3p']).inUnitsOf(Units.eV)
    params['Arsenic', 'onsite_spin_split'] = ATK_W(Arsenic, ['3p', '3p', '3p', '3p']).inUnitsOf(Units.eV)

    # Off-site parameters
    params['Gallium', 'Antimony', 'hamiltonian_matrix_element', 's0s0s'] = FreeParameter(-1.3671)
    params['Gallium', 'Antimony', 'hamiltonian_matrix_element', 's0p0s'] = FreeParameter(2.7093)
    params['Antimony', 'Gallium', 'hamiltonian_matrix_element', 's0p0s'] = FreeParameter(2.5624)
    params['Gallium', 'Antimony', 'hamiltonian_matrix_element', 's0d0s'] = FreeParameter(-2.4274)
    params['Antimony', 'Gallium', 'hamiltonian_matrix_element', 's0d0s'] = FreeParameter(-2.6143)
    params['Antimony', 'Gallium', 'hamiltonian_matrix_element', 's1s0s'] = FreeParameter(-1.9813)
    params['Antimony', 'Gallium', 'hamiltonian_matrix_element', 's0s1s'] = FreeParameter(-1.6622)
    params['Gallium', 'Antimony', 'hamiltonian_matrix_element', 's1p0s'] = FreeParameter(2.4596)
    params['Antimony', 'Gallium', 'hamiltonian_matrix_element', 's1p0s'] = FreeParameter(3.0164)
    params['Gallium', 'Antimony', 'hamiltonian_matrix_element', 's1d0s'] = FreeParameter(-0.8007)
    params['Antimony', 'Gallium', 'hamiltonian_matrix_element', 's1d0s'] = FreeParameter(-0.8557)
    params['Gallium', 'Antimony', 'hamiltonian_matrix_element', 's1s1s'] = FreeParameter(-3.2355)
    params['Gallium', 'Antimony', 'hamiltonian_matrix_element', 'p0p0p'] = FreeParameter(-1.6809)
    params['Gallium', 'Antimony', 'hamiltonian_matrix_element', 'p0p0s'] = FreeParameter(4.4500)
    params['Gallium', 'Antimony', 'hamiltonian_matrix_element', 'p0d0p'] = FreeParameter(1.8670)
    params['Gallium', 'Antimony', 'hamiltonian_matrix_element', 'p0d0s'] = FreeParameter(-2.2429)
    params['Antimony', 'Gallium', 'hamiltonian_matrix_element', 'p0d0s'] = FreeParameter(-2.0377)
    params['Antimony', 'Gallium', 'hamiltonian_matrix_element', 'p0d0p'] = FreeParameter(1.9790)
    params['Gallium', 'Antimony', 'hamiltonian_matrix_element', 'd0d0s'] = FreeParameter(-1.2492)
    params['Gallium', 'Antimony', 'hamiltonian_matrix_element', 'd0d0p'] = FreeParameter(2.1970)
    params['Gallium', 'Antimony', 'hamiltonian_matrix_element', 'd0d0d'] = FreeParameter(-1.7451)

    # Off-site parameters
    params['Gallium', 'Arsenic', 'hamiltonian_matrix_element', 's0s0s'] = FreeParameter(-1.6187)
    params['Gallium', 'Arsenic', 'hamiltonian_matrix_element', 's0p0s'] = FreeParameter(2.9382)
    params['Arsenic', 'Gallium', 'hamiltonian_matrix_element', 's0p0s'] = FreeParameter(2.4912)
    params['Gallium', 'Arsenic', 'hamiltonian_matrix_element', 's0d0s'] = FreeParameter(-2.4095)
    params['Arsenic', 'Gallium', 'hamiltonian_matrix_element', 's0d0s'] = FreeParameter(-2.7333)
    params['Arsenic', 'Gallium', 'hamiltonian_matrix_element', 's1s0s'] = FreeParameter(-1.9927)
    params['Arsenic', 'Gallium', 'hamiltonian_matrix_element', 's0s1s'] = FreeParameter(-1.5648)
    params['Gallium', 'Arsenic', 'hamiltonian_matrix_element', 's1p0s'] = FreeParameter(2.2086)
    params['Arsenic', 'Gallium', 'hamiltonian_matrix_element', 's1p0s'] = FreeParameter(2.1835)
    params['Gallium', 'Arsenic', 'hamiltonian_matrix_element', 's1d0s'] = FreeParameter(-0.6486)
    params['Arsenic', 'Gallium', 'hamiltonian_matrix_element', 's1d0s'] = FreeParameter(-0.6906)
    params['Gallium', 'Arsenic', 'hamiltonian_matrix_element', 's1s1s'] = FreeParameter(-3.6761)
    params['Gallium', 'Arsenic', 'hamiltonian_matrix_element', 'p0p0p'] = FreeParameter(-1.4572)
    params['Gallium', 'Arsenic', 'hamiltonian_matrix_element', 'p0p0s'] = FreeParameter(4.4094)
    params['Gallium', 'Arsenic', 'hamiltonian_matrix_element', 'p0d0p'] = FreeParameter(2.079)
    params['Gallium', 'Arsenic', 'hamiltonian_matrix_element', 'p0d0s'] = FreeParameter(-1.8002)
    params['Arsenic', 'Gallium', 'hamiltonian_matrix_element', 'p0d0s'] = FreeParameter(-1.7811)
    params['Arsenic', 'Gallium', 'hamiltonian_matrix_element', 'p0d0p'] = FreeParameter(1.7821)
    params['Gallium', 'Arsenic', 'hamiltonian_matrix_element', 'd0d0s'] = FreeParameter(-1.1409)
    params['Gallium', 'Arsenic', 'hamiltonian_matrix_element', 'd0d0p'] = FreeParameter(2.2030)
    params['Gallium', 'Arsenic', 'hamiltonian_matrix_element', 'd0d0d'] = FreeParameter(-1.9770)

    # Strain scaling
    params['Gallium', 'Antimony', 'hamiltonian_matrix_element_eta', 's0s0s'] = 2.
    params['Gallium', 'Antimony', 'hamiltonian_matrix_element_eta', 's0p0s'] = 2.
    params['Antimony', 'Gallium', 'hamiltonian_matrix_element_eta', 's0p0s'] = 2.
    params['Gallium', 'Antimony', 'hamiltonian_matrix_element_eta', 's0d0s'] = 2.
    params['Antimony', 'Gallium', 'hamiltonian_matrix_element_eta', 's0d0s'] = 2.
    params['Antimony', 'Gallium', 'hamiltonian_matrix_element_eta', 's1s0s'] = 2.
    params['Antimony', 'Gallium', 'hamiltonian_matrix_element_eta', 's0s1s'] = 2.
    params['Gallium', 'Antimony', 'hamiltonian_matrix_element_eta', 's1p0s'] = 2.
    params['Antimony', 'Gallium', 'hamiltonian_matrix_element_eta', 's1p0s'] = 2.
    params['Gallium', 'Antimony', 'hamiltonian_matrix_element_eta', 's1d0s'] = 2.
    params['Antimony', 'Gallium', 'hamiltonian_matrix_element_eta', 's1d0s'] = 2.
    params['Gallium', 'Antimony', 'hamiltonian_matrix_element_eta', 's1s1s'] = 2.
    params['Gallium', 'Antimony', 'hamiltonian_matrix_element_eta', 'p0p0p'] = 2.
    params['Gallium', 'Antimony', 'hamiltonian_matrix_element_eta', 'p0p0s'] = 2.
    params['Gallium', 'Antimony', 'hamiltonian_matrix_element_eta', 'p0d0p'] = 2.
    params['Gallium', 'Antimony', 'hamiltonian_matrix_element_eta', 'p0d0s'] = 2.
    params['Antimony', 'Gallium', 'hamiltonian_matrix_element_eta', 'p0d0s'] = 2.
    params['Antimony', 'Gallium', 'hamiltonian_matrix_element_eta', 'p0d0p'] = 2.
    params['Gallium', 'Antimony', 'hamiltonian_matrix_element_eta', 'd0d0s'] = 2.
    params['Gallium', 'Antimony', 'hamiltonian_matrix_element_eta', 'd0d0p'] = 2.
    params['Gallium', 'Antimony', 'hamiltonian_matrix_element_eta', 'd0d0d'] = 2.

    # Strain scaling
    params['Gallium', 'Arsenic', 'hamiltonian_matrix_element_eta', 's0s0s'] = 2.
    params['Gallium', 'Arsenic', 'hamiltonian_matrix_element_eta', 's0p0s'] = 2.
    params['Arsenic', 'Gallium', 'hamiltonian_matrix_element_eta', 's0p0s'] = 2.
    params['Gallium', 'Arsenic', 'hamiltonian_matrix_element_eta', 's0d0s'] = 2.
    params['Arsenic', 'Gallium', 'hamiltonian_matrix_element_eta', 's0d0s'] = 2.
    params['Arsenic', 'Gallium', 'hamiltonian_matrix_element_eta', 's1s0s'] = 2.
    params['Arsenic', 'Gallium', 'hamiltonian_matrix_element_eta', 's0s1s'] = 2.
    params['Gallium', 'Arsenic', 'hamiltonian_matrix_element_eta', 's1p0s'] = 2.
    params['Arsenic', 'Gallium', 'hamiltonian_matrix_element_eta', 's1p0s'] = 2.
    params['Gallium', 'Arsenic', 'hamiltonian_matrix_element_eta', 's1d0s'] = 2.
    params['Arsenic', 'Gallium', 'hamiltonian_matrix_element_eta', 's1d0s'] = 2.
    params['Gallium', 'Arsenic', 'hamiltonian_matrix_element_eta', 's1s1s'] = 2.
    params['Gallium', 'Arsenic', 'hamiltonian_matrix_element_eta', 'p0p0p'] = 2.
    params['Gallium', 'Arsenic', 'hamiltonian_matrix_element_eta', 'p0p0s'] = 2.
    params['Gallium', 'Arsenic', 'hamiltonian_matrix_element_eta', 'p0d0p'] = 2.
    params['Gallium', 'Arsenic', 'hamiltonian_matrix_element_eta', 'p0d0s'] = 2.
    params['Arsenic', 'Gallium', 'hamiltonian_matrix_element_eta', 'p0d0s'] = 2.
    params['Arsenic', 'Gallium', 'hamiltonian_matrix_element_eta', 'p0d0p'] = 2.
    params['Gallium', 'Arsenic', 'hamiltonian_matrix_element_eta', 'd0d0s'] = 2.
    params['Gallium', 'Arsenic', 'hamiltonian_matrix_element_eta', 'd0d0p'] = 2.
    params['Gallium', 'Arsenic', 'hamiltonian_matrix_element_eta', 'd0d0d'] = 2.

    # Build a dummy initial parameters dictionary.
    initial_parameters = params.asPlainDictionary()
    hamiltonian_parametrization = parameterDictionaryToHamiltonianParametrization(initial_parameters)
    # K-point sampling same as DFT calculation:
    k_point_sampling = KpointDensity(
        density_a=4.0 * Units.Angstrom,
        density_b=4.0 * Units.Angstrom,
        density_c=4.0 * Units.Angstrom
    )
    numerical_accuracy_parameters = NumericalAccuracyParameters(k_point_sampling=k_point_sampling)
    calculator = SemiEmpiricalCalculator(
        hamiltonian_parametrization=hamiltonian_parametrization,
        numerical_accuracy_parameters=numerical_accuracy_parameters,
        spin_polarization=SpinOrbit
    )

    # Fit.
    fitter = SemiEmpiricalFitter(
        targets=[
            reference_gaas, reference_gasb],
        semi_empirical_calculator=calculator,
        parameters=params,
        optimizer=optimizer,
        optimizer_kwargs=optimizer_kwargs,
        dictionary_to_parametrization_method=parameterDictionaryToHamiltonianParametrization,
        filename_prefix=output_path + '/fitting_results/start_basis_set'
        )
    parameters = fitter.update()
    hamiltonian_parametrization = parameterDictionaryToHamiltonianParametrization(parameters)

    k_point_sampling = KpointDensity(
        density_a=4.0 * Units.Angstrom,
        density_b=4.0 * Units.Angstrom,
        density_c=4.0 * Units.Angstrom
    )
    numerical_accuracy_parameters = NumericalAccuracyParameters(k_point_sampling=k_point_sampling)

    calculator = SemiEmpiricalCalculator(
        hamiltonian_parametrization=hamiltonian_parametrization,
        numerical_accuracy_parameters=numerical_accuracy_parameters,
        spin_polarization=SpinOrbit)
    reference_gaas.plotComparison(calculator, output_path + '/fitting_results/GaAs_fit.png')
    reference_gasb.plotComparison(calculator, output_path + '/fitting_results/GaSb_fit.png')


def fitAlAs_and_AlSb(filepath_alas, filepath_alsb, output_path, optimizer, optimizer_kwargs):

    target_bands_alas = nlread(filepath_alas, Bandstructure)[0]
    reference_alas = ReferenceBandstructureFromMemory(
        target_bands_alas,
        object_id='AlAs',
        emin_emax=(-10.0, 10.0) * Units.eV
    )

    target_bands_alsb = nlread(filepath_alsb, Bandstructure)[0]
    reference_alsb = ReferenceBandstructureFromMemory(
        target_bands_alsb,
        object_id='AlSb',
        emin_emax=(-10.0, 10.0) * Units.eV
    )

    # Create and initialize the fitting parameters.
    params = TargetParameters()
    params['elements'] = [Aluminium, Arsenic, Antimony]

    # Couple of atoms
    a_sb = 6.1355
    d_n1_sb = numpy.sqrt(3.0) / 4.0 * a_sb
    d_n2_sb = numpy.sqrt(2.0) / 2.0 * a_sb
    a_as = 5.6613
    d_n1_as = numpy.sqrt(3.0) / 4.0 * a_as
    d_n2_as = numpy.sqrt(2.0) / 2.0 * a_as

    params['Aluminium', 'Antimony', 'nearest_neighbor_distance'] = d_n1_sb
    params['Aluminium', 'Antimony', 'second_neighbor_distance'] = d_n2_sb
    params['Antimony', 'Aluminium', 'nearest_neighbor_distance'] = d_n1_sb
    params['Antimony', 'Aluminium', 'second_neighbor_distance'] = d_n2_sb

    params['Aluminium', 'Arsenic', 'nearest_neighbor_distance'] = d_n1_as
    params['Aluminium', 'Arsenic', 'second_neighbor_distance'] = d_n2_as
    params['Arsenic', 'Aluminium', 'nearest_neighbor_distance'] = d_n1_as
    params['Arsenic', 'Aluminium', 'second_neighbor_distance'] = d_n2_as

    # On site terms
    params['Antimony', 'orbitals'] = ['s0', 'p0', 'd0', 's1']
    params['Aluminium', 'orbitals'] = ['s0', 'p0', 'd0', 's1']
    params['Arsenic', 'orbitals'] = ['s0', 'p0', 'd0', 's1']

    params['Aluminium', 'number_of_valence_electrons'] = 3
    params['Antimony', 'number_of_valence_electrons'] = 5
    params['Arsenic', 'number_of_valence_electrons'] = 5

    params['Aluminium', 'ionization_potential', 's0'] = FreeParameter(0.9574)
    params['Aluminium', 'ionization_potential', 'p0'] = FreeParameter(6.3386)
    params['Aluminium', 'ionization_potential', 'd0'] = FreeParameter((13.0570 + 11.4691)/2.)
    params['Aluminium', 'ionization_potential', 's1'] = FreeParameter((19.5133 + 16.4173)/2.)
    params['Antimony', 'ionization_potential', 's0'] = FreeParameter(-4.9565)
    params['Antimony', 'ionization_potential', 'p0'] = FreeParameter(4.0739)
    params['Antimony', 'ionization_potential', 'd0'] = BoundParameter(('Aluminium', 'ionization_potential', 'd0'))
    params['Antimony', 'ionization_potential', 's1'] = BoundParameter(('Aluminium', 'ionization_potential', 's1'))
    params['Arsenic', 'ionization_potential', 's0'] = FreeParameter(-5.9819)
    params['Arsenic', 'ionization_potential', 'p0'] = FreeParameter(3.5826)
    params['Arsenic', 'ionization_potential', 'd0'] = BoundParameter(('Aluminium', 'ionization_potential', 'd0'))
    params['Arsenic', 'ionization_potential', 's1'] = BoundParameter(('Aluminium', 'ionization_potential', 's1'))

    params['Aluminium', 'onsite_spin_orbit_split'] = [0., 2 * (0.0072 + 0.0079)/2., 0., 0.]
    params['Antimony', 'onsite_spin_orbit_split'] = [0., 2 * 0.3912, 0., 0.]
    params['Arsenic', 'onsite_spin_orbit_split'] = [0., 2 * 0.1721, 0., 0.]

    params['Aluminium', 'onsite_hartree_shift'] = ATK_U(Aluminium, ['3p', '3p', '3p', '3p']).inUnitsOf(Units.eV)
    params['Antimony', 'onsite_hartree_shift'] = ATK_U(Antimony, ['3p', '3p', '3p', '3p']).inUnitsOf(Units.eV)
    params['Arsenic', 'onsite_hartree_shift'] = ATK_U(Arsenic, ['3p', '3p', '3p', '3p']).inUnitsOf(Units.eV)

    params['Aluminium', 'onsite_spin_split'] = ATK_W(Aluminium, ['3p', '3p', '3p', '3p']).inUnitsOf(Units.eV)
    params['Antimony', 'onsite_spin_split'] = ATK_W(Antimony, ['3p', '3p', '3p', '3p']).inUnitsOf(Units.eV)
    params['Arsenic', 'onsite_spin_split'] = ATK_W(Arsenic, ['3p', '3p', '3p', '3p']).inUnitsOf(Units.eV)

    # Off-site parameters
    params['Aluminium', 'Antimony', 'hamiltonian_matrix_element', 's0s0s'] = FreeParameter(-1.6179)
    params['Aluminium', 'Antimony', 'hamiltonian_matrix_element', 's0p0s'] = FreeParameter(2.9334)
    params['Antimony', 'Aluminium', 'hamiltonian_matrix_element', 's0p0s'] = FreeParameter(2.5918)
    params['Aluminium', 'Antimony', 'hamiltonian_matrix_element', 's0d0s'] = FreeParameter(-2.0008)
    params['Antimony', 'Aluminium', 'hamiltonian_matrix_element', 's0d0s'] = FreeParameter(-2.7920)
    params['Antimony', 'Aluminium', 'hamiltonian_matrix_element', 's1s0s'] = FreeParameter(-1.2097)
    params['Antimony', 'Aluminium', 'hamiltonian_matrix_element', 's0s1s'] = FreeParameter(-1.6983)
    params['Aluminium', 'Antimony', 'hamiltonian_matrix_element', 's1p0s'] = FreeParameter(1.8889)
    params['Antimony', 'Aluminium', 'hamiltonian_matrix_element', 's1p0s'] = FreeParameter(2.4649)
    params['Aluminium', 'Antimony', 'hamiltonian_matrix_element', 's1d0s'] = FreeParameter(-0.7878)
    params['Antimony', 'Aluminium', 'hamiltonian_matrix_element', 's1d0s'] = FreeParameter(-0.7307)
    params['Aluminium', 'Antimony', 'hamiltonian_matrix_element', 's1s1s'] = FreeParameter(-3.3145)
    params['Aluminium', 'Antimony', 'hamiltonian_matrix_element', 'p0p0p'] = FreeParameter(-1.5273)
    params['Aluminium', 'Antimony', 'hamiltonian_matrix_element', 'p0p0s'] = FreeParameter(4.1042)
    params['Aluminium', 'Antimony', 'hamiltonian_matrix_element', 'p0d0p'] = FreeParameter(1.8364)
    params['Aluminium', 'Antimony', 'hamiltonian_matrix_element', 'p0d0s'] = FreeParameter(-1.9726)
    params['Antimony', 'Aluminium', 'hamiltonian_matrix_element', 'p0d0s'] = FreeParameter(-1.9819)
    params['Antimony', 'Aluminium', 'hamiltonian_matrix_element', 'p0d0p'] = FreeParameter(2.1292)
    params['Aluminium', 'Antimony', 'hamiltonian_matrix_element', 'd0d0s'] = FreeParameter(-1.1395)
    params['Aluminium', 'Antimony', 'hamiltonian_matrix_element', 'd0d0p'] = FreeParameter(2.1206)
    params['Aluminium', 'Antimony', 'hamiltonian_matrix_element', 'd0d0d'] = FreeParameter(-1.7260)

    # Off-site parameters
    params['Aluminium', 'Arsenic', 'hamiltonian_matrix_element', 's0s0s'] = FreeParameter(-1.7292)
    params['Aluminium', 'Arsenic', 'hamiltonian_matrix_element', 's0p0s'] = FreeParameter(2.7435)
    params['Arsenic', 'Aluminium', 'hamiltonian_matrix_element', 's0p0s'] = FreeParameter(2.5175)
    params['Aluminium', 'Arsenic', 'hamiltonian_matrix_element', 's0d0s'] = FreeParameter(-2.3869)
    params['Arsenic', 'Aluminium', 'hamiltonian_matrix_element', 's0d0s'] = FreeParameter(-2.5535)
    params['Arsenic', 'Aluminium', 'hamiltonian_matrix_element', 's1s0s'] = FreeParameter(-1.6167)
    params['Arsenic', 'Aluminium', 'hamiltonian_matrix_element', 's0s1s'] = FreeParameter(-1.2688)
    params['Aluminium', 'Arsenic', 'hamiltonian_matrix_element', 's1p0s'] = FreeParameter(2.1989)
    params['Arsenic', 'Aluminium', 'hamiltonian_matrix_element', 's1p0s'] = FreeParameter(2.1190)
    params['Aluminium', 'Arsenic', 'hamiltonian_matrix_element', 's1d0s'] = FreeParameter(-0.7442)
    params['Arsenic', 'Aluminium', 'hamiltonian_matrix_element', 's1d0s'] = FreeParameter(-0.7064)
    params['Aluminium', 'Arsenic', 'hamiltonian_matrix_element', 's1s1s'] = FreeParameter(-3.604)
    params['Aluminium', 'Arsenic', 'hamiltonian_matrix_element', 'p0p0p'] = FreeParameter(-1.3398)
    params['Aluminium', 'Arsenic', 'hamiltonian_matrix_element', 'p0p0s'] = FreeParameter(4.2460)
    params['Aluminium', 'Arsenic', 'hamiltonian_matrix_element', 'p0d0p'] = FreeParameter(2.0928)
    params['Aluminium', 'Arsenic', 'hamiltonian_matrix_element', 'p0d0s'] = FreeParameter(-1.7601)
    params['Arsenic', 'Aluminium', 'hamiltonian_matrix_element', 'p0d0s'] = FreeParameter(-1.7240)
    params['Arsenic', 'Aluminium', 'hamiltonian_matrix_element', 'p0d0p'] = FreeParameter(1.7776)
    params['Aluminium', 'Arsenic', 'hamiltonian_matrix_element', 'd0d0s'] = FreeParameter(-1.2175)
    params['Aluminium', 'Arsenic', 'hamiltonian_matrix_element', 'd0d0p'] = FreeParameter(2.1693)
    params['Aluminium', 'Arsenic', 'hamiltonian_matrix_element', 'd0d0d'] = FreeParameter(-1.7540)

    # Strain scaling
    params['Aluminium', 'Antimony', 'hamiltonian_matrix_element_eta', 's0s0s'] = 2.
    params['Aluminium', 'Antimony', 'hamiltonian_matrix_element_eta', 's0p0s'] = 2.
    params['Antimony', 'Aluminium', 'hamiltonian_matrix_element_eta', 's0p0s'] = 2.
    params['Aluminium', 'Antimony', 'hamiltonian_matrix_element_eta', 's0d0s'] = 2.
    params['Antimony', 'Aluminium', 'hamiltonian_matrix_element_eta', 's0d0s'] = 2.
    params['Antimony', 'Aluminium', 'hamiltonian_matrix_element_eta', 's1s0s'] = 2.
    params['Antimony', 'Aluminium', 'hamiltonian_matrix_element_eta', 's0s1s'] = 2.
    params['Aluminium', 'Antimony', 'hamiltonian_matrix_element_eta', 's1p0s'] = 2.
    params['Antimony', 'Aluminium', 'hamiltonian_matrix_element_eta', 's1p0s'] = 2.
    params['Aluminium', 'Antimony', 'hamiltonian_matrix_element_eta', 's1d0s'] = 2.
    params['Antimony', 'Aluminium', 'hamiltonian_matrix_element_eta', 's1d0s'] = 2.
    params['Aluminium', 'Antimony', 'hamiltonian_matrix_element_eta', 's1s1s'] = 2.
    params['Aluminium', 'Antimony', 'hamiltonian_matrix_element_eta', 'p0p0p'] = 2.
    params['Aluminium', 'Antimony', 'hamiltonian_matrix_element_eta', 'p0p0s'] = 2.
    params['Aluminium', 'Antimony', 'hamiltonian_matrix_element_eta', 'p0d0p'] = 2.
    params['Aluminium', 'Antimony', 'hamiltonian_matrix_element_eta', 'p0d0s'] = 2.
    params['Antimony', 'Aluminium', 'hamiltonian_matrix_element_eta', 'p0d0s'] = 2.
    params['Antimony', 'Aluminium', 'hamiltonian_matrix_element_eta', 'p0d0p'] = 2.
    params['Aluminium', 'Antimony', 'hamiltonian_matrix_element_eta', 'd0d0s'] = 2.
    params['Aluminium', 'Antimony', 'hamiltonian_matrix_element_eta', 'd0d0p'] = 2.
    params['Aluminium', 'Antimony', 'hamiltonian_matrix_element_eta', 'd0d0d'] = 2.

    # Strain scaling
    params['Aluminium', 'Arsenic', 'hamiltonian_matrix_element_eta', 's0s0s'] = 2.
    params['Aluminium', 'Arsenic', 'hamiltonian_matrix_element_eta', 's0p0s'] = 2.
    params['Arsenic', 'Aluminium', 'hamiltonian_matrix_element_eta', 's0p0s'] = 2.
    params['Aluminium', 'Arsenic', 'hamiltonian_matrix_element_eta', 's0d0s'] = 2.
    params['Arsenic', 'Aluminium', 'hamiltonian_matrix_element_eta', 's0d0s'] = 2.
    params['Arsenic', 'Aluminium', 'hamiltonian_matrix_element_eta', 's1s0s'] = 2.
    params['Arsenic', 'Aluminium', 'hamiltonian_matrix_element_eta', 's0s1s'] = 2.
    params['Aluminium', 'Arsenic', 'hamiltonian_matrix_element_eta', 's1p0s'] = 2.
    params['Arsenic', 'Aluminium', 'hamiltonian_matrix_element_eta', 's1p0s'] = 2.
    params['Aluminium', 'Arsenic', 'hamiltonian_matrix_element_eta', 's1d0s'] = 2.
    params['Arsenic', 'Aluminium', 'hamiltonian_matrix_element_eta', 's1d0s'] = 2.
    params['Aluminium', 'Arsenic', 'hamiltonian_matrix_element_eta', 's1s1s'] = 2.
    params['Aluminium', 'Arsenic', 'hamiltonian_matrix_element_eta', 'p0p0p'] = 2.
    params['Aluminium', 'Arsenic', 'hamiltonian_matrix_element_eta', 'p0p0s'] = 2.
    params['Aluminium', 'Arsenic', 'hamiltonian_matrix_element_eta', 'p0d0p'] = 2.
    params['Aluminium', 'Arsenic', 'hamiltonian_matrix_element_eta', 'p0d0s'] = 2.
    params['Arsenic', 'Aluminium', 'hamiltonian_matrix_element_eta', 'p0d0s'] = 2.
    params['Arsenic', 'Aluminium', 'hamiltonian_matrix_element_eta', 'p0d0p'] = 2.
    params['Aluminium', 'Arsenic', 'hamiltonian_matrix_element_eta', 'd0d0s'] = 2.
    params['Aluminium', 'Arsenic', 'hamiltonian_matrix_element_eta', 'd0d0p'] = 2.
    params['Aluminium', 'Arsenic', 'hamiltonian_matrix_element_eta', 'd0d0d'] = 2.

    # Build a dummy initial parameters dictionary.
    initial_parameters = params.asPlainDictionary()
    hamiltonian_parametrization = parameterDictionaryToHamiltonianParametrization(initial_parameters)
    # K-point sampling same as DFT calculation:
    k_point_sampling = KpointDensity(
        density_a=4.0 * Units.Angstrom,
        density_b=4.0 * Units.Angstrom,
        density_c=4.0 * Units.Angstrom
    )
    numerical_accuracy_parameters = NumericalAccuracyParameters(k_point_sampling=k_point_sampling)
    calculator = SemiEmpiricalCalculator(
        hamiltonian_parametrization=hamiltonian_parametrization,
        numerical_accuracy_parameters=numerical_accuracy_parameters,
        spin_polarization=SpinOrbit
    )

    # Fit.
    fitter = SemiEmpiricalFitter(
        targets=[
            reference_alas, reference_alsb],
        semi_empirical_calculator=calculator,
        parameters=params,
        optimizer=optimizer,
        optimizer_kwargs=optimizer_kwargs,
        dictionary_to_parametrization_method=parameterDictionaryToHamiltonianParametrization,
        filename_prefix=output_path + '/fitting_results/start_basis_set'
        )
    parameters = fitter.update()
    hamiltonian_parametrization = parameterDictionaryToHamiltonianParametrization(parameters)

    k_point_sampling = KpointDensity(
        density_a=4.0 * Units.Angstrom,
        density_b=4.0 * Units.Angstrom,
        density_c=4.0 * Units.Angstrom
    )
    numerical_accuracy_parameters = NumericalAccuracyParameters(k_point_sampling=k_point_sampling)

    calculator = SemiEmpiricalCalculator(
        hamiltonian_parametrization=hamiltonian_parametrization,
        numerical_accuracy_parameters=numerical_accuracy_parameters,
        spin_polarization=SpinOrbit)
    reference_alas.plotComparison(calculator, output_path + '/fitting_results/AlAs_fit.png')
    reference_alsb.plotComparison(calculator, output_path + '/fitting_results/AlSb_fit.png')


def main(compounds, input_path, output_path="."):
    if str(compounds).strip() == 'gasb':
        fitGaSb(
            input_path + '/fitting_targets/GaSb.hdf5',
            output_path,
            optimizer='least_squares',
            optimizer_kwargs={'max_nfev': 100, 'xtol': 1e-4})

    elif str(compounds).strip() == 'gaas':
        fitGaAs(
            input_path + '/fitting_targets/GaAs.hdf5',
            output_path,
            optimizer='least_squares',
            optimizer_kwargs={'max_nfev': 100, 'xtol': 1e-4})

    elif str(compounds).strip() == 'gaas_gasb':
        fitGaAs_and_GaSb(
            input_path + '/fitting_targets/GaAs.hdf5',
            input_path + '/fitting_targets/GaSb.hdf5',
            output_path,
            optimizer='least_squares',
            optimizer_kwargs={'max_nfev': 200, 'xtol': 1e-4})

    elif str(compounds).strip() == 'alas_alsb':
        fitAlAs_and_AlSb(
            input_path + '/fitting_targets/AlAs.hdf5',
            input_path + '/fitting_targets/AlSb.hdf5',
            output_path,
            optimizer='least_squares',
            optimizer_kwargs={'max_nfev': 200, 'xtol': 1e-4})

    else:
        raise NLValueError("\nInvalid argument. Possible arguments are:\n- gaas\n- gasb\n- gaas_gasb\n- alas_alsb")


if __name__ == '__main__':

    try:
        compounds, input_path, output_path = sys.argv[1], sys.argv[2], sys.argv[3]

    except IndexError:
        raise ValueError("Please enter the following arguments to the script:\n"
                         "1. compounds to fit. Possible compounds are:\n- gaas\n- gasb\n- gaas_gasb\n- alas_alsb\n"
                         "2. input path\n"
                         "3. output path"
                         )
    if not os.path.exists(output_path + "/fitting_results"):
        os.makedirs(output_path + "/fitting_results")

    main(compounds, input_path, output_path)
