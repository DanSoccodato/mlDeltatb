import numpy
import textwrap
import unittest

from QuantumATK import *

from .EmpiricalSlaterKosterUtilities import parameterDictionaryToHamiltonianParametrization


class EmpiricalSlaterKosterUtilitiesTest(unittest.TestCase):
    """ Test module for EmpiricalSlaterKosterUtilities """

    def compareSlaterKosterTableFromScript(self, target_basis_set, reference_basis_set):
        """ Compare two basis sets where one is provided as script. """
        if type(reference_basis_set) is str:
            result = locals().copy()
            exec(reference_basis_set, globals(), result)
            basis_set = result['basis_set']
        else:
            basis_set = reference_basis_set
        onsite_dictionary_1 = target_basis_set.onsiteParameters()
        onsite_dictionary_2 = basis_set.onsiteParameters()
        self.assertEqual(
            sorted(onsite_dictionary_1.keys()),
            sorted(onsite_dictionary_2.keys()))

        for key_1, key_2 in zip(
            sorted(onsite_dictionary_1.keys()),
            sorted(onsite_dictionary_2.keys())):
            onsite_1 = onsite_dictionary_1[key_1]
            onsite_2 = onsite_dictionary_2[key_2]
            self.assertEqual(onsite_1.angularMomenta(), onsite_2.angularMomenta())
            self.assertEqual(onsite_1.fillingMethod(), onsite_2.fillingMethod())
            self.assertAlmostEqual(onsite_1.vacuumLevel(), onsite_2.vacuumLevel())
            #XXX assertSequencesAlmostEqual not support in unittest.
            # self.assertSequencesAlmostEqual(onsite_1.ionizationPotential(), onsite_2.ionizationPotential())
            # self.assertSequencesAlmostEqual(onsite_1.occupations(), onsite_2.occupations())
            # self.assertSequencesAlmostEqual(onsite_1.onsiteHartreeShift(), onsite_2.onsiteHartreeShift())
            # self.assertSequencesAlmostEqual(onsite_1.onsiteSpinSplit(), onsite_2.onsiteSpinSplit())

        offsite_dictionary_1 = target_basis_set.offsiteParameters()
        offsite_dictionary_2 = basis_set.offsiteParameters()
        self.assertEqual(
            sorted(offsite_dictionary_1.keys()),
            sorted(offsite_dictionary_2.keys()))
        for key_1, key_2 in zip(
            sorted(offsite_dictionary_1.keys()),
            sorted(offsite_dictionary_2.keys())):
            offsite_1 = offsite_dictionary_1[key_1]
            offsite_2 = offsite_dictionary_2[key_2]
            for entry_1, entry_2 in zip(offsite_1, offsite_2):
                self.assertEqual(len(entry_1), len(entry_2))
                self.assertAlmostEqual(entry_1[0], entry_2[0])
                self.assertAlmostEqual(entry_1[1], entry_2[1])
                if len(entry_1) > 2:
                    self.assertAlmostEqual(entry_1[2], entry_2[2])


    def testOneElementSpds(self):
        """ Test that we can convert a dictionary for single element spds parametrization """
        params = dict()
        params['elements'] = [Silicon]
        params['Silicon', 'orbitals'] = ['s0', 'p0', 'd0', 's1']
        params['Silicon', 'number_of_valence_electrons'] = 4.
        params['Silicon', 'onsite_spin_orbit_split'] = [0., 0., 0., 0.]
        params['Silicon', 'onsite_hartree_shift'] = ATK_U(
            Silicon, ['3p', '3p', '3p', '3p'], 'ncp').inUnitsOf(eV)

        params['Silicon', 'ionization_potential', 's0'] = -10.
        params['Silicon', 'ionization_potential', 'p0'] = -8.
        params['Silicon', 'ionization_potential', 'd0'] = -6.
        params['Silicon', 'ionization_potential', 's1'] = -4.

        params['Silicon', 'hamiltonian_matrix_element', 's0s0s'] = -1.
        params['Silicon', 'hamiltonian_matrix_element', 's0p0s'] = -2.
        params['Silicon', 'hamiltonian_matrix_element', 's0d0s'] = -3.
        params['Silicon', 'hamiltonian_matrix_element', 's1s1s'] = -4.
        params['Silicon', 'hamiltonian_matrix_element', 's1p0s'] = -5.
        params['Silicon', 'hamiltonian_matrix_element', 's1d0s'] = -6.

        params['Silicon', 'hamiltonian_matrix_element', 'p0p0s'] = -7.
        params['Silicon', 'hamiltonian_matrix_element', 'p0p0p'] = -8.
        params['Silicon', 'hamiltonian_matrix_element', 'p0d0s'] = -9.
        params['Silicon', 'hamiltonian_matrix_element', 'p0d0p'] = -10.

        params['Silicon', 'hamiltonian_matrix_element', 'd0d0s'] = -11.
        params['Silicon', 'hamiltonian_matrix_element', 'd0d0p'] = -12.
        params['Silicon', 'hamiltonian_matrix_element', 'd0d0d'] = -13.

        params['Silicon', 'hamiltonian_matrix_element_eta', 's0s0s'] = -0.
        params['Silicon', 'hamiltonian_matrix_element_eta', 's0p0s'] = -0.
        params['Silicon', 'hamiltonian_matrix_element_eta', 's0d0s'] = -0.
        params['Silicon', 'hamiltonian_matrix_element_eta', 's1s1s'] = -0.
        params['Silicon', 'hamiltonian_matrix_element_eta', 's1p0s'] = -0.
        params['Silicon', 'hamiltonian_matrix_element_eta', 's1d0s'] = -0.

        params['Silicon', 'hamiltonian_matrix_element_eta', 'p0p0s'] = -0.
        params['Silicon', 'hamiltonian_matrix_element_eta', 'p0p0p'] = -0.
        params['Silicon', 'hamiltonian_matrix_element_eta', 'p0d0s'] = -0.
        params['Silicon', 'hamiltonian_matrix_element_eta', 'p0d0p'] = -0.

        params['Silicon', 'hamiltonian_matrix_element_eta', 'd0d0s'] = -0.
        params['Silicon', 'hamiltonian_matrix_element_eta', 'd0d0p'] = -0.
        params['Silicon', 'hamiltonian_matrix_element_eta', 'd0d0d'] = -0.

        params['Silicon', 'nearest_neighbor_distance'] = 2.
        params['Silicon', 'second_neighbor_distance'] = 3.

        hamiltonian_parametrization = parameterDictionaryToHamiltonianParametrization(params)
        basis_set = hamiltonian_parametrization.basisSet()

        reference_string = textwrap.dedent("""\
        from QuantumATK import *

        silicon_onsite_term = SlaterKosterOnsiteParameters(
            element=PeriodicTable.Silicon,
            angular_momenta=[ 0 , 1 , 2 , 0 ],
            occupations=[ 0.4 , 1.2 , 2.0 , 0.4 ],
            filling_method=SphericalSymmetric,
            ionization_potential=[ -10.0*eV , -8.0*eV , -6.0*eV , -4.0*eV ],
            onsite_hartree_shift=[ 6.657*eV , 6.657*eV , 6.657*eV , 6.657*eV ],
            onsite_spin_split=[[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]*eV,
            onsite_spin_orbit_split=[0.0, 0.0, 0.0, 0.0]*eV,
            vacuum_level=0.0*Hartree,
            )

        basis_set = SlaterKosterTable(
            silicon=silicon_onsite_term,
            si_si_s1s1s=[
                (1.6*Angstrom, -4.0*eV),
                (1.62*Angstrom, -4.0*eV),
                (1.64*Angstrom, -4.0*eV),
                (1.66*Angstrom, -4.0*eV),
                (1.68*Angstrom, -4.0*eV),
                (1.7*Angstrom, -4.0*eV),
                (1.72*Angstrom, -4.0*eV),
                (1.74*Angstrom, -4.0*eV),
                (1.76*Angstrom, -4.0*eV),
                (1.78*Angstrom, -4.0*eV),
                (1.8*Angstrom, -4.0*eV),
                (1.82*Angstrom, -4.0*eV),
                (1.8399999999999999*Angstrom, -4.0*eV),
                (1.8599999999999999*Angstrom, -4.0*eV),
                (1.88*Angstrom, -4.0*eV),
                (1.9*Angstrom, -4.0*eV),
                (1.92*Angstrom, -4.0*eV),
                (1.94*Angstrom, -4.0*eV),
                (1.96*Angstrom, -4.0*eV),
                (1.98*Angstrom, -4.0*eV),
                (2.0*Angstrom, -4.0*eV),
                (2.02*Angstrom, -4.0*eV),
                (2.04*Angstrom, -4.0*eV),
                (2.06*Angstrom, -4.0*eV),
                (2.08*Angstrom, -4.0*eV),
                (2.1*Angstrom, -4.0*eV),
                (2.12*Angstrom, -4.0*eV),
                (2.14*Angstrom, -4.0*eV),
                (2.16*Angstrom, -4.0*eV),
                (2.1799999999999997*Angstrom, -4.0*eV),
                (2.2*Angstrom, -4.0*eV),
                (2.2199999999999998*Angstrom, -4.0*eV),
                (2.24*Angstrom, -4.0*eV),
                (2.26*Angstrom, -4.0*eV),
                (2.2800000000000002*Angstrom, -4.0*eV),
                (2.3*Angstrom, -4.0*eV),
                (2.32*Angstrom, -4.0*eV),
                (2.34*Angstrom, -4.0*eV),
                (2.36*Angstrom, -4.0*eV),
                (2.38*Angstrom, -4.0*eV),
                (2.4*Angstrom, -4.0*eV),
                (2.5*Angstrom, 0.0*eV),
            ],
            si_si_s1p0s=[
                (1.6*Angstrom, -5.0*eV),
                (1.62*Angstrom, -5.0*eV),
                (1.64*Angstrom, -5.0*eV),
                (1.66*Angstrom, -5.0*eV),
                (1.68*Angstrom, -5.0*eV),
                (1.7*Angstrom, -5.0*eV),
                (1.72*Angstrom, -5.0*eV),
                (1.74*Angstrom, -5.0*eV),
                (1.76*Angstrom, -5.0*eV),
                (1.78*Angstrom, -5.0*eV),
                (1.8*Angstrom, -5.0*eV),
                (1.82*Angstrom, -5.0*eV),
                (1.8399999999999999*Angstrom, -5.0*eV),
                (1.8599999999999999*Angstrom, -5.0*eV),
                (1.88*Angstrom, -5.0*eV),
                (1.9*Angstrom, -5.0*eV),
                (1.92*Angstrom, -5.0*eV),
                (1.94*Angstrom, -5.0*eV),
                (1.96*Angstrom, -5.0*eV),
                (1.98*Angstrom, -5.0*eV),
                (2.0*Angstrom, -5.0*eV),
                (2.02*Angstrom, -5.0*eV),
                (2.04*Angstrom, -5.0*eV),
                (2.06*Angstrom, -5.0*eV),
                (2.08*Angstrom, -5.0*eV),
                (2.1*Angstrom, -5.0*eV),
                (2.12*Angstrom, -5.0*eV),
                (2.14*Angstrom, -5.0*eV),
                (2.16*Angstrom, -5.0*eV),
                (2.1799999999999997*Angstrom, -5.0*eV),
                (2.2*Angstrom, -5.0*eV),
                (2.2199999999999998*Angstrom, -5.0*eV),
                (2.24*Angstrom, -5.0*eV),
                (2.26*Angstrom, -5.0*eV),
                (2.2800000000000002*Angstrom, -5.0*eV),
                (2.3*Angstrom, -5.0*eV),
                (2.32*Angstrom, -5.0*eV),
                (2.34*Angstrom, -5.0*eV),
                (2.36*Angstrom, -5.0*eV),
                (2.38*Angstrom, -5.0*eV),
                (2.4*Angstrom, -5.0*eV),
                (2.5*Angstrom, 0.0*eV),
            ],
            si_si_s1d0s=[
                (1.6*Angstrom, -6.0*eV),
                (1.62*Angstrom, -6.0*eV),
                (1.64*Angstrom, -6.0*eV),
                (1.66*Angstrom, -6.0*eV),
                (1.68*Angstrom, -6.0*eV),
                (1.7*Angstrom, -6.0*eV),
                (1.72*Angstrom, -6.0*eV),
                (1.74*Angstrom, -6.0*eV),
                (1.76*Angstrom, -6.0*eV),
                (1.78*Angstrom, -6.0*eV),
                (1.8*Angstrom, -6.0*eV),
                (1.82*Angstrom, -6.0*eV),
                (1.8399999999999999*Angstrom, -6.0*eV),
                (1.8599999999999999*Angstrom, -6.0*eV),
                (1.88*Angstrom, -6.0*eV),
                (1.9*Angstrom, -6.0*eV),
                (1.92*Angstrom, -6.0*eV),
                (1.94*Angstrom, -6.0*eV),
                (1.96*Angstrom, -6.0*eV),
                (1.98*Angstrom, -6.0*eV),
                (2.0*Angstrom, -6.0*eV),
                (2.02*Angstrom, -6.0*eV),
                (2.04*Angstrom, -6.0*eV),
                (2.06*Angstrom, -6.0*eV),
                (2.08*Angstrom, -6.0*eV),
                (2.1*Angstrom, -6.0*eV),
                (2.12*Angstrom, -6.0*eV),
                (2.14*Angstrom, -6.0*eV),
                (2.16*Angstrom, -6.0*eV),
                (2.1799999999999997*Angstrom, -6.0*eV),
                (2.2*Angstrom, -6.0*eV),
                (2.2199999999999998*Angstrom, -6.0*eV),
                (2.24*Angstrom, -6.0*eV),
                (2.26*Angstrom, -6.0*eV),
                (2.2800000000000002*Angstrom, -6.0*eV),
                (2.3*Angstrom, -6.0*eV),
                (2.32*Angstrom, -6.0*eV),
                (2.34*Angstrom, -6.0*eV),
                (2.36*Angstrom, -6.0*eV),
                (2.38*Angstrom, -6.0*eV),
                (2.4*Angstrom, -6.0*eV),
                (2.5*Angstrom, 0.0*eV),
            ],
            si_si_s0s0s=[
                (1.6*Angstrom, -1.0*eV),
                (1.62*Angstrom, -1.0*eV),
                (1.64*Angstrom, -1.0*eV),
                (1.66*Angstrom, -1.0*eV),
                (1.68*Angstrom, -1.0*eV),
                (1.7*Angstrom, -1.0*eV),
                (1.72*Angstrom, -1.0*eV),
                (1.74*Angstrom, -1.0*eV),
                (1.76*Angstrom, -1.0*eV),
                (1.78*Angstrom, -1.0*eV),
                (1.8*Angstrom, -1.0*eV),
                (1.82*Angstrom, -1.0*eV),
                (1.8399999999999999*Angstrom, -1.0*eV),
                (1.8599999999999999*Angstrom, -1.0*eV),
                (1.88*Angstrom, -1.0*eV),
                (1.9*Angstrom, -1.0*eV),
                (1.92*Angstrom, -1.0*eV),
                (1.94*Angstrom, -1.0*eV),
                (1.96*Angstrom, -1.0*eV),
                (1.98*Angstrom, -1.0*eV),
                (2.0*Angstrom, -1.0*eV),
                (2.02*Angstrom, -1.0*eV),
                (2.04*Angstrom, -1.0*eV),
                (2.06*Angstrom, -1.0*eV),
                (2.08*Angstrom, -1.0*eV),
                (2.1*Angstrom, -1.0*eV),
                (2.12*Angstrom, -1.0*eV),
                (2.14*Angstrom, -1.0*eV),
                (2.16*Angstrom, -1.0*eV),
                (2.1799999999999997*Angstrom, -1.0*eV),
                (2.2*Angstrom, -1.0*eV),
                (2.2199999999999998*Angstrom, -1.0*eV),
                (2.24*Angstrom, -1.0*eV),
                (2.26*Angstrom, -1.0*eV),
                (2.2800000000000002*Angstrom, -1.0*eV),
                (2.3*Angstrom, -1.0*eV),
                (2.32*Angstrom, -1.0*eV),
                (2.34*Angstrom, -1.0*eV),
                (2.36*Angstrom, -1.0*eV),
                (2.38*Angstrom, -1.0*eV),
                (2.4*Angstrom, -1.0*eV),
                (2.5*Angstrom, 0.0*eV),
            ],
            si_si_s0p0s=[
                (1.6*Angstrom, -2.0*eV),
                (1.62*Angstrom, -2.0*eV),
                (1.64*Angstrom, -2.0*eV),
                (1.66*Angstrom, -2.0*eV),
                (1.68*Angstrom, -2.0*eV),
                (1.7*Angstrom, -2.0*eV),
                (1.72*Angstrom, -2.0*eV),
                (1.74*Angstrom, -2.0*eV),
                (1.76*Angstrom, -2.0*eV),
                (1.78*Angstrom, -2.0*eV),
                (1.8*Angstrom, -2.0*eV),
                (1.82*Angstrom, -2.0*eV),
                (1.8399999999999999*Angstrom, -2.0*eV),
                (1.8599999999999999*Angstrom, -2.0*eV),
                (1.88*Angstrom, -2.0*eV),
                (1.9*Angstrom, -2.0*eV),
                (1.92*Angstrom, -2.0*eV),
                (1.94*Angstrom, -2.0*eV),
                (1.96*Angstrom, -2.0*eV),
                (1.98*Angstrom, -2.0*eV),
                (2.0*Angstrom, -2.0*eV),
                (2.02*Angstrom, -2.0*eV),
                (2.04*Angstrom, -2.0*eV),
                (2.06*Angstrom, -2.0*eV),
                (2.08*Angstrom, -2.0*eV),
                (2.1*Angstrom, -2.0*eV),
                (2.12*Angstrom, -2.0*eV),
                (2.14*Angstrom, -2.0*eV),
                (2.16*Angstrom, -2.0*eV),
                (2.1799999999999997*Angstrom, -2.0*eV),
                (2.2*Angstrom, -2.0*eV),
                (2.2199999999999998*Angstrom, -2.0*eV),
                (2.24*Angstrom, -2.0*eV),
                (2.26*Angstrom, -2.0*eV),
                (2.2800000000000002*Angstrom, -2.0*eV),
                (2.3*Angstrom, -2.0*eV),
                (2.32*Angstrom, -2.0*eV),
                (2.34*Angstrom, -2.0*eV),
                (2.36*Angstrom, -2.0*eV),
                (2.38*Angstrom, -2.0*eV),
                (2.4*Angstrom, -2.0*eV),
                (2.5*Angstrom, 0.0*eV),
            ],
            si_si_s0d0s=[
                (1.6*Angstrom, -3.0*eV),
                (1.62*Angstrom, -3.0*eV),
                (1.64*Angstrom, -3.0*eV),
                (1.66*Angstrom, -3.0*eV),
                (1.68*Angstrom, -3.0*eV),
                (1.7*Angstrom, -3.0*eV),
                (1.72*Angstrom, -3.0*eV),
                (1.74*Angstrom, -3.0*eV),
                (1.76*Angstrom, -3.0*eV),
                (1.78*Angstrom, -3.0*eV),
                (1.8*Angstrom, -3.0*eV),
                (1.82*Angstrom, -3.0*eV),
                (1.8399999999999999*Angstrom, -3.0*eV),
                (1.8599999999999999*Angstrom, -3.0*eV),
                (1.88*Angstrom, -3.0*eV),
                (1.9*Angstrom, -3.0*eV),
                (1.92*Angstrom, -3.0*eV),
                (1.94*Angstrom, -3.0*eV),
                (1.96*Angstrom, -3.0*eV),
                (1.98*Angstrom, -3.0*eV),
                (2.0*Angstrom, -3.0*eV),
                (2.02*Angstrom, -3.0*eV),
                (2.04*Angstrom, -3.0*eV),
                (2.06*Angstrom, -3.0*eV),
                (2.08*Angstrom, -3.0*eV),
                (2.1*Angstrom, -3.0*eV),
                (2.12*Angstrom, -3.0*eV),
                (2.14*Angstrom, -3.0*eV),
                (2.16*Angstrom, -3.0*eV),
                (2.1799999999999997*Angstrom, -3.0*eV),
                (2.2*Angstrom, -3.0*eV),
                (2.2199999999999998*Angstrom, -3.0*eV),
                (2.24*Angstrom, -3.0*eV),
                (2.26*Angstrom, -3.0*eV),
                (2.2800000000000002*Angstrom, -3.0*eV),
                (2.3*Angstrom, -3.0*eV),
                (2.32*Angstrom, -3.0*eV),
                (2.34*Angstrom, -3.0*eV),
                (2.36*Angstrom, -3.0*eV),
                (2.38*Angstrom, -3.0*eV),
                (2.4*Angstrom, -3.0*eV),
                (2.5*Angstrom, 0.0*eV),
            ],
            si_si_p0p0s=[
                (1.6*Angstrom, -7.0*eV),
                (1.62*Angstrom, -7.0*eV),
                (1.64*Angstrom, -7.0*eV),
                (1.66*Angstrom, -7.0*eV),
                (1.68*Angstrom, -7.0*eV),
                (1.7*Angstrom, -7.0*eV),
                (1.72*Angstrom, -7.0*eV),
                (1.74*Angstrom, -7.0*eV),
                (1.76*Angstrom, -7.0*eV),
                (1.78*Angstrom, -7.0*eV),
                (1.8*Angstrom, -7.0*eV),
                (1.82*Angstrom, -7.0*eV),
                (1.8399999999999999*Angstrom, -7.0*eV),
                (1.8599999999999999*Angstrom, -7.0*eV),
                (1.88*Angstrom, -7.0*eV),
                (1.9*Angstrom, -7.0*eV),
                (1.92*Angstrom, -7.0*eV),
                (1.94*Angstrom, -7.0*eV),
                (1.96*Angstrom, -7.0*eV),
                (1.98*Angstrom, -7.0*eV),
                (2.0*Angstrom, -7.0*eV),
                (2.02*Angstrom, -7.0*eV),
                (2.04*Angstrom, -7.0*eV),
                (2.06*Angstrom, -7.0*eV),
                (2.08*Angstrom, -7.0*eV),
                (2.1*Angstrom, -7.0*eV),
                (2.12*Angstrom, -7.0*eV),
                (2.14*Angstrom, -7.0*eV),
                (2.16*Angstrom, -7.0*eV),
                (2.1799999999999997*Angstrom, -7.0*eV),
                (2.2*Angstrom, -7.0*eV),
                (2.2199999999999998*Angstrom, -7.0*eV),
                (2.24*Angstrom, -7.0*eV),
                (2.26*Angstrom, -7.0*eV),
                (2.2800000000000002*Angstrom, -7.0*eV),
                (2.3*Angstrom, -7.0*eV),
                (2.32*Angstrom, -7.0*eV),
                (2.34*Angstrom, -7.0*eV),
                (2.36*Angstrom, -7.0*eV),
                (2.38*Angstrom, -7.0*eV),
                (2.4*Angstrom, -7.0*eV),
                (2.5*Angstrom, 0.0*eV),
            ],
            si_si_p0p0p=[
                (1.6*Angstrom, -8.0*eV),
                (1.62*Angstrom, -8.0*eV),
                (1.64*Angstrom, -8.0*eV),
                (1.66*Angstrom, -8.0*eV),
                (1.68*Angstrom, -8.0*eV),
                (1.7*Angstrom, -8.0*eV),
                (1.72*Angstrom, -8.0*eV),
                (1.74*Angstrom, -8.0*eV),
                (1.76*Angstrom, -8.0*eV),
                (1.78*Angstrom, -8.0*eV),
                (1.8*Angstrom, -8.0*eV),
                (1.82*Angstrom, -8.0*eV),
                (1.8399999999999999*Angstrom, -8.0*eV),
                (1.8599999999999999*Angstrom, -8.0*eV),
                (1.88*Angstrom, -8.0*eV),
                (1.9*Angstrom, -8.0*eV),
                (1.92*Angstrom, -8.0*eV),
                (1.94*Angstrom, -8.0*eV),
                (1.96*Angstrom, -8.0*eV),
                (1.98*Angstrom, -8.0*eV),
                (2.0*Angstrom, -8.0*eV),
                (2.02*Angstrom, -8.0*eV),
                (2.04*Angstrom, -8.0*eV),
                (2.06*Angstrom, -8.0*eV),
                (2.08*Angstrom, -8.0*eV),
                (2.1*Angstrom, -8.0*eV),
                (2.12*Angstrom, -8.0*eV),
                (2.14*Angstrom, -8.0*eV),
                (2.16*Angstrom, -8.0*eV),
                (2.1799999999999997*Angstrom, -8.0*eV),
                (2.2*Angstrom, -8.0*eV),
                (2.2199999999999998*Angstrom, -8.0*eV),
                (2.24*Angstrom, -8.0*eV),
                (2.26*Angstrom, -8.0*eV),
                (2.2800000000000002*Angstrom, -8.0*eV),
                (2.3*Angstrom, -8.0*eV),
                (2.32*Angstrom, -8.0*eV),
                (2.34*Angstrom, -8.0*eV),
                (2.36*Angstrom, -8.0*eV),
                (2.38*Angstrom, -8.0*eV),
                (2.4*Angstrom, -8.0*eV),
                (2.5*Angstrom, 0.0*eV),
            ],
            si_si_p0d0s=[
                (1.6*Angstrom, -9.0*eV),
                (1.62*Angstrom, -9.0*eV),
                (1.64*Angstrom, -9.0*eV),
                (1.66*Angstrom, -9.0*eV),
                (1.68*Angstrom, -9.0*eV),
                (1.7*Angstrom, -9.0*eV),
                (1.72*Angstrom, -9.0*eV),
                (1.74*Angstrom, -9.0*eV),
                (1.76*Angstrom, -9.0*eV),
                (1.78*Angstrom, -9.0*eV),
                (1.8*Angstrom, -9.0*eV),
                (1.82*Angstrom, -9.0*eV),
                (1.8399999999999999*Angstrom, -9.0*eV),
                (1.8599999999999999*Angstrom, -9.0*eV),
                (1.88*Angstrom, -9.0*eV),
                (1.9*Angstrom, -9.0*eV),
                (1.92*Angstrom, -9.0*eV),
                (1.94*Angstrom, -9.0*eV),
                (1.96*Angstrom, -9.0*eV),
                (1.98*Angstrom, -9.0*eV),
                (2.0*Angstrom, -9.0*eV),
                (2.02*Angstrom, -9.0*eV),
                (2.04*Angstrom, -9.0*eV),
                (2.06*Angstrom, -9.0*eV),
                (2.08*Angstrom, -9.0*eV),
                (2.1*Angstrom, -9.0*eV),
                (2.12*Angstrom, -9.0*eV),
                (2.14*Angstrom, -9.0*eV),
                (2.16*Angstrom, -9.0*eV),
                (2.1799999999999997*Angstrom, -9.0*eV),
                (2.2*Angstrom, -9.0*eV),
                (2.2199999999999998*Angstrom, -9.0*eV),
                (2.24*Angstrom, -9.0*eV),
                (2.26*Angstrom, -9.0*eV),
                (2.2800000000000002*Angstrom, -9.0*eV),
                (2.3*Angstrom, -9.0*eV),
                (2.32*Angstrom, -9.0*eV),
                (2.34*Angstrom, -9.0*eV),
                (2.36*Angstrom, -9.0*eV),
                (2.38*Angstrom, -9.0*eV),
                (2.4*Angstrom, -9.0*eV),
                (2.5*Angstrom, 0.0*eV),
            ],
            si_si_p0d0p=[
                (1.6*Angstrom, -10.0*eV),
                (1.62*Angstrom, -10.0*eV),
                (1.64*Angstrom, -10.0*eV),
                (1.66*Angstrom, -10.0*eV),
                (1.68*Angstrom, -10.0*eV),
                (1.7*Angstrom, -10.0*eV),
                (1.72*Angstrom, -10.0*eV),
                (1.74*Angstrom, -10.0*eV),
                (1.76*Angstrom, -10.0*eV),
                (1.78*Angstrom, -10.0*eV),
                (1.8*Angstrom, -10.0*eV),
                (1.82*Angstrom, -10.0*eV),
                (1.8399999999999999*Angstrom, -10.0*eV),
                (1.8599999999999999*Angstrom, -10.0*eV),
                (1.88*Angstrom, -10.0*eV),
                (1.9*Angstrom, -10.0*eV),
                (1.92*Angstrom, -10.0*eV),
                (1.94*Angstrom, -10.0*eV),
                (1.96*Angstrom, -10.0*eV),
                (1.98*Angstrom, -10.0*eV),
                (2.0*Angstrom, -10.0*eV),
                (2.02*Angstrom, -10.0*eV),
                (2.04*Angstrom, -10.0*eV),
                (2.06*Angstrom, -10.0*eV),
                (2.08*Angstrom, -10.0*eV),
                (2.1*Angstrom, -10.0*eV),
                (2.12*Angstrom, -10.0*eV),
                (2.14*Angstrom, -10.0*eV),
                (2.16*Angstrom, -10.0*eV),
                (2.1799999999999997*Angstrom, -10.0*eV),
                (2.2*Angstrom, -10.0*eV),
                (2.2199999999999998*Angstrom, -10.0*eV),
                (2.24*Angstrom, -10.0*eV),
                (2.26*Angstrom, -10.0*eV),
                (2.2800000000000002*Angstrom, -10.0*eV),
                (2.3*Angstrom, -10.0*eV),
                (2.32*Angstrom, -10.0*eV),
                (2.34*Angstrom, -10.0*eV),
                (2.36*Angstrom, -10.0*eV),
                (2.38*Angstrom, -10.0*eV),
                (2.4*Angstrom, -10.0*eV),
                (2.5*Angstrom, 0.0*eV),
            ],
            si_si_d0d0s=[
                (1.6*Angstrom, -11.0*eV),
                (1.62*Angstrom, -11.0*eV),
                (1.64*Angstrom, -11.0*eV),
                (1.66*Angstrom, -11.0*eV),
                (1.68*Angstrom, -11.0*eV),
                (1.7*Angstrom, -11.0*eV),
                (1.72*Angstrom, -11.0*eV),
                (1.74*Angstrom, -11.0*eV),
                (1.76*Angstrom, -11.0*eV),
                (1.78*Angstrom, -11.0*eV),
                (1.8*Angstrom, -11.0*eV),
                (1.82*Angstrom, -11.0*eV),
                (1.8399999999999999*Angstrom, -11.0*eV),
                (1.8599999999999999*Angstrom, -11.0*eV),
                (1.88*Angstrom, -11.0*eV),
                (1.9*Angstrom, -11.0*eV),
                (1.92*Angstrom, -11.0*eV),
                (1.94*Angstrom, -11.0*eV),
                (1.96*Angstrom, -11.0*eV),
                (1.98*Angstrom, -11.0*eV),
                (2.0*Angstrom, -11.0*eV),
                (2.02*Angstrom, -11.0*eV),
                (2.04*Angstrom, -11.0*eV),
                (2.06*Angstrom, -11.0*eV),
                (2.08*Angstrom, -11.0*eV),
                (2.1*Angstrom, -11.0*eV),
                (2.12*Angstrom, -11.0*eV),
                (2.14*Angstrom, -11.0*eV),
                (2.16*Angstrom, -11.0*eV),
                (2.1799999999999997*Angstrom, -11.0*eV),
                (2.2*Angstrom, -11.0*eV),
                (2.2199999999999998*Angstrom, -11.0*eV),
                (2.24*Angstrom, -11.0*eV),
                (2.26*Angstrom, -11.0*eV),
                (2.2800000000000002*Angstrom, -11.0*eV),
                (2.3*Angstrom, -11.0*eV),
                (2.32*Angstrom, -11.0*eV),
                (2.34*Angstrom, -11.0*eV),
                (2.36*Angstrom, -11.0*eV),
                (2.38*Angstrom, -11.0*eV),
                (2.4*Angstrom, -11.0*eV),
                (2.5*Angstrom, 0.0*eV),
            ],
            si_si_d0d0p=[
                (1.6*Angstrom, -12.0*eV),
                (1.62*Angstrom, -12.0*eV),
                (1.64*Angstrom, -12.0*eV),
                (1.66*Angstrom, -12.0*eV),
                (1.68*Angstrom, -12.0*eV),
                (1.7*Angstrom, -12.0*eV),
                (1.72*Angstrom, -12.0*eV),
                (1.74*Angstrom, -12.0*eV),
                (1.76*Angstrom, -12.0*eV),
                (1.78*Angstrom, -12.0*eV),
                (1.8*Angstrom, -12.0*eV),
                (1.82*Angstrom, -12.0*eV),
                (1.8399999999999999*Angstrom, -12.0*eV),
                (1.8599999999999999*Angstrom, -12.0*eV),
                (1.88*Angstrom, -12.0*eV),
                (1.9*Angstrom, -12.0*eV),
                (1.92*Angstrom, -12.0*eV),
                (1.94*Angstrom, -12.0*eV),
                (1.96*Angstrom, -12.0*eV),
                (1.98*Angstrom, -12.0*eV),
                (2.0*Angstrom, -12.0*eV),
                (2.02*Angstrom, -12.0*eV),
                (2.04*Angstrom, -12.0*eV),
                (2.06*Angstrom, -12.0*eV),
                (2.08*Angstrom, -12.0*eV),
                (2.1*Angstrom, -12.0*eV),
                (2.12*Angstrom, -12.0*eV),
                (2.14*Angstrom, -12.0*eV),
                (2.16*Angstrom, -12.0*eV),
                (2.1799999999999997*Angstrom, -12.0*eV),
                (2.2*Angstrom, -12.0*eV),
                (2.2199999999999998*Angstrom, -12.0*eV),
                (2.24*Angstrom, -12.0*eV),
                (2.26*Angstrom, -12.0*eV),
                (2.2800000000000002*Angstrom, -12.0*eV),
                (2.3*Angstrom, -12.0*eV),
                (2.32*Angstrom, -12.0*eV),
                (2.34*Angstrom, -12.0*eV),
                (2.36*Angstrom, -12.0*eV),
                (2.38*Angstrom, -12.0*eV),
                (2.4*Angstrom, -12.0*eV),
                (2.5*Angstrom, 0.0*eV),
            ],
            si_si_d0d0d=[
                (1.6*Angstrom, -13.0*eV),
                (1.62*Angstrom, -13.0*eV),
                (1.64*Angstrom, -13.0*eV),
                (1.66*Angstrom, -13.0*eV),
                (1.68*Angstrom, -13.0*eV),
                (1.7*Angstrom, -13.0*eV),
                (1.72*Angstrom, -13.0*eV),
                (1.74*Angstrom, -13.0*eV),
                (1.76*Angstrom, -13.0*eV),
                (1.78*Angstrom, -13.0*eV),
                (1.8*Angstrom, -13.0*eV),
                (1.82*Angstrom, -13.0*eV),
                (1.8399999999999999*Angstrom, -13.0*eV),
                (1.8599999999999999*Angstrom, -13.0*eV),
                (1.88*Angstrom, -13.0*eV),
                (1.9*Angstrom, -13.0*eV),
                (1.92*Angstrom, -13.0*eV),
                (1.94*Angstrom, -13.0*eV),
                (1.96*Angstrom, -13.0*eV),
                (1.98*Angstrom, -13.0*eV),
                (2.0*Angstrom, -13.0*eV),
                (2.02*Angstrom, -13.0*eV),
                (2.04*Angstrom, -13.0*eV),
                (2.06*Angstrom, -13.0*eV),
                (2.08*Angstrom, -13.0*eV),
                (2.1*Angstrom, -13.0*eV),
                (2.12*Angstrom, -13.0*eV),
                (2.14*Angstrom, -13.0*eV),
                (2.16*Angstrom, -13.0*eV),
                (2.1799999999999997*Angstrom, -13.0*eV),
                (2.2*Angstrom, -13.0*eV),
                (2.2199999999999998*Angstrom, -13.0*eV),
                (2.24*Angstrom, -13.0*eV),
                (2.26*Angstrom, -13.0*eV),
                (2.2800000000000002*Angstrom, -13.0*eV),
                (2.3*Angstrom, -13.0*eV),
                (2.32*Angstrom, -13.0*eV),
                (2.34*Angstrom, -13.0*eV),
                (2.36*Angstrom, -13.0*eV),
                (2.38*Angstrom, -13.0*eV),
                (2.4*Angstrom, -13.0*eV),
                (2.5*Angstrom, 0.0*eV),
            ],
            )
        """)

        self.compareSlaterKosterTableFromScript(basis_set, reference_string)

    def testTwoElementSp(self):
        """ Test that we can convert a dictionary for two element sp parametrization """
        params = dict()
        params['elements'] = [Silicon, Germanium]

        params['Silicon', 'orbitals'] = ['s0', 'p0']
        params['Silicon', 'number_of_valence_electrons'] = 4.
        params['Silicon', 'onsite_spin_orbit_split'] = [0., 0.]
        params['Silicon', 'onsite_hartree_shift'] = ATK_U(
            Silicon, ['3p', '3p'], 'ncp').inUnitsOf(eV)

        params['Silicon', 'ionization_potential', 's0'] = -10.
        params['Silicon', 'ionization_potential', 'p0'] = -8.
        params['Silicon', 'hamiltonian_matrix_element', 's0s0s'] = -1.
        params['Silicon', 'hamiltonian_matrix_element', 's0p0s'] = -2.
        params['Silicon', 'hamiltonian_matrix_element', 'p0p0s'] = -7.
        params['Silicon', 'hamiltonian_matrix_element', 'p0p0p'] = -8.
        params['Silicon', 'eta', 's0s0s'] = -0.
        params['Silicon', 'eta', 's0p0s'] = -0.
        params['Silicon', 'eta', 'p0p0s'] = -0.
        params['Silicon', 'eta', 'p0p0p'] = -0.

        params['Silicon', 'nearest_neighbor_distance'] = 2.
        params['Silicon', 'second_neighbor_distance'] = 3.

        params['Germanium', 'orbitals'] = ['s0', 'p0']
        params['Germanium', 'number_of_valence_electrons'] = 4.
        params['Germanium', 'onsite_spin_orbit_split'] = [0., 0.]
        params['Germanium', 'onsite_hartree_shift'] = ATK_U(
            Silicon, ['3p', '3p'], 'ncp').inUnitsOf(eV)

        params['Germanium', 'ionization_potential', 's0'] = -10.5
        params['Germanium', 'ionization_potential', 'p0'] = -8.5
        params['Germanium', 'hamiltonian_matrix_element', 's0s0s'] = -1.5
        params['Germanium', 'hamiltonian_matrix_element', 's0p0s'] = -2.5
        params['Germanium', 'hamiltonian_matrix_element', 'p0p0s'] = -7.5
        params['Germanium', 'hamiltonian_matrix_element', 'p0p0p'] = -8.5
        params['Germanium', 'hamiltonian_matrix_element_eta', 's0s0s'] = -2
        params['Germanium', 'hamiltonian_matrix_element_eta', 's0p0s'] = -2
        params['Germanium', 'hamiltonian_matrix_element_eta', 'p0p0s'] = -2
        params['Germanium', 'hamiltonian_matrix_element_eta', 'p0p0p'] = -2

        params['Germanium', 'nearest_neighbor_distance'] = 2.2
        params['Germanium', 'second_neighbor_distance'] = 3.2

        # Interaction terms.
        params['Silicon', 'Germanium', 'hamiltonian_matrix_element', 's0s0s'] = 1.0
        params['Silicon', 'Germanium', 'hamiltonian_matrix_element', 's0p0s'] = 2.0
        params['Silicon', 'Germanium', 'hamiltonian_matrix_element', 'p0s0s'] = 3.0
        params['Silicon', 'Germanium', 'hamiltonian_matrix_element', 'p0p0s'] = 4.0
        params['Silicon', 'Germanium', 'hamiltonian_matrix_element', 'p0p0p'] = 5.0
        params['Silicon', 'Germanium', 'hamiltonian_matrix_element_eta', 's0s0s'] = 1.0
        params['Silicon', 'Germanium', 'hamiltonian_matrix_element_eta', 's0p0s'] = 2.0
        params['Silicon', 'Germanium', 'hamiltonian_matrix_element_eta', 'p0s0s'] = 3.0
        params['Silicon', 'Germanium', 'hamiltonian_matrix_element_eta', 'p0p0s'] = 4.0
        params['Silicon', 'Germanium', 'hamiltonian_matrix_element_eta', 'p0p0p'] = 5.0

        params['Silicon', 'Germanium', 'nearest_neighbor_distance'] = 2.1
        params['Silicon', 'Germanium', 'second_neighbor_distance'] = 3.1


        hamiltonian_parametrization = parameterDictionaryToHamiltonianParametrization(params)
        basis_set = hamiltonian_parametrization.basisSet()

        reference_string = textwrap.dedent("""\
        from QuantumATK import *

        germanium_onsite_term = SlaterKosterOnsiteParameters(
            element=PeriodicTable.Germanium,
            angular_momenta=[ 0 , 1 ],
            occupations=[ 1.0 , 3.0 ],
            filling_method=SphericalSymmetric,
            ionization_potential=[ -10.5*eV , -8.5*eV ],
            onsite_hartree_shift=[ 6.657*eV , 6.657*eV ],
            onsite_spin_split=[[0.0, 0.0], [0.0, 0.0]]*eV,
            onsite_spin_orbit_split=[0.0, 0.0]*eV,
            vacuum_level=0.0*Hartree,
            )

        silicon_onsite_term = SlaterKosterOnsiteParameters(
            element=PeriodicTable.Silicon,
            angular_momenta=[ 0 , 1 ],
            occupations=[ 1.0 , 3.0 ],
            filling_method=SphericalSymmetric,
            ionization_potential=[ -10.0*eV , -8.0*eV ],
            onsite_hartree_shift=[ 6.657*eV , 6.657*eV ],
            onsite_spin_split=[[0.0, 0.0], [0.0, 0.0]]*eV,
            onsite_spin_orbit_split=[0.0, 0.0]*eV,
            vacuum_level=0.0*Hartree,
            )

        basis_set = SlaterKosterTable(
            germanium=germanium_onsite_term,
            silicon=silicon_onsite_term,
            si_si_s0s0s=[
                (1.6*Angstrom, -1.0*eV),
                (1.62*Angstrom, -1.0*eV),
                (1.64*Angstrom, -1.0*eV),
                (1.66*Angstrom, -1.0*eV),
                (1.68*Angstrom, -1.0*eV),
                (1.7*Angstrom, -1.0*eV),
                (1.72*Angstrom, -1.0*eV),
                (1.74*Angstrom, -1.0*eV),
                (1.76*Angstrom, -1.0*eV),
                (1.78*Angstrom, -1.0*eV),
                (1.8*Angstrom, -1.0*eV),
                (1.82*Angstrom, -1.0*eV),
                (1.8399999999999999*Angstrom, -1.0*eV),
                (1.8599999999999999*Angstrom, -1.0*eV),
                (1.88*Angstrom, -1.0*eV),
                (1.9*Angstrom, -1.0*eV),
                (1.92*Angstrom, -1.0*eV),
                (1.94*Angstrom, -1.0*eV),
                (1.96*Angstrom, -1.0*eV),
                (1.98*Angstrom, -1.0*eV),
                (2.0*Angstrom, -1.0*eV),
                (2.02*Angstrom, -1.0*eV),
                (2.04*Angstrom, -1.0*eV),
                (2.06*Angstrom, -1.0*eV),
                (2.08*Angstrom, -1.0*eV),
                (2.1*Angstrom, -1.0*eV),
                (2.12*Angstrom, -1.0*eV),
                (2.14*Angstrom, -1.0*eV),
                (2.16*Angstrom, -1.0*eV),
                (2.1799999999999997*Angstrom, -1.0*eV),
                (2.2*Angstrom, -1.0*eV),
                (2.2199999999999998*Angstrom, -1.0*eV),
                (2.24*Angstrom, -1.0*eV),
                (2.26*Angstrom, -1.0*eV),
                (2.2800000000000002*Angstrom, -1.0*eV),
                (2.3*Angstrom, -1.0*eV),
                (2.32*Angstrom, -1.0*eV),
                (2.34*Angstrom, -1.0*eV),
                (2.36*Angstrom, -1.0*eV),
                (2.38*Angstrom, -1.0*eV),
                (2.4*Angstrom, -1.0*eV),
                (2.5*Angstrom, 0.0*eV),
            ],
            si_si_s0p0s=[
                (1.6*Angstrom, -2.0*eV),
                (1.62*Angstrom, -2.0*eV),
                (1.64*Angstrom, -2.0*eV),
                (1.66*Angstrom, -2.0*eV),
                (1.68*Angstrom, -2.0*eV),
                (1.7*Angstrom, -2.0*eV),
                (1.72*Angstrom, -2.0*eV),
                (1.74*Angstrom, -2.0*eV),
                (1.76*Angstrom, -2.0*eV),
                (1.78*Angstrom, -2.0*eV),
                (1.8*Angstrom, -2.0*eV),
                (1.82*Angstrom, -2.0*eV),
                (1.8399999999999999*Angstrom, -2.0*eV),
                (1.8599999999999999*Angstrom, -2.0*eV),
                (1.88*Angstrom, -2.0*eV),
                (1.9*Angstrom, -2.0*eV),
                (1.92*Angstrom, -2.0*eV),
                (1.94*Angstrom, -2.0*eV),
                (1.96*Angstrom, -2.0*eV),
                (1.98*Angstrom, -2.0*eV),
                (2.0*Angstrom, -2.0*eV),
                (2.02*Angstrom, -2.0*eV),
                (2.04*Angstrom, -2.0*eV),
                (2.06*Angstrom, -2.0*eV),
                (2.08*Angstrom, -2.0*eV),
                (2.1*Angstrom, -2.0*eV),
                (2.12*Angstrom, -2.0*eV),
                (2.14*Angstrom, -2.0*eV),
                (2.16*Angstrom, -2.0*eV),
                (2.1799999999999997*Angstrom, -2.0*eV),
                (2.2*Angstrom, -2.0*eV),
                (2.2199999999999998*Angstrom, -2.0*eV),
                (2.24*Angstrom, -2.0*eV),
                (2.26*Angstrom, -2.0*eV),
                (2.2800000000000002*Angstrom, -2.0*eV),
                (2.3*Angstrom, -2.0*eV),
                (2.32*Angstrom, -2.0*eV),
                (2.34*Angstrom, -2.0*eV),
                (2.36*Angstrom, -2.0*eV),
                (2.38*Angstrom, -2.0*eV),
                (2.4*Angstrom, -2.0*eV),
                (2.5*Angstrom, 0.0*eV),
            ],
            si_si_p0p0s=[
                (1.6*Angstrom, -7.0*eV),
                (1.62*Angstrom, -7.0*eV),
                (1.64*Angstrom, -7.0*eV),
                (1.66*Angstrom, -7.0*eV),
                (1.68*Angstrom, -7.0*eV),
                (1.7*Angstrom, -7.0*eV),
                (1.72*Angstrom, -7.0*eV),
                (1.74*Angstrom, -7.0*eV),
                (1.76*Angstrom, -7.0*eV),
                (1.78*Angstrom, -7.0*eV),
                (1.8*Angstrom, -7.0*eV),
                (1.82*Angstrom, -7.0*eV),
                (1.8399999999999999*Angstrom, -7.0*eV),
                (1.8599999999999999*Angstrom, -7.0*eV),
                (1.88*Angstrom, -7.0*eV),
                (1.9*Angstrom, -7.0*eV),
                (1.92*Angstrom, -7.0*eV),
                (1.94*Angstrom, -7.0*eV),
                (1.96*Angstrom, -7.0*eV),
                (1.98*Angstrom, -7.0*eV),
                (2.0*Angstrom, -7.0*eV),
                (2.02*Angstrom, -7.0*eV),
                (2.04*Angstrom, -7.0*eV),
                (2.06*Angstrom, -7.0*eV),
                (2.08*Angstrom, -7.0*eV),
                (2.1*Angstrom, -7.0*eV),
                (2.12*Angstrom, -7.0*eV),
                (2.14*Angstrom, -7.0*eV),
                (2.16*Angstrom, -7.0*eV),
                (2.1799999999999997*Angstrom, -7.0*eV),
                (2.2*Angstrom, -7.0*eV),
                (2.2199999999999998*Angstrom, -7.0*eV),
                (2.24*Angstrom, -7.0*eV),
                (2.26*Angstrom, -7.0*eV),
                (2.2800000000000002*Angstrom, -7.0*eV),
                (2.3*Angstrom, -7.0*eV),
                (2.32*Angstrom, -7.0*eV),
                (2.34*Angstrom, -7.0*eV),
                (2.36*Angstrom, -7.0*eV),
                (2.38*Angstrom, -7.0*eV),
                (2.4*Angstrom, -7.0*eV),
                (2.5*Angstrom, 0.0*eV),
            ],
            si_si_p0p0p=[
                (1.6*Angstrom, -8.0*eV),
                (1.62*Angstrom, -8.0*eV),
                (1.64*Angstrom, -8.0*eV),
                (1.66*Angstrom, -8.0*eV),
                (1.68*Angstrom, -8.0*eV),
                (1.7*Angstrom, -8.0*eV),
                (1.72*Angstrom, -8.0*eV),
                (1.74*Angstrom, -8.0*eV),
                (1.76*Angstrom, -8.0*eV),
                (1.78*Angstrom, -8.0*eV),
                (1.8*Angstrom, -8.0*eV),
                (1.82*Angstrom, -8.0*eV),
                (1.8399999999999999*Angstrom, -8.0*eV),
                (1.8599999999999999*Angstrom, -8.0*eV),
                (1.88*Angstrom, -8.0*eV),
                (1.9*Angstrom, -8.0*eV),
                (1.92*Angstrom, -8.0*eV),
                (1.94*Angstrom, -8.0*eV),
                (1.96*Angstrom, -8.0*eV),
                (1.98*Angstrom, -8.0*eV),
                (2.0*Angstrom, -8.0*eV),
                (2.02*Angstrom, -8.0*eV),
                (2.04*Angstrom, -8.0*eV),
                (2.06*Angstrom, -8.0*eV),
                (2.08*Angstrom, -8.0*eV),
                (2.1*Angstrom, -8.0*eV),
                (2.12*Angstrom, -8.0*eV),
                (2.14*Angstrom, -8.0*eV),
                (2.16*Angstrom, -8.0*eV),
                (2.1799999999999997*Angstrom, -8.0*eV),
                (2.2*Angstrom, -8.0*eV),
                (2.2199999999999998*Angstrom, -8.0*eV),
                (2.24*Angstrom, -8.0*eV),
                (2.26*Angstrom, -8.0*eV),
                (2.2800000000000002*Angstrom, -8.0*eV),
                (2.3*Angstrom, -8.0*eV),
                (2.32*Angstrom, -8.0*eV),
                (2.34*Angstrom, -8.0*eV),
                (2.36*Angstrom, -8.0*eV),
                (2.38*Angstrom, -8.0*eV),
                (2.4*Angstrom, -8.0*eV),
                (2.5*Angstrom, 0.0*eV),
            ],
            si_ge_s0s0s=[
                (1.6800000000000002*Angstrom, 1.25*eV),
                (1.7010000000000003*Angstrom, 1.2345679012345678*eV),
                (1.722*Angstrom, 1.2195121951219512*eV),
                (1.7429999999999999*Angstrom, 1.2048192771084338*eV),
                (1.764*Angstrom, 1.1904761904761905*eV),
                (1.785*Angstrom, 1.1764705882352942*eV),
                (1.806*Angstrom, 1.1627906976744187*eV),
                (1.827*Angstrom, 1.1494252873563218*eV),
                (1.848*Angstrom, 1.1363636363636365*eV),
                (1.8690000000000002*Angstrom, 1.1235955056179776*eV),
                (1.8900000000000001*Angstrom, 1.1111111111111112*eV),
                (1.9110000000000003*Angstrom, 1.0989010989010988*eV),
                (1.932*Angstrom, 1.0869565217391306*eV),
                (1.9529999999999998*Angstrom, 1.0752688172043012*eV),
                (1.974*Angstrom, 1.0638297872340425*eV),
                (1.9949999999999999*Angstrom, 1.0526315789473684*eV),
                (2.016*Angstrom, 1.0416666666666667*eV),
                (2.037*Angstrom, 1.0309278350515465*eV),
                (2.058*Angstrom, 1.0204081632653061*eV),
                (2.079*Angstrom, 1.0101010101010102*eV),
                (2.1*Angstrom, 1.0*eV),
                (2.121*Angstrom, 0.9900990099009901*eV),
                (2.1420000000000003*Angstrom, 0.9803921568627451*eV),
                (2.1630000000000003*Angstrom, 0.970873786407767*eV),
                (2.184*Angstrom, 0.9615384615384615*eV),
                (2.205*Angstrom, 0.9523809523809523*eV),
                (2.2260000000000004*Angstrom, 0.9433962264150942*eV),
                (2.2470000000000003*Angstrom, 0.9345794392523364*eV),
                (2.2680000000000002*Angstrom, 0.9259259259259258*eV),
                (2.2889999999999997*Angstrom, 0.9174311926605506*eV),
                (2.3100000000000005*Angstrom, 0.9090909090909091*eV),
                (2.331*Angstrom, 0.900900900900901*eV),
                (2.3520000000000003*Angstrom, 0.8928571428571428*eV),
                (2.3729999999999998*Angstrom, 0.8849557522123894*eV),
                (2.3940000000000006*Angstrom, 0.8771929824561403*eV),
                (2.415*Angstrom, 0.8695652173913044*eV),
                (2.436*Angstrom, 0.8620689655172414*eV),
                (2.457*Angstrom, 0.8547008547008548*eV),
                (2.4779999999999998*Angstrom, 0.8474576271186441*eV),
                (2.499*Angstrom, 0.8403361344537815*eV),
                (2.52*Angstrom, 0.8333333333333334*eV),
                (2.6*Angstrom, 0.0*eV),
            ],
            si_ge_s0p0s=[
                (1.6800000000000002*Angstrom, 3.1249999999999996*eV),
                (1.7010000000000003*Angstrom, 3.048315805517451*eV),
                (1.722*Angstrom, 2.9744199881023206*eV),
                (1.7429999999999999*Angstrom, 2.903178980984178*eV),
                (1.764*Angstrom, 2.8344671201814062*eV),
                (1.785*Angstrom, 2.768166089965398*eV),
                (1.806*Angstrom, 2.7041644131963225*eV),
                (1.827*Angstrom, 2.642356982428326*eV),
                (1.848*Angstrom, 2.5826446280991737*eV),
                (1.8690000000000002*Angstrom, 2.524933720489837*eV),
                (1.8900000000000001*Angstrom, 2.4691358024691357*eV),
                (1.9110000000000003*Angstrom, 2.415167250332085*eV),
                (1.932*Angstrom, 2.362948960302458*eV),
                (1.9529999999999998*Angstrom, 2.3124060585038735*eV),
                (1.974*Angstrom, 2.2634676324128566*eV),
                (1.9949999999999999*Angstrom, 2.21606648199446*eV),
                (2.016*Angstrom, 2.170138888888889*eV),
                (2.037*Angstrom, 2.125624402168137*eV),
                (2.058*Angstrom, 2.0824656393169514*eV),
                (2.079*Angstrom, 2.0406081012141617*eV),
                (2.1*Angstrom, 2.0*eV),
                (2.121*Angstrom, 1.9605920988138417*eV),
                (2.1420000000000003*Angstrom, 1.9223375624759709*eV),
                (2.1630000000000003*Angstrom, 1.8851918182675087*eV),
                (2.184*Angstrom, 1.8491124260355027*eV),
                (2.205*Angstrom, 1.8140589569160996*eV),
                (2.2260000000000004*Angstrom, 1.7799928800284797*eV),
                (2.2470000000000003*Angstrom, 1.7468774565464231*eV),
                (2.2680000000000002*Angstrom, 1.7146776406035664*eV),
                (2.2889999999999997*Angstrom, 1.6833599865331206*eV),
                (2.3100000000000005*Angstrom, 1.652892561983471*eV),
                (2.331*Angstrom, 1.62324486648811*eV),
                (2.3520000000000003*Angstrom, 1.5943877551020407*eV),
                (2.3729999999999998*Angstrom, 1.5662933667475922*eV),
                (2.3940000000000006*Angstrom, 1.5389350569405968*eV),
                (2.415*Angstrom, 1.512287334593573*eV),
                (2.436*Angstrom, 1.4863258026159336*eV),
                (2.457*Angstrom, 1.4610271020527432*eV),
                (2.4779999999999998*Angstrom, 1.4363688595231257*eV),
                (2.499*Angstrom, 1.412329637737448*eV),
                (2.52*Angstrom, 1.3888888888888888*eV),
                (2.6*Angstrom, 0.0*eV),
            ],
            si_ge_p0s0s=[
                (1.6800000000000002*Angstrom, 5.859374999999998*eV),
                (1.7010000000000003*Angstrom, 5.645029269476762*eV),
                (1.722*Angstrom, 5.441012173357904*eV),
                (1.7429999999999999*Angstrom, 5.246709001778635*eV),
                (1.764*Angstrom, 5.061548428895368*eV),
                (1.785*Angstrom, 4.884998982291879*eV),
                (1.806*Angstrom, 4.71656583697033*eV),
                (1.827*Angstrom, 4.555787900738493*eV),
                (1.848*Angstrom, 4.4022351615326825*eV),
                (1.8690000000000002*Angstrom, 4.255506270488489*eV),
                (1.8900000000000001*Angstrom, 4.11522633744856*eV),
                (1.9110000000000003*Angstrom, 3.9810449181298107*eV),
                (1.932*Angstrom, 3.8526341744061816*eV),
                (1.9529999999999998*Angstrom, 3.7296871911352802*eV),
                (1.974*Angstrom, 3.6119164347013673*eV),
                (1.9949999999999999*Angstrom, 3.4990523399912528*eV),
                (2.016*Angstrom, 3.3908420138888893*eV),
                (2.037*Angstrom, 3.2870480445899024*eV),
                (2.058*Angstrom, 3.1874474071177827*eV),
                (2.079*Angstrom, 3.0918304563850936*eV),
                (2.1*Angstrom, 3.0*eV),
                (2.121*Angstrom, 2.911770443782933*eV),
                (2.1420000000000003*Angstrom, 2.826967003641133*eV),
                (2.1630000000000003*Angstrom, 2.7454249780594786*eV),
                (2.184*Angstrom, 2.6669890760127446*eV),
                (2.205*Angstrom, 2.591512795594428*eV),
                (2.2260000000000004*Angstrom, 2.5188578490969054*eV),
                (2.2470000000000003*Angstrom, 2.448893630672556*eV),
                (2.2680000000000002*Angstrom, 2.3814967230605086*eV),
                (2.2889999999999997*Angstrom, 2.316550440183194*eV),
                (2.3100000000000005*Angstrom, 2.2539444027047324*eV),
                (2.331*Angstrom, 2.1935741439028518*eV),
                (2.3520000000000003*Angstrom, 2.1353407434402327*eV),
                (2.3729999999999998*Angstrom, 2.079150486833087*eV),
                (2.3940000000000006*Angstrom, 2.0249145486060485*eV),
                (2.415*Angstrom, 1.9725486972959647*eV),
                (2.436*Angstrom, 1.9219730206240524*eV),
                (2.457*Angstrom, 1.873111669298389*eV),
                (2.4779999999999998*Angstrom, 1.8258926180378716*eV),
                (2.499*Angstrom, 1.780247442526195*eV),
                (2.52*Angstrom, 1.7361111111111114*eV),
                (2.6*Angstrom, 0.0*eV),
            ],
            si_ge_p0p0s=[
                (1.6800000000000002*Angstrom, 9.765624999999998*eV),
                (1.7010000000000003*Angstrom, 9.292229250167507*eV),
                (1.722*Angstrom, 8.84717426562261*eV),
                (1.7429999999999999*Angstrom, 8.428448195628329*eV),
                (1.764*Angstrom, 8.034203855389473*eV),
                (1.785*Angstrom, 7.6627435016343215*eV),
                (1.806*Angstrom, 7.312505173597411*eV),
                (1.827*Angstrom, 6.982050422587729*eV),
                (1.848*Angstrom, 6.6700532750495185*eV),
                (1.8690000000000002*Angstrom, 6.3752902928666515*eV),
                (1.8900000000000001*Angstrom, 6.096631611034903*eV),
                (1.9110000000000003*Angstrom, 5.8330328470766455*eV),
                (1.932*Angstrom, 5.583527788994466*eV),
                (1.9529999999999998*Angstrom, 5.34722177940542*eV),
                (1.974*Angstrom, 5.123285722980663*eV),
                (1.9949999999999999*Angstrom, 4.910950652619302*eV),
                (2.016*Angstrom, 4.709502797067902*eV),
                (2.037*Angstrom, 4.518279099092649*eV),
                (2.058*Angstrom, 4.3366631389357595*eV),
                (2.079*Angstrom, 4.164081422740868*eV),
                (2.1*Angstrom, 4.0*eV),
                (2.121*Angstrom, 3.843921377931265*eV),
                (2.1420000000000003*Angstrom, 3.695381704106057*eV),
                (2.1630000000000003*Angstrom, 3.553948191662755*eV),
                (2.184*Angstrom, 3.419216764118903*eV),
                (2.205*Angstrom, 3.2908098991675274*eV),
                (2.2260000000000004*Angstrom, 3.1683746529520818*eV),
                (2.2470000000000003*Angstrom, 3.0515808481901003*eV),
                (2.2680000000000002*Angstrom, 2.940119411185813*eV),
                (2.2889999999999997*Angstrom, 2.833700844260788*eV),
                (2.3100000000000005*Angstrom, 2.732053821460282*eV),
                (2.331*Angstrom, 2.6349238965800024*eV),
                (2.3520000000000003*Angstrom, 2.5420723136193244*eV),
                (2.3729999999999998*Angstrom, 2.453274910717507*eV),
                (2.3940000000000006*Angstrom, 2.3683211094807577*eV),
                (2.415*Angstrom, 2.2870129823721337*eV),
                (2.436*Angstrom, 2.2091643915218993*eV),
                (2.457*Angstrom, 2.1346001929326373*eV),
                (2.4779999999999998*Angstrom, 2.0631555006077646*eV),
                (2.499*Angstrom, 1.9946750056315912*eV),
                (2.52*Angstrom, 1.9290123456790125*eV),
                (2.6*Angstrom, 0.0*eV),
            ],
            si_ge_p0p0p=[
                (1.6800000000000002*Angstrom, 15.258789062499996*eV),
                (1.7010000000000003*Angstrom, 14.339859953962202*eV),
                (1.722*Angstrom, 13.48654613661983*eV),
                (1.7429999999999999*Angstrom, 12.69344607775351*eV),
                (1.764*Angstrom, 11.95566049909148*eV),
                (1.785*Angstrom, 11.268740443579883*eV),
                (1.806*Angstrom, 10.628641240693911*eV),
                (1.827*Angstrom, 10.031681641649037*eV),
                (1.848*Angstrom, 9.474507492968067*eV),
                (1.8690000000000002*Angstrom, 8.95405940009361*eV),
                (1.8900000000000001*Angstrom, 8.467543904215143*eV),
                (1.9110000000000003*Angstrom, 8.012407756973413*eV),
                (1.932*Angstrom, 7.586314930699003*eV),
                (1.9529999999999998*Angstrom, 7.18712604758793*eV),
                (1.974*Angstrom, 6.812879950772159*eV),
                (1.9949999999999999*Angstrom, 6.461777174499083*eV),
                (2.016*Angstrom, 6.132165100348831*eV),
                (2.037*Angstrom, 5.8225246122327965*eV),
                (2.058*Angstrom, 5.531458085377244*eV),
                (2.079*Angstrom, 5.257678564066752*eV),
                (2.1*Angstrom, 5.0*eV),
                (2.121*Angstrom, 4.757328438033744*eV),
                (2.1420000000000003*Angstrom, 4.528654049149579*eV),
                (2.1630000000000003*Angstrom, 4.31304392192082*eV),
                (2.184*Angstrom, 4.109635533796759*eV),
                (2.205*Angstrom, 3.917630832342294*eV),
                (2.2260000000000004*Angstrom, 3.736290864330285*eV),
                (2.2470000000000003*Angstrom, 3.5649308974183413*eV),
                (2.2680000000000002*Angstrom, 3.4029159851687645*eV),
                (2.2889999999999997*Angstrom, 3.24965693149173*eV),
                (2.3100000000000005*Angstrom, 3.1046066152957748*eV),
                (2.331*Angstrom, 2.967256640292796*eV),
                (2.3520000000000003*Angstrom, 2.8371342785929956*eV),
                (2.3729999999999998*Angstrom, 2.713799679997243*eV),
                (2.3940000000000006*Angstrom, 2.596843321799076*eV),
                (2.415*Angstrom, 2.4858836764914494*eV),
                (2.436*Angstrom, 2.3805650770710125*eV),
                (2.457*Angstrom, 2.280555761680168*eV),
                (2.4779999999999998*Angstrom, 2.1855460811522933*eV),
                (2.499*Angstrom, 2.095246854655033*eV),
                (2.52*Angstrom, 2.009387860082305*eV),
                (2.6*Angstrom, 0.0*eV),
            ],
            ge_ge_s0s0s=[
                (1.7600000000000002*Angstrom, -0.9600000000000002*eV),
                (1.7820000000000003*Angstrom, -0.9841500000000002*eV),
                (1.804*Angstrom, -1.0085999999999997*eV),
                (1.826*Angstrom, -1.03335*eV),
                (1.848*Angstrom, -1.0583999999999998*eV),
                (1.87*Angstrom, -1.08375*eV),
                (1.8920000000000001*Angstrom, -1.1094*eV),
                (1.9140000000000001*Angstrom, -1.1353499999999999*eV),
                (1.9360000000000002*Angstrom, -1.1616*eV),
                (1.9580000000000002*Angstrom, -1.18815*eV),
                (1.9800000000000002*Angstrom, -1.215*eV),
                (2.0020000000000002*Angstrom, -1.24215*eV),
                (2.024*Angstrom, -1.2695999999999998*eV),
                (2.046*Angstrom, -1.2973499999999998*eV),
                (2.068*Angstrom, -1.3254*eV),
                (2.09*Angstrom, -1.35375*eV),
                (2.112*Angstrom, -1.3824*eV),
                (2.134*Angstrom, -1.4113499999999999*eV),
                (2.156*Angstrom, -1.4405999999999999*eV),
                (2.178*Angstrom, -1.47015*eV),
                (2.2*Angstrom, -1.5*eV),
                (2.2220000000000004*Angstrom, -1.5301500000000001*eV),
                (2.244*Angstrom, -1.5606000000000002*eV),
                (2.2660000000000005*Angstrom, -1.59135*eV),
                (2.2880000000000003*Angstrom, -1.6224000000000003*eV),
                (2.3100000000000005*Angstrom, -1.65375*eV),
                (2.3320000000000003*Angstrom, -1.6854000000000002*eV),
                (2.3540000000000005*Angstrom, -1.7173500000000002*eV),
                (2.3760000000000003*Angstrom, -1.7496000000000003*eV),
                (2.3979999999999997*Angstrom, -1.7821499999999995*eV),
                (2.4200000000000004*Angstrom, -1.8150000000000002*eV),
                (2.4419999999999997*Angstrom, -1.8481499999999995*eV),
                (2.4640000000000004*Angstrom, -1.8816000000000004*eV),
                (2.4859999999999998*Angstrom, -1.9153499999999994*eV),
                (2.5080000000000005*Angstrom, -1.9494000000000002*eV),
                (2.53*Angstrom, -1.9837499999999997*eV),
                (2.552*Angstrom, -2.0183999999999997*eV),
                (2.574*Angstrom, -2.05335*eV),
                (2.596*Angstrom, -2.0886*eV),
                (2.618*Angstrom, -2.1241499999999998*eV),
                (2.64*Angstrom, -2.1599999999999997*eV),
                (2.7*Angstrom, 0.0*eV),
            ],
            ge_ge_s0p0s=[
                (1.7600000000000002*Angstrom, -1.6000000000000003*eV),
                (1.7820000000000003*Angstrom, -1.6402500000000002*eV),
                (1.804*Angstrom, -1.6809999999999996*eV),
                (1.826*Angstrom, -1.7222499999999998*eV),
                (1.848*Angstrom, -1.7639999999999998*eV),
                (1.87*Angstrom, -1.80625*eV),
                (1.8920000000000001*Angstrom, -1.849*eV),
                (1.9140000000000001*Angstrom, -1.89225*eV),
                (1.9360000000000002*Angstrom, -1.936*eV),
                (1.9580000000000002*Angstrom, -1.98025*eV),
                (1.9800000000000002*Angstrom, -2.025*eV),
                (2.0020000000000002*Angstrom, -2.07025*eV),
                (2.024*Angstrom, -2.1159999999999997*eV),
                (2.046*Angstrom, -2.16225*eV),
                (2.068*Angstrom, -2.209*eV),
                (2.09*Angstrom, -2.2562499999999996*eV),
                (2.112*Angstrom, -2.3040000000000003*eV),
                (2.134*Angstrom, -2.3522499999999997*eV),
                (2.156*Angstrom, -2.401*eV),
                (2.178*Angstrom, -2.45025*eV),
                (2.2*Angstrom, -2.5*eV),
                (2.2220000000000004*Angstrom, -2.55025*eV),
                (2.244*Angstrom, -2.601*eV),
                (2.2660000000000005*Angstrom, -2.65225*eV),
                (2.2880000000000003*Angstrom, -2.704*eV),
                (2.3100000000000005*Angstrom, -2.75625*eV),
                (2.3320000000000003*Angstrom, -2.809*eV),
                (2.3540000000000005*Angstrom, -2.8622500000000004*eV),
                (2.3760000000000003*Angstrom, -2.9160000000000004*eV),
                (2.3979999999999997*Angstrom, -2.970249999999999*eV),
                (2.4200000000000004*Angstrom, -3.0250000000000004*eV),
                (2.4419999999999997*Angstrom, -3.0802499999999995*eV),
                (2.4640000000000004*Angstrom, -3.1360000000000006*eV),
                (2.4859999999999998*Angstrom, -3.192249999999999*eV),
                (2.5080000000000005*Angstrom, -3.2490000000000006*eV),
                (2.53*Angstrom, -3.3062499999999995*eV),
                (2.552*Angstrom, -3.3639999999999994*eV),
                (2.574*Angstrom, -3.4222499999999996*eV),
                (2.596*Angstrom, -3.481*eV),
                (2.618*Angstrom, -3.5402499999999995*eV),
                (2.64*Angstrom, -3.5999999999999996*eV),
                (2.7*Angstrom, 0.0*eV),
            ],
            ge_ge_p0p0s=[
                (1.7600000000000002*Angstrom, -4.800000000000001*eV),
                (1.7820000000000003*Angstrom, -4.920750000000001*eV),
                (1.804*Angstrom, -5.042999999999999*eV),
                (1.826*Angstrom, -5.1667499999999995*eV),
                (1.848*Angstrom, -5.292*eV),
                (1.87*Angstrom, -5.418749999999999*eV),
                (1.8920000000000001*Angstrom, -5.547*eV),
                (1.9140000000000001*Angstrom, -5.67675*eV),
                (1.9360000000000002*Angstrom, -5.808*eV),
                (1.9580000000000002*Angstrom, -5.94075*eV),
                (1.9800000000000002*Angstrom, -6.075*eV),
                (2.0020000000000002*Angstrom, -6.210750000000001*eV),
                (2.024*Angstrom, -6.347999999999999*eV),
                (2.046*Angstrom, -6.48675*eV),
                (2.068*Angstrom, -6.627*eV),
                (2.09*Angstrom, -6.76875*eV),
                (2.112*Angstrom, -6.912*eV),
                (2.134*Angstrom, -7.056749999999999*eV),
                (2.156*Angstrom, -7.202999999999999*eV),
                (2.178*Angstrom, -7.350750000000001*eV),
                (2.2*Angstrom, -7.5*eV),
                (2.2220000000000004*Angstrom, -7.65075*eV),
                (2.244*Angstrom, -7.803000000000001*eV),
                (2.2660000000000005*Angstrom, -7.95675*eV),
                (2.2880000000000003*Angstrom, -8.112*eV),
                (2.3100000000000005*Angstrom, -8.26875*eV),
                (2.3320000000000003*Angstrom, -8.427000000000001*eV),
                (2.3540000000000005*Angstrom, -8.58675*eV),
                (2.3760000000000003*Angstrom, -8.748000000000001*eV),
                (2.3979999999999997*Angstrom, -8.910749999999998*eV),
                (2.4200000000000004*Angstrom, -9.075000000000001*eV),
                (2.4419999999999997*Angstrom, -9.240749999999998*eV),
                (2.4640000000000004*Angstrom, -9.408000000000003*eV),
                (2.4859999999999998*Angstrom, -9.576749999999997*eV),
                (2.5080000000000005*Angstrom, -9.747000000000002*eV),
                (2.53*Angstrom, -9.91875*eV),
                (2.552*Angstrom, -10.091999999999999*eV),
                (2.574*Angstrom, -10.266749999999998*eV),
                (2.596*Angstrom, -10.443*eV),
                (2.618*Angstrom, -10.62075*eV),
                (2.64*Angstrom, -10.799999999999999*eV),
                (2.7*Angstrom, 0.0*eV),
            ],
            ge_ge_p0p0p=[
                (1.7600000000000002*Angstrom, -5.44*eV),
                (1.7820000000000003*Angstrom, -5.576850000000001*eV),
                (1.804*Angstrom, -5.715399999999999*eV),
                (1.826*Angstrom, -5.85565*eV),
                (1.848*Angstrom, -5.997599999999999*eV),
                (1.87*Angstrom, -6.141249999999999*eV),
                (1.8920000000000001*Angstrom, -6.2866*eV),
                (1.9140000000000001*Angstrom, -6.43365*eV),
                (1.9360000000000002*Angstrom, -6.5824*eV),
                (1.9580000000000002*Angstrom, -6.73285*eV),
                (1.9800000000000002*Angstrom, -6.885000000000001*eV),
                (2.0020000000000002*Angstrom, -7.038850000000001*eV),
                (2.024*Angstrom, -7.194399999999999*eV),
                (2.046*Angstrom, -7.351649999999999*eV),
                (2.068*Angstrom, -7.510599999999999*eV),
                (2.09*Angstrom, -7.67125*eV),
                (2.112*Angstrom, -7.833600000000001*eV),
                (2.134*Angstrom, -7.997649999999999*eV),
                (2.156*Angstrom, -8.1634*eV),
                (2.178*Angstrom, -8.33085*eV),
                (2.2*Angstrom, -8.5*eV),
                (2.2220000000000004*Angstrom, -8.67085*eV),
                (2.244*Angstrom, -8.8434*eV),
                (2.2660000000000005*Angstrom, -9.01765*eV),
                (2.2880000000000003*Angstrom, -9.193600000000002*eV),
                (2.3100000000000005*Angstrom, -9.37125*eV),
                (2.3320000000000003*Angstrom, -9.550600000000001*eV),
                (2.3540000000000005*Angstrom, -9.73165*eV),
                (2.3760000000000003*Angstrom, -9.914400000000002*eV),
                (2.3979999999999997*Angstrom, -10.098849999999997*eV),
                (2.4200000000000004*Angstrom, -10.285000000000002*eV),
                (2.4419999999999997*Angstrom, -10.472849999999998*eV),
                (2.4640000000000004*Angstrom, -10.662400000000002*eV),
                (2.4859999999999998*Angstrom, -10.853649999999996*eV),
                (2.5080000000000005*Angstrom, -11.046600000000002*eV),
                (2.53*Angstrom, -11.241249999999999*eV),
                (2.552*Angstrom, -11.437599999999998*eV),
                (2.574*Angstrom, -11.635649999999998*eV),
                (2.596*Angstrom, -11.8354*eV),
                (2.618*Angstrom, -12.036849999999998*eV),
                (2.64*Angstrom, -12.239999999999998*eV),
                (2.7*Angstrom, 0.0*eV),
            ],
            )
        """)

        self.compareSlaterKosterTableFromScript(basis_set, reference_string)

    def testTwoElementSpRequiresNoRedundantDistance(self):
        """ Test that we don't need to specify nearest neighbor distance for non-interacting elements """
        # A dummy parametrization where Si and Ge do not interact.
        params = dict()
        params['elements'] = [Silicon, Germanium]

        params['Silicon', 'orbitals'] = ['s0', 'p0']
        params['Silicon', 'number_of_valence_electrons'] = 4.
        params['Silicon', 'onsite_spin_orbit_split'] = [0., 0.]
        params['Silicon', 'onsite_hartree_shift'] = ATK_U(
            Silicon, ['3p', '3p'], 'ncp').inUnitsOf(eV)

        params['Silicon', 'ionization_potential', 's0'] = -10.
        params['Silicon', 'ionization_potential', 'p0'] = -8.
        params['Silicon', 'hamiltonian_matrix_element', 's0s0s'] = -1.
        params['Silicon', 'hamiltonian_matrix_element', 's0p0s'] = -2.
        params['Silicon', 'hamiltonian_matrix_element', 'p0p0s'] = -7.
        params['Silicon', 'hamiltonian_matrix_element', 'p0p0p'] = -8.
        params['Silicon', 'eta', 's0s0s'] = -0.
        params['Silicon', 'eta', 's0p0s'] = -0.
        params['Silicon', 'eta', 'p0p0s'] = -0.
        params['Silicon', 'eta', 'p0p0p'] = -0.

        params['Silicon', 'nearest_neighbor_distance'] = 2.
        params['Silicon', 'second_neighbor_distance'] = 3.

        params['Germanium', 'orbitals'] = ['s0', 'p0']
        params['Germanium', 'number_of_valence_electrons'] = 4.
        params['Germanium', 'onsite_spin_orbit_split'] = [0., 0.]
        params['Germanium', 'onsite_hartree_shift'] = ATK_U(
            Silicon, ['3p', '3p'], 'ncp').inUnitsOf(eV)

        params['Germanium', 'ionization_potential', 's0'] = -10.5
        params['Germanium', 'ionization_potential', 'p0'] = -8.5
        params['Germanium', 'hamiltonian_matrix_element', 's0s0s'] = -1.5
        params['Germanium', 'hamiltonian_matrix_element', 's0p0s'] = -2.5
        params['Germanium', 'hamiltonian_matrix_element', 'p0p0s'] = -7.5
        params['Germanium', 'hamiltonian_matrix_element', 'p0p0p'] = -8.5
        params['Germanium', 'hamiltonian_matrix_element_eta', 's0s0s'] = -2
        params['Germanium', 'hamiltonian_matrix_element_eta', 's0p0s'] = -2
        params['Germanium', 'hamiltonian_matrix_element_eta', 'p0p0s'] = -2
        params['Germanium', 'hamiltonian_matrix_element_eta', 'p0p0p'] = -2

        params['Germanium', 'nearest_neighbor_distance'] = 2.2
        params['Germanium', 'second_neighbor_distance'] = 3.2

        # No interaction terms.
        hamiltonian_parametrization = parameterDictionaryToHamiltonianParametrization(params)
        # The basis set should be created and do not complain about missing distance specification.
        basis_set = hamiltonian_parametrization.basisSet()
        # check that there are no Si-Ge entries.
        for offsite_key in basis_set.offsiteParameters().keys():
            self.assertNotIn('si_ge', offsite_key)
            self.assertNotIn('ge_si', offsite_key)


    def testBoykinParameters(self):
        """ Test that we can use the utility to reproduce a built-in Boykin Silicon basis """
        #FIXME: does not recognize that ss1s and s0s1s are the same thing when comparing the tables.
        # Create and initialize the fitting parameters.
        params = dict()
        params['elements'] = [Silicon]
        params['Silicon', 'orbitals'] = ['s0', 'p', 'd', 's1']
        params['Silicon', 'number_of_valence_electrons'] = 4.
        params['Silicon', 'onsite_spin_orbit_split'] = [0., 0.01989, 0., 0.]
        params['Silicon', 'onsite_spin_split'] = ATK_W(
            PeriodicTable.Silicon, [ "3p", "3p", "3p", "3s" ]).inUnitsOf(eV)
        params['Silicon', 'onsite_hartree_shift'] = ATK_U(
            Silicon, ['3p', '3p', '3p', '3p'], 'ncp').inUnitsOf(eV)
        params['Silicon', 'ionization_potential', 's0'] = -2.15168
        params['Silicon', 'ionization_potential', 'p'] =  4.22925
        params['Silicon', 'ionization_potential', 'd'] = 13.78950
        params['Silicon', 'ionization_potential', 's1'] = 19.11650

        # The sign in the initial guess is only to keep track of whch values are positive or
        # or negative in the Boykin parametrization, in case we want to play with the range.
        params['Silicon', 'hamiltonian_matrix_element', 's0s0s'] = -1.95933
        params['Silicon', 'hamiltonian_matrix_element', 's1s1s'] = -4.24135
        params['Silicon', 'hamiltonian_matrix_element', 's0s1s'] = -1.52230
        params['Silicon', 'hamiltonian_matrix_element', 's0ps'] = 3.02562
        params['Silicon', 'hamiltonian_matrix_element', 's1ps'] = 3.15565
        params['Silicon', 'hamiltonian_matrix_element', 's0ds'] = -2.28485
        params['Silicon', 'hamiltonian_matrix_element', 's1ds'] = -0.80993
        params['Silicon', 'hamiltonian_matrix_element', 'pps'] = 4.10364
        params['Silicon', 'hamiltonian_matrix_element', 'ppp'] = -1.51801
        params['Silicon', 'hamiltonian_matrix_element', 'pds'] = -1.35554
        params['Silicon', 'hamiltonian_matrix_element', 'pdp'] = 2.38479
        params['Silicon', 'hamiltonian_matrix_element', 'dds'] = -1.68136
        params['Silicon', 'hamiltonian_matrix_element', 'ddp'] = 2.58880
        params['Silicon', 'hamiltonian_matrix_element', 'ddd'] = -1.81400

        params['Silicon', 'hamiltonian_matrix_element_eta', 's0s0s'] = 0.56247
        params['Silicon', 'hamiltonian_matrix_element_eta', 's1s1s'] = 0.19237
        params['Silicon', 'hamiltonian_matrix_element_eta', 's0s1s'] = 0.13203
        params['Silicon', 'hamiltonian_matrix_element_eta', 's0ps'] = 2.36548
        params['Silicon', 'hamiltonian_matrix_element_eta', 's1ps'] = 0.34492
        params['Silicon', 'hamiltonian_matrix_element_eta', 's0ds'] = 2.56720
        params['Silicon', 'hamiltonian_matrix_element_eta', 's1ds'] = 1.08601
        params['Silicon', 'hamiltonian_matrix_element_eta', 'pps'] = 0.20000
        params['Silicon', 'hamiltonian_matrix_element_eta', 'ppp'] = 1.67770
        params['Silicon', 'hamiltonian_matrix_element_eta', 'pds'] = 0.20000
        params['Silicon', 'hamiltonian_matrix_element_eta', 'pdp'] = 4.43250
        params['Silicon', 'hamiltonian_matrix_element_eta', 'dds'] = 0.10000
        params['Silicon', 'hamiltonian_matrix_element_eta', 'ddp'] = 6.00000
        params['Silicon', 'hamiltonian_matrix_element_eta', 'ddd'] = 5.99970

        a = 5.431
        params['Silicon', 'nearest_neighbor_distance'] = numpy.sqrt(3.0) / 4.0 * a
        params['Silicon', 'second_neighbor_distance'] = numpy.sqrt(2.0) / 2.0 * a

        hamiltonian_parametrization = (
            parameterDictionaryToHamiltonianParametrization(
                params))

        self.compareSlaterKosterTableFromScript(
            hamiltonian_parametrization.basisSet(), Boykin.Si_Basis)


if __name__ == '__main__':
    unittest.main()
