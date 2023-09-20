import numpy
import unittest

from QuantumATK import *

from .Descriptor import Descriptor, CoulombMatrixDescriptor, CountSpeciesDescriptor


class DescriptorTest(unittest.TestCase):

    def testGetNeighbour(self):
        a = 5.87476
        lattice = FaceCenteredCubic(a * Angstrom)
        fractional_coordinates = [[0., 0., 0.],
                                  [0.25, 0.25, 0.25]]

        # Set up configurations
        gaas = BulkConfiguration(
            bravais_lattice=lattice,
            elements=[Gallium, Arsenic],
            fractional_coordinates=fractional_coordinates
        )

        d_n1 = numpy.sqrt(3.0) / 4.0 * a
        d_n2 = numpy.sqrt(2.0) / 2.0 * a
        r_cut = (d_n1 + d_n2)/2. * Angstrom

        generic_descriptor = Descriptor(r_cut=r_cut)
        reduced_neighbours = generic_descriptor.getNeighbours(gaas)

        # Each atom must have 4 neighbours
        for neighbourhood in reduced_neighbours:
            self.assertEqual(len(neighbourhood), 4)

        # Repeat configuration for testing
        atoms = gaas.elements()

        # Neighbours of Gallium must be Arsenic
        for n in reduced_neighbours[0]:
            self.assertIs(atoms[n], Arsenic)

        # Neighbours of Arsenic must be Gallium
        for n in reduced_neighbours[1]:
            self.assertIs(atoms[n], Gallium)

    def testGetNeighbourDefaultDistance(self):
        a = 5.87476
        lattice = FaceCenteredCubic(a * Angstrom)
        fractional_coordinates = [[0., 0., 0.],
                                  [0.25, 0.25, 0.25]]

        # Set up configurations
        gaas = BulkConfiguration(
            bravais_lattice=lattice,
            elements=[Gallium, Arsenic],
            fractional_coordinates=fractional_coordinates
        )

        generic_descriptor = Descriptor()
        reduced_neighbours = generic_descriptor.getNeighbours(gaas)

        # Each atom must have 4 neighbours
        for neighbourhood in reduced_neighbours:
            self.assertEqual(len(neighbourhood), 4)

        # Repeat configuration for testing
        atoms = gaas.elements()

        # Neighbours of Gallium must be Arsenic
        for n in reduced_neighbours[0]:
            self.assertIs(atoms[n], Arsenic)

        # Neighbours of Arsenic must be Gallium
        for n in reduced_neighbours[1]:
            self.assertIs(atoms[n], Gallium)

    def testGetSecondNeighbour(self):
        a = 5.87476
        lattice = FaceCenteredCubic(a * Angstrom)
        fractional_coordinates = [[0., 0., 0.],
                                  [0.25, 0.25, 0.25]]

        # Set up configurations
        gaas = BulkConfiguration(
            bravais_lattice=lattice,
            elements=[Gallium, Arsenic],
            fractional_coordinates=fractional_coordinates
        )

        d_n2 = numpy.sqrt(2.0) / 2.0 * a
        r_cut = (d_n2 + 0.5) * Angstrom

        count_descriptor = CountSpeciesDescriptor(r_cut=r_cut).computeFeatures(gaas)

        # 13 Gallium 4 Arsenic
        self.assertEqual(list(count_descriptor[0]), [13, 4])
        # 4 Gallium 13 Arsenic
        self.assertEqual(list(count_descriptor[1]), [4, 13])

    def testCountSpeciesDescriptor(self):
        a = 5.87476
        lattice = FaceCenteredCubic(a * Angstrom)
        fractional_coordinates = [[0., 0., 0.],
                                  [0.25, 0.25, 0.25]]

        # Set up configurations
        gaas = BulkConfiguration(
            bravais_lattice=lattice,
            elements=[Gallium, Arsenic],
            fractional_coordinates=fractional_coordinates
        )

        d_n1 = numpy.sqrt(3.0) / 4.0 * a
        d_n2 = numpy.sqrt(2.0) / 2.0 * a
        r_cut = (d_n1 + d_n2) / 2. * Angstrom

        count_descriptor = CountSpeciesDescriptor(r_cut=r_cut).computeFeatures(gaas)

        # 1 Gallium 4 Arsenic
        self.assertEqual(list(count_descriptor[0]), [1, 4])
        # 4 Gallium 1 Arsenic
        self.assertEqual(list(count_descriptor[1]), [4, 1])

    def testCoulombMatrixDescriptor(self):
        a = 5.87476
        lattice = FaceCenteredCubic(a * Angstrom)
        fractional_coordinates = [[0., 0., 0.],
                                  [0.25, 0.25, 0.25]]

        # Set up configurations
        gaas = BulkConfiguration(
            bravais_lattice=lattice,
            elements=[Gallium, Arsenic],
            fractional_coordinates=fractional_coordinates
        )

        d_n1 = numpy.sqrt(3.0) / 4.0 * a
        d_n2 = numpy.sqrt(2.0) / 2.0 * a
        r_cut = (d_n1 + d_n2) / 2. * Angstrom

        descriptor = CoulombMatrixDescriptor(r_cut=r_cut)
        neighbours = descriptor.getNeighbours(gaas)

        coulomb_descriptor = descriptor.computeFeatures(gaas)

        for i, line in enumerate(coulomb_descriptor):
            m = len(neighbours[i]) + 1
            self.assertEqual(len(line), 2*m-1)

        target_value = 0.5 * Gallium.atomicNumber() ** 2.4
        self.assertEqual(coulomb_descriptor[0][0], target_value)

    def testCoulombMatrixExceptions(self):
        a = 5.87476
        lattice = FaceCenteredCubic(a * Angstrom)
        fractional_coordinates = [[0., 0., 0.],
                                  [0.25, 0.25, 0.25]]

        # Set up configurations
        gaas = BulkConfiguration(
            bravais_lattice=lattice,
            elements=[Gallium, Arsenic],
            fractional_coordinates=fractional_coordinates
        )

        descriptor = CoulombMatrixDescriptor()

        with self.assertRaises(TypeError):
            descriptor.computeFeatures(gaas, -1)

        with self.assertRaises(TypeError):
            descriptor.computeFeatures(gaas, True)


if __name__ == '__main__':
    unittest.main()
