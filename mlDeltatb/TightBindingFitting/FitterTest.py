import contextlib
import numpy
import tempfile
import os
import unittest

from QuantumATK import *

from .Fitter import FreeParameter
from .Fitter import ReferenceBandstructureFromMemory
from .Fitter import TargetParameters


#XXX Might be a common utility.
@contextlib.contextmanager
def temporaryFile(suffix, dir):
    with tempfile.NamedTemporaryFile(suffix='.hdf5', dir=dir) as temporary_file:
        filename = temporary_file.name
    try:
        yield filename
    finally:
        os.remove(filename)


class TargetParametersTest(unittest.TestCase):
    """ Test module fot TargetParameters"""
    def testFreeParameter(self):
        """ Test the FreeParameter object basic functionality """
        foo = FreeParameter(1.0)
        self.assertEqual(foo.value(), 1.0)
        self.assertEqual(foo.bounds()[0], -numpy.inf)
        self.assertEqual(foo.bounds()[1], numpy.inf)
        with self.assertRaisesRegex(ValueError, "The parameter value must be*"):
            FreeParameter('a')
        with self.assertRaisesRegex(ValueError, "The parameter value must be*"):
            FreeParameter(1)

    def testConstruction(self):
        """ Test that the object can be constructed and used as a dictionary """
        foo = TargetParameters()
        foo['bar'] = 0
        foo['fox'] = FreeParameter(1.0)

    def testSetAsFreeParameters(self):
        """ Test the functionality to change parameters to Free parameters """
        bar = TargetParameters()
        bar['A'] = 0.
        bar['A', 'B'] = 1.
        bar['C', 0] = 2.

        bar.setAsFreeParameters(lambda key: ('A' in key))
        self.assertIsInstance(bar['A'], FreeParameter)
        self.assertEqual(bar['A'].value(), 0.)
        self.assertIsInstance(bar['A', 'B'], FreeParameter)
        self.assertEqual(bar['A', 'B'].value(), 1.)
        self.assertEqual(bar['C', 0], 2.)
        bar.setAsFreeParameters(lambda key: key == ('C', 0))
        self.assertIsInstance(bar['C', 0], FreeParameter)

    def testSetBounds(self):
        """ Test the setBounds functionality """
        foo = TargetParameters()
        foo['A'] = FreeParameter(0.)
        foo['A', 'B'] = FreeParameter(1., -5., 5.)
        foo['C', 0] = 2.

        foo.setBounds('A', -1.0, 1.0)
        self.assertEqual(foo['C', 0], 2.)


class FitterTest(unittest.TestCase):

    def testBandstructureFromMemory(self):
        # -----------------------------------------------------------------------------
        # Si, fcc
        # -----------------------------------------------------------------------------
        # Set up lattice
        lattice = FaceCenteredCubic(5.4306*Angstrom)

        # Define elements
        elements = [Silicon, Silicon]

        # Define coordinates
        fractional_coordinates = [[ 0.  ,  0.  ,  0.  ],
                                [ 0.25,  0.25,  0.25]]

        # Set up configuration
        bulk_configuration = BulkConfiguration(
            bravais_lattice=lattice,
            elements=elements,
            fractional_coordinates=fractional_coordinates
            )

        k_point_sampling = KpointDensity(
            density_a=4.0*Angstrom,
            )
        numerical_accuracy_parameters=NumericalAccuracyParameters(k_point_sampling=k_point_sampling)
        calculator = HuckelCalculator(numerical_accuracy_parameters=numerical_accuracy_parameters)
        bulk_configuration.setCalculator(calculator)

        # Create a Bandstructure from Huckel calculator
        reference_bandstructure = Bandstructure(configuration=bulk_configuration)

        reference = ReferenceBandstructureFromMemory(
            reference_bandstructure,
        )

        # We evaluate te residual using the same calculator. It should be a vector of zeros.
        residual = reference.evaluateResidual(bulk_configuration.calculator())
        self.assertAlmostEqual(numpy.linalg.norm(residual), 0.0)
        # Now we change calculator, we should get a finite residual.
        calculator = LCAOCalculator()
        residual = reference.evaluateResidual(calculator)
        self.assertNotAlmostEqual(numpy.linalg.norm(residual), 0.0)


if __name__ == '__main__':
    unittest.main()
