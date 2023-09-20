import numpy
import scipy
import pickle
import pprint
import abc
from collections import OrderedDict
import uuid

# Particle Swarm Optimization support
try:
    from pyswarm import pso
    has_pyswarm = True
except:
    has_pyswarm = False
# Genetic Algorithm support
try:
    import pygad
    has_pygad = True
except:
    has_pygad = False

from QuantumATK import *

"""
A module with utilities to fit a tight binding set to
a collection of reference data.
"""

################################################################################
#
# Fitting target utilities
#
################################################################################


class FittingTargetInterface:
    def __init__(self,
                 weight=1.0,
                 object_id=str(uuid.uuid4())):
        self._weight = weight
        self._object_id = object_id

    def objectId(self):
        return self._object_id

    def weight(self):
        return self._weight

    @abc.abstractmethod
    def evaluateResidual(self, semi_empirical_calculator):
        pass


################################################################################
#
# Parameter dictionary
#
################################################################################

class FreeParameter:
    """
    A class describing a free parameter, subject to optimization. Free parameters
    can only contain real scalar values.
    """
    def __init__(self, value=0.0, min_bound=-numpy.inf, max_bound=numpy.inf):
        """
        :param value: the initial value. In the case of search on interval, min_bound and
            max_bound are defined as decrement/increment with respect to this value.
        :param value: float

        :min_bound: the minimum bound, as distance from the initial value (i.e. min_value = value + min_bound)
        :param min_bound: float

        :max_bound: the minimum bound, as distance from the initial value (i.e. max_value = value + max_bound)
        :param max_bound: float
        """
        #XXX something a bit more robust to check if float
        if ((not isinstance(value, float)) or value + min_bound > value + max_bound):
            raise ValueError(
                "The parameter value must be a float; min_bound and max_bound must define a positive ordered interval.")
        self._value = value
        self._bounds = (min_bound + value, max_bound + value)

    def value(self):
        return self._value

    def bounds(self):
        return self._bounds


class BoundParameter:
    """ A class describing a parameter, bound to another free parameter identified by a dictionary key. """
    def __init__(self, key):
        self._bound_key = key

    def boundKey(self):
        return self._bound_key

class TargetParameters(OrderedDict):
    """
    A dictionary-like class to declare all parameters, including the free parameters to be
    optimized and the non-free parameters required for a given hamiltonian parametrization.

    Keys are required to follow this syntax:
    <parameter>, (<Elements> ... )
    """
    def setAsFreeParameters(self, key_filter=None, min_bound=-numpy.inf, max_bound=numpy.inf):
        """
        Set all float values in the dictionary as Free parameters

        :param key_filter:
            A filter function to be applied to each key of the dictionary.
            The parameter is set as free only if the filter returns True.
            |DEFAULT| `lambda: True`
        :type key_filter:
            A lambda function.
        """
        if key_filter is None:
            key_filter = lambda: True
        for key, value in self.items():
            #XXX pretty sure this check must be improved.
            if (isinstance(value, float) and key_filter(key)):
                self[key] = FreeParameter(self[key], min_bound, max_bound)

    def freeItems(self):
        # XXX Return a generator instead?
        return [x for x in self.items() if isinstance(x[1], FreeParameter)]

    def setBounds(self, key, min_bound, max_bound):
        if not isinstance(self[key], FreeParameter):
            raise ValueError("Cannot set bounds for non FreeParameter value")
        self[key] = FreeParameter(
            self[key].value(), min_bound=min_bound, max_bound=max_bound)

    def asPlainDictionary(self):
        """
        Return a dictionary where all FreeParameter end BoundParameter entries are
        substituted by their current value.
        """
        result = dict()
        for key, value in self.items():
            #XXX pretty sure this check must be improved.
            if isinstance(value, FreeParameter):
                result[key] = value.value()
            elif isinstance(value, BoundParameter):
                bound_value = self[value.boundKey()]
                if not isinstance(bound_value, FreeParameter):
                    raise ValueError(
                        "Bound parameter {} is not bound to a FreeParametr".format(key))
                result[key] = bound_value.value()
            else:
                result[key] = value

        return result

    def numberOfDofs(self):
        """
        Return the number of degrees of freedom.
        """
        ndofs = len([x for x in self.values() if isinstance(x, FreeParameter)])
        return ndofs


class ReferenceBandstructureFromMemory(FittingTargetInterface):
    def __init__(self,
                 bandstructure,
                 weight=1.0,
                 object_id=str(uuid.uuid4()),
                 emin_emax=(-1.0, 1.0)*eV):
        super(ReferenceBandstructureFromMemory, self).__init__(weight=weight,
                                                               object_id=object_id
                                                               )

        self._target_bandstructure = bandstructure
        self.__emin_emax = emin_emax

    def targetQuantity(self):
        return self._target_bandstructure

    def _energyRange(self):
        """ Return the target energy range relative to valence edge as energy zero. """
        target_bandstructure = self.targetQuantity()
        emin, emax = (
            self.__emin_emax[0],
            self.__emin_emax[1] + target_bandstructure.indirectBandGap())
        return (emin, emax)

    def evaluateTarget(self, semi_empirical_calculator):
        configuration = self._target_bandstructure._configuration()
        kpoints = self._target_bandstructure.kpoints()
        configuration.setCalculator(semi_empirical_calculator)
        setVerbosity(SilentLog)

        configuration.update()
        result = Bandstructure(configuration, kpoints=kpoints)
        return result

    def evaluateResidual(self, semi_empirical_calculator):
        def shiftedBandArray(bands):
            """
            Shift bands such that the top of valence is at zero and return as ndarray.
            Crop values out of the energy range.
            """
            emin, emax = (x.inUnitsOf(eV) for x in self._energyRange())
            if emin > emax:
                raise ValueError("Energy range has emin is larger than emax")
            bands_array = bands.evaluate().inUnitsOf(eV)
            band_edge = numpy.max(numpy.where(bands_array < 0, bands_array, -numpy.inf))
            bands_array -= band_edge
            bands_array = numpy.where(bands_array < emax, bands_array, emax)
            bands_array = numpy.where(bands_array > emin, bands_array, emin)
            return bands_array

        # Get the target data
        target_bandstructure = self.targetQuantity()
        target_bands = shiftedBandArray(target_bandstructure)

        # Get custom bands
        bands = self.evaluateTarget(semi_empirical_calculator)

        if bands is None:
            # The calculation went wrong. Return a dummy large residual.
            return [1e3]
        else:
            custom_bands = shiftedBandArray(bands)

        # Determine which bands to compare. Make sure to take care of the following cases:
        #   - overall more bands available in one calculator
        #   - more valence bands available in one calculator (e.g. semi-core states)
        target_occupied_bands = self.targetQuantity()._numberOfOccupiedBands()[0]
        custom_occupied_bands = bands._numberOfOccupiedBands()[0]

        offset_target = max(0, target_occupied_bands - custom_occupied_bands)
        offset_custom = max(0, custom_occupied_bands - target_occupied_bands)

        nband = min(
            target_bands.shape[1] - offset_target,
            custom_bands.shape[1] - offset_custom)

        target_bands = target_bands[
            :,
            offset_target:offset_target + nband]

        custom_bands = custom_bands[
            :,
            offset_custom:offset_custom + nband]

        # Return the whole array as list. The Fitter deals with the cost function dimensionality.
        return (custom_bands - target_bands).flatten().tolist()

    def plotComparison(self, custom_calculator, path='tmp.png'):
        """
        Generate a 2D plot with the energies for the reference data and the custom calculator.
        """
        import os
        os.environ['QT_QPA_PLATFORM'] = 'offscreen'
        import pylab

        custom_bands = self.evaluateTarget(custom_calculator)
        target_bands = self.targetQuantity()

        emin = self.__emin_emax[0] + target_bands.valenceBandEdge()
        emax = self.__emin_emax[1] + target_bands.conductionBandEdge()

        if isMainProcess():
            pylab.plot(target_bands.evaluate()[:, :].inUnitsOf(eV), 'b-', label='Target bands')
            pylab.plot(custom_bands.evaluate()[:, :].inUnitsOf(eV), 'r--', label='Fitted bands')
            pylab.title('Bandstructure fit: {}'.format(self.objectId()))
            pylab.ylim((emin.inUnitsOf(eV), emax.inUnitsOf(eV)))
            # pylab.legend()
            pylab.savefig(path)
            pylab.close()


################################################################################
#
# The Fitter!
#
################################################################################

class SemiEmpiricalFitter():
    def __init__(
            self,
            targets,
            semi_empirical_calculator,
            parameters,
            dictionary_to_parametrization_method,
            optimizer='least_squares',
            optimizer_kwargs={},
            filename_prefix='fitter_output'):
        """
        A class to perform calculations for a serie of references and minimize
        a given error function (currently a weighted RMSE).

        :param targets:
            A list of target studies.
        :type targets:
            list

        :param semi_empirical_calculator:
            The semi_empirical_calculator to be used.
        :type semi_empirical_calculator:
            :class:~.`SemiEmpiricalCalculator`

        :param parameters:
            A dictionary of fitting parameters.
        :type parameters:
            Dictionary with ``FreeParameter`` or float values.
        """
        # Save the reference calculations studies.
        self._targets = targets

        self._calculator = semi_empirical_calculator
        self._parameters = parameters

        # Initialize a bookkeeping og free parameters.
        self._free_variables_keys = []
        self._initial_guess = []
        self._min_bounds = []
        self._max_bounds = []
        # Items are ordered as insertion in py3, the loop is deterministic.
        for key, value in self._parameters.freeItems():
            self._free_variables_keys.append(key)
            self._initial_guess.append(value.value())
            self._min_bounds.append(value.bounds()[0])
            self._max_bounds.append(value.bounds()[1])

        self._filename_prefix = filename_prefix
        self._optimizer_kwargs = optimizer_kwargs
        self._dictionary_to_parametrization = dictionary_to_parametrization_method
        self._optimizer = optimizer
        # Get the weights upfront, so we can normalize them.
        self._weights = numpy.array([t.weight() for t in targets])
        self._weights /= numpy.linalg.norm(self._weights)

    def freeParametersToCalculator(self, x):
        """
        Build the calculator with given free parameters.

        :param x: The array of free parameters
        :type param: As accepted from ``freeParametersToDictionary``
        """
        param_dict = self.freeParametersToDictionary(x)

        hamiltonian_parametrization = self._dictionary_to_parametrization(param_dict)

        # Deal with pair_potential not in hamiltonian parametrization.
        try:
            calculator = self._calculator()(
                hamiltonian_parametrization=hamiltonian_parametrization[0],
                pair_potentials=hamiltonian_parametrization[1])
        except TypeError:
            calculator = self._calculator()(
                hamiltonian_parametrization=hamiltonian_parametrization)

        return calculator

    def freeParametersToDictionary(self, x):
        """
        Translate the array of free variables to a parameter dictionary.

        :param x: The array of free parameters.
        :type x: array of floats
        """
        # Make a copy of the parameter dictionary.
        updated_parameters = dict(self._parameters)
        # Set all values as fixed.
        for index, key in enumerate(self._free_variables_keys):
            updated_parameters[key] = x[index]
        # Set the bound values.
        for key, value in updated_parameters.items():
            if isinstance(value, BoundParameter):
                updated_parameters[key] = updated_parameters[value.boundKey()]

        return updated_parameters

    def _evaluateTargetResiduals(self, x):
        """
        Evaluate the residual between the reference calculation and the optimized one.

        :param x: The array of free floating parameters
        :type param: As accepted from ``freeParametersToDictionary``

        :return: The list of residual vector (as list) per each target
        :rtype: a list of list of floats.
        """
        # Make sure that all processes are aligned when setting parameters and
        # when retrieving residuals.
        fitting_calculator = self.freeParametersToCalculator(x)
        residual_per_target = [t.evaluateResidual(fitting_calculator) for t in self._targets]
        return residual_per_target

    def evaluateResidual(self, x):
        """
        Evaluate the total residual between the reference calculation and the optimized one.

        :param x: The array of free parameters
        :type param: As accepted from ``freeParametersToDictionary``
        """
        result = []
        residual_per_target = self._evaluateTargetResiduals(x)
        for weight, residual in zip(self._weights, residual_per_target):
            prefactor = weight / numpy.sqrt(len(residual))
            result += [x * prefactor for x in residual]
        return result

    def evaluateCostFunction(self, x):
        """
        Evaluate the cost function as weighted RMSE.

        :param x: The array of free parameters
        :type param: As accepted from ``freeParametersToDictionary``
        """
        #XXX Maybe this could be a user defined function.
        cost = 0.
        residual_per_target = self._evaluateTargetResiduals(x)
        for weight, residual in zip(self._weights, residual_per_target):
            residual = numpy.array(residual)
            cost += numpy.linalg.norm(residual) * weight / numpy.sqrt(len(residual))
        return cost

    def update(self):
        """
        Run the fitting procedure.
        """
        if isMainProcess():
            print('Starting fitting.')
            print('Saving to {}'.format(self._filename_prefix))
            print('Initial parameters:')
            pp = pprint.PrettyPrinter(indent=4)
            pp.pprint(self._parameters)
            print('\nTargets:')
            for target in self._targets:
                print(target.objectId())

        if self._optimizer == 'least_squares':
            # Assign our own defaults.
            optimizer_kwargs = {
                'bounds': (self._min_bounds, self._max_bounds),
                'verbose': 2,
                'max_nfev': 120,
                'ftol': 1e-3
            }
            optimizer_kwargs.update(self._optimizer_kwargs)

            result = scipy.optimize.least_squares(
                self.evaluateResidual,
                self._initial_guess,
                **optimizer_kwargs)

            param_dict = self.freeParametersToDictionary(result.x)
            # Save some stuff.
            if isMainProcess():
                print('\nFitting terminated')
                print('Cost : {}\n'.format(result.cost))
                print('Residuals: {}\n'.format(result.fun))
                print('Number of function evaluations: {}\n'.format(result.nfev))
                print('Result:')

        elif self._optimizer == 'pso':

            if not has_pyswarm:
                raise RuntimeError(
                    "pyswarm is not installed. Install it with `atkpython -m pip install pyswarm`")

            ndofs = len(self._free_variables_keys)
            optimizer_kwargs = {
                'lb': self._min_bounds,
                'ub': self._max_bounds,
                'debug': True,
                'swarmsize': min(ndofs * 50, 100),
                'minstep': 1e-4,
                'minfunc': 1e-4,
                'maxiter': 50
            }
            optimizer_kwargs.update(self._optimizer_kwargs)

            xopt, fopt = pso(
                self.evaluateCostFunction,
                **optimizer_kwargs)

            param_dict = self.freeParametersToDictionary(xopt)
            # Save some stuff.
            cost = self.evaluateCostFunction(xopt)
            if isMainProcess():
                print('\nFitting terminated')
                print('Cost : {}\n'.format(cost))
                # print('Number of function evaluations: {}\n'.format(result.nfev))
                print('Result:')

        elif self._optimizer == 'differential_evolution':
            ndofs = len(self._free_variables_keys)
            optimizer_kwargs = {
                'bounds': [(xmin, xmax) for xmin, xmax in zip(self._min_bounds, self._max_bounds)],
                'maxiter': 500,
                'popsize': ndofs * 10,
                'tol': 1e-4,
                'disp': True
            }

            optimizer_kwargs.update(self._optimizer_kwargs)

            def callback(xk):
                if isMainProcess():
                    print('xk: {}'.format(xk))

            result = scipy.optimize.differential_evolution(
                self.evaluateCostFunction,
                **optimizer_kwargs)

            param_dict = self.freeParametersToDictionary(result.x)
            # Save some stuff.
            if isMainProcess():
                print('\nFitting terminated')
                print('Number of function evaluations: {}\n'.format(result.nfev))
                print('Result:')

        elif self._optimizer == 'ga':

            if not has_pygad:
                raise RuntimeError(
                    "pygad is not installed. Install it with `atkpython -m pip install pygad`")

            ndofs = len(self._free_variables_keys)
            gene_space = [{'low': x, 'high': y} for x, y in zip(self._min_bounds, self._max_bounds)]
            optimizer_kwargs = {
                'num_generations': 50,
                'sol_per_pop': len(self._initial_guess) * 300,
                'num_parents_mating': 2,
                'stop_criteria': "saturate_5",
            }
            optimizer_kwargs.update(self._optimizer_kwargs)

            def fitness_func(sol, sol_idx):
                return 1.0 / self.evaluateCostFunction(sol)

            def callback_gen(ga_instance):
                if isMainProcess():
                    print("Generation : ", ga_instance.generations_completed)
                    print("Fitness of the best solution :", ga_instance.best_solution()[1])
                    print("Parameters of the best solution : {solution}".format(
                        solution=ga_instance.best_solution()[0]), flush=True)

            ga = pygad.GA(
                fitness_func=fitness_func,
                num_genes=len(self._initial_guess),
                gene_space=gene_space,
                callback_generation=callback_gen,
                **optimizer_kwargs)
            ga.run()

            solution, solution_fitness, solution_idx = ga.best_solution()

            param_dict = self.freeParametersToDictionary(solution)
            # Save some stuff.
            cost = self.evaluateCostFunction(solution)
            if isMainProcess():
                print("Parameters of the best solution : {solution}".format(solution=solution))
                print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))
                print('\nFitting terminated')

                print('Cost : {}\n'.format(cost))
                print('Result:')

        else:
            raise ValueError("Unknown optimizer")

        hamiltonian_parametrization = self._dictionary_to_parametrization(param_dict)

        # Deal with pair_potentials.
        try:
            hamiltonian_parametrization = hamiltonian_parametrization[0]
        except TypeError:
            hamiltonian_parametrization = hamiltonian_parametrization


        if isMainProcess():
            pp = pprint.PrettyPrinter(indent=4)
            pp.pprint(param_dict)
            with open(self._filename_prefix + '.pickle', 'wb') as fd:
                # TODO Generalize pickle.dump((result, param_dict), fd)
                pickle.dump((param_dict), fd)
            with open(self._filename_prefix + '.py', 'w') as fd:
                fd.write('from QuantumATK import * \n')
                fd.write(hamiltonian_parametrization._script())

        self._calculator._stopSeriesOfCalculations(None)

        return param_dict
