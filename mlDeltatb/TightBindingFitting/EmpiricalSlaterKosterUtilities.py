from QuantumATK import *
import numpy


def parameterDictionaryToSKTable(params):
    """
    A function to translate a list of parameters in an orthogonal nearest-neighbor
    slater basis set.

    :param params:
        The input parameters for the model.
    :type params:
        A dictionary of float.

    The parameter dictionary should contain the following keys (all length parameters in
    Angstrom and energy parameters in eV):

    - A list of elements which should be parametrized
        params['elements'] = [Silicon, Hydrogen, Carbon ...]

    Per each element:
    - List of orbitals. Each orbital should have also an index 0, 1 ... (to distinguish s, s* etc.)
        params['Silicon', 'orbitals'] = ['s0', 'p0', 'd0', 's1']
    - Valence electrons
        params['Silicon', 'number_of_valence_electrons'] = 4
    - Reference nearest-neighbor distance for this element-element interactions.
        params['Silicon', 'nearest_neighbor_distance'] = ...
    - Reference second-neighbor distance for this element-element interactions
      (used to determine the interaction range)
        params['Silicon', 'second_neighbor_distance'] = ...
    - Entry for each interatomic hamiltonian matrix element
        params['Silicon', 'hamiltonian_matrix_element', 's0s0s'] = ...  -> s s sigma
        params['Silicon', 'hamiltonian_matrix_element', 's0s1s'] = ...  -> s s* sigma
    - Entry for each Harrison scaling parameter "eta". If not scaling is needed, set to zero.
        params['Silicon', 'eta', 's0s0s'] = ...  -> s s sigma
        params['Silicon', 'eta', 's0s1s'] = ...  -> s s* sigma
    - The value of the ionization potential per each element and shell in eV
        params['Silicon', 'ionization_potential', , <orbital>] = -1.0
    - The value of spin orbit split constants in eV (list, see `SlaterKosterOnsiteParameters` documentation)
        params['Silicon', 'onsite_spin_orbit_split'] = [0., 0.02] * eV
    - The value of the onsite Hartree shift in eV (list with one entry per shell)
        params['Silicon', 'onsite_hartree_shift'] = 5
    - The value of spin split constants in eV (list, see `SlaterKosterOnsiteParameters` documentation)
      (optional, it can be left to None)
        params['Silicon', 'onsite_spin_split'] = [0., 0.02] * eV

    Per each element pair:
    - Entry for each interatomic hamiltonian matrix element
        params['Silicon', 'Germanium', 'hamiltonian_matrix_element', 's0s0s'] = ...  -> s s sigma
        params['Silicon', 'Germanium', 'hamiltonian_matrix_element', 's0s1s'] = ...  -> s s* sigma
    - Entry for each Harrison scaling parameter "eta". If not scaling is needed, set to zero.
        params['Silicon', 'Germanium', 'hamiltonian_matrix_element_eta', 's0s0s'] = ...  -> s s sigma
        params['Silicon', 'Germanium', 'hamiltonian_matrix_element_eta', 's0s1s'] = ...  -> s s* sigma
    - Reference nearest-neighbor distance for this element-element interactions.
        params['Silicon', 'Germanium', 'nearest_neighbor_distance'] = ...
    - Reference second-neighbor distance for this element-element interactions
      (used to determine the interaction range)
        params['Silicon', 'Germanium', 'second_neighbor_distance'] = ...

    IMPORTANT: all entries for element pair must have the same element ordering, i.e.
    they must be either all ['Silicon', 'Germanium' ...] keys, or ['Germanium', 'Silicon' ...].

    Note: instead of s0, p0 etc. also s, p, d can be specified, as in the input for
    SlaterKosterTable.
    """
    #XXX Add a check upfront on the keys.

    string_to_angular_momentum = {'s': 0, 'p': 1, 'd': 2, 'f': 3}
    elements = params['elements']
    onsite_terms = []
    onsite_names = []
    offsite_terms = []
    offsite_names = []

    # Process monoatomic terms.
    for element in elements:
        element_string = element.name()
        element_symbol = element.symbol()
        ionization_potential = [
            params[element_string, 'ionization_potential', key]
            for key in params[element_string, 'orbitals']]

        try:
            if params[element_string, 'onsite_spin_split'] is not None:
                onsite_spin_split = params[element_string, 'onsite_spin_split'] * Units.eV
            else:
                onsite_spin_split = None
        except KeyError:
            onsite_spin_split = None

        onsite_term = SlaterKosterOnsiteParameters(
            element=element,
            angular_momenta=[
                string_to_angular_momentum[x[0]] for x in params[element_string, 'orbitals']],
            number_of_valence_electrons = params[element_string, 'number_of_valence_electrons'],
            filling_method=SphericalSymmetric,
            ionization_potential=ionization_potential * Units.eV,
            onsite_hartree_shift=params[element_string, 'onsite_hartree_shift'] * Units.eV,
            onsite_spin_split=onsite_spin_split,
            onsite_spin_orbit_split=params[element_string, 'onsite_spin_orbit_split'] * Units.eV,
            )

        onsite_terms.append(onsite_term)
        onsite_names.append(element_string.lower())

        # Same-element interactions.
        prefix_string = f'{element_symbol}_{element_symbol}_'.lower()
        offsite_entries = [
            (key, val) for key, val in params.items() if element_string in key
            and key[1] == 'hamiltonian_matrix_element']
        offsite_values = {'{}'.format(key[2]): val for key, val in offsite_entries}
        eta_entries = [
            (key, val) for key, val in params.items() if element_string in key
            and key[1] == 'hamiltonian_matrix_element_eta']
        eta_values = {'{}'.format(key[2]): val for key, val in eta_entries}
        overlap_entries = [
            (key, val) for key, val in params.items() if element_string in key
            and key[1] == 'overlap_matrix_element']
        overlap_values = {'{}'.format(key[2]): val for key, val in overlap_entries}
        overlap_eta_entries = [
            (key, val) for key, val in params.items() if element_string in key
            and key[1] == 'overlap_matrix_element_eta']
        overlap_eta_values = {'{}'.format(key[2]): val for key, val in overlap_eta_entries}

        # First neighbor distance
        if offsite_entries and element_string in offsite_entries[0][0]:
            d1 = params[element_string, 'nearest_neighbor_distance'] * Units.Angstrom
            d2 = params[element_string, 'second_neighbor_distance'] * Units.Angstrom
            epsilon = numpy.linspace(-0.20, 0.20, 41)
            distances = [d1 * (1.0 + x) for x in epsilon] + [0.5 * (d1 + d2)]

            for key, offsite in offsite_values.items():
                offsite_names.append(f'{prefix_string}{key}')
                try:
                    eta = eta_values[key]
                except KeyError:
                    eta = 0.
                try:
                    overlap = overlap_values[key]
                except KeyError:
                    overlap = None
                try:
                    overlap_eta = overlap_eta_values[key]
                except KeyError:
                    overlap_eta = 0.

                entry = [offsite*eV / (1.0 + x)**eta for x in epsilon] + [0. * eV]
                if overlap is None:
                    offsite_terms.append(list(zip(distances, entry)))
                else:
                    offsite_terms.append(
                        list(zip(
                            distances,
                            entry,
                            [overlap / (1.0 + x)**overlap_eta for x in epsilon] + [0.])))

    # Process heteroatomic interactions. Loop directly on pair-keys of the
    # input dictionary.
    all_elements = [x.name() for x in params['elements']]
    element_pairs = [k for k in params.keys()
        if len(k) > 1 and k[0] in all_elements and k[1] in all_elements]
    element_pairs = set(element_pairs)

    for element_pair in element_pairs:
        first_element = element_pair[0]
        second_element = element_pair[1]
        first_element_symbol = [x for x in elements if x.name() == first_element][0].symbol()
        second_element_symbol = [x for x in elements if x.name() == second_element][0].symbol()

        prefix_string = f'{first_element_symbol}_{second_element_symbol}_'.lower()
        offsite_entries = [
            (key, val) for key, val in params.items()
            if key[0] == first_element and key[1] == second_element
            and key[2] == 'hamiltonian_matrix_element']
        offsite_values = {'{}'.format(key[3]): val for key, val in offsite_entries}
        overlap_entries = [
            (key, val) for key, val in params.items()
            if key[0] == first_element and key[1] == second_element
            and key[2] == 'overlap_matrix_element']
        overlap_values = {'{}'.format(key[3]): val for key, val in overlap_entries}

        eta_entries = [
            (key, val) for key, val in params.items()
            if key[0] == first_element and key[1] == second_element
            and key[2] == 'hamiltonian_matrix_element_eta']
        eta_values = {'{}'.format(key[3]): val for key, val in eta_entries}
        overlap_eta_entries = [
            (key, val) for key, val in params.items()
            if key[0] == first_element and key[1] == second_element
            and key[2] == 'overlap_matrix_element_eta']
        overlap_eta_values = {'{}'.format(key[3]): val for key, val in overlap_eta_entries}

        # First neighbor distance
        d1 = params[first_element, second_element, 'nearest_neighbor_distance'] * Units.Angstrom
        d2 = params[first_element, second_element, 'second_neighbor_distance'] * Units.Angstrom
        epsilon = numpy.linspace(-0.20, 0.20, 41)
        distances = [ d1 * (1.0 + x) for x in epsilon ] + [ 0.5* (d1 + d2) ]

        for key, offsite in offsite_values.items():
            offsite_names.append(f'{prefix_string}{key}')
            try:
                eta = eta_values[key]
            except KeyError:
                eta = 0.
            try:
                overlap = overlap_values[key]
            except KeyError:
                overlap = None
            try:
                overlap_eta = overlap_eta_values[key]
            except KeyError:
                overlap_eta = 0.

            entry = [offsite*eV / (1.0 + x)**eta for x in epsilon] + [0. * eV]
            if overlap is None:
                offsite_terms.append(list(zip(distances, entry)))
            else:
                offsite_terms.append(
                    list(zip(
                        distances,
                        entry,
                        [overlap / (1.0 + x)**overlap_eta for x in epsilon] + [0.])))

    table_args = {a: b for a, b in zip(onsite_names, onsite_terms)}
    table_args.update({a: b for a, b in zip(offsite_names, offsite_terms)})

    basis_set = SlaterKosterTable(**table_args)

    return basis_set


def parameterDictionaryToHamiltonianParametrization(params):

    basis_set = parameterDictionaryToSKTable(params)

    return SlaterKosterHamiltonianParametrization(basis_set=basis_set)
