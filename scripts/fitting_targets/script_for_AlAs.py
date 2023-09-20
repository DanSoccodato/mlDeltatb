# Set up lattice
lattice = FaceCenteredCubic(5.6613 * Angstrom)
experimental_BG = 2.153

# Define elements
elements = [Aluminum, Arsenic]

# Define coordinates
fractional_coordinates = [[0., 0., 0.],
                          [0.250008844079, 0.250008844079, 0.250008844079]]

# Set up configuration
alas = BulkConfiguration(
    bravais_lattice=lattice,
    elements=elements,
    fractional_coordinates=fractional_coordinates
)


# ----------------------------------------
# Exchange-Correlation
# ----------------------------------------
exchange_correlation = HSECustomExchangeCorrelation(
    screening_length=0.11 * 1 / Bohr,
    exx_fraction=0.25,
    number_of_spins=4,
    spin_orbit=True)

k_point_sampling = KpointDensity(
    density_a=4.0 * Angstrom,
    density_b=4.0 * Angstrom,
    density_c=4.0 * Angstrom
)

numerical_accuracy_parameters = NumericalAccuracyParameters(
    k_point_sampling=k_point_sampling
)

# ----------------------------------------
# Basis Set
# ----------------------------------------

checkpoint_handler = NoCheckpointHandler

calculator = LCAOCalculator(
    exchange_correlation=exchange_correlation,
    numerical_accuracy_parameters=numerical_accuracy_parameters,
    checkpoint_handler=checkpoint_handler
)

# %% Set Calculator

alas.setCalculator(calculator)

alas.update()

# %% Bandstructure

bandstructure = Bandstructure(
    configuration=alas,
    route=['L', 'G', 'X', 'U', 'G']
)

nlsave('AlAs.hdf5', bandstructure)
