lattice = FaceCenteredCubic(6.0959*Angstrom)

# Define elements
elements = [Gallium, Antimony]

# Define coordinates
fractional_coordinates = [[ 0.  ,  0.  ,  0.  ],
                          [ 0.25,  0.25,  0.25]]

# Set up configuration
gasb = BulkConfiguration(
    bravais_lattice=lattice,
    elements=elements,
    fractional_coordinates=fractional_coordinates
    )


# %% Set LCAOCalculator

# %% LCAOCalculator

#----------------------------------------
# Exchange-Correlation
#----------------------------------------

exchange_correlation = HSECustomExchangeCorrelation(
    screening_length=0.11*1/Bohr,
    exx_fraction=0.39,
    number_of_spins=4,
    spin_orbit=True)

k_point_sampling = KpointDensity(
    density_a=4.0*Angstrom,
    density_b=4.0*Angstrom,
    density_c=4.0*Angstrom
)

numerical_accuracy_parameters = NumericalAccuracyParameters(
    k_point_sampling=k_point_sampling
)

checkpoint_handler = NoCheckpointHandler

# ----------------------------------------
# Basis Set
# ----------------------------------------
basis_set = [
    BasisGGAPseudoDojoSO.Gallium_Medium,
    BasisGGAPseudoDojoSO.Antimony_Medium,
]

calculator = LCAOCalculator(
    exchange_correlation=exchange_correlation,
    numerical_accuracy_parameters=numerical_accuracy_parameters,
    checkpoint_handler=checkpoint_handler
)


# %% Set Calculator

gasb.setCalculator(calculator)
gasb.update()

# %% Bandstructure
bandstructure = Bandstructure(
    configuration=gasb,
    route=['L', 'G', 'X', 'U', 'G']
)

nlsave('GaSb.hdf5', bandstructure)
