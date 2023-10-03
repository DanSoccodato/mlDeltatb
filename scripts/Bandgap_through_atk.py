import sys

from QuantumATK import *
import numpy
from mlDeltatb.BackpropOptimization.ATKUtils import attachCalculator
from mlDeltatb.Common.DatasetProcessing import readBandstructureFiles


def main(dataset_path, input_path, output_path):

    concentrations = ['10', '30', '40', '60', '80']

    for index, concentration in enumerate(concentrations):

        directory = dataset_path + "/" + concentration + "/"
        print(f"  Processing concentration: {concentration}")
        bandstructures = readBandstructureFiles(directory)
        configurations = [bs._configuration() for bs in bandstructures]

        gaps = []

        for conf in configurations:
            conf = attachCalculator(conf, input_path + "/start_basis_set.pickle")
            conf.update()

            bands = Bandstructure(
                configuration=conf,
                route=['L', 'G', 'X'],
                bands_above_fermi_level=10
            )

            gaps.append(bands.indirectBandGap())

        f = f"{output_path}/{concentration}_tb_gaps.dat"
        numpy.savetxt(f, gaps, fmt=['%.4f'])


if __name__ == "__main__":
    try:
        args = sys.argv[1], sys.argv[2], sys.argv[3]

    except IndexError:
        raise ValueError("\nPossible input arguments:\n"
                         "1. dataset path;\n"
                         "2. input path;\n"
                         "3. output path.\n"
                         "Please provide all arguments.")

    main(*args)