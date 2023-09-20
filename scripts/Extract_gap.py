import sys
import os
import fnmatch
import numpy as np
from QuantumATK import *


def getGapStatistics(directory):
    gaps = []

    for file in os.listdir(directory):
        if fnmatch.fnmatch(file, "*_band_*.hdf5"):
            bandstructure = nlread(directory + file, Bandstructure)
            gaps.append(bandstructure[0].indirectBandGap().inUnitsOf(eV))

    gaps = np.array(gaps)
    mean_gap = np.mean(gaps)
    std_gap = np.std(gaps)

    return gaps, mean_gap, std_gap


def main():
    input_path = sys.argv[1]
    concentration = sys.argv[2]
    directory = input_path + str(concentration) + "/"

    gaps, mean_gap, std_gap = getGapStatistics(directory)

    print("\nConcentration: {:.3f}".format(float(concentration)))
    print("\nMean band gap: {:.3f}".format(mean_gap))
    print("\nSigma: {:.3f}".format(std_gap))

    with open(directory + "Bandgap_report.dat", "a") as f:
        f.write("Mean band gap: {:.3f}".format(mean_gap))
        f.write("\nsigma: {:.3f}\n".format(std_gap))
        f.write("\nEnsemble realizations:\n")
        np.savetxt(f, gaps)


if __name__ == "__main__":
    main()
