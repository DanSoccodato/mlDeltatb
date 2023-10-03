from QuantumATK import *
import pickle

from mlDeltatb.TightBindingFitting.EmpiricalSlaterKosterUtilities import parameterDictionaryToSKTable


def basisSetFromFile(basis_set_path):
    with open(basis_set_path, "rb") as f:
        params_dictionary = pickle.load(f)
    basis_set = parameterDictionaryToSKTable(params_dictionary)
    return basis_set
