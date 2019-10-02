from .layers import *
from .models import *

custom_layers = {
    'MessageLayer': MessageLayer,
    'EdgeNetwork': EdgeNetwork,
    'ReduceAtomToMol': ReduceAtomToMol,
    'ReduceAtomOrBondToMol': ReduceAtomOrBondToMol,
    'ReduceBondToAtom': ReduceBondToAtom,
    'GatherAtomToBond': GatherAtomToBond,
    'GatherMolToAtomOrBond': GatherMolToAtomOrBond,
    'Embedding2D': Embedding2D,
    'masked_mean_squared_error': masked_mean_squared_error,
    'masked_mean_absolute_error': masked_mean_absolute_error,
    'masked_log_cosh': masked_log_cosh
}
