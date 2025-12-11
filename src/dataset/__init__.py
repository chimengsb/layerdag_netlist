
from .circuit_dag import *
from .general import DAGDataset
from .blif_tile import get_blif_dataset


def load_dataset(name):
    if name == 'blif':
        return get_blif_dataset()
    else:
        raise ValueError(f'Unknown dataset: {name}')
