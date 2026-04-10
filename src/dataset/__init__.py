# from .layer_dag import *
from .circuit_dag import *
from .general import DAGDataset
from .tpu_tile import get_tpu_tile
from .dag_tile import get_circuit_bench

# def load_dataset(dataset_name):
#     if dataset_name == 'tpu_tile':
#         return get_tpu_tile()
#     else:
#         return NotImplementedError


def load_dataset(name):
    if name == 'tpu_tile':
        return get_tpu_tile()
    elif name == 'circuit_bench':
        return get_circuit_bench()
    else:
        raise ValueError(f'Unknown dataset: {name}')
