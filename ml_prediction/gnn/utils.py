import torch
import random
import numpy as np
from torch_geometric.data import Data
from ase.neighborlist import natural_cutoffs, neighbor_list


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass


def convert_to_graph(atoms, y, idx, **extra_props):
    cov_cutoffs = np.array(natural_cutoffs(atoms, mult=1, H=3, O=3))
    i, j, d = neighbor_list('ijd', atoms, cutoff=cov_cutoffs, self_interaction=False)
    edge_index = torch.tensor([i, j], dtype=torch.long)
    edge_attr = torch.tensor(d, dtype=torch.float32)
    pos = torch.tensor(atoms.get_positions(), dtype=torch.float32)
    atom_numbers = torch.tensor(atoms.numbers, dtype=torch.long)
    graph_data = Data(
        edge_index=edge_index,
        edge_attr=edge_attr,
        pos=pos,
        y=torch.tensor([y], dtype=torch.float32),
        z=atom_numbers,
        idx=idx,
        **extra_props,
    )
    return graph_data
