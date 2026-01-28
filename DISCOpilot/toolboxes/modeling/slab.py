import math
from typing import Tuple

from pymatgen.core import Structure
from pymatgen.core.surface import SlabGenerator
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer


def ensure_structure(struct_or_dict) -> Structure:
    if isinstance(struct_or_dict, Structure):
        return struct_or_dict
    return Structure.from_dict(struct_or_dict)


def generate_slab(
    struct_or_dict,
    hkl: Tuple[int, int, int],
    min_size: float,
    vacuum: float,
) -> Structure:
    bulk = ensure_structure(struct_or_dict)
    sga = SpacegroupAnalyzer(bulk)
    std = sga.get_conventional_standard_structure()
    slab_gen = SlabGenerator(
        std,
        miller_index=tuple(int(x) for x in hkl),
        min_slab_size=float(min_size),
        min_vacuum_size=float(vacuum),
        center_slab=True,
        reorient_lattice=True,
    )
    slabs = slab_gen.get_slabs()
    if not slabs:
        raise RuntimeError("Failed to generate slab")
    slab = slabs[0]
    a, b = slab.lattice.abc[:2]
    ra, rb = math.ceil(min_size / a), math.ceil(min_size / b)
    if ra > 1 or rb > 1:
        slab.make_supercell([ra, rb, 1])
    return slab
