from typing import Tuple

import numpy as np
from pymatgen.analysis.adsorption import AdsorbateSiteFinder
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor
from ase.build import molecule


def ensure_structure(struct_or_dict) -> Structure:
    if isinstance(struct_or_dict, Structure):
        return struct_or_dict
    return Structure.from_dict(struct_or_dict)


def _normalize_site_label(site_label: str) -> str:
    label = (site_label or "").strip().lower()
    if label in {"top", "ontop"}:
        return "ontop"
    if label in {"bridge"}:
        return "bridge"
    if label in {"fcc", "hcp", "hollow"}:
        return "hollow"
    return "hollow"


def pick_adsorption_site(slab: Structure, site_label: str) -> Tuple[str, np.ndarray]:
    target = _normalize_site_label(site_label)
    asf = AdsorbateSiteFinder(slab)
    sites_dict = asf.find_adsorption_sites(distance=2.0, symm_reduce=0.1)

    # Prefer the requested site type if available.
    if target in sites_dict and sites_dict[target]:
        return target, np.array(sites_dict[target][0])

    # Fallback: choose first available site.
    for key, coords in sites_dict.items():
        if coords:
            return key, np.array(coords[0])

    raise RuntimeError("No adsorption sites found")


def add_adsorbate(slab_struct, adsorbate: str, site_label: str) -> Structure:
    slab = ensure_structure(slab_struct)
    site_type, coord = pick_adsorption_site(slab, site_label)

    ase_mol = molecule(adsorbate)
    pmg_mol = AseAtomsAdaptor.get_molecule(ase_mol)

    new_struct = AdsorbateSiteFinder(slab).add_adsorbate(pmg_mol, coord, reorient=True)
    return new_struct
