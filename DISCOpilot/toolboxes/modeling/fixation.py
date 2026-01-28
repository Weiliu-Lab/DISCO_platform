from typing import Tuple

from pymatgen.core import Structure


def ensure_structure(struct_or_dict) -> Structure:
    if isinstance(struct_or_dict, Structure):
        return struct_or_dict
    return Structure.from_dict(struct_or_dict)


def apply_fixation(struct_or_dict, z_frac: float) -> Tuple[dict, str]:
    try:
        z_val = float(z_frac)
    except Exception as exc:
        raise ValueError("Invalid z fraction") from exc
    if z_val < 0.0 or z_val > 1.0:
        raise ValueError("Z fraction must be between 0 and 1")

    struct = ensure_structure(struct_or_dict)
    frac_z = struct.frac_coords[:, 2]
    fix_mask = [bool(z < z_val) for z in frac_z]

    struct_dict = struct.as_dict()
    sites = struct_dict.get("sites", [])
    for i, site in enumerate(sites):
        props = site.get("properties") or {}
        fixed = bool(fix_mask[i])
        props["fix"] = fixed
        props["selective_dynamics"] = [not fixed, not fixed, not fixed]
        site["properties"] = props

    info = f"Fix z< {z_val:g} ({sum(fix_mask)}/{len(fix_mask)})"
    return struct_dict, info
