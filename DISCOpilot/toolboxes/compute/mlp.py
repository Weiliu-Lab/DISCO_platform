from typing import Dict, Optional

from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor
from ase.constraints import FixAtoms
from ase.optimize import BFGS

HAS_DEEPMD = False
try:  # optional dependency
    from deepmd.calculator import DP

    HAS_DEEPMD = True
except Exception:
    DP = None


def run_mlp_relaxation(
    structure_dict: Dict,
    model_path: str,
    fmax: float = 0.05,
    steps: int = 50,
) -> Dict[str, object]:
    if not HAS_DEEPMD:
        return {"status": "error", "error": "DeepMD not available"}

    try:
        struct = Structure.from_dict(structure_dict)
        atoms = AseAtomsAdaptor.get_atoms(struct)
        fix_prop = struct.site_properties.get("fix") if struct.site_properties else None
        if fix_prop:
            fix_indices = [i for i, val in enumerate(fix_prop) if bool(val)]
            if fix_indices:
                atoms.set_constraint(FixAtoms(indices=fix_indices))

        atoms.calc = DP(model=model_path)
        traj_e = []
        dyn = BFGS(atoms)
        dyn.attach(lambda: traj_e.append(atoms.get_potential_energy()), interval=1)
        dyn.run(fmax=fmax, steps=steps)

        return {
            "status": "ok",
            "energy": traj_e[-1] if traj_e else None,
            "trajectory": traj_e,
            "structure": AseAtomsAdaptor.get_structure(atoms).as_dict(),
        }
    except Exception as exc:
        return {"status": "error", "error": str(exc)}
