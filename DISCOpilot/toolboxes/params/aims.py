from typing import Dict


DEFAULT_AIMS_PARAMS: Dict[str, object] = {
    "xc": "pbe",
    "fmax": 0.02,
    "charge": 0,
    "acc_rho": 1e-4,
    "acc_eev": 1e-3,
    "acc_etot": 1e-6,
    "sc_iter": 200,
    "kgrid": [4, 4, 1],
}


def merge_aims_params(overrides: Dict[str, object] | None) -> Dict[str, object]:
    params = DEFAULT_AIMS_PARAMS.copy()
    if overrides:
        params.update(overrides)
    return params
