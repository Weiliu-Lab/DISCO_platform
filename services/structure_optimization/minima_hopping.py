"""
Approximate minima hopping search for surface adsorption structures.

This implementation is intentionally lightweight: if a real ASE calculator is
provided it performs short local relaxations; otherwise it falls back to a
geometry-based surrogate score so the Planner-Expert workflow can be developed
and demonstrated before the full production calculator is wired in.
"""

from __future__ import annotations

import math
import random
from typing import Dict, List, Optional

import numpy as np
from ase import Atoms
from ase.constraints import FixAtoms
from ase.optimize import BFGS
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor


class MinimaHoppingSearch:
    """Approximate minima hopping style exploration for adsorption systems."""

    def __init__(
        self,
        slab: Structure,
        adsorbate: Atoms,
        calculator=None,
        hop_count: int = 12,
        local_steps: int = 30,
        height_window=(1.8, 3.4),
    ):
        self.slab_pmg = slab
        self.slab_ase = AseAtomsAdaptor.get_atoms(slab)
        self.adsorbate = adsorbate.copy()
        self.calculator = calculator
        self.hop_count = int(hop_count)
        self.local_steps = int(local_steps)
        self.height_window = height_window

        positions = self.slab_ase.get_positions()
        self.x_range = (positions[:, 0].min(), positions[:, 0].max())
        self.y_range = (positions[:, 1].min(), positions[:, 1].max())
        self.z_max = positions[:, 2].max()
        self._setup_constraints()

    def _setup_constraints(self):
        z_coords = self.slab_ase.get_positions()[:, 2]
        z_sorted = np.sort(np.unique(np.round(z_coords, decimals=6)))
        if len(z_sorted) > 2:
            z_threshold = z_sorted[1]
            mask = z_coords <= z_threshold
            self.slab_ase.set_constraint(FixAtoms(mask=mask))

    def _random_individual(self) -> Dict[str, float]:
        return {
            "x": random.uniform(*self.x_range),
            "y": random.uniform(*self.y_range),
            "height": random.uniform(*self.height_window),
            "theta": random.uniform(0, 2 * math.pi),
            "phi": random.uniform(0, math.pi),
            "psi": random.uniform(0, 2 * math.pi),
        }

    def _perturb(self, individual: Dict[str, float]) -> Dict[str, float]:
        trial = individual.copy()
        trial["x"] = float(np.clip(trial["x"] + random.uniform(-0.8, 0.8), *self.x_range))
        trial["y"] = float(np.clip(trial["y"] + random.uniform(-0.8, 0.8), *self.y_range))
        trial["height"] = float(
            np.clip(trial["height"] + random.uniform(-0.4, 0.4), *self.height_window)
        )
        trial["theta"] = (trial["theta"] + random.uniform(-0.6, 0.6)) % (2 * math.pi)
        trial["phi"] = float(np.clip(trial["phi"] + random.uniform(-0.5, 0.5), 0.0, math.pi))
        trial["psi"] = (trial["psi"] + random.uniform(-0.6, 0.6)) % (2 * math.pi)
        return trial

    def _individual_to_atoms(self, individual: Dict[str, float]) -> Atoms:
        slab_copy = self.slab_ase.copy()
        ads_copy = self.adsorbate.copy()

        ads_copy.euler_rotate(
            phi=individual["phi"],
            theta=individual["theta"],
            psi=individual["psi"],
            center="COM",
        )

        com = ads_copy.get_center_of_mass()
        target = np.array(
            [individual["x"], individual["y"], self.z_max + individual["height"]]
        )
        ads_copy.translate(target - com)

        combined = slab_copy + ads_copy
        if self.calculator is not None:
            combined.calc = self.calculator
        return combined

    def _surrogate_energy(self, atoms: Atoms) -> float:
        slab_count = len(self.slab_ase)
        slab_pos = atoms.get_positions()[:slab_count]
        ads_pos = atoms.get_positions()[slab_count:]

        pairwise = np.linalg.norm(slab_pos[:, None, :] - ads_pos[None, :, :], axis=2)
        min_dist = float(pairwise.min())
        mean_dist = float(np.sort(pairwise, axis=0)[:4].mean())
        ads_height = float(ads_pos[:, 2].mean() - slab_pos[:, 2].max())

        if min_dist < 1.45:
            return 1e4 + (1.45 - min_dist) * 1e3

        repulsion = 0.02 * np.sum((2.0 / np.clip(pairwise, 1.6, None)) ** 12)
        attraction = -0.15 * np.sum((2.7 / np.clip(pairwise, 2.0, None)) ** 6)
        height_penalty = 0.6 * (ads_height - 2.2) ** 2
        contact_balance = 0.4 * abs(mean_dist - 2.6)
        return float(repulsion + attraction + height_penalty + contact_balance)

    def _local_refine(self, atoms: Atoms) -> float:
        if self.calculator is None:
            return self._surrogate_energy(atoms)

        optimizer = BFGS(atoms, logfile=None)
        optimizer.run(fmax=0.08, steps=self.local_steps)
        return float(atoms.get_potential_energy())

    def search(
        self,
        initial_candidates: Optional[List[Dict]] = None,
        top_k: int = 5,
    ) -> List[Dict]:
        candidates = initial_candidates or []
        working_population: List[Dict] = []

        for candidate in candidates:
            working_population.append(candidate.get("individual") or self._random_individual())

        if not working_population:
            working_population = [self._random_individual() for _ in range(max(4, self.hop_count // 2))]

        explored: List[Dict] = []

        for seed_index, seed in enumerate(working_population):
            current = seed
            for hop_index in range(max(1, self.hop_count)):
                trial = current if hop_index == 0 else self._perturb(current)
                atoms = self._individual_to_atoms(trial)
                energy = self._local_refine(atoms)
                explored.append(
                    {
                        "structure": AseAtomsAdaptor.get_structure(atoms),
                        "energy": energy,
                        "individual": trial,
                        "label": f"minhop_seed{seed_index}_hop{hop_index}",
                    }
                )
                current = trial

        explored.sort(key=lambda item: item["energy"])
        return explored[: max(1, int(top_k))]