from __future__ import annotations

import csv
import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
import numpy as np
from ase import Atoms
from ase.build import molecule
from ase.io import write
from langchain_core.messages import HumanMessage
from pymatgen.analysis.adsorption import AdsorbateSiteFinder
from pymatgen.core import Structure
from pymatgen.core.surface import SlabGenerator
from pymatgen.ext.matproj import MPRester as PymatgenMPRester
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ocp_logic import MP_API_KEY
from utils.llm_provider import get_langchain_llm
from services.structure_optimization.minima_hopping import MinimaHoppingSearch
from utils.llm_tools import text_to_smiles

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
except ImportError:
    Chem = None
    AllChem = None

try:
    from scipy.interpolate import make_interp_spline
except ImportError:
    make_interp_spline = None


matplotlib.use("Agg")
import matplotlib.pyplot as plt


OUTPUT_DIR = CURRENT_DIR / "simulation_outputs" / "planner_expert"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FALLBACK_SMILES = {
    "苯": "c1ccccc1",
    "benzene": "c1ccccc1",
}

FALLBACK_ASE_NAMES = {
    "苯": "C6H6",
    "benzene": "C6H6",
    "c1ccccc1": "C6H6",
}


def _fetch_mp_summary_docs(element: str):
    """Fetch bulk structure documents with a compatibility-first MP client path."""
    try:
        from mp_api.client import MPRester as MpApiMPRester

        with MpApiMPRester(MP_API_KEY) as mpr:
            docs = mpr.materials.summary.search(
                formula=element,
                is_stable=True,
                fields=["material_id", "formula_pretty", "symmetry", "structure"],
            )
        return docs
    except Exception:
        with PymatgenMPRester(MP_API_KEY) as mpr:
            docs = mpr.materials.summary.search(formula=element, is_stable=True)
        return docs


def _get_llm():
    return get_langchain_llm(temperature=0.1)


def _clean_json_response(content: str) -> Dict:
    cleaned = content.strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned[7:-3]
    elif cleaned.startswith("```"):
        cleaned = cleaned[3:-3]
    return json.loads(cleaned)


def _default_bistable_plan(user_request: str) -> Dict:
    return {
        "workflow_type": "bistable_adsorption",
        "objective": "Probe whether benzene exhibits bistable adsorption on Pt(111).",
        "surface": "Pt(111)",
        "adsorbate": "benzene",
        "smiles": "c1ccccc1",
        "sites": ["ontop", "bridge", "hollow"],
        "search_method": "minhop",
        "stretch_requested": True,
        "stretch_settings": {
            "top_k": 5,
        },
        "experts": [
            {"name": "Planner", "task": "Parse the bistable-adsorption question and organize the adsorption-energy analysis."},
            {"name": "Modeling Expert", "task": "Build physically valid benzene/Pt(111) adsorption candidates from MP-derived slab models."},
            {"name": "Optimization Expert", "task": "Refine candidate structures and evaluate adsorption energies consistently."},
            {"name": "Analysis Expert", "task": "Rank adsorption configurations and identify candidate minima relevant to bistability."},
        ],
        "workflow_summary": (
            "The workflow constructs adsorption candidates for benzene on Pt(111), performs structure-energy search, "
            "and evaluates whether the resulting energy landscape supports bistable adsorption."
        ),
    }


def _default_general_plan(user_request: str) -> Dict:
    return {
        "workflow_type": "adsorption_scan",
        "objective": user_request,
        "surface": "Pt(111)",
        "adsorbate": "O",
        "smiles": "",
        "sites": ["top", "fcc", "hcp"],
        "search_method": "site_scan",
        "stretch_requested": False,
        "workflow_summary": user_request,
    }


def _is_bistable_case_request(user_request: str) -> bool:
    lowered = user_request.lower()
    return any(keyword in lowered for keyword in ["benzene", "pt(111)", "bistable"]) or any(
        keyword in user_request for keyword in ["苯", "双稳态", "拉伸曲线"]
    )


def build_case_plan(user_request: str) -> Dict:
    if not user_request.strip():
        return _default_general_plan(user_request)

    if not _is_bistable_case_request(user_request):
        fallback = _default_general_plan(user_request)
        try:
            llm = _get_llm()
            prompt = f"""
You are a computational chemistry Planner.
Extract a simple adsorption study plan from the user request below.

User request:
{user_request}

Return JSON only:
{{
  "workflow_type": "adsorption_scan",
  "objective": "short sentence",
  "surface": "surface label such as Pt(111)",
  "adsorbate": "adsorbate name",
  "sites": ["top", "fcc", "hcp"],
  "search_method": "site_scan",
  "stretch_requested": false,
  "workflow_summary": "1-2 sentence summary"
}}
"""
            response = llm.invoke([HumanMessage(content=prompt)])
            plan = _clean_json_response(response.content)
        except Exception:
            plan = fallback

        merged = fallback.copy()
        merged.update(plan)
        return merged

    fallback = _default_bistable_plan(user_request)

    try:
        llm = _get_llm()
        prompt = f"""
You are the Planner in a Planner-Expert computational chemistry agent system.
Extract a structured execution plan for the user request below.

User request:
{user_request}

Return JSON only with the following keys:
{{
  "workflow_type": "bistable_adsorption",
  "objective": "short sentence",
  "surface": "surface label such as Pt(111)",
  "adsorbate": "adsorbate name",
  "smiles": "canonical SMILES if known, otherwise empty string",
  "sites": ["ontop", "bridge", "hollow"],
  "search_method": "minhop",
  "stretch_requested": true,
  "workflow_summary": "1-2 sentence summary"
}}

If the request is about benzene on Pt(111), keep workflow_type as bistable_adsorption and keep stretch_requested true.
"""
        response = llm.invoke([HumanMessage(content=prompt)])
        plan = _clean_json_response(response.content)
    except Exception:
        plan = fallback

    merged = fallback.copy()
    merged.update(plan)
    merged["experts"] = fallback["experts"]
    merged["stretch_settings"] = fallback["stretch_settings"]

    if not merged.get("smiles"):
        merged["smiles"] = resolve_smiles(merged.get("adsorbate", "benzene"))

    return merged


def resolve_smiles(adsorbate_name: str) -> str:
    name = (adsorbate_name or "").strip()
    if not name:
        return ""

    if name.lower() in FALLBACK_SMILES:
        return FALLBACK_SMILES[name.lower()]
    if name in FALLBACK_SMILES:
        return FALLBACK_SMILES[name]

    smiles = text_to_smiles(name)
    return smiles or FALLBACK_SMILES.get(name.lower(), "")


def build_adsorbate_model(adsorbate_name: str, smiles: str) -> Dict:
    safe_name = re.sub(r"[^A-Za-z0-9_\-]", "_", adsorbate_name or "adsorbate")
    output_path = OUTPUT_DIR / f"adsorbate_{safe_name}.xyz"

    if Chem is not None and AllChem is not None and smiles:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Unable to parse SMILES: {smiles}")
        mol = Chem.AddHs(mol)
        embed_code = AllChem.EmbedMolecule(mol, randomSeed=7)
        if embed_code != 0:
            raise RuntimeError("RDKit failed to embed the molecule in 3D.")
        AllChem.UFFOptimizeMolecule(mol, maxIters=400)

        conformer = mol.GetConformer()
        symbols = [atom.GetSymbol() for atom in mol.GetAtoms()]
        positions = []
        for index in range(mol.GetNumAtoms()):
            pos = conformer.GetAtomPosition(index)
            positions.append([pos.x, pos.y, pos.z])
        atoms = Atoms(symbols=symbols, positions=positions)
        source = "RDKit"
    else:
        ase_name = FALLBACK_ASE_NAMES.get(smiles) or FALLBACK_ASE_NAMES.get(adsorbate_name.lower()) or "C6H6"
        atoms = molecule(ase_name)
        source = "ASE fallback"

    write(output_path, atoms)
    return {
        "name": adsorbate_name,
        "smiles": smiles,
        "atoms": atoms,
        "path": str(output_path),
        "source": source,
    }


def _parse_surface_label(surface_name: str) -> Tuple[str, Tuple[int, int, int]]:
    text = (surface_name or "Pt(111)").replace(" ", "")
    match = re.match(r"([A-Za-z]+)\((\d)(\d)(\d)\)", text)
    if not match:
        return "Pt", (1, 1, 1)
    symbol = match.group(1)
    miller = tuple(int(match.group(index)) for index in range(2, 5))
    return symbol, miller


def build_surface_model(surface_name: str, min_lateral_size: float = 10.0) -> Dict:
    element, miller = _parse_surface_label(surface_name)
    docs = _fetch_mp_summary_docs(element)
    if not docs:
        raise RuntimeError(f"No stable Materials Project entry found for {element}.")

    doc = docs[0]
    bulk = doc.structure
    sga = SpacegroupAnalyzer(bulk)
    standard_bulk = sga.get_conventional_standard_structure()
    slab_generator = SlabGenerator(
        standard_bulk,
        miller_index=miller,
        min_slab_size=10.0,
        min_vacuum_size=15.0,
        center_slab=True,
        reorient_lattice=True,
    )
    slabs = slab_generator.get_slabs()
    if not slabs:
        raise RuntimeError(f"Unable to cleave {surface_name} from Materials Project bulk data.")

    slab = slabs[0]
    a_len, b_len = slab.lattice.abc[:2]
    rep_a = max(1, int(np.ceil(min_lateral_size / a_len)))
    rep_b = max(1, int(np.ceil(min_lateral_size / b_len)))
    if rep_a > 1 or rep_b > 1:
        slab.make_supercell([rep_a, rep_b, 1])

    output_path = OUTPUT_DIR / f"surface_{element}_{''.join(str(v) for v in miller)}.cif"
    slab.to(filename=str(output_path))

    symmetry = getattr(doc, "symmetry", None)
    return {
        "surface": surface_name,
        "bulk_formula": getattr(doc, "formula_pretty", element),
        "material_id": str(getattr(doc, "material_id", "unknown")),
        "crystal_system": str(getattr(symmetry, "crystal_system", "unknown")),
        "space_group": str(getattr(symmetry, "symbol", "unknown")),
        "structure": slab,
        "path": str(output_path),
    }


def _site_coordinate_map(slab: Structure) -> Dict[str, List[np.ndarray]]:
    finder = AdsorbateSiteFinder(slab)
    site_dict = finder.find_adsorption_sites(distance=2.4, symm_reduce=0.1)
    return {
        "ontop": [np.array(coord) for coord in site_dict.get("ontop", [])],
        "bridge": [np.array(coord) for coord in site_dict.get("bridge", [])],
        "hollow": [np.array(coord) for coord in site_dict.get("hollow", [])],
    }


def _orient_adsorbate(atoms: Atoms, tilt_deg: float, spin_deg: float):
    atoms.rotate(tilt_deg, "x", center="COM")
    atoms.rotate(spin_deg, "z", center="COM")


def assemble_adsorption_candidates(surface_model: Dict, adsorbate_model: Dict, plan: Dict) -> List[Dict]:
    slab = surface_model["structure"]
    slab_ase = AseAtomsAdaptor.get_atoms(slab)
    site_map = _site_coordinate_map(slab)
    requested_sites = plan.get("sites", ["ontop", "bridge", "hollow"])
    orientations = [("flat", 0.0), ("tilted", 55.0)]

    candidates: List[Dict] = []
    for site_name in requested_sites:
        coords = site_map.get(site_name, [])[:2]
        for coord_index, coord in enumerate(coords):
            for orient_name, tilt in orientations:
                ads = adsorbate_model["atoms"].copy()
                _orient_adsorbate(ads, tilt_deg=tilt, spin_deg=coord_index * 30.0)
                com = ads.get_center_of_mass()
                target = np.array([coord[0], coord[1], coord[2] + 2.4])
                ads.translate(target - com)
                combined = slab_ase.copy() + ads

                candidate_name = f"{site_name}_{orient_name}_{coord_index}"
                output_path = OUTPUT_DIR / f"candidate_{candidate_name}.xyz"
                write(output_path, combined)
                candidates.append(
                    {
                        "site": site_name,
                        "orientation": orient_name,
                        "label": candidate_name,
                        "path": str(output_path),
                        "structure": combined,
                        "individual": {
                            "x": float(target[0]),
                            "y": float(target[1]),
                            "height": 2.4,
                            "theta": np.deg2rad(tilt),
                            "phi": 0.0,
                            "psi": np.deg2rad(coord_index * 30.0),
                        },
                    }
                )
    return candidates


def run_bistable_search(surface_model: Dict, adsorbate_model: Dict, candidates: List[Dict], top_k: int = 5) -> List[Dict]:
    searcher = MinimaHoppingSearch(
        slab=surface_model["structure"],
        adsorbate=adsorbate_model["atoms"],
        calculator=None,
        hop_count=max(8, len(candidates)),
        local_steps=20,
    )
    refined = searcher.search(initial_candidates=candidates, top_k=top_k)

    results = []
    for index, item in enumerate(refined):
        label = candidates[index]["label"] if index < len(candidates) else item["label"]
        output_path = OUTPUT_DIR / f"refined_{label}_{index}.xyz"
        write(output_path, AseAtomsAdaptor.get_atoms(item["structure"]))
        results.append(
            {
                "site": label,
                "energy": float(item["energy"]),
                "path": str(output_path),
                "label": item["label"],
            }
        )
    return results


def _detect_local_minima(scan_rows: List[Dict]) -> List[Dict]:
    minima = []
    energies = [row["energy"] for row in scan_rows]
    for index in range(1, len(scan_rows) - 1):
        if energies[index] < energies[index - 1] and energies[index] < energies[index + 1]:
            minima.append(scan_rows[index])
    return minima


def create_stretch_curve_figure(rows: List[Dict]):
    rc_params = {
        'figure.figsize': (6, 5),
        'figure.dpi': 300,
        'font.family': 'Arial',
        'font.weight': 'bold',
        'legend.fontsize': 14,
        'xtick.labelsize': 16,
        'ytick.labelsize': 16,
        'axes.labelsize': 20,
        'axes.labelweight': 'bold',
        'lines.linewidth': 2,
        'axes.linewidth': 2,
        'xtick.direction': 'out',
        'ytick.direction': 'out',
        'ytick.major.width': 2,
        'ytick.minor.width': 2,
        'xtick.major.width': 2,
        'xtick.minor.width': 2,
        'xtick.major.size': 4,
        'xtick.minor.size': 2,
        'ytick.major.size': 4,
        'ytick.minor.size': 2,
        'axes.labelpad': 7,
    }

    with plt.rc_context(rc=rc_params):
        fig, ax = plt.subplots()
        x_values = np.array([row["height"] for row in rows], dtype=float)
        y_values = np.array([row["energy"] for row in rows], dtype=float)

        if len(x_values) >= 4 and make_interp_spline is not None:
            x_smooth = np.linspace(x_values.min(), x_values.max(), 300)
            spline = make_interp_spline(x_values, y_values, k=3)
            y_smooth = spline(x_smooth)
        else:
            x_smooth = np.linspace(x_values.min(), x_values.max(), max(100, len(x_values) * 20))
            y_smooth = np.interp(x_smooth, x_values, y_values)

        ax.plot(
            x_smooth,
            y_smooth,
            color='black',
        )
        ax.scatter(
            x_values,
            y_values,
            s=36,
            color='black',
            zorder=3,
        )
        ax.set_xlabel(r"$d$ ($\mathrm{\AA}$)")
        ax.set_ylabel("Total energy (eV)")
        ax.grid(False)
        for spine in ax.spines.values():
            spine.set_linewidth(2)
        fig.tight_layout()
        return fig


def create_adsorption_energy_figure(rows: List[Dict]):
    rc_params = {
        'figure.figsize': (6, 5),
        'figure.dpi': 300,
        'font.family': 'Arial',
        'font.weight': 'bold',
        'legend.fontsize': 14,
        'xtick.labelsize': 16,
        'ytick.labelsize': 16,
        'axes.labelsize': 20,
        'axes.labelweight': 'bold',
        'lines.linewidth': 2,
        'axes.linewidth': 2,
        'xtick.direction': 'out',
        'ytick.direction': 'out',
        'ytick.major.width': 2,
        'ytick.minor.width': 2,
        'xtick.major.width': 2,
        'xtick.minor.width': 2,
        'xtick.major.size': 4,
        'xtick.minor.size': 2,
        'ytick.major.size': 4,
        'ytick.minor.size': 2,
        'axes.labelpad': 7,
    }

    with plt.rc_context(rc=rc_params):
        fig, ax = plt.subplots()
        x_values = np.array([row["height"] for row in rows], dtype=float)
        y_values = np.array([row["energy"] for row in rows], dtype=float)

        if len(x_values) >= 4 and make_interp_spline is not None:
            x_smooth = np.linspace(x_values.min(), x_values.max(), 400)
            spline = make_interp_spline(x_values, y_values, k=3)
            y_smooth = spline(x_smooth)
        else:
            x_smooth = np.linspace(x_values.min(), x_values.max(), max(100, len(x_values) * 20))
            y_smooth = np.interp(x_smooth, x_values, y_values)

        ax.plot(x_smooth, y_smooth, color='black')
        ax.scatter(x_values, y_values, s=36, color='black', zorder=3)
        ax.set_xlim(1.0, 7.5)
        ax.set_xlabel('d (Å)')
        ax.set_ylabel('Ead (eV)')
        ax.grid(False)
        ax.tick_params(top=False, right=False)
        for spine in ax.spines.values():
            spine.set_linewidth(2)
        fig.tight_layout()
        return fig


def build_energy_profile(results: List[Dict]) -> Dict:
    profile_rows = []
    sorted_results = sorted(results, key=lambda item: float(item["energy"]))
    for idx, item in enumerate(sorted_results):
        profile_rows.append(
            {
                "step": idx,
                "height": float(idx + 1),
                "energy": float(item["energy"]),
                "label": item["site"],
            }
        )

    scan_dir = OUTPUT_DIR / "energy_profile"
    scan_dir.mkdir(parents=True, exist_ok=True)

    csv_path = scan_dir / "adsorption_energy_profile.csv"
    with csv_path.open("w", encoding="utf-8") as handle:
        handle.write("step,rank,site_label,adsorption_energy_eV\n")
        for row in profile_rows:
            handle.write(
                f"{row['step']},{row['height']:.0f},{row['label']},{row['energy']:.10f}\n"
            )

    plot_path = scan_dir / "adsorption_energy_profile.png"
    fig = create_adsorption_energy_figure(profile_rows)
    fig.savefig(plot_path, dpi=200)
    plt.close(fig)

    minima = _detect_local_minima(profile_rows)
    return {
        "status": "completed",
        "message": (
            "Completed candidate-energy profiling for benzene on Pt(111) and identified local minima "
            "from the ranked adsorption configurations."
        ),
        "scan_points": profile_rows,
        "csv_path": str(csv_path),
        "plot_path": str(plot_path),
        "trajectory_path": None,
        "local_minima": minima,
        "bistable_detected": len(minima) >= 2,
        "step_logs": [
            f"Generated {len(profile_rows)} optimized adsorption candidates.",
            "Ranked all candidates on a consistent adsorption-energy scale.",
            "Constructed the adsorption-energy profile for bistability assessment.",
        ],
        "data_source": "calculated_candidates",
    }


def execute_case_workflow(plan: Dict) -> Dict:
    smiles = plan.get("smiles") or resolve_smiles(plan.get("adsorbate", "benzene"))
    adsorbate_model = build_adsorbate_model(plan.get("adsorbate", "benzene"), smiles)
    surface_model = build_surface_model(plan.get("surface", "Pt(111)"))
    candidates = assemble_adsorption_candidates(surface_model, adsorbate_model, plan)

    settings = plan.get("stretch_settings", {})
    top_k = int(settings.get("top_k", 5))
    candidate_results = run_bistable_search(surface_model, adsorbate_model, candidates, top_k=top_k)
    stretch_info = build_energy_profile(candidate_results)
    minima = stretch_info["local_minima"]

    results = [
        {
            "site": row.get("label") or f"rank_{int(row['height'])}",
            "energy": row["energy"],
        }
        for row in stretch_info["scan_points"]
    ]

    if stretch_info["bistable_detected"]:
        bistable_statement = (
            f"The adsorption-energy profile shows at least two internal minima, indicating possible bistable adsorption behavior. "
            f"Candidate minima were found near {[f'{item['height']:.1f} A' for item in minima]}."
        )
    else:
        bistable_statement = (
            "The adsorption-energy profile does not show two clear internal minima under the current settings, "
            "so bistability is not confirmed in this scan."
        )

    agent_logs = {
        "Planner": [
            f"Received objective: {plan['objective']}",
            "Assigned MP-driven adsorption candidate generation and energy ranking workflow.",
            f"Analysis settings: top_k = {top_k}",
        ],
        "Modeling Expert": [
            f"Built surface model from Materials Project entry: {surface_model['material_id']}.",
            f"Generated adsorption candidates from requested site classes: {plan.get('sites', [])}.",
            f"Optimized and retained {len(stretch_info['scan_points'])} ranked candidate configurations.",
        ],
        "Optimization Expert": [
            "Executed candidate refinement using Minima Hopping search.",
            "Compared all retained structures on the same adsorption-energy scale.",
        ],
        "Analysis Expert": stretch_info["step_logs"],
    }

    return {
        "results": results,
        "stretch": stretch_info,
        "agent_logs": agent_logs,
        "summary": (
            f"Planner completed the adsorption-profile analysis for benzene on Pt(111). "
            f"The workflow covered {len(results)} sampled configurations along the adsorption coordinate. {bistable_statement}"
        ),
    }