from typing import Dict, List, Tuple

from llm.client import DeepSeekClient
from llm.parse import coerce_float, coerce_list, extract_json
from toolboxes.modeling.adsorption import add_adsorbate
from toolboxes.modeling.fixation import apply_fixation
from toolboxes.modeling.mp_search import smart_search_mp
from toolboxes.modeling.slab import generate_slab
from toolboxes.modeling.surface import parse_surface_label


DEFAULT_SITES = ["top", "fcc", "hcp"]
STRUCTURE_TYPES = {"bulk", "surface", "adsorption"}


class ModelingAgent:
    def __init__(self, mp_api_key: str, llm_client: DeepSeekClient | None):
        self.mp_api_key = mp_api_key
        self.llm_client = llm_client

    def plan(self, user_request: str) -> Tuple[Dict, List[str]]:
        if not self.llm_client:
            raise RuntimeError("Missing LLM client for modeling.")

        content = self.llm_client.chat(
            [
                {
                    "role": "system",
                    "content": (
                        "You are the modeling agent. Decide the structure type "
                        "(bulk, surface, adsorption) and basic modeling parameters. "
                        "Return JSON only."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f'The user request is: "{user_request}".\n\n'
                        "Return JSON with keys:\n"
                        "- structure_type: bulk, surface, or adsorption\n"
                        "- formula: chemical formula, e.g. MgO\n"
                        "- surface: surface label, e.g. MgO(111) (if surface/adsorption)\n"
                        "- adsorbate: e.g. O, CO (if adsorption)\n"
                        "- sites: list of adsorption sites (if adsorption)\n"
                        "- slab: {\"min_size\": 10.0, \"vacuum\": 15.0} (if surface/adsorption)\n"
                        "- fixation: {\"z_frac\": 0.5} (if surface/adsorption)\n"
                    ),
                },
            ]
        )
        data = extract_json(content)

        formula = (
            str(
                data.get("formula")
                or data.get("composition")
                or data.get("material")
                or ""
            )
            .strip()
        )
        surface = str(data.get("surface") or "").strip()
        adsorbate = str(data.get("adsorbate") or "").strip()
        sites = coerce_list(data.get("sites"))

        if not formula and surface:
            formula, _ = parse_surface_label(surface)

        structure_type = str(data.get("structure_type") or "").lower().strip()
        if structure_type not in STRUCTURE_TYPES:
            if adsorbate:
                structure_type = "adsorption"
            elif surface:
                structure_type = "surface"
            else:
                structure_type = "bulk"

        if structure_type == "adsorption" and not adsorbate:
            structure_type = "surface"

        slab = data.get("slab") or {}
        fixation = data.get("fixation") or {}

        plan = {
            "structure_type": structure_type,
            "formula": formula,
            "surface": surface,
            "adsorbate": adsorbate,
            "sites": sites,
            "slab": {
                "min_size": coerce_float(slab.get("min_size"), 10.0),
                "vacuum": coerce_float(slab.get("vacuum"), 15.0),
            },
            "fixation": {
                "z_frac": coerce_float(fixation.get("z_frac"), 0.5),
            },
        }
        logs = [f"Modeling plan: {structure_type}"]
        if not formula:
            logs.append("Warning: formula missing in modeling plan.")
        return plan, logs

    def build(self, plan: Dict) -> Tuple[List[Dict], List[str]]:
        logs: List[str] = []
        structure_type = str(plan.get("structure_type") or "surface").lower().strip()
        surface_label = plan.get("surface") or ""
        adsorbate = plan.get("adsorbate") or ""
        sites = plan.get("sites") or []
        slab_cfg = plan.get("slab") or {}
        fixation_cfg = plan.get("fixation") or {}
        formula = plan.get("formula") or ""

        if not formula and surface_label:
            formula, _ = parse_surface_label(surface_label)

        if not formula:
            return [], ["Missing formula for MP search."]

        mp_res, mp_msg = smart_search_mp(self.mp_api_key, formula, limit=1)
        if not mp_res:
            return [], logs + [f"MP search failed: {mp_msg}"]

        bulk_struct = mp_res["structure"]
        logs.append(f"MP structure loaded: {mp_res.get('meta', {}).get('material_id', '')}")

        if structure_type == "bulk":
            return (
                [
                    {
                        "site": "bulk",
                        "structure": bulk_struct,
                        "meta": {"formula": formula, "structure_type": "bulk"},
                    }
                ],
                logs,
            )

        if not surface_label:
            surface_label = formula

        formula_label, hkl = parse_surface_label(surface_label)
        logs.append(f"Surface parsed: {formula_label} ({hkl[0]}{hkl[1]}{hkl[2]})")

        slab = generate_slab(
            bulk_struct,
            hkl=tuple(hkl),
            min_size=float(slab_cfg.get("min_size", 10.0)),
            vacuum=float(slab_cfg.get("vacuum", 15.0)),
        )

        slab_dict, fix_info = apply_fixation(slab, fixation_cfg.get("z_frac", 0.5))
        logs.append(f"Fixation applied: {fix_info}")

        if structure_type == "surface" or not adsorbate:
            return (
                [
                    {
                        "site": "slab",
                        "structure": slab_dict,
                        "meta": {
                            "surface": surface_label,
                            "structure_type": "surface",
                        },
                    }
                ],
                logs,
            )

        results: List[Dict] = []
        for site in sites or DEFAULT_SITES:
            try:
                ads_struct = add_adsorbate(slab_dict, adsorbate, site)
                results.append(
                    {
                        "site": site,
                        "structure": ads_struct.as_dict(),
                        "meta": {
                            "surface": surface_label,
                            "adsorbate": adsorbate,
                            "site": site,
                            "structure_type": "adsorption",
                        },
                    }
                )
            except Exception as exc:
                logs.append(f"Adsorption failed for {site}: {exc}")

        return results, logs
