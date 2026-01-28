from typing import Dict, Tuple

try:
    from mp_api.client import MPRester
except Exception:  # pragma: no cover - optional dependency
    MPRester = None


def smart_search_mp(api_key: str, query_str: str, limit: int = 1) -> Tuple[Dict, str]:
    """Search Materials Project and return a structure dict + metadata."""
    if MPRester is None:
        return {}, "mp_api not available"

    query_str = (query_str or "").strip()
    if not query_str:
        return {}, "Empty query"

    filters = {}
    if query_str.startswith("mp-") or query_str.startswith("mvc-"):
        filters["material_ids"] = [query_str]
        search_mode = "ID"
    elif "-" in query_str and not any(char.isdigit() for char in query_str):
        filters["chemsys"] = query_str
        search_mode = "Chemsys"
    else:
        filters["formula"] = query_str
        search_mode = "Formula"

    try:
        with MPRester(api_key) as mpr:
            docs = mpr.materials.summary.search(
                **filters,
                fields=[
                    "material_id",
                    "formula_pretty",
                    "symmetry",
                    "structure",
                    "energy_per_atom",
                    "is_stable",
                ],
            )
            if not docs:
                return {}, f"No results for: {query_str}"

            doc = docs[:limit][0]
            struct = getattr(doc, "structure", None)
            if struct is None:
                return {}, "No structure found in MP results"

            return (
                {
                    "structure": struct.as_dict(),
                    "meta": {
                        "material_id": str(getattr(doc, "material_id", "")),
                        "formula": getattr(doc, "formula_pretty", ""),
                        "search_mode": search_mode,
                        "energy": getattr(doc, "energy_per_atom", None),
                        "is_stable": getattr(doc, "is_stable", False),
                    },
                },
                "ok",
            )
    except Exception as exc:
        return {}, f"MP search error: {exc}"
