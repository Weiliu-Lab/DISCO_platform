import re
from typing import List, Tuple


def parse_surface_label(surface: str) -> Tuple[str, List[int]]:
    """Parse a surface label like 'Pt(111)' into (formula, [h, k, l])."""
    if not surface:
        return "", [1, 1, 1]

    surface = surface.strip()
    match = re.match(r"^([A-Za-z0-9]+)\s*\((\d)(\d)(\d)\)$", surface)
    if match:
        formula = match.group(1)
        hkl = [int(match.group(2)), int(match.group(3)), int(match.group(4))]
        return formula, hkl

    return surface, [1, 1, 1]
