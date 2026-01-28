import base64
import io
import os

import ase.io
import pandas as pd


def _decode_maybe_base64(content_string: str) -> str:
    if not content_string:
        return ""
    try:
        decoded = base64.b64decode(content_string)
        return decoded.decode("utf-8", errors="replace")
    except Exception:
        # Allow non-base64 input (e.g., raw text) and keep original content.
        return str(content_string)


def _guess_ase_format(text: str, filename=None) -> str:
    if filename:
        ext = os.path.splitext(str(filename).lower())[1]
        if ext == ".cif":
            return "cif"
        if ext in {".vasp", ".poscar", ".contcar"}:
            return "vasp"
    head = (text or "")[:1200]
    if "data_" in head or "_cell_" in head:
        return "cif"
    return "vasp"


def parse_atoms(content_string: str, filename=None):
    """解析上传内容为 ASE Atoms；不依赖 pymatgen。"""
    text = _decode_maybe_base64(content_string)
    if not text.strip():
        raise ValueError("结构文件内容为空")
    fmt = _guess_ase_format(text, filename)
    try:
        return ase.io.read(io.StringIO(text), format=fmt)
    except Exception as e:
        raise ValueError(f"结构解析失败: filename={filename!r} fmt={fmt!r} ({e})") from e


def atoms_to_ctk_structure(atoms):
    """仅用于 crystal_toolkit 可视化：在边界处把 ASE Atoms 转成 pymatgen Structure。"""
    if atoms is None:
        return None
    from pymatgen.io.ase import AseAtomsAdaptor

    return AseAtomsAdaptor.get_structure(atoms)


def parse_structure_for_viewer(content_string: str, filename=None):
    """给 crystal_toolkit viewer 用的解析入口（内部会在边界处使用 pymatgen 转换）。"""
    atoms = parse_atoms(content_string, filename)
    return atoms_to_ctk_structure(atoms)


def parse_csv_content(content_string):
    if not content_string:
        raise ValueError("CSV 未上传")
    try:
        decoded = base64.b64decode(content_string.split(",")[1])
        df = pd.read_csv(io.StringIO(decoded.decode("utf-8")), sep=None, engine="python")
        if df is None or df.empty:
            raise ValueError("CSV 内容为空")
        # 清理第一列(Key)
        df.iloc[:, 0] = df.iloc[:, 0].astype(str).str.strip()
        df.set_index(df.columns[0], inplace=True)
        return df, len(df)
    except Exception as e:
        raise ValueError(f"CSV 解析失败: {e}") from e
