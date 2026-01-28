#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import csv
import fnmatch
import re
from pathlib import Path

import numpy as np
import pandas as pd

MODELS_CSV_NAME = "models.csv"
ALL_MODELS_CSV_NAME = "all_models_rmse_complexity.csv"
RMSE_MAX_DEFAULT = 50.0


def read_desc_dim_from_sisso_in(path: Path) -> int | None:
    if not path.exists():
        return None
    text = path.read_text(encoding="utf-8", errors="replace")
    m = re.search(r"desc_dim\s*=\s*(\d+)", text)
    return int(m.group(1)) if m else None


def read_ops_from_sisso_in(path: Path) -> list[str]:
    if not path.exists():
        return ["+", "-", "*", "/"]

    text = path.read_text(encoding="utf-8", errors="replace")
    m = re.search(r"ops\s*=\s*'(.*?)'", text, flags=re.DOTALL)
    if not m:
        return ["+", "-", "*", "/"]

    ops_str = m.group(1)
    ops = re.findall(r"\((.*?)\)", ops_str)
    ops = [op for op in ops if op]
    return ops or ["+", "-", "*", "/"]


def count_ops_in_expr(expr: str, ops_sorted: list[str]) -> int:
    s = expr
    total = 0
    for op in ops_sorted:
        pattern = re.escape(op)
        matches = re.findall(pattern, s)
        c = len(matches)
        if c > 0:
            total += c
            s = re.sub(pattern, " ", s)
    return total


def _is_sisso_job_dir(job_dir: Path) -> bool:
    return (job_dir / "Models").is_dir() and (job_dir / "SIS_subspaces" / "Uspace.expressions").is_file()


def _find_job_dirs(root: Path, prefer_pattern: str = "d*_c*") -> list[Path]:
    children = [p for p in root.iterdir() if p.is_dir()]

    pattern_dirs = sorted([p for p in children if fnmatch.fnmatch(p.name, prefer_pattern)])
    if pattern_dirs:
        return pattern_dirs

    job_dirs: list[Path] = []
    if _is_sisso_job_dir(root):
        job_dirs.append(root)
    job_dirs.extend(sorted([p for p in children if _is_sisso_job_dir(p)]))
    return job_dirs


def _select_top_file(models_dir: Path, desc_dim: int | None) -> tuple[Path, int]:
    top_candidates: list[tuple[Path, int | None]] = []
    for path in models_dir.iterdir():
        if not path.is_file():
            continue
        fname = path.name
        if not fname.startswith("top"):
            continue
        if fname.endswith("_coeff"):
            continue
        if "_D" not in fname:
            continue

        m = re.search(r"_D(\d+)", fname)
        d_dim = int(m.group(1)) if m else None
        top_candidates.append((path, d_dim))

    if not top_candidates:
        raise RuntimeError(f"No top*_Dxxx file found in {models_dir.as_posix()}")

    if desc_dim is not None:
        filtered = [c for c in top_candidates if c[1] == desc_dim or c[1] is None]
        if filtered:
            top_candidates = filtered

    top_file, top_d = max(top_candidates, key=lambda c: c[0].stat().st_mtime)
    if desc_dim is None:
        if top_d is None:
            raise RuntimeError("Cannot infer desc_dim from SISSO.in or top*_Dxxx filename")
        desc_dim = top_d

    return top_file, desc_dim


def _load_expressions(uspace_expr_file: Path) -> list[str]:
    if not uspace_expr_file.exists():
        raise RuntimeError(f"Missing file: {uspace_expr_file.as_posix()}")

    exprs: list[str] = []
    with uspace_expr_file.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            raw = line.rstrip("\n")
            expr = raw.split("SIS_score")[0].strip() if "SIS_score" in raw else raw.strip()
            exprs.append(expr)
    return exprs


def _parse_top_file(top_file: Path, desc_dim: int, n_expr: int) -> tuple[np.ndarray, list[int]]:
    raw_feature_rows: list[list[int]] = []
    with top_file.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            if any(key in s for key in ["Rank", "F_ID", "Feature", "RMSE", "MAE", "Error", "SISSO"]):
                continue

            ints_in_line: list[int] = []
            for p in s.split():
                p_clean = p.strip(",")
                if p_clean.lstrip("+-").isdigit():
                    ints_in_line.append(int(p_clean))
            if ints_in_line:
                raw_feature_rows.append(ints_in_line)

    if not raw_feature_rows:
        raise RuntimeError(f"No usable rows found in top file: {top_file.as_posix()}")

    feature_ids_list: list[list[int]] = []
    keep_row_idx: list[int] = []
    for row_idx, ints in enumerate(raw_feature_rows):
        if len(ints) >= desc_dim + 1:
            candidate = ints[1 : 1 + desc_dim]
        elif len(ints) == desc_dim:
            candidate = ints
        else:
            continue

        if not all(1 <= i <= n_expr for i in candidate):
            continue

        feature_ids_list.append(candidate)
        keep_row_idx.append(row_idx)

    if not feature_ids_list:
        raise RuntimeError(f"No valid models parsed from top file: {top_file.as_posix()}")

    return np.array(feature_ids_list, dtype=int), keep_row_idx


def _load_coefficients(coeff_file: Path, keep_row_idx: list[int]) -> np.ndarray | None:
    if not coeff_file.exists():
        return None

    coeff_rows: list[list[float]] = []
    with coeff_file.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            if any(key in s for key in ["c0", "Coeff", "Rank"]):
                continue
            try:
                floats = [float(p) for p in s.split()]
            except ValueError:
                continue
            if floats:
                coeff_rows.append(floats)

    if not coeff_rows:
        return None

    coeff_raw_full = np.array(coeff_rows, dtype=float)
    return coeff_raw_full[keep_row_idx, :]


def _load_uspace_data(job_dir: Path) -> np.ndarray | None:
    for candidate in [
        job_dir / "SIS_subspaces" / "Uspace_t001.dat",
        job_dir / "SIS_subspaces" / "Uspace.dat",
    ]:
        if candidate.exists():
            try:
                return np.loadtxt(candidate.as_posix())
            except Exception:
                continue
    return None


def _rmse(y: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y - y_pred) ** 2)))


def _maxae(y: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.max(np.abs(y - y_pred)))


def extract_models_csv(job_dir: Path) -> Path:
    job_dir = job_dir.resolve()
    models_dir = job_dir / "Models"
    uspace_expr_file = job_dir / "SIS_subspaces" / "Uspace.expressions"
    sisso_in_file = job_dir / "SISSO.in"

    if not models_dir.is_dir():
        raise RuntimeError(f"Missing Models/ in job dir: {job_dir.as_posix()}")

    desc_dim = read_desc_dim_from_sisso_in(sisso_in_file)
    top_file, desc_dim = _select_top_file(models_dir, desc_dim)
    coeff_file = Path(f"{top_file.as_posix()}_coeff")

    ops_sorted = sorted(read_ops_from_sisso_in(sisso_in_file), key=len, reverse=True)

    exprs = _load_expressions(uspace_expr_file)
    feature_ids_all, keep_row_idx = _parse_top_file(top_file, desc_dim, n_expr=len(exprs))
    nmodels = int(feature_ids_all.shape[0])

    coeff_raw = _load_coefficients(coeff_file, keep_row_idx)
    has_coeff = coeff_raw is not None

    data = _load_uspace_data(job_dir)
    has_error = False

    if has_coeff and data is not None and coeff_raw.shape[1] >= desc_dim + 1:
        has_error = True
        y_true = data[:, 0]
        X_feat = data[:, 1:]

        c0_all = coeff_raw[:, 0]
        c_all = coeff_raw[:, 1 : 1 + desc_dim]
        n_iter = min(nmodels, coeff_raw.shape[0])
    else:
        if has_coeff and coeff_raw.shape[1] >= desc_dim + 1:
            c0_all = coeff_raw[:, 0]
            c_all = coeff_raw[:, 1 : 1 + desc_dim]
            n_iter = min(nmodels, coeff_raw.shape[0])
        else:
            n_iter = nmodels

    out_csv = job_dir / MODELS_CSV_NAME
    rows_written = 0
    with out_csv.open("w", newline="", encoding="utf-8") as fout:
        writer = csv.writer(fout)

        header = ["rank"]
        header += [f"expr_{i+1}" for i in range(desc_dim)]
        header += [f"C_comp_{i+1}" for i in range(desc_dim)]
        header.append("C_total")
        if has_coeff and coeff_raw.shape[1] >= desc_dim + 1:
            header.append("c0")
            header += [f"c{i+1}" for i in range(desc_dim)]
        if has_error:
            header += ["RMSE", "MaxAE"]

        writer.writerow(header)
        rows_written += 1

        for rank in range(int(n_iter)):
            ids = feature_ids_all[rank]
            expr_list = [exprs[i - 1] for i in ids]

            comp_complexities = [count_ops_in_expr(e, ops_sorted) for e in expr_list]
            total_complexity = int(sum(comp_complexities))

            row: list[object] = [rank + 1]
            row += expr_list
            row += comp_complexities
            row.append(total_complexity)

            if has_coeff and coeff_raw.shape[1] >= desc_dim + 1:
                c0 = float(c0_all[rank])
                cs = c_all[rank]

                row.append(float(f"{c0:.6f}"))
                for cj in cs:
                    row.append(float(f"{float(cj):.6f}"))

                if has_error:
                    cols = ids - 1
                    Dmat = X_feat[:, cols]
                    y_pred = c0 + np.dot(Dmat, cs)
                    row.append(float(f"{_rmse(y_true, y_pred):.6f}"))
                    row.append(float(f"{_maxae(y_true, y_pred):.6f}"))

            writer.writerow(row)
            rows_written += 1

    print("[DONE] SISSO_extract")
    print("  job_dir:", job_dir.as_posix())
    print("  wrote:", out_csv.as_posix())
    print("  top file used:", top_file.as_posix())
    print("  desc_dim:", desc_dim)
    print("  models written (excluding header):", rows_written - 1)
    print("  has_coeff:", has_coeff, "has_error:", has_error)

    return out_csv


def build_all_models_csv(root: Path, job_dirs: list[Path], rmse_max: float) -> Path:
    rows: list[pd.DataFrame] = []
    for job_dir in job_dirs:
        csv_path = job_dir / MODELS_CSV_NAME
        folder = job_dir.name
        if not csv_path.exists():
            raise RuntimeError(f"Missing {MODELS_CSV_NAME} in: {job_dir.as_posix()}")

        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            raise RuntimeError(f"Failed to read {csv_path.as_posix()}: {e}") from e

        if "RMSE" not in df.columns:
            print(f"[WARN] skip {csv_path.as_posix()} (no RMSE column)")
            continue

        comp_cols = [c for c in df.columns if c.startswith("C_comp_")]
        if not comp_cols:
            print(f"[WARN] skip {csv_path.as_posix()} (no C_comp_* columns)")
            continue

        df["RMSE"] = pd.to_numeric(df["RMSE"], errors="coerce")
        for c in comp_cols:
            df[c] = pd.to_numeric(df[c], errors="coerce")

        df["complexity"] = df[comp_cols].sum(axis=1, skipna=True) + 1

        expr_cols = sorted(
            [c for c in df.columns if c.startswith("expr_")],
            key=lambda x: int(x.split("_")[1]),
        )
        if expr_cols:
            df["descriptor"] = df[expr_cols].astype(str).agg(" | ".join, axis=1)
            df["desc_dim"] = len(expr_cols)
        else:
            df["descriptor"] = ""
            df["desc_dim"] = 0

        if "rank" in df.columns:
            df["rank"] = pd.to_numeric(df["rank"], errors="coerce")
        else:
            df["rank"] = pd.NA

        df["folder"] = folder

        keep = df[["folder", "rank", "desc_dim", "complexity", "RMSE", "descriptor"]].copy()
        keep = keep.dropna(subset=["complexity", "RMSE"])
        keep = keep[keep["RMSE"] <= float(rmse_max)].copy()
        rows.append(keep)

    if not rows:
        raise RuntimeError("No usable models.csv found (or missing RMSE/C_comp_* columns).")

    all_df = pd.concat(rows, ignore_index=True)
    out_path = root / ALL_MODELS_CSV_NAME
    all_df.to_csv(out_path, index=False)

    print("[DONE] SISSO_extract summary")
    print("  wrote:", out_path.as_posix())
    print("  job_dirs:", len(job_dirs))
    print(f"  filter: RMSE <= {float(rmse_max)}")

    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract SISSO models and build summary CSV.")
    parser.add_argument("--root", default=".", help="Job dir or parent dir containing SISSO subdirs.")
    parser.add_argument("--rmse-max", type=float, default=RMSE_MAX_DEFAULT, help="Summary CSV RMSE filter.")
    parser.add_argument("--no-summary", action="store_true", help="Only generate per-job models.csv, skip summary CSV.")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    if not root.exists():
        raise FileNotFoundError(f"Root does not exist: {root.as_posix()}")

    job_dirs = _find_job_dirs(root)
    if not job_dirs:
        raise RuntimeError(f"No SISSO job dirs found under: {root.as_posix()}")

    # If subdirs look like d*_c*, be strict: they should all be valid job dirs.
    if any(fnmatch.fnmatch(p.name, "d*_c*") for p in job_dirs):
        for p in job_dirs:
            if not _is_sisso_job_dir(p):
                raise RuntimeError(f"Invalid SISSO subdir (missing Models/ or Uspace.expressions): {p.as_posix()}")

    for job_dir in job_dirs:
        extract_models_csv(job_dir)

    if not args.no_summary:
        build_all_models_csv(root, job_dirs, rmse_max=float(args.rmse_max))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
