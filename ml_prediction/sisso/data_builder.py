import os

import pandas as pd


class SissoTrainDataBuilder:
    def __init__(self, elements_df):
        self.elements_df = elements_df
        # 建立 Symbol -> Properties 的映射（按需读取）
        if not isinstance(self.elements_df, pd.DataFrame) or self.elements_df.empty:
            raise ValueError("elements_properties_all.csv 未正确加载（DataFrame 为空）")
        if "symbol" not in self.elements_df.columns:
            raise ValueError("elements_properties_all.csv 缺少必要的 symbol")

        self._symbol_to_row = {str(r["symbol"]): r for _, r in self.elements_df.iterrows()}

    def get_atom_prop(self, symbol, prop):
        """严格读取元素属性：缺失即报错，不做 0/默认兜底。"""
        # 兼容列名映射
        prop_map = {
            "Radius": "atomic_radius",
            "Electronegativity": "en_pauling",
            "Mass": "atomic_mass",
        }
        real_prop = prop_map.get(prop, prop)

        row = self._symbol_to_row.get(str(symbol))
        if row is None:
            raise KeyError(f"元素属性表中不存在 symbol={symbol!r}")
        if real_prop not in row.index:
            raise KeyError(f"元素属性表缺少列 {real_prop}")
        val = row.get(real_prop)
        if val is None or (isinstance(val, float) and pd.isna(val)):
            raise ValueError(f"元素属性缺失 symbol={symbol!r} col={real_prop}")
        try:
            return float(val)
        except Exception as e:
            raise ValueError(
                f"元素属性无法转为数值 symbol={symbol!r} col={real_prop} val={val!r}"
            ) from e

    def build_train_dat(self, structs, targets_map, indices, feats, parser_func):
        """
        True feature extraction from structures.
        structs: list of {'filename':..., 'content':...}
        targets_map: dict of {clean_filename: target_value}
        indices: list of 1-based atom indices (e.g. [1, 2])
        """
        lines = []
        prop_name = "Property"

        if not feats or len(feats) <= 0:
            raise ValueError("未选择特征列（feats 不能为空）")
        feature_list = feats

        # 1. 生成表头 (Header)
        header_cols = ["materials", prop_name]
        for idx in indices:
            for feat in feature_list:
                header_cols.append(f"Atom{idx}_{feat}")

        lines.append(" ".join(header_cols))

        # 2. 生成数据行
        valid_count = 0
        error_logs = []

        for s in structs:
            fname_raw = s.get("filename", "")
            # dcc.Upload 在部分浏览器/系统下可能带路径：C:\fakepath\a.cif
            fname = os.path.basename(str(fname_raw)).strip()
            # 尝试匹配 key
            key_candidates = [
                fname,
                os.path.splitext(fname)[0],
                fname.split(".")[0],
            ]

            target_val = None
            matched_key = None
            for k in key_candidates:
                if k in targets_map:
                    target_val = targets_map[k]
                    matched_key = k
                    break

            if target_val is None:
                raise ValueError(
                    f"目标CSV中找不到结构对应的key: filename={fname!r} candidates={key_candidates}"
                )

            # Parse atoms (ASE)
            atoms = parser_func(s.get("content"), fname)

            row = [matched_key, str(target_val)]

            # Extract features for each requested atom index (1-based)
            natoms = len(atoms)
            for atom_idx in indices:
                if atom_idx < 1 or atom_idx > natoms:
                    raise ValueError(
                        f"原子索引越界: filename={fname!r} atom_idx={atom_idx} natoms={natoms}"
                    )

                symbol = atoms[atom_idx - 1].symbol
                for feat in feature_list:
                    val = self.get_atom_prop(symbol, feat)
                    row.append(f"{val:.4f}")

            lines.append(" ".join(row))
            valid_count += 1

        return "\n".join(lines), valid_count, error_logs
