import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class SissoConfigManager:
    def __init__(self, filepath):
        # 强制使用模板文件作为唯一来源，不提供内置默认模板
        self.filepath = filepath
        if not self.filepath:
            raise FileNotFoundError("未提供 SISSO.in 模板路径")

        p = Path(self.filepath)
        if not p.is_absolute():
            p = (PROJECT_ROOT / p).resolve()
        if not p.exists():
            raise FileNotFoundError(f"找不到 SISSO.in 模板文件: {p}")

        with open(p, "r", encoding="utf-8") as f:
            self.raw_content = f.read()

    def update_template(self, params):
        content = self.raw_content
        if "nsample" in params:
            content = re.sub(r"(nsample\s*=\s*)\d+", f"\\g<1>{params['nsample']}", content)
        if "nsf" in params:
            content = re.sub(r"(nsf\s*=\s*)\d+", f"\\g<1>{params['nsf']}", content)
        if "desc_dim" in params:
            content = re.sub(r"(desc_dim\s*=\s*)\d+", f"\\g<1>{params['desc_dim']}", content)
        if "fcomplexity" in params:
            content = re.sub(r"(fcomplexity\s*=\s*)\d+", f"\\g<1>{params['fcomplexity']}", content)
        if "ops" in params:
            content = re.sub(r"(ops\s*=\s*)'.*?'", f"ops='{params['ops']}'", content)
        return content
