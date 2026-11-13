"""Copy the charts and screenshots the README shows into docs/images/.

Run after build_report.py (which creates the charts in report/assets/) and after you have taken your screenshots:
    python scripts/export_readme_assets.py
"""
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "images"
OUT.mkdir(parents=True, exist_ok=True)
PAIRS = [
    ("report/assets/chart_overall.png", "mrr_overall.png"),
    ("report/assets/chart_by_type.png", "mrr_by_type.png"),
    ("report/assets/chart_gate.png", "gate.png"),
    ("report/screenshots/01_app_overview.png", "app_overview.png"),
    ("report/screenshots/02_paraphrase.png", "app_paraphrase.png"),
    ("report/screenshots/05_out_of_scope.png", "app_out_of_scope.png"),
]
for src, dst in PAIRS:
    s = ROOT / src
    if s.exists():
        shutil.copy2(s, OUT / dst)
        print(f"copied  {src}  ->  docs/images/{dst}")
    else:
        print(f"MISSING {src}  (README image docs/images/{dst} will be broken until it exists)")
print("\nCheck every image for personal details (your Windows user name in a path, e-mail, tokens) before committing.")
