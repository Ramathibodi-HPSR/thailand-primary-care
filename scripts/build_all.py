"""
Run the full PC_MAP data pipeline, from raw GIS export to published
dashboard bundle.

Usage:
  python scripts/build_all.py

Requires data/raw/coverage.parquet, data/raw/hospitals_confirmed.csv and
data/raw/province_population.json to already be in place (see README.md --
these come from an external ArcGIS buffer analysis, not from this repo).
"""

import runpy
from pathlib import Path

STEPS = [
    "01_build_coverage_totals.py",
    "02_build_public_coverage_totals.py",
    "03_build_province_coverage.py",
    "04_build_facility_lookup.py",
    "05_prepare_dashboard_bundle.py",
]


def main():
    scripts_dir = Path(__file__).resolve().parent
    for step in STEPS:
        print(f"\n=== {step} ===")
        runpy.run_path(str(scripts_dir / step), run_name="__main__")


if __name__ == "__main__":
    main()
