"""
Run the v2 pipeline (task F), from the NHSO registry, WorldPop and the tambon
boundaries to coverage and access tables. Replaces the ArcGIS buffer run.

Usage:
  python scripts/v2/build_v2.py

Requires the two registry snapshots in data/raw/ and the PC_GIS folder next to
this repo (or PC_GIS_DIR set); see config.py. Takes about 12 minutes, most of it
the road graph (16) and the travel-time runs (17).
"""
import runpy
from pathlib import Path

STEPS = [
    "10_population_grid.py",
    "11_facilities.py",
    "12_nearest_coverage.py",
    "13_compare_arcgis.py",
    "14_change_over_time.py",
    "15_wealth_equity.py",
    "16_road_graph.py",
    "17_travel_time.py",
    "18_e2sfca.py",
    "19_mclp.py",
    "20_dashboard_bundle.py",
]


def main():
    scripts_dir = Path(__file__).resolve().parent
    for step in STEPS:
        print(f"\n=== {step} ===")
        runpy.run_path(str(scripts_dir / step), run_name="__main__")


if __name__ == "__main__":
    main()
