"""
Build population-covered-by-radius series, public facilities only.

Input:
  data/processed/coverage_public.parquet   (produced by 01_filter_public_facilities.py)

Output:
  dashboard/data/coverage_distance_public.json   { "0.1": <people>, ... }
"""

import json

import pandas as pd

from common import COVERAGE_PUBLIC_PARQUET, DASHBOARD_DATA, KEEP_RADII_SET, radius_key


def main():
    DASHBOARD_DATA.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(COVERAGE_PUBLIC_PARQUET, columns=["radius_km", "pop"])
    df = df[df["radius_km"].isin(KEEP_RADII_SET)]

    totals = df.groupby("radius_km")["pop"].sum().sort_index()
    out = {radius_key(r): int(v) for r, v in totals.items()}

    out_path = DASHBOARD_DATA / "coverage_distance_public.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"Saved: {out_path} ({len(out)} radius steps)")


if __name__ == "__main__":
    main()
