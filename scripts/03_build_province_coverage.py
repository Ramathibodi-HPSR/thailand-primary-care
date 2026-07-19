"""
Build population-covered-by-radius, broken down by province, as % of each
province's total population. This is an analysis product (not currently
consumed by the dashboard UI) kept alongside the pipeline for reuse.

Input:
  data/raw/coverage.parquet
  data/raw/province_population.json

Output:
  data/processed/coverage_province.json
"""

import json

import pandas as pd

from common import (
    COVERAGE_RAW_PARQUET,
    COVERAGE_PROVINCE_JSON,
    PROVINCE_POPULATION_JSON,
    KEEP_RADII_SET,
    DATA_PROCESSED,
)


def main():
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(COVERAGE_RAW_PARQUET, columns=["prov_name", "radius_km", "pop"])
    df = df[df["radius_km"].isin(KEEP_RADII_SET)]

    totals = (
        df.groupby(["prov_name", "radius_km"], as_index=False)["pop"]
        .sum()
        .sort_values(["prov_name", "radius_km"])
    )

    with open(PROVINCE_POPULATION_JSON, "r", encoding="utf-8") as f:
        province_population_raw = json.load(f)
    province_population = {
        row["province_name_th"]: float(row["population"]) for row in province_population_raw
    }

    out = {}
    for row in totals.itertuples(index=False):
        prov = row.prov_name
        radius = f"{row.radius_km:.1f}"
        pop = int(row.pop)

        total = province_population.get(prov)
        pct = None if not total else min(100.0, round(pop / total * 100.0, 2))

        out.setdefault(prov, {})[radius] = {"pop": pop, "pct": pct}

    with open(COVERAGE_PROVINCE_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"Saved: {COVERAGE_PROVINCE_JSON} ({len(out)} provinces)")


if __name__ == "__main__":
    main()
