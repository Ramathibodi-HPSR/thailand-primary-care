"""
Build the per-facility, per-radius population lookup used for map popups.

Input:
  data/raw/coverage.parquet

Output:
  dashboard/data/coverage_lookup.json
    { "radii": ["0.1", ..., "10.0"], "facilities": { "<pc_id>": [pop, ...] } }

Published as one compact matrix rather than one JSON object per
(facility, radius) row: the old JSON-Lines format repeated the "pc_id",
"radius_km" and "pop" keys on every one of ~558k lines, and required the
browser to run ~558k individual JSON.parse() calls on load. A single
JSON.parse() over one array-of-numbers-per-facility object is both a much
smaller payload and dramatically faster to load. pop is rounded to the
nearest person -- the UI only ever displays an integer, so sub-person
precision just bloats the file.
"""

import json

import pandas as pd

from common import COVERAGE_RAW_PARQUET, DASHBOARD_DATA, KEEP_RADII, radius_key


def main():
    DASHBOARD_DATA.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(COVERAGE_RAW_PARQUET, columns=["pc_id", "radius_km", "pop"])
    df = df[df["radius_km"].isin(set(KEEP_RADII))].copy()
    df["pop"] = df["pop"].round().astype(int)

    radii_keys = [radius_key(r) for r in KEEP_RADII]
    pivot = df.pivot_table(index="pc_id", columns="radius_km", values="pop", aggfunc="sum")
    pivot = pivot.reindex(columns=KEEP_RADII, fill_value=0)

    facilities = {
        str(pc_id): [int(v) for v in row]
        for pc_id, row in zip(pivot.index, pivot.to_numpy())
    }

    out = {"radii": radii_keys, "facilities": facilities}

    out_path = DASHBOARD_DATA / "coverage_lookup.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, separators=(",", ":"))

    print(f"Saved: {out_path} ({len(facilities):,} facilities x {len(radii_keys)} radii)")


if __name__ == "__main__":
    main()
