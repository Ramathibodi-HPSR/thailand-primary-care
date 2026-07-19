"""
Build the per-facility, per-radius population lookup used for map popups.

Input:
  data/raw/coverage.parquet

Output:
  dashboard/data/coverage_lookup.json   (JSON Lines: {"pc_id","radius_km","pop"})

Only the tiered KEEP_RADII steps are published (see common.py), and pop is
rounded to the nearest person -- the UI only ever displays an integer, so
sub-person precision just bloats the file.
"""

import pandas as pd

from common import COVERAGE_RAW_PARQUET, DASHBOARD_DATA, KEEP_RADII_SET


def main():
    DASHBOARD_DATA.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(COVERAGE_RAW_PARQUET, columns=["pc_id", "radius_km", "pop"])
    df = df[df["radius_km"].isin(KEEP_RADII_SET)].copy()
    df["pop"] = df["pop"].round().astype(int)
    df = df.sort_values(["pc_id", "radius_km"])

    out_path = DASHBOARD_DATA / "coverage_lookup.json"
    with open(out_path, "w", encoding="utf-8") as f:
        for row in df.itertuples(index=False):
            f.write(
                '{"pc_id":"%s","radius_km":%.1f,"pop":%d}\n'
                % (row.pc_id, row.radius_km, row.pop)
            )

    print(f"Saved: {out_path} ({len(df):,} rows)")


if __name__ == "__main__":
    main()
