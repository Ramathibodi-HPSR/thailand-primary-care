"""
Copy the facility registry into the dashboard's data bundle, so
dashboard/ is a self-contained folder that GitHub Pages can serve on its own.

Input:
  data/raw/hospitals_confirmed.csv

Output:
  dashboard/data/hospitals_confirmed.csv

Drops sub_type_name: it's not read anywhere in dashboard/script.js (only
clinic_type drives marker color/layer/popup), and it's the single largest
column in the source file.
"""

import pandas as pd

from common import HOSPITALS_CSV, DASHBOARD_DATA

DASHBOARD_COLUMNS = ["pc_id", "pc_name", "clinic_type", "lat", "lon"]


def main():
    DASHBOARD_DATA.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(HOSPITALS_CSV, dtype={"pc_id": str})
    df = df[DASHBOARD_COLUMNS]

    out_path = DASHBOARD_DATA / "hospitals_confirmed.csv"
    df.to_csv(out_path, index=False)

    print(f"Saved: {out_path} ({len(df):,} facilities)")


if __name__ == "__main__":
    main()
