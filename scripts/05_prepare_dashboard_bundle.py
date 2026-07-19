"""
Copy the facility registry into the dashboard's data bundle, so
dashboard/ is a self-contained folder that GitHub Pages can serve on its own.

Input:
  data/raw/hospitals_confirmed.csv

Output:
  dashboard/data/hospitals_confirmed.csv
"""

import pandas as pd

from common import HOSPITALS_CSV, DASHBOARD_DATA


def main():
    DASHBOARD_DATA.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(HOSPITALS_CSV, dtype={"pc_id": str})
    df = df.drop(columns=[c for c in df.columns if c.startswith("Unnamed")], errors="ignore")

    out_path = DASHBOARD_DATA / "hospitals_confirmed.csv"
    df.to_csv(out_path, index=False)

    print(f"Saved: {out_path} ({len(df):,} facilities)")


if __name__ == "__main__":
    main()
