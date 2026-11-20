"""
Check the Python replica against the ArcGIS run behind the live map.

Input:
  data/processed/v2/coverage_curve.csv, data/processed/v2/coverage_facility.parquet
  data/raw/coverage.parquet, data/raw/coverage_public.parquet
Output:
  data/processed/v2/compare_arcgis.csv   ArcGIS and Python totals by radius, all types and class P
  printed summary, including the facility-level agreement at 1, 2 and 5 km
"""
import numpy as np
import pandas as pd

from config import LIVE_COVERAGE, LIVE_COVERAGE_PUBLIC, PROCESSED


def main():
    curve = pd.read_csv(PROCESSED / "coverage_curve.csv")
    rep = curve[curve["scenario"] == "live_replica"]
    rows = []
    for group, path in [("all", LIVE_COVERAGE), ("P", LIVE_COVERAGE_PUBLIC)]:
        arc = pd.read_parquet(path, columns=["pc_id", "radius_km", "pop"])
        arc["radius_km"] = arc["radius_km"].round(1)
        tot = arc.groupby("radius_km")["pop"].sum()
        py = rep[rep["group"] == group].set_index("radius_km")["pop"]
        t = pd.DataFrame({"arcgis": tot, "python": py}).dropna()
        t["diff_pct"] = (t["python"] / t["arcgis"] - 1) * 100
        rows.append(t.assign(group=group).reset_index())
        print(f"\n== {group}: totals (million people) ==")
        print((t.loc[[0.1, 0.5, 1.0, 2.0, 5.0] + ([10.0] if 10.0 in t.index else [])]
               .assign(arcgis=lambda d: d["arcgis"] / 1e6, python=lambda d: d["python"] / 1e6)
               .round(2)).to_string())

        fa = pd.read_parquet(PROCESSED / "coverage_facility.parquet")
        fa = fa[(fa["scenario"] == "live_replica") & (fa["group"] == group)]
        arc["hcode"] = arc["pc_id"].astype(str).str.zfill(5)
        for r in [1.0, 2.0, 5.0]:
            m = arc[arc["radius_km"] == r][["hcode", "pop"]].merge(
                fa[fa["radius_km"].round(1) == r][["hcode", "pop"]], on="hcode",
                suffixes=("_arc", "_py"))
            m = m[(m["pop_arc"] > 0) | (m["pop_py"] > 0)]
            rel = (m["pop_py"] - m["pop_arc"]).abs() / m["pop_arc"].clip(lower=1)
            print(f"   {r:.0f} km: {len(m):,} units, correlation {np.corrcoef(m['pop_arc'], m['pop_py'])[0, 1]:.4f}, "
                  f"median abs diff {(m['pop_py'] - m['pop_arc']).abs().median():,.0f} people, "
                  f"within 10% for {(rel <= 0.10).mean():.1%}")
    pd.concat(rows).to_csv(PROCESSED / "compare_arcgis.csv", index=False)
    print(f"\nSaved: {PROCESSED / 'compare_arcgis.csv'}")


if __name__ == "__main__":
    main()
