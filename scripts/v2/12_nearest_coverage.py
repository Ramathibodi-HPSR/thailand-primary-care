"""
Distance from every populated 100 m cell to the nearest facility (KD-tree on
the unit sphere), in place of the ArcGIS buffer run. Each cell's people go to
its nearest facility, so nobody is counted twice, as in the ArcGIS run.

Scenarios:
  live_replica  the 24,256 units and coordinates behind the live map
                (hospitals_confirmed.csv; units without coordinates dropped,
                as ArcGIS did). Only used to check this script against ArcGIS.
  fixed_2025    units in the 2025-11-29 snapshot, coordinates from 11_facilities.py
  current_2026  units in the 2026-11-20 snapshot, coordinates from 11_facilities.py
                (both without prison health units)

Groups: all (every kept type), P (NHSO class P), P_public (class P owned by
the state), innovation (pharmacies, nurse clinics, doctor clinics).

Input:
  data/interim/v2/popgrid.parquet, data/processed/v2/facilities.csv,
  data/raw/hospitals_confirmed.csv
Output (data/processed/v2/):
  coverage_curve.csv        scenario, group, radius_km, pop, pct (of 69.8 million)
  coverage_province.csv     same by province of residence (denominator = people living there)
  coverage_facility.parquet people allocated to each facility within each radius
  access_tambon.csv         per tambon: people, mean distance, share within 1, 2, 5 km
  access_h3r7.csv           the same per H3 resolution-7 hexagon
"""
import h3
import numpy as np
import pandas as pd

from config import FACILITIES, LIVE_FACILITIES, POPGRID, PROCESSED, RADII_KM
from geo import nearest

GROUPS = {
    "all": lambda f: f.index == f.index,
    "P": lambda f: f["type"].eq("P"),
    "P_public": lambda f: f["type"].eq("P") & f["ownership"].eq("public"),
    "innovation": lambda f: f["type"].ne("P"),
}
ACCESS_KM = [1, 2, 5]


def scenarios():
    live = pd.read_csv(LIVE_FACILITIES, dtype={"pc_id": str}).dropna(subset=["lat", "lon"])
    live = live.assign(
        hcode=live["pc_id"].str.zfill(5),
        type=live["clinic_type"].map({"Public Facility": "P", "Pharmacy": "pharmacy",
                                      "Nurse Clinic": "nurse_clinic", "Doctor Clinic": "doctor_clinic"}),
        ownership=np.where(live["sub_type_name"].isin(["คลินิก (เอกชน)", "โรงพยาบาล (เอกชน)"]),
                           "private", "public"),
    )[["hcode", "type", "ownership", "sub_type_name", "lon", "lat"]]
    # Prison health units are left out; the replica keeps them, as ArcGIS did
    fac = pd.read_csv(FACILITIES, dtype={"hcode": str}).dropna(subset=["lon", "lat"])
    fac = fac[fac["public_access"]]
    yield "live_replica", live.reset_index(drop=True)
    # Each snapshot keeps its own type and ownership for a unit
    for name, s in [("fixed_2025", "2025"), ("current_2026", "2026")]:
        f = fac[fac[f"in_{s}"]].rename(columns={f"type_{s}": "t", f"ownership_{s}": "o"})
        yield name, f[["hcode", "t", "o", "sub_type_name", "lon", "lat"]].rename(
            columns={"t": "type", "o": "ownership"}).reset_index(drop=True)


def main():
    PROCESSED.mkdir(parents=True, exist_ok=True)
    grid = pd.read_parquet(POPGRID)
    pop = grid["pop"].to_numpy()
    total = pop.sum()
    grid["prov"] = grid["tambon"].str[:2]
    grid["h3"] = [h3.latlng_to_cell(y, x, 7) for x, y in zip(grid["lon"], grid["lat"])]
    nbin = len(RADII_KM) + 1  # bin k holds cells with distance in ((k-1)*100 m, k*100 m]; last bin = beyond 10 km
    prov_codes, prov_idx = np.unique(grid["prov"], return_inverse=True)

    curves, provs, alloc, tambon, hexes = [], [], [], [], []
    for scen, fac in scenarios():
        for group, sel in GROUPS.items():
            f = fac[sel(fac)].reset_index(drop=True)
            # Units that share one coordinate split that point's people equally
            pts, f["pt"], n_at = np.unique(f[["lon", "lat"]].to_numpy(), axis=0,
                                           return_inverse=True, return_counts=True)
            idx, dist = nearest(grid["lon"], grid["lat"], pts[:, 0], pts[:, 1])
            b = np.minimum(np.maximum(np.ceil(dist / 100).astype(int), 1), nbin) - 1

            covered = np.cumsum(np.bincount(b, weights=pop, minlength=nbin))[:-1]
            curves.append(pd.DataFrame({"scenario": scen, "group": group, "n_facilities": len(f),
                                        "radius_km": RADII_KM, "pop": covered,
                                        "pct": covered / total * 100}))

            pc = np.bincount(prov_idx * nbin + b, weights=pop, minlength=len(prov_codes) * nbin)
            pc = np.cumsum(pc.reshape(len(prov_codes), nbin), axis=1)[:, :-1]
            provs.append(pd.DataFrame({
                "scenario": scen, "group": group,
                "radius_km": np.tile(RADII_KM, len(prov_codes)),
                "prov": np.repeat(prov_codes, len(RADII_KM)), "pop_covered": pc.ravel()}))
            if group in ("all", "P"):
                fa = np.bincount(idx * nbin + b, weights=pop, minlength=len(pts) * nbin)
                fa = np.cumsum(fa.reshape(len(pts), nbin), axis=1)[:, :-1]
                fa = fa[f["pt"]] / n_at[f["pt"], None]
                alloc.append(pd.DataFrame({
                    "scenario": scen, "group": group,
                    "radius_km": np.tile(RADII_KM, len(f)),
                    "hcode": np.repeat(f["hcode"].to_numpy(), len(RADII_KM)), "pop": fa.ravel()}))

            g = grid[["prov", "tambon", "h3"]].assign(pop=pop, d=dist)

            if scen != "live_replica":
                g["wd"] = g["pop"] * g["d"]
                for k in ACCESS_KM:
                    g[f"in{k}"] = g["pop"] * (g["d"] <= k * 1000)
                cols = ["pop", "wd"] + [f"in{k}" for k in ACCESS_KM]
                for key, out in [("tambon", tambon), ("h3", hexes)]:
                    a = g.groupby(key)[cols].sum()
                    a["mean_km"] = a["wd"] / a["pop"] / 1000
                    for k in ACCESS_KM:
                        a[f"share_{k}km"] = a[f"in{k}"] / a["pop"]
                    a = a.drop(columns=["wd"] + [f"in{k}" for k in ACCESS_KM]).reset_index()
                    out.append(a.assign(scenario=scen, group=group))
            print(f"{scen:13s} {group:10s} {len(f):6,d} units  within 1 km {covered[9] / 1e6:5.1f} M "
                  f"({covered[9] / total:.1%})  within 5 km {covered[49] / 1e6:5.1f} M ({covered[49] / total:.1%})")

    pd.concat(curves).to_csv(PROCESSED / "coverage_curve.csv", index=False)
    pp = grid.groupby("prov")["pop"].sum().rename("pop_resident")
    pr = pd.concat(provs).merge(pp, left_on="prov", right_index=True)
    pr["pct"] = pr["pop_covered"] / pr["pop_resident"] * 100
    pr.to_csv(PROCESSED / "coverage_province.csv", index=False)
    pd.concat(alloc).to_parquet(PROCESSED / "coverage_facility.parquet", index=False)
    pd.concat(tambon).to_csv(PROCESSED / "access_tambon.csv", index=False)
    pd.concat(hexes).to_csv(PROCESSED / "access_h3r7.csv", index=False)
    print(f"Saved: {PROCESSED}/coverage_curve.csv, coverage_province.csv, coverage_facility.parquet, "
          f"access_tambon.csv, access_h3r7.csv")


if __name__ == "__main__":
    main()
