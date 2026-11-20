"""
Task A4 (RQ2): are there enough units for the people around them, when each
unit can serve only so many? Gaussian two-step floating catchment area index
(the Gaussian form of E2SFCA, Luo & Qi 2009; Dai 2010), every unit with the
same capacity, read as units per 10,000 people.

  step 1  R_j = 1 / sum_i P_i W(d_ij)           for cells i within d0 of unit j
  step 2  A_i = sum_j R_j W(d_ij) * 10,000      for units j within d0 of cell i
  W(d) = (exp(-d^2 / 2 d0^2) - exp(-1/2)) / (1 - exp(-1/2)) for d <= d0, else 0

Distance is straight-line, not road time. The plan asked for a 30-minute road
catchment, but A3 found that almost everyone is within 15 minutes by motor
vehicle, so a 30-minute catchment would smooth the whole country into one
number, and road times between every unit and every cell would take hours on
this machine. A 2 km catchment is close to a 30-minute walk on roads (A3) and
5 km to a short drive. A road-time version is later work.

Networks as in 14_change_over_time.py (no prison units, no class Z).

Input:
  data/interim/v2/popgrid.parquet, data/processed/v2/facilities.csv,
  PC_GIS/data/raw/rwi/tha_relative_wealth_index.csv
Output (data/processed/v2/):
  a4_summary.csv    per network, group and catchment: mean index, people with index 0,
                    concentration index across wealth, mean by wealth quintile
  a4_tambon.csv     population-weighted mean index per tambon (current network)
  a4_h3r7.csv       the same per H3 resolution-7 hexagon
"""
import h3
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from config import EARTH_RADIUS_M, FACILITIES, POPGRID, PROCESSED, RWI_CSV
from geo import chord_to_m, nearest, to_xyz

CATCHMENT_KM = [2, 5]
CHUNK = 2000


def gaussian(d, d0):
    e = np.exp(-0.5)
    return np.where(d <= d0, (np.exp(-0.5 * (d / d0) ** 2) - e) / (1 - e), 0.0)


def pairs(cell_tree, fac_xyz, d0_m):
    """All (unit, cell, metres) pairs within d0, built in chunks of units."""
    chord = 2 * np.sin(d0_m / EARTH_RADIUS_M / 2)
    fj, ci, dm = [], [], []
    for s in range(0, len(fac_xyz), CHUNK):
        t = cKDTree(fac_xyz[s:s + CHUNK])
        m = t.sparse_distance_matrix(cell_tree, chord, output_type="coo_matrix")
        fj.append(m.row + s)
        ci.append(m.col)
        dm.append(chord_to_m(m.data))
    return np.concatenate(fj), np.concatenate(ci), np.concatenate(dm)


def concentration_index(h, w, rank):
    mu = np.average(h, weights=w)
    return 2 * np.average((h - mu) * (rank - 0.5), weights=w) / mu


def main():
    grid = pd.read_parquet(POPGRID)
    pop = grid["pop"].to_numpy()
    cell_tree = cKDTree(to_xyz(grid["lon"], grid["lat"]))

    rwi = pd.read_csv(RWI_CSV)
    j, d = nearest(grid["lon"], grid["lat"], rwi["longitude"], rwi["latitude"])
    r = np.where(d <= 3000, rwi["rwi"].to_numpy()[j], np.nan)
    has = ~np.isnan(r)
    order = np.argsort(r[has], kind="stable")
    w = pop[has][order]
    rank = np.empty(has.sum())
    rank[order] = (np.cumsum(w) - w / 2) / w.sum()
    q = np.minimum((rank * 5).astype(int), 4) + 1

    fac = pd.read_csv(FACILITIES, dtype={"hcode": str}, low_memory=False).dropna(subset=["lon", "lat"])
    fac = fac[fac["public_access"]]
    fac["fy0"] = pd.to_numeric(fac["datein_fyear_2025"], errors="coerce")
    p25 = fac[fac["in_2025"] & fac["type_2025"].eq("P")]
    inn25 = fac[fac["in_2025"] & fac["type_2025"].ne("P")]
    cur = fac[fac["in_2026"]]
    nets = {
        ("before", "all"): pd.concat([p25, inn25[inn25["fy0"] <= 2023]]),
        ("peak_2025", "all"): fac[fac["in_2025"]],
        ("current_2026", "all"): cur,
        ("current_2026", "P"): cur[cur["type_2026"].eq("P")],
    }

    grid["h3"] = [h3.latlng_to_cell(y, x, 7) for x, y in zip(grid["lon"], grid["lat"])]
    rows, tam, hexes = [], [], []
    for d0 in CATCHMENT_KM:
        for (net, group), f in nets.items():
            fj, ci, dm = pairs(cell_tree, to_xyz(f["lon"], f["lat"]), d0 * 1000)
            wt = gaussian(dm, d0 * 1000)
            demand = np.bincount(fj, weights=pop[ci] * wt, minlength=len(f))
            ratio = np.divide(1.0, demand, out=np.zeros(len(f)), where=demand > 0)
            a = np.bincount(ci, weights=ratio[fj] * wt, minlength=len(grid)) * 1e4
            row = {"network": net, "group": group, "catchment_km": d0, "n_units": len(f),
                   "pairs": len(fj), "mean_index": np.average(a, weights=pop),
                   "units_per_10k_national": len(f) / pop.sum() * 1e4,
                   "people_index_0": pop[a == 0].sum() / pop.sum(),
                   "concentration_index": concentration_index(a[has], pop[has], rank)}
            for k in range(1, 6):
                m = q == k
                row[f"mean_Q{k}"] = np.average(a[has][m], weights=pop[has][m])
            rows.append(row)
            if net == "current_2026":
                c = pd.DataFrame({"pop": pop, "wa": pop * a, "tambon": grid["tambon"], "h3": grid["h3"]})
                for key, sink in [("tambon", tam), ("h3", hexes)]:
                    g = c.groupby(key)[["pop", "wa"]].sum()
                    sink.append(pd.DataFrame({key: g.index, "pop": g["pop"].to_numpy(),
                                              "index_per_10k": (g["wa"] / g["pop"]).to_numpy(),
                                              "group": group, "catchment_km": d0}))
            print(f"d0 {d0} km {net:13s} {group:4s} {len(f):6,d} units  pairs {len(fj):11,d}  "
                  f"mean {row['mean_index']:.2f} per 10k  index 0 for {row['people_index_0']:.1%}  "
                  f"CI {row['concentration_index']:+.3f}")
    out = pd.DataFrame(rows)
    out.to_csv(PROCESSED / "a4_summary.csv", index=False)
    pd.concat(tam).to_csv(PROCESSED / "a4_tambon.csv", index=False)
    pd.concat(hexes).to_csv(PROCESSED / "a4_h3r7.csv", index=False)
    print("\nmean index (units per 10,000 people) by wealth quintile (Q1 poorest):")
    print(out.set_index(["catchment_km", "network", "group"])[[f"mean_Q{k}" for k in range(1, 6)]].round(2).to_string())
    print(f"\nSaved: {PROCESSED}/a4_summary.csv, a4_tambon.csv, a4_h3r7.csv")


if __name__ == "__main__":
    main()
