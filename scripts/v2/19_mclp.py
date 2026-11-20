"""
Task A5 (RQ3): if k new units were placed to reach as many uncovered people as
possible, how many more people would be within reach than the market managed,
and where should the next public units go?

Maximal covering location problem (Church & ReVelle 1974), solved greedily:
each step adds the candidate site that covers the most people still beyond the
radius. Greedy reaches at least 1 - 1/e (about 63%) of the best possible
coverage, and usually much more. Straight-line radii 2 and 5 km, as in A1.

Candidate sites: the population-weighted centre of the uncovered people in each
H3 resolution-8 hexagon (about 0.7 km2) that has any.

  benchmark  start from the "before" network (14_change_over_time.py) and place
             k = the number of innovation units that entered in FY2024-25;
             compare with the people those real entrants newly reached
  policy     start from the current network and place 50, 100 or 500 new units

Input:
  data/interim/v2/popgrid.parquet, data/processed/v2/facilities.csv, a1_units.csv
Output (data/processed/v2/):
  a5_benchmark.csv  radius, k, people newly covered: greedy planner and the real entrants
  a5_curve.csv      people newly covered against k (planner), for each start and radius
  a5_sites.csv      the first 500 sites from the current network, in order, with the
                    people each adds, tambon and province
"""
import heapq

import h3
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from config import EARTH_RADIUS_M, FACILITIES, POPGRID, PROCESSED
from geo import nearest, to_xyz

RADII_KM = [2, 5]
POLICY_K = [50, 100, 500]
CHUNK = 5000


def coverage_lists(cand_xyz, cell_xyz, r_m):
    """CSR-style arrays: for each candidate, the cells within r_m."""
    chord = 2 * np.sin(r_m / EARTH_RADIUS_M / 2)
    cell_tree = cKDTree(cell_xyz)
    rows, cols = [], []
    for s in range(0, len(cand_xyz), CHUNK):
        m = cKDTree(cand_xyz[s:s + CHUNK]).sparse_distance_matrix(cell_tree, chord, output_type="coo_matrix")
        rows.append(m.row + s)
        cols.append(m.col)
    rows, cols = np.concatenate(rows), np.concatenate(cols)
    o = np.argsort(rows, kind="stable")
    ptr = np.concatenate([[0], np.cumsum(np.bincount(rows, minlength=len(cand_xyz)))])
    return ptr, cols[o]


def greedy(ptr, cols, w, k):
    """Lazy greedy maximal covering: order of picks and the people each adds."""
    covered = np.zeros(len(w), dtype=bool)
    gains = np.add.reduceat(w[cols], ptr[:-1]) if len(cols) else np.zeros(len(ptr) - 1)
    gains[ptr[:-1] == ptr[1:]] = 0
    heap = [(-g, j) for j, g in enumerate(gains) if g > 0]
    heapq.heapify(heap)
    picks, added = [], []
    while heap and len(picks) < k:
        g, j = heapq.heappop(heap)
        c = cols[ptr[j]:ptr[j + 1]]
        fresh = w[c][~covered[c]].sum()
        if heap and fresh < -heap[0][0] - 1e-9:
            if fresh > 0:
                heapq.heappush(heap, (-fresh, j))
            continue
        if fresh <= 0:
            break
        covered[c] = True
        picks.append(j)
        added.append(fresh)
    return np.array(picks, dtype=int), np.array(added)


def main():
    grid = pd.read_parquet(POPGRID)
    pop = grid["pop"].to_numpy()
    fac = pd.read_csv(FACILITIES, dtype={"hcode": str}, low_memory=False).dropna(subset=["lon", "lat"])
    fac = fac[fac["public_access"]]
    fac["fy0"] = pd.to_numeric(fac["datein_fyear_2025"], errors="coerce")
    p25 = fac[fac["in_2025"] & fac["type_2025"].eq("P")]
    inn25 = fac[fac["in_2025"] & fac["type_2025"].ne("P")]
    before = pd.concat([p25, inn25[inn25["fy0"] <= 2023]])
    entrants = inn25[inn25["fy0"].isin([2024, 2025])]
    current = fac[fac["in_2026"]]
    _, d_before = nearest(grid["lon"], grid["lat"], before["lon"], before["lat"])
    _, d_after = nearest(grid["lon"], grid["lat"], pd.concat([before, entrants])["lon"],
                         pd.concat([before, entrants])["lat"])
    _, d_cur = nearest(grid["lon"], grid["lat"], current["lon"], current["lat"])
    r8 = np.array([h3.latlng_to_cell(y, x, 8) for x, y in zip(grid["lon"], grid["lat"])])

    bench, curves, sites = [], [], []
    for start, d_start, k_max in [("before", d_before, len(entrants)), ("current_2026", d_cur, max(POLICY_K))]:
        for r in RADII_KM:
            unc = d_start > r * 1000
            u = grid[unc].assign(h=r8[unc], wx=lambda t: t["lon"] * t["pop"], wy=lambda t: t["lat"] * t["pop"])
            c = u.groupby("h")[["pop", "wx", "wy"]].sum()
            c = c[c["pop"] > 0]
            cand = pd.DataFrame({"h3r8": c.index, "lon": c["wx"] / c["pop"], "lat": c["wy"] / c["pop"]})
            ptr, cols = coverage_lists(to_xyz(cand["lon"], cand["lat"]), to_xyz(u["lon"], u["lat"]), r * 1000)
            picks, added = greedy(ptr, cols, u["pop"].to_numpy(), k_max)
            cum = np.cumsum(added)
            curves.append(pd.DataFrame({"start": start, "radius_km": r, "k": np.arange(1, len(cum) + 1),
                                        "people_newly_covered": cum}))
            uncovered_total = u["pop"].sum()
            if start == "before":
                real = pop[(d_before > r * 1000) & (d_after <= r * 1000)].sum()
                kk = len(entrants)
                plan = cum[min(kk, len(cum)) - 1]
                bench.append({"radius_km": r, "k": kk, "people_uncovered_before": uncovered_total,
                              "real_entrants": real, "greedy_planner": plan,
                              "planner_sites_needed_to_match_real": int(np.searchsorted(cum, real) + 1),
                              "real_over_planner": real / plan})
                print(f"benchmark r={r} km: {kk:,} entrants newly reached {real / 1e6:.2f} M; "
                      f"greedy planner with the same k {plan / 1e6:.2f} M "
                      f"(real = {real / plan:.0%}); planner matches the real gain with "
                      f"{bench[-1]['planner_sites_needed_to_match_real']:,} sites")
            else:
                for k in POLICY_K:
                    print(f"policy r={r} km: {k:4d} new units cover {cum[min(k, len(cum)) - 1] / 1e6:.2f} M "
                          f"of {uncovered_total / 1e6:.2f} M people now beyond {r} km")
                s = cand.iloc[picks[:max(POLICY_K)]].reset_index(drop=True)
                s["order"] = np.arange(1, len(s) + 1)
                s["people_added"] = added[:len(s)]
                k_cell, _ = nearest(s["lon"], s["lat"], grid["lon"], grid["lat"])
                s["tambon"] = grid["tambon"].to_numpy()[k_cell]
                s["prov"] = s["tambon"].str[:2]
                sites.append(s.assign(radius_km=r))

    pd.DataFrame(bench).to_csv(PROCESSED / "a5_benchmark.csv", index=False)
    pd.concat(curves).to_csv(PROCESSED / "a5_curve.csv", index=False)
    sites = pd.concat(sites)
    sites.to_csv(PROCESSED / "a5_sites.csv", index=False)
    top = sites[(sites["radius_km"] == 2) & (sites["order"] <= 100)]["prov"].value_counts().head(8)
    print("\nprovinces with most of the first 100 sites (2 km):", top.to_dict())
    print(f"Saved: {PROCESSED}/a5_benchmark.csv, a5_curve.csv, a5_sites.csv")


if __name__ == "__main__":
    main()
