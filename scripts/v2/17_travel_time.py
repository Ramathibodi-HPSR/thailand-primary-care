"""
Task A3, step 2: travel time (minutes) from every populated 100 m cell to the
nearest facility over the road graph from 16_road_graph.py, by motorised
transport (road class speeds) and on foot (every road at walking speed).

Each cell and each facility joins the graph at its nearest road node; that
last stretch off the road counts at walking speed (OFF_ROAD_KMH). One Dijkstra
run per network, from a virtual source joined to every facility's node, gives
every node its time to the nearest facility.

Networks: current (2026-11-20) all units, class P, class P public, innovation
units; peak (2025-11-29) and before, as in 14_change_over_time.py. All without
prison health units (D2) and class Z (D3).

Input:
  data/interim/v2/popgrid.parquet, road_nodes.parquet, road_edges.parquet,
  data/processed/v2/facilities.csv, PC_GIS/data/raw/rwi/tha_relative_wealth_index.csv
Output (data/processed/v2/):
  a3_time_curve.csv    share of people within 0-120 minutes, by network, group and mode
  a3_quintiles.csv     share within 5, 10, 15, 30, 60 minutes by wealth quintile (as in 15_wealth_equity.py)
  a3_tambon.csv        per tambon: people, mean minutes, share within 5-60 minutes
  a3_h3r7.csv          the same per H3 resolution-7 hexagon
  a3_check.csv         straight-line km against road minutes, for a sanity check
"""
import time

import h3
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree

from config import FACILITIES, INTERIM, POPGRID, PROCESSED, RWI_CSV
from geo import chord_to_m, nearest, to_xyz

OFF_ROAD_KMH = 5
MINUTES = [5, 10, 15, 30, 60]


def snap(tree, lon, lat):
    chord, idx = tree.query(to_xyz(lon, lat), k=1, workers=-1)
    return idx, chord_to_m(chord) / 1000 / OFF_ROAD_KMH * 60


def main():
    t0 = time.time()
    nodes = pd.read_parquet(INTERIM / "road_nodes.parquet")
    edges = pd.read_parquet(INTERIM / "road_edges.parquet")
    n = len(nodes)
    tree = cKDTree(to_xyz(nodes["lon"], nodes["lat"]))

    grid = pd.read_parquet(POPGRID)
    pop = grid["pop"].to_numpy()
    total = pop.sum()
    cell_node, cell_off = snap(tree, grid["lon"], grid["lat"])

    fac = pd.read_csv(FACILITIES, dtype={"hcode": str}, low_memory=False).dropna(subset=["lon", "lat"])
    fac = fac[fac["public_access"]]
    fac["fy0"] = pd.to_numeric(fac["datein_fyear_2025"], errors="coerce")
    p25 = fac[fac["in_2025"] & fac["type_2025"].eq("P")]
    inn25 = fac[fac["in_2025"] & fac["type_2025"].ne("P")]
    cur = fac[fac["in_2026"]]
    nets = {
        ("current_2026", "all"): cur,
        ("current_2026", "P"): cur[cur["type_2026"].eq("P")],
        ("current_2026", "P_public"): cur[cur["type_2026"].eq("P") & cur["ownership_2026"].eq("public")],
        ("current_2026", "innovation"): cur[cur["type_2026"].ne("P")],
        ("peak_2025", "all"): fac[fac["in_2025"]],
        ("before", "all"): pd.concat([p25, inn25[inn25["fy0"] <= 2023]]),
    }

    # Graph plus one extra node (index n) for the virtual source; a tiny constant
    # keeps zero-length edges as edges
    base_u = edges["u"].to_numpy()
    base_v = edges["v"].to_numpy()
    base_w = edges["minutes"].to_numpy().astype(float) + 1e-6
    print(f"graph and snapping ready in {time.time() - t0:.0f}s; median cell off-road "
          f"{np.median(cell_off):.1f} min, 95th percentile {np.percentile(cell_off, 95):.1f} min")

    # Wealth quintiles, the same rule as 15_wealth_equity.py
    rwi = pd.read_csv(RWI_CSV)
    j, d = nearest(grid["lon"], grid["lat"], rwi["longitude"], rwi["latitude"])
    r = np.where(d <= 3000, rwi["rwi"].to_numpy()[j], np.nan)
    order = np.argsort(np.where(np.isnan(r), np.inf, r), kind="stable")
    ok = ~np.isnan(r[order])
    w = pop[order][ok]
    rank = (np.cumsum(w) - w / 2) / w.sum()
    q = np.zeros(len(grid), dtype=int)
    q[order[ok]] = np.minimum((rank * 5).astype(int), 4) + 1

    grid["h3"] = [h3.latlng_to_cell(y, x, 7) for x, y in zip(grid["lon"], grid["lat"])]
    base_km = edges["km"].to_numpy().astype(float) + 1e-9
    curve, quint, tam, hexes = [], [], [], []

    def summarise(net, group, mode, n_units, minutes):
        reach = np.isfinite(minutes)
        m = np.where(reach, minutes, np.inf)
        steps = np.arange(0, 121)
        o = np.argsort(m)
        cum = np.concatenate([[0.0], np.cumsum(pop[o])])
        within = cum[np.searchsorted(m[o], steps, side="right")]
        curve.append(pd.DataFrame({"network": net, "group": group, "mode": mode, "n_units": n_units,
                                   "minutes": steps, "pop": within, "pct": within / total * 100}))
        cells = pd.DataFrame({"pop": pop, "q": q, "tambon": grid["tambon"], "h3": grid["h3"]})
        for k in MINUTES:
            cells[f"in{k}"] = pop * (m <= k)
        cells["wm"] = pop * np.where(reach, minutes, 0)
        cells["pop_reach"] = pop * reach
        cols = ["pop", "pop_reach", "wm"] + [f"in{k}" for k in MINUTES]
        a = cells[cells["q"] > 0].groupby("q")[cols].sum()
        out = pd.DataFrame({"network": net, "group": group, "mode": mode, "quintile": a.index,
                            "people": a["pop"], "mean_minutes": a["wm"] / a["pop_reach"]})
        for k in MINUTES:
            out[f"share_{k}min"] = a[f"in{k}"] / a["pop"]
        quint.append(out)
        if net == "current_2026":
            for key, sink in [("tambon", tam), ("h3", hexes)]:
                a = cells.groupby(key)[cols].sum()
                a["mean_minutes"] = a["wm"] / a["pop_reach"]
                for k in MINUTES:
                    a[f"share_{k}min"] = a[f"in{k}"] / a["pop"]
                sink.append(a[["pop", "mean_minutes"] + [f"share_{k}min" for k in MINUTES]]
                            .reset_index().assign(group=group, mode=mode))
        print(f"{net:13s} {group:10s} {mode:9s} {n_units:6,d} units  " +
              "  ".join(f"<= {k} min {pop[m <= k].sum() / total:.1%}" for k in MINUTES) +
              f"  unreachable {pop[~reach].sum() / total:.2%}")

    for (net, group), f in nets.items():
        f_node, f_off = snap(tree, f["lon"], f["lat"])
        g = pd.DataFrame({"node": f_node, "off": f_off}).groupby("node")["off"].min()
        u = np.concatenate([base_u, np.full(len(g), n)])
        v = np.concatenate([base_v, g.index.to_numpy()])
        # Motorised: road minutes at the class speeds; walking: road km at OFF_ROAD_KMH
        for mode, w_road, w_src, to_min in [
            ("motorised", base_w, g.to_numpy() + 1e-6, lambda d: d + cell_off),
            ("walking", base_km, g.to_numpy() / 60 * OFF_ROAD_KMH + 1e-9,
             lambda d: d / OFF_ROAD_KMH * 60 + cell_off),
        ]:
            graph = coo_matrix((np.concatenate([w_road, w_src]), (u, v)), shape=(n + 1, n + 1)).tocsr()
            minutes = to_min(dijkstra(graph, directed=False, indices=n)[cell_node])
            summarise(net, group, mode, len(f), minutes)
            if (net, group, mode) == ("current_2026", "all", "motorised"):
                _, km = nearest(grid["lon"], grid["lat"], f["lon"], f["lat"])
                s = np.random.default_rng(1).choice(len(grid), 20000, replace=False, p=pop / total)
                pd.DataFrame({"km_straight": km[s] / 1000, "minutes_road": minutes[s]}) \
                    .to_csv(PROCESSED / "a3_check.csv", index=False)

    pd.concat(curve).to_csv(PROCESSED / "a3_time_curve.csv", index=False)
    qq = pd.concat(quint)
    qq.to_csv(PROCESSED / "a3_quintiles.csv", index=False)
    pd.concat(tam).to_csv(PROCESSED / "a3_tambon.csv", index=False)
    pd.concat(hexes).to_csv(PROCESSED / "a3_h3r7.csv", index=False)
    for mode, k in [("motorised", 5), ("motorised", 10), ("walking", 30)]:
        print(f"\n{mode}: share within {k} min by wealth quintile (Q1 poorest):")
        print(qq[qq["mode"] == mode].pivot_table(index=["network", "group"], columns="quintile",
                                                 values=f"share_{k}min").round(3).to_string())
    print("\nmean minutes by wealth quintile:")
    print(qq.pivot_table(index=["mode", "network", "group"], columns="quintile",
                         values="mean_minutes").round(1).to_string())
    print(f"\nSaved a3_* in {PROCESSED} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
