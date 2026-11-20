"""
Task A2 (RQ1, RQ2): how access, the units that entered in FY2024-25 and the
units that moved to class Z line up with how wealthy an area is.

Wealth is Meta's Relative Wealth Index (2.4 km tiles, April 2021). Each 100 m
populated cell takes the index of the nearest tile centre within 3 km; people
are then ranked by it and cut into fifths (Q1 = poorest fifth of people).
A tile's index has a standard error about as large as the spread between tiles,
so read quintiles and the concentration index, not single tiles.

Networks are the same as in 14_change_over_time.py: before, peak (2025-11-29),
current (2026-11-20), without prison health units (D2) and class Z (D3).

Input:
  data/interim/v2/popgrid.parquet, data/processed/v2/facilities.csv,
  data/processed/v2/a1_units.csv, PC_GIS/data/raw/rwi/tha_relative_wealth_index.csv
Output (data/processed/v2/):
  a2_quintiles.csv       per wealth quintile: people, share within 1, 2, 5 km and mean
                         distance, for each network and group
  a2_concentration.csv   concentration index of access (within 1, 2, 5 km) by network and group
  a2_curve.csv           concentration curve points (current network, within 2 km)
  a2_units.csv           units per 100,000 people by wealth quintile: class P, entrants
                         FY2024-25 by type, units that moved to class Z
  a2_people.csv          people newly within / losing 2 km by wealth quintile
"""
import numpy as np
import pandas as pd

from config import FACILITIES, POPGRID, PROCESSED, RWI_CSV
from geo import nearest

MAX_TILE_M = 3000
R_KM = [1, 2, 5]


def concentration_index(h, w, rank):
    """Standard concentration index of h over a fractional rank, with weights w."""
    mu = np.average(h, weights=w)
    return 2 * np.average((h - mu) * (rank - 0.5), weights=w) / mu


def main():
    grid = pd.read_parquet(POPGRID)
    rwi = pd.read_csv(RWI_CSV)
    j, d = nearest(grid["lon"], grid["lat"], rwi["longitude"], rwi["latitude"])
    grid["rwi"] = np.where(d <= MAX_TILE_M, rwi["rwi"].to_numpy()[j], np.nan)
    unmatched = grid.loc[grid["rwi"].isna(), "pop"].sum() / grid["pop"].sum()
    g = grid.dropna(subset=["rwi"]).sort_values("rwi").reset_index(drop=True)
    pop = g["pop"].to_numpy()
    cum = np.cumsum(pop)
    g["rank"] = (cum - pop / 2) / pop.sum()
    g["q"] = np.minimum((g["rank"] * 5).astype(int), 4) + 1
    total = pop.sum()

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
        ("current_2026", "P_public"): cur[cur["type_2026"].eq("P") & cur["ownership_2026"].eq("public")],
        ("current_2026", "innovation"): cur[cur["type_2026"].ne("P")],
    }

    quint, conc, dist = [], [], {}
    for (net, group), f in nets.items():
        _, dd = nearest(g["lon"], g["lat"], f["lon"], f["lat"])
        dist[(net, group)] = dd
        q = g.assign(d=dd)
        for r in R_KM:
            q[f"in{r}"] = q["pop"] * (q["d"] <= r * 1000)
            conc.append({"network": net, "group": group, "radius_km": r,
                         "share_within": q[f"in{r}"].sum() / total,
                         "concentration_index": concentration_index((dd <= r * 1000).astype(float), pop, g["rank"])})
        q["wd"] = q["pop"] * q["d"] / 1000
        a = q.groupby("q")[["pop", "wd"] + [f"in{r}" for r in R_KM]].sum()
        out = pd.DataFrame({"network": net, "group": group, "quintile": a.index, "people": a["pop"],
                            "mean_km": a["wd"] / a["pop"]})
        for r in R_KM:
            out[f"share_{r}km"] = a[f"in{r}"] / a["pop"]
        quint.append(out)
    pd.concat(quint).to_csv(PROCESSED / "a2_quintiles.csv", index=False)
    conc = pd.DataFrame(conc)
    conc.to_csv(PROCESSED / "a2_concentration.csv", index=False)

    # Concentration curve, current network, people within 2 km
    h = (dist[("current_2026", "all")] <= 2000) * pop
    pts = np.linspace(0, 1, 101)
    curve = pd.DataFrame({"cum_pop": pts,
                          "cum_within_2km": np.interp(pts, np.concatenate([[0], cum / total]),
                                                      np.concatenate([[0], np.cumsum(h) / h.sum()]))})
    curve.to_csv(PROCESSED / "a2_curve.csv", index=False)

    # Units per 100,000 people in each wealth quintile (the quintile of the cell nearest the unit)
    def unit_q(f):
        k, _ = nearest(f["lon"], f["lat"], g["lon"], g["lat"])
        return g["q"].to_numpy()[k]

    a1 = pd.read_csv(PROCESSED / "a1_units.csv", dtype={"hcode": str})
    new = a1[a1["entry"] == "FY2024-25"]
    sets = {"class_P_2025": p25, "moved_to_Z_by_2026": a1[a1["left_to"] == "class_Z"]}
    for t in ["nurse_clinic", "pharmacy", "doctor_clinic"]:
        sets[f"entrants_{t}"] = new[new["type_2025"] == t]
    people_q = g.groupby("q")["pop"].sum()
    units = pd.DataFrame({name: pd.Series(unit_q(f)).value_counts().reindex(range(1, 6), fill_value=0)
                          / people_q * 1e5 for name, f in sets.items()})
    units.index.name = "quintile"
    units.to_csv(PROCESSED / "a2_units.csv")

    m = 2000
    d0, d1, d2 = dist[("before", "all")], dist[("peak_2025", "all")], dist[("current_2026", "all")]
    ppl = pd.DataFrame({
        "people": people_q,
        "newly_within_2km": g[(d0 > m) & (d1 <= m)].groupby("q")["pop"].sum(),
        "lost_2km": g[(d1 <= m) & (d2 > m)].groupby("q")["pop"].sum(),
    }).fillna(0)
    ppl.index.name = "quintile"
    ppl.to_csv(PROCESSED / "a2_people.csv")

    print(f"people with no RWI tile within {MAX_TILE_M / 1000:.0f} km: {unmatched:.2%}")
    print("\nconcentration index (positive = access concentrated among the better-off):")
    print(conc.pivot_table(index=["network", "group"], columns="radius_km",
                           values="concentration_index").round(3).to_string())
    print("\nshare within 2 km by wealth quintile (Q1 poorest):")
    qq = pd.concat(quint)
    print(qq.pivot_table(index=["network", "group"], columns="quintile", values="share_2km").round(3).to_string())
    print("\nunits per 100,000 people by wealth quintile:\n" + units.round(1).to_string())
    print("\npeople (thousand) by wealth quintile:\n" + (ppl / 1e3).round(0).to_string())


if __name__ == "__main__":
    main()
