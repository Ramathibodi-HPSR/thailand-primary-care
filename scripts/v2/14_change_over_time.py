"""
Task A1 (RQ1): where did the contracted private units (pharmacies, nurse
clinics, doctor clinics) go when they entered in fiscal years 2024-25
(ปีงบ 2567-68), and where did the ones that left by November 2026 sit?

Networks (coordinates from 11_facilities.py, the same for a unit in every network):
  before   class P of 2025 + innovation units of 2025 that entered by FY2023
  peak     every unit in the 2025-11-29 snapshot (the peak before units fell)
  current  every unit in the 2026-11-20 snapshot (the current network;
           class Z is not counted)
  FY2020..FY2025: class P of 2025 + innovation units of 2025 that entered by that year.
  Units that left before November 2025 are not in any snapshot, so the early
  years undercount (survivorship).

Entry year is datein_fyear of the 2025 snapshot: the Thai fiscal year in
Gregorian years (October starts the next year). The 2026 snapshot resets
datein when a unit signs a new contract, so it is not used for entry.

Input:
  data/interim/v2/popgrid.parquet, data/processed/v2/facilities.csv
Output (data/processed/v2/):
  a1_year_curve.csv  coverage by entry year and radius
  a1_units.csv       one row per 2025 innovation unit: entry year, stayed or left,
                     distance to the nearest class P unit and to the "before" network,
                     density quintile of where it sits
  a1_people.csv      people newly within r km (before -> peak), and losing it (peak -> current)
  a1_density.csv     units and people by population-density quintile
  a1_province.csv    the same counts by province
"""
import h3
import numpy as np
import pandas as pd

from config import FACILITIES, POPGRID, PROCESSED, RADII_KM
from geo import nearest

ENTRY_YEARS = [2024, 2025]          # ปีงบ 2567-68
SERIES_YEARS = list(range(2020, 2026))
R_KM = [1, 2, 5]


def dist_to(grid, f):
    _, d = nearest(grid["lon"], grid["lat"], f["lon"], f["lat"])
    return d


def curve(d, pop, total):
    out = {}
    for r in [0.5] + R_KM:
        out[f"pct_{r}km"] = pop[d <= r * 1000].sum() / total * 100
    return out


def main():
    grid = pd.read_parquet(POPGRID)
    pop = grid["pop"].to_numpy()
    total = pop.sum()
    fac = pd.read_csv(FACILITIES, dtype={"hcode": str, "catm6": str}, low_memory=False) \
        .dropna(subset=["lon", "lat"])
    fac = fac[fac["public_access"]]  # no prison health units
    fac["fy0"] = pd.to_numeric(fac["datein_fyear_2025"], errors="coerce")

    p25 = fac[fac["in_2025"] & fac["type_2025"].eq("P")]
    inn = fac[fac["in_2025"] & fac["type_2025"].ne("P")].copy()
    inn["fate"] = np.select([inn["in_2026"] & inn["type_2026"].ne("P"), inn["in_2026"]],
                            ["stayed", "became_P"], "left")
    # Why a unit left the R classes: most moved to class Z, which is not counted
    inn["left_to"] = np.where(inn["fate"] != "left", "",
                              np.where(inn["class_z_2026"], "class_Z", "other_or_none"))
    before = pd.concat([p25, inn[inn["fy0"] <= ENTRY_YEARS[0] - 1]])
    peak = fac[fac["in_2025"]]
    current = fac[fac["in_2026"]]

    d_p = dist_to(grid, p25)
    d_before, d_peak, d_cur = dist_to(grid, before), dist_to(grid, peak), dist_to(grid, current)

    # Coverage by entry year
    rows = []
    for y in SERIES_YEARS:
        net = pd.concat([p25, inn[inn["fy0"] <= y]])
        rows.append({"network": f"FY{y}", "n_units": len(net), **curve(dist_to(grid, net), pop, total)})
    for name, net, d in [("class_P_2025", p25, d_p), ("before", before, d_before),
                         ("peak_2025", peak, d_peak), ("current_2026", current, d_cur)]:
        rows.append({"network": name, "n_units": len(net), **curve(d, pop, total)})
    pd.DataFrame(rows).to_csv(PROCESSED / "a1_year_curve.csv", index=False)

    # Each innovation unit: how far from class P and from the network before the expansion
    _, inn["km_to_P"] = nearest(inn["lon"], inn["lat"], p25["lon"], p25["lat"])
    _, inn["km_to_before"] = nearest(inn["lon"], inn["lat"], before["lon"], before["lat"])
    inn[["km_to_P", "km_to_before"]] /= 1000
    # A unit already in the "before" network is its own nearest point there
    inn.loc[inn["fy0"] <= ENTRY_YEARS[0] - 1, "km_to_before"] = np.nan

    # Population-density quintiles: people per H3 r7 hexagon (about 5.2 km2),
    # cut so each quintile holds a fifth of the people
    grid["h3"] = [h3.latlng_to_cell(y, x, 7) for x, y in zip(grid["lon"], grid["lat"])]
    hex_pop = grid.groupby("h3")["pop"].sum().sort_values()
    cum = hex_pop.cumsum() / hex_pop.sum()
    hex_q = pd.Series(np.minimum((cum.to_numpy() * 5).astype(int), 4) + 1, index=hex_pop.index)
    grid["dq"] = grid["h3"].map(hex_q)
    inn["h3"] = [h3.latlng_to_cell(y, x, 7) for x, y in zip(inn["lon"], inn["lat"])]
    inn["dq"] = inn["h3"].map(hex_q).fillna(1).astype(int)  # a hexagon with no people is lowest
    inn["entry"] = np.select([inn["fy0"].isin(ENTRY_YEARS), inn["fy0"] < ENTRY_YEARS[0]],
                             ["FY2024-25", "before FY2024"], "unknown")
    inn[["hcode", "hname", "type_2025", "prov", "fy0", "entry", "fate", "left_to", "km_to_P", "km_to_before",
         "dq", "xy_source", "lon", "lat"]].to_csv(PROCESSED / "a1_units.csv", index=False)

    # People gaining access with the expansion, and losing it with the exits
    ppl = []
    for r in R_KM:
        m = r * 1000
        ppl.append({
            "radius_km": r,
            "newly_within_before_to_peak": pop[(d_before > m) & (d_peak <= m)].sum(),
            "lost_peak_to_current": pop[(d_peak <= m) & (d_cur > m)].sum(),
            "within_before": pop[d_before <= m].sum(),
            "within_peak": pop[d_peak <= m].sum(),
            "within_current": pop[d_cur <= m].sum(),
            "people_beyond_P": pop[d_p > m].sum(),
        })
    ppl = pd.DataFrame(ppl)
    ppl.to_csv(PROCESSED / "a1_people.csv", index=False)

    # Units by density quintile, against the people in each quintile (20% each)
    new = inn[inn["entry"] == "FY2024-25"]
    dens = pd.DataFrame({
        "people_share": grid.groupby("dq")["pop"].sum() / total,
        "class_P_2025": p25.assign(h3=[h3.latlng_to_cell(y, x, 7) for x, y in zip(p25["lon"], p25["lat"])])
            ["h3"].map(hex_q).fillna(1).astype(int).value_counts(normalize=True),
        "entrants_FY2024_25": new["dq"].value_counts(normalize=True),
        "left_by_2026": inn[inn["fate"] == "left"]["dq"].value_counts(normalize=True),
        "stayed": inn[inn["fate"] == "stayed"]["dq"].value_counts(normalize=True),
    }).fillna(0).sort_index()
    dens.index.name = "density_quintile"
    dens.to_csv(PROCESSED / "a1_density.csv")

    grid["prov"] = grid["tambon"].str[:2]
    prov = pd.DataFrame({
        "residents": grid.groupby("prov")["pop"].sum(),
        "newly_within_2km": grid[(d_before > 2000) & (d_peak <= 2000)].groupby("prov")["pop"].sum(),
        "lost_within_2km": grid[(d_peak <= 2000) & (d_cur > 2000)].groupby("prov")["pop"].sum(),
        "entrants_FY2024_25": new.assign(p=new["hcode"].map(fac.set_index("hcode")["catm6"]).str[:2])
            .groupby("p").size(),
        "left_by_2026": inn[inn["fate"] == "left"].assign(
            p=lambda d: d["hcode"].map(fac.set_index("hcode")["catm6"]).str[:2]).groupby("p").size(),
    }).fillna(0)
    prov.index.name = "prov"
    prov.to_csv(PROCESSED / "a1_province.csv")

    # Summary
    share_beyond = {r: pop[d_p > r * 1000].sum() / total for r in [2, 5]}
    print(f"2025 innovation units: {len(inn):,} (entered FY2024-25: {len(new):,}; left by Nov 2026: "
          f"{(inn['fate'] == 'left').sum():,}, of which to class Z {(inn['left_to'] == 'class_Z').sum():,})")
    print(pd.crosstab(inn["type_2025"], [inn["entry"], inn["fate"]]).to_string())
    for r in [2, 5]:
        print(f"\nbeyond {r} km of class P: people {share_beyond[r]:.1%} | entrants FY2024-25 "
              f"{(new['km_to_P'] > r).mean():.1%} | all left {(inn.loc[inn['fate'] == 'left', 'km_to_P'] > r).mean():.1%} "
              f"| stayed {(inn.loc[inn['fate'] == 'stayed', 'km_to_P'] > r).mean():.1%}")
    print("\nentrants FY2024-25, km to nearest class P: median "
          f"{new['km_to_P'].median():.2f}, km to the 'before' network: median {new['km_to_before'].median():.2f}")
    print("\npeople (million):\n" + (ppl.set_index("radius_km") / 1e6).round(2).to_string())
    print("\nshare by density quintile (1 = sparsest fifth of people):\n" + dens.round(3).to_string())
    print("\n" + pd.read_csv(PROCESSED / "a1_year_curve.csv").round(2).to_string(index=False))


if __name__ == "__main__":
    main()
