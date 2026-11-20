"""
Build the data bundle for dashboard v2 from the outputs of steps 10-19.
Everything the page shows comes from here, including every number in its
text (facts.json), so the prose cannot drift from the data.

Published data are aggregates (H3 resolution-6 hexagons, about 36 km2) and the
public NHSO facility registry; no data about individuals.

Input:
  data/interim/v2/popgrid.parquet, the adm3 boundaries (names only)
  data/processed/v2/facilities.csv and the a1_* to a5_* tables
Output (dashboard/data/):
  facts.json   every number quoted in the page text
  charts.json  series for the charts
  hex6.json    per H3 r6 hexagon: people, wealth rank, share within 2 km (before, peak,
               current), walking minutes, availability index, people beyond 2 km
  units.csv    facilities in the 2025 or 2026 network (no prison units): type t, ownership o,
               entry fiscal year y, status s (c current, z moved to class Z, x left otherwise),
               a = in the 2025 snapshot, c = coordinate is a tambon or district centroid
  sites.csv    the first 500 sites a planner would add to the current network (2 km)
  provinces.geojson  simplified province outlines, drawn as the map background so the
                     page needs no tile service
"""
import json

import h3
import numpy as np
import geopandas as gpd
import pandas as pd
import pyogrio

from config import ADM1_SHP, ADM3_SHP, FACILITIES, POPGRID, PROCESSED, ROOT, RWI_CSV
from geo import nearest

OUT = ROOT / "dashboard" / "data"


def r(x, nd=1):
    return float(round(float(x), nd))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    grid = pd.read_parquet(POPGRID)
    pop = grid["pop"].to_numpy()
    total = pop.sum()
    fac = pd.read_csv(FACILITIES, dtype={"hcode": str, "catm6": str}, low_memory=False).dropna(subset=["lon", "lat"])
    fac = fac[fac["public_access"]]
    fac["fy0"] = pd.to_numeric(fac["datein_fyear_2025"], errors="coerce")
    p25 = fac[fac["in_2025"] & fac["type_2025"].eq("P")]
    inn25 = fac[fac["in_2025"] & fac["type_2025"].ne("P")]
    nets = {"before": pd.concat([p25, inn25[inn25["fy0"] <= 2023]]),
            "peak": fac[fac["in_2025"]], "current": fac[fac["in_2026"]]}

    # Hexagon layer
    rwi = pd.read_csv(RWI_CSV)
    j, d = nearest(grid["lon"], grid["lat"], rwi["longitude"], rwi["latitude"])
    rv = np.where(d <= 3000, rwi["rwi"].to_numpy()[j], np.nan)
    ok = ~np.isnan(rv)
    order = np.argsort(rv[ok], kind="stable")
    w = pop[ok][order]
    rank = np.full(len(grid), np.nan)
    rr = np.empty(ok.sum())
    rr[order] = (np.cumsum(w) - w / 2) / w.sum()
    rank[ok] = rr
    hx = pd.DataFrame({"h6": [h3.latlng_to_cell(y, x, 6) for x, y in zip(grid["lon"], grid["lat"])],
                       "pop": pop, "wrank": np.nan_to_num(rank) * pop, "pop_rank": ok * pop})
    for name, f in nets.items():
        _, dd = nearest(grid["lon"], grid["lat"], f["lon"], f["lat"])
        hx[f"in2_{name}"] = pop * (dd <= 2000)
    hx = hx.groupby("h6").sum()
    a3 = pd.read_csv(PROCESSED / "a3_h3r7.csv")
    a3 = a3[(a3["group"] == "all") & (a3["mode"] == "walking")]
    a3["h6"] = [h3.cell_to_parent(c, 6) for c in a3["h3"]]
    a3["wm"] = a3["mean_minutes"] * a3["pop"]
    walk = a3.groupby("h6")[["wm", "pop"]].sum()
    a4 = pd.read_csv(PROCESSED / "a4_h3r7.csv")
    a4 = a4[(a4["group"] == "all") & (a4["catchment_km"] == 2)]
    a4["h6"] = [h3.cell_to_parent(c, 6) for c in a4["h3"]]
    a4["wi"] = a4["index_per_10k"] * a4["pop"]
    avail = a4.groupby("h6")[["wi", "pop"]].sum()
    hexes = []
    for h, row in hx.iterrows():
        p = row["pop"]
        if p < 1:
            continue
        hexes.append([h, int(round(p)),
                      r(row["wrank"] / row["pop_rank"], 3) if row["pop_rank"] > 0 else None,
                      r(row["in2_before"] / p, 3), r(row["in2_peak"] / p, 3), r(row["in2_current"] / p, 3),
                      r(walk.loc[h, "wm"] / walk.loc[h, "pop"], 1) if h in walk.index else None,
                      r(avail.loc[h, "wi"] / avail.loc[h, "pop"], 2) if h in avail.index else None,
                      int(round(p - row["in2_current"]))])
    with open(OUT / "hex6.json", "w") as fh:
        json.dump({"columns": ["h3", "people", "wealth_rank", "within2_before", "within2_peak",
                               "within2_current", "walk_minutes", "availability", "beyond2_current"],
                   "rows": hexes}, fh, separators=(",", ":"))

    # Units
    a1 = pd.read_csv(PROCESSED / "a1_units.csv", dtype={"hcode": str})
    fate = a1.set_index("hcode")[["fate", "left_to"]]
    u = fac[fac["in_2025"] | fac["in_2026"]].copy()
    u["t"] = u["type_2026"].combine_first(u["type_2025"]).map(
        {"P": "P", "pharmacy": "ph", "nurse_clinic": "nc", "doctor_clinic": "dc"})
    u["y"] = u["fy0"].combine_first(pd.to_numeric(u["datein_fyear_2026"], errors="coerce").where(~u["in_2025"]))
    u = u.join(fate, on="hcode")
    u["s"] = np.select([u["in_2026"], u["left_to"].eq("class_Z")], ["c", "z"], "x")
    u["o"] = np.where(u["ownership"].eq("private"), "pri", "pub")
    u["c"] = u["xy_source"].str.contains("centroid").astype(int)
    u["a"] = u["in_2025"].astype(int)
    u = u[["hcode", "hname", "t", "o", "y", "s", "a", "c", "lat", "lon"]].copy()
    u["lat"], u["lon"] = u["lat"].round(5), u["lon"].round(5)
    u["y"] = u["y"].astype("Int64")
    u.to_csv(OUT / "units.csv", index=False)

    # Sites
    tam = pyogrio.read_dataframe(ADM3_SHP, read_geometry=False,
                                 columns=["ADM3_PCODE", "ADM3_EN", "ADM3_TH", "ADM2_EN", "ADM2_TH", "ADM1_EN", "ADM1_TH"])
    tam["tambon"] = tam["ADM3_PCODE"].str[2:]
    st = pd.read_csv(PROCESSED / "a5_sites.csv", dtype={"tambon": str})
    st = st[st["radius_km"] == 2].merge(tam, on="tambon", how="left")
    st = st[["order", "lat", "lon", "people_added", "ADM3_EN", "ADM2_EN", "ADM1_EN", "ADM3_TH", "ADM2_TH", "ADM1_TH"]]
    st["lat"], st["lon"], st["people_added"] = st["lat"].round(5), st["lon"].round(5), st["people_added"].round(0).astype(int)
    st.to_csv(OUT / "sites.csv", index=False)

    # Charts
    yc = pd.read_csv(PROCESSED / "a1_year_curve.csv")
    q2 = pd.read_csv(PROCESSED / "a2_quintiles.csv")
    q3 = pd.read_csv(PROCESSED / "a3_quintiles.csv")
    a4s = pd.read_csv(PROCESSED / "a4_summary.csv")
    tc = pd.read_csv(PROCESSED / "a3_time_curve.csv")
    mc = pd.read_csv(PROCESSED / "a5_curve.csv")
    au = pd.read_csv(PROCESSED / "a2_units.csv", index_col=0)
    cc = pd.read_csv(PROCESSED / "a2_curve.csv")
    con = pd.read_csv(PROCESSED / "a2_concentration.csv")
    bm = pd.read_csv(PROCESSED / "a5_benchmark.csv").set_index("radius_km")
    pe = pd.read_csv(PROCESSED / "a1_people.csv").set_index("radius_km")

    pol = mc[(mc["start"] == "current_2026") & (mc["radius_km"] == 2)]
    pol5 = pol[(pol["k"] % 5 == 0) | (pol["k"] == 1)]  # keeps k = 50, 100 and 500 exactly

    def qrow(df, net, group, col, extra=None):
        m = (df["network"] == net) & (df["group"] == group)
        if extra:
            for k, v in extra.items():
                m &= df[k] == v
        return [r(x * 100 if "share" in col else x, 1) for x in df[m].sort_values("quintile")[col]]

    # FY2020-24 from the entry-year series; the FY2025 point is the whole November 2025
    # network (the peak), so the chart and the text quote the same number. The series'
    # own FY2025 point leaves out the units with no entry year.
    years = pd.concat([yc[yc["network"].isin([f"FY{y}" for y in range(2020, 2025)])],
                       yc[yc["network"] == "peak_2025"].assign(network="FY2025")])
    charts = {
        "years": {"fy": [int(x[2:]) + 543 for x in years["network"]],
                  "units": years["n_units"].tolist(),
                  "within1": [r(x) for x in years["pct_1km"]], "within2": [r(x) for x in years["pct_2km"]],
                  "current_within1": r(yc.loc[yc["network"] == "current_2026", "pct_1km"].iloc[0]),
                  "current_within2": r(yc.loc[yc["network"] == "current_2026", "pct_2km"].iloc[0]),
                  "current_units": int(yc.loc[yc["network"] == "current_2026", "n_units"].iloc[0])},
        "units_per_100k": {c: [r(x) for x in au[c]] for c in au.columns},
        "within2_by_quintile": {k: qrow(q2, net, g, "share_2km") for k, (net, g) in {
            "before": ("before", "all"), "peak": ("peak_2025", "all"), "current": ("current_2026", "all"),
            "current_P": ("current_2026", "P"), "current_innovation": ("current_2026", "innovation")}.items()},
        "concentration_curve": {"x": [r(x, 3) for x in cc["cum_pop"]], "y": [r(x, 3) for x in cc["cum_within_2km"]]},
        "walk30_by_quintile": {k: qrow(q3, net, "all", "share_30min", {"mode": "walking"}) for k, net in
                               {"before": "before", "peak": "peak_2025", "current": "current_2026"}.items()},
        "walk_minutes_by_quintile": qrow(q3, "current_2026", "all", "mean_minutes", {"mode": "walking"}),
        "drive10_by_quintile": qrow(q3, "current_2026", "all", "share_10min", {"mode": "motorised"}),
        "time_curve": {mode: [r(x) for x in tc[(tc["network"] == "current_2026") & (tc["group"] == "all")
                                              & (tc["mode"] == mode)].sort_values("minutes")["pct"]]
                       for mode in ["motorised", "walking"]},
        "availability_by_quintile": {k: [r(a4s[(a4s["network"] == net) & (a4s["group"] == g) & (a4s["catchment_km"] == 2)]
                                           [f"mean_Q{i}"].iloc[0], 2) for i in range(1, 6)]
                                     for k, (net, g) in {"before": ("before", "all"), "peak": ("peak_2025", "all"),
                                                         "current": ("current_2026", "all"),
                                                         "current_P": ("current_2026", "P")}.items()},
        "planner": {"k": mc[(mc["start"] == "before") & (mc["radius_km"] == 2)]["k"].iloc[::50].tolist() + [int(bm.loc[2, "k"])],
                    "people": [r(x / 1e6, 3) for x in mc[(mc["start"] == "before") & (mc["radius_km"] == 2)]
                               ["people_newly_covered"].iloc[::50]] + [r(bm.loc[2, "greedy_planner"] / 1e6, 3)]},
        "policy": {"k": pol5["k"].tolist(), "people": [r(x / 1e6, 3) for x in pol5["people_newly_covered"]]},
    }
    with open(OUT / "charts.json", "w") as fh:
        json.dump(charts, fh, separators=(",", ":"))

    # Facts quoted in the text
    new = a1[a1["entry"] == "FY2024-25"]
    beyond2 = {t: (new.loc[new["type_2025"] == t, "km_to_P"] > 2).mean() * 100
               for t in ["nurse_clinic", "pharmacy", "doctor_clinic"]}
    _, dP = nearest(grid["lon"], grid["lat"], p25["lon"], p25["lat"])
    ci = con[con["radius_km"] == 2].set_index(["network", "group"])["concentration_index"]
    pol = mc[(mc["start"] == "current_2026") & (mc["radius_km"] == 2)].set_index("k")["people_newly_covered"]
    unc2 = pop[nearest(grid["lon"], grid["lat"], nets["current"]["lon"], nets["current"]["lat"])[1] > 2000].sum()
    facts = {
        "population_m": r(total / 1e6, 1),
        "units_current": int(len(nets["current"])),
        "units_P_current": int((nets["current"]["type_2026"] == "P").sum()),
        "units_innovation_2025": int(len(inn25)),
        "units_innovation_current": int((nets["current"]["type_2026"] != "P").sum()),
        "entrants": int(len(new)),
        "entrants_nurse": int((new["type_2025"] == "nurse_clinic").sum()),
        "entrants_pharmacy": int((new["type_2025"] == "pharmacy").sum()),
        "entrants_doctor": int((new["type_2025"] == "doctor_clinic").sum()),
        "left": int((a1["fate"] == "left").sum()),
        "moved_to_Z": int((a1["left_to"] == "class_Z").sum()),
        "entrants_left_pct_nurse": r((new.loc[new["type_2025"] == "nurse_clinic", "fate"] == "left").mean() * 100, 0),
        "entrants_left_pct_pharmacy": r((new.loc[new["type_2025"] == "pharmacy", "fate"] == "left").mean() * 100, 0),
        "within1_before": r(yc.loc[yc["network"] == "before", "pct_1km"].iloc[0]),
        "within1_peak": r(yc.loc[yc["network"] == "peak_2025", "pct_1km"].iloc[0]),
        "within1_current": r(yc.loc[yc["network"] == "current_2026", "pct_1km"].iloc[0]),
        "newly_within1_m": r(pe.loc[1, "newly_within_before_to_peak"] / 1e6, 2),
        "lost_within1_m": r(pe.loc[1, "lost_peak_to_current"] / 1e6, 2),
        "lost_share_of_gain_1km": r(pe.loc[1, "lost_peak_to_current"] / pe.loc[1, "newly_within_before_to_peak"] * 100, 0),
        "people_beyond2_P_pct": r(pop[dP > 2000].sum() / total * 100, 1),
        "entrants_beyond2_nurse_pct": r(beyond2["nurse_clinic"], 1),
        "entrants_beyond2_pharmacy_pct": r(beyond2["pharmacy"], 1),
        "entrants_beyond2_doctor_pct": r(beyond2["doctor_clinic"], 1),
        "nurse_per_100k_Q1": r(au.loc[1, "entrants_nurse_clinic"]), "nurse_per_100k_Q5": r(au.loc[5, "entrants_nurse_clinic"]),
        "pharmacy_per_100k_Q1": r(au.loc[1, "entrants_pharmacy"]), "pharmacy_per_100k_Q5": r(au.loc[5, "entrants_pharmacy"]),
        "within2_current_Q1": charts["within2_by_quintile"]["current"][0],
        "within2_current_Q5": charts["within2_by_quintile"]["current"][4],
        "ci2_before": r(ci[("before", "all")], 3), "ci2_peak": r(ci[("peak_2025", "all")], 3),
        "ci2_current": r(ci[("current_2026", "all")], 3),
        "drive15_current": charts["time_curve"]["motorised"][15],
        "walk30_current": charts["time_curve"]["walking"][30],
        "walk30_Q1": charts["walk30_by_quintile"]["current"][0], "walk30_Q5": charts["walk30_by_quintile"]["current"][4],
        "walk_minutes_Q1": charts["walk_minutes_by_quintile"][0], "walk_minutes_Q5": charts["walk_minutes_by_quintile"][4],
        "avail_Q1": charts["availability_by_quintile"]["current"][0], "avail_Q5": charts["availability_by_quintile"]["current"][4],
        "market_newly_within2_m": r(bm.loc[2, "real_entrants"] / 1e6, 2),
        "planner_newly_within2_m": r(bm.loc[2, "greedy_planner"] / 1e6, 2),
        "market_share_of_planner_pct": r(bm.loc[2, "real_over_planner"] * 100, 0),
        "planner_over_market": r(1 / bm.loc[2, "real_over_planner"], 1),
        "planner_sites_to_match": int(bm.loc[2, "planner_sites_needed_to_match_real"]),
        "beyond2_current_m": r(unc2 / 1e6, 2),
        "policy_100_m": r(pol.loc[100] / 1e6, 2), "policy_500_m": r(pol.loc[500] / 1e6, 2),
        "registry_old": "29 Nov 2025", "registry_new": "20 Nov 2026",
        "updated": pd.Timestamp.today().strftime("%-d %b %Y"),
    }
    with open(OUT / "facts.json", "w") as fh:
        json.dump(facts, fh, indent=1, ensure_ascii=False)
    prov = gpd.read_file(ADM1_SHP)[["ADM1_PCODE", "ADM1_EN", "ADM1_TH", "geometry"]]
    prov["geometry"] = prov.geometry.simplify(0.005, preserve_topology=True)
    prov.to_file(OUT / "provinces.geojson", driver="GeoJSON", COORDINATE_PRECISION=4)

    sizes = {p.name: p.stat().st_size for p in OUT.iterdir()}
    print(f"Saved: {OUT} " + ", ".join(f"{k} {v / 1e6:.2f} MB" for k, v in sorted(sizes.items())))
    print(json.dumps(facts, ensure_ascii=False))


if __name__ == "__main__":
    main()
