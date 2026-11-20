"""
Build one facility table from both NHSO registry snapshots, check every
coordinate against the tambon the registry gives, and pick one coordinate per
unit.

Coordinate rule, per unit:
  1. the 2026-11-20 snapshot coordinate, if it lies within 2 km of the unit's
     registered tambon;
  2. otherwise the 2025-11-29 coordinate, under the same test;
  3. otherwise the population-weighted centroid of the registered tambon,
     flagged as xy_source = "tambon_centroid".

Input:
  data/raw/hosmap_2025-11-29.csv, data/raw/hosmap_2026-11-20.csv
  data/interim/v2/tambons.parquet (from 10_population_grid.py)
Output:
  data/processed/v2/facilities.csv
"""
import geopandas as gpd
import numpy as np
import pandas as pd

from config import (FACILITIES, MAX_OFF_TAMBON_M, NO_PUBLIC_ACCESS, NOT_A_UNIT, PROCESSED, REGISTRY,
                    TAMBONS, TYPE_CODES, TYPE_ORDER, UTM)

KEEP_COLS = ["hcode", "hname", "hclass", "type", "ownership", "htype_name", "sub_type_name",
             "status_uc", "datein", "datein_fyear", "prov", "ampr", "tumbon", "catm6",
             "lon_raw", "lat_raw", "xy_check", "off_m"]


def classify(hclass):
    codes = {c.strip() for c in str(hclass).split(",")} if pd.notna(hclass) else set()
    for t in TYPE_ORDER:
        if codes & set(TYPE_CODES[t]):
            return t
    return None


def check_xy(df, tam):
    """Distance (m) from each coordinate to its registered tambon, and a label."""
    lon = pd.to_numeric(df["longitude"], errors="coerce")
    lat = pd.to_numeric(df["latitude"], errors="coerce")
    df["lon_raw"], df["lat_raw"] = lon, lat
    check = pd.Series("missing", index=df.index)
    has = lon.notna() & lat.notna()
    check[has & ((lon == 0) | (lat == 0))] = "zero"
    in_box = has & lat.between(5.5, 20.6) & lon.between(97.3, 105.7)
    check[has & ~in_box & (check != "zero")] = "outside_thailand"

    pts = gpd.GeoSeries(gpd.points_from_xy(lon.where(in_box), lat.where(in_box)),
                        index=df.index, crs="EPSG:4326").to_crs(UTM)
    poly = tam.set_index("tambon").geometry.to_crs(UTM)
    known = in_box & df["catm6"].isin(poly.index)
    off = pd.Series(np.nan, index=df.index)
    off[known] = pts[known].distance(gpd.GeoSeries(poly.loc[df.loc[known, "catm6"]].to_numpy(),
                                                   index=df.index[known], crs=UTM), align=True)

    # Registry tambon code not in the 2025 boundaries: test against the district
    districts = tam.assign(d=tam["tambon"].str[:4]).dissolve(by="d").geometry.to_crs(UTM)
    by_district = in_box & ~known & df["catm6"].str[:4].isin(districts.index)
    off[by_district] = pts[by_district].distance(
        gpd.GeoSeries(districts.loc[df.loc[by_district, "catm6"].str[:4]].to_numpy(),
                      index=df.index[by_district], crs=UTM), align=True)

    check[in_box & off.eq(0)] = "in_tambon"
    check[in_box & off.gt(0) & off.le(MAX_OFF_TAMBON_M)] = "near_tambon"
    check[in_box & off.gt(MAX_OFF_TAMBON_M)] = "off_tambon"
    check[in_box & off.isna()] = "no_tambon_code"
    check[by_district & off.le(MAX_OFF_TAMBON_M)] = "in_district"
    df["xy_check"], df["off_m"] = check, off.round(0)
    return df


def load(snapshot, path, tam):
    df = pd.read_csv(path, dtype=str, low_memory=False)
    df["hcode"] = df["hcode"].str.zfill(5)
    df["type"] = df["hclass"].map(classify)
    df = df[df["type"].notna()].copy()
    df["ownership"] = np.where(df["htype_name"].eq("เอกชน"), "private", "public")
    df["datein_fyear"] = pd.to_numeric(df["datein_fyear"], errors="coerce").astype("Int64")
    df = check_xy(df, tam)
    return df[KEEP_COLS].add_suffix(f"_{snapshot[:4]}").rename(columns={f"hcode_{snapshot[:4]}": "hcode"})


def main():
    PROCESSED.mkdir(parents=True, exist_ok=True)
    tam = gpd.read_parquet(TAMBONS)
    (s_old, p_old), (s_new, p_new) = sorted(REGISTRY.items())
    old, new = load(s_old, p_old, tam), load(s_new, p_new, tam)
    o, n = s_old[:4], s_new[:4]
    df = old.merge(new, on="hcode", how="outer")
    df[f"in_{o}"] = df[f"type_{o}"].notna()
    df[f"in_{n}"] = df[f"type_{n}"].notna()
    # Class Z in each raw snapshot (not kept as a type; see config.py)
    for s, path in ((o, p_old), (n, p_new)):
        raw = pd.read_csv(path, dtype=str, usecols=["hcode", "hclass"])
        z = raw[raw["hclass"].fillna("").str.split(",").apply(lambda c: "Z" in [i.strip() for i in c])]
        df[f"class_z_{s}"] = df["hcode"].isin(z["hcode"].str.zfill(5))

    # Descriptive fields from the latest snapshot that has the unit
    for c in ["hname", "type", "ownership", "htype_name", "sub_type_name", "datein",
              "datein_fyear", "prov", "ampr", "tumbon", "catm6"]:
        df[c] = df[f"{c}_{n}"].combine_first(df[f"{c}_{o}"])

    df["public_access"] = ~df["sub_type_name"].isin(NO_PUBLIC_ACCESS) & ~df["hcode"].isin(NOT_A_UNIT)

    ok = {"in_tambon", "near_tambon", "in_district"}
    use_new = df[f"xy_check_{n}"].isin(ok)
    use_old = ~use_new & df[f"xy_check_{o}"].isin(ok)
    df["xy_source"] = np.select([use_new, use_old], [f"registry_{s_new}", f"registry_{s_old}"],
                                "tambon_centroid")
    df["lon"] = np.where(use_new, df[f"lon_raw_{n}"], np.where(use_old, df[f"lon_raw_{o}"], np.nan))
    df["lat"] = np.where(use_new, df[f"lat_raw_{n}"], np.where(use_old, df[f"lat_raw_{o}"], np.nan))

    cent = tam.set_index("tambon")[["c_lon", "c_lat"]]
    fill = df["lon"].isna() & df["catm6"].isin(cent.index)
    df.loc[fill, "lon"] = cent.loc[df.loc[fill, "catm6"], "c_lon"].to_numpy()
    df.loc[fill, "lat"] = cent.loc[df.loc[fill, "catm6"], "c_lat"].to_numpy()
    # Tambon code unknown to the 2025 boundaries: centroid of its district instead
    dist = tam.assign(d=tam["tambon"].str[:4], wx=tam["c_lon"] * tam["pop"], wy=tam["c_lat"] * tam["pop"]) \
        .groupby("d")[["wx", "wy", "pop"]].sum()
    fill = df["lon"].isna() & df["catm6"].str[:4].isin(dist.index)
    d = dist.loc[df.loc[fill, "catm6"].str[:4]]
    df.loc[fill, "lon"] = (d["wx"] / d["pop"]).to_numpy()
    df.loc[fill, "lat"] = (d["wy"] / d["pop"]).to_numpy()
    df.loc[fill, "xy_source"] = "district_centroid"
    df.loc[df["lon"].isna(), "xy_source"] = "none"

    cols = ["hcode", "hname", "type", "ownership", "htype_name", "sub_type_name", "public_access", "datein",
            "datein_fyear", "prov", "ampr", "tumbon", "catm6", f"in_{o}", f"in_{n}",
            f"type_{o}", f"type_{n}", f"ownership_{o}", f"ownership_{n}", f"class_z_{o}", f"class_z_{n}",
            # datein is reset when a unit signs a new contract (most units still in
            # the 2026 snapshot have a 2026 datein), so keep each snapshot's own
            f"datein_{o}", f"datein_fyear_{o}", f"datein_{n}", f"datein_fyear_{n}",
            f"hclass_{o}", f"hclass_{n}", f"status_uc_{o}", f"status_uc_{n}",
            "lon", "lat", "xy_source", f"xy_check_{o}", f"off_m_{o}", f"xy_check_{n}", f"off_m_{n}",
            f"lon_raw_{o}", f"lat_raw_{o}", f"lon_raw_{n}", f"lat_raw_{n}"]
    df = df[cols].sort_values("hcode")
    df.to_csv(FACILITIES, index=False, encoding="utf-8")

    print(f"Saved: {FACILITIES} ({len(df):,} units in either snapshot)")
    for s in (o, n):
        sub = df[df[f"in_{s}"]]
        print(f"\n{s}: {len(sub):,} units\n" + pd.crosstab(sub["type"], sub[f"xy_check_{s}"]).to_string())
    print("\nchosen coordinate, by type (units in either snapshot):")
    print(pd.crosstab(df["type"], df["xy_source"]).to_string())


if __name__ == "__main__":
    main()
