"""
Turn the WorldPop raster into a table of populated 100 m cells, each with the
tambon it lies in (OCHA/RTSD adm3, 2025-06-28).

Input:
  PC_GIS/Dashboard/Data/tha_ppp_2020_UNadj_constrained.tif
  PC_GIS/data/raw/tha_adm/tha_admbnda_adm3_rtsd_20250628.shp
Output:
  data/interim/v2/popgrid.parquet   one row per populated cell: lon, lat, pop, tambon (6 digits)
  data/interim/v2/tambons.parquet   one row per tambon: names, codes, population and
                                    population-weighted centroid
"""
import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize

from config import ADM3_SHP, INTERIM, POPGRID, TAMBONS, WORLDPOP_TIF
from geo import nearest


def main():
    INTERIM.mkdir(parents=True, exist_ok=True)

    with rasterio.open(WORLDPOP_TIF) as src:
        pop = src.read(1)
        nodata = src.nodata
        transform = src.transform
        shape = (src.height, src.width)

    rows, cols = np.nonzero((pop != nodata) & (pop > 0))
    values = pop[rows, cols].astype("float64")
    del pop
    lon, lat = rasterio.transform.xy(transform, rows, cols, offset="center")
    lon, lat = np.asarray(lon), np.asarray(lat)

    tam = gpd.read_file(ADM3_SHP)[["ADM3_PCODE", "ADM3_TH", "ADM3_EN", "ADM2_PCODE", "ADM2_TH",
                                   "ADM1_PCODE", "ADM1_TH", "geometry"]]
    tam["tambon"] = tam["ADM3_PCODE"].str[2:]
    tam = tam.reset_index(drop=True)

    # Cell centre rule: a cell belongs to the polygon that contains its centre
    label = rasterize(
        ((geom, i + 1) for i, geom in enumerate(tam.geometry)),
        out_shape=shape, transform=transform, fill=0, dtype="int32",
    )
    idx = label[rows, cols] - 1
    del label

    # Populated cells whose centre falls outside every polygon (coast, rivers)
    # go to the tambon of the nearest labelled cell.
    out = idx < 0
    if out.any():
        j, _ = nearest(lon[out], lat[out], lon[~out], lat[~out])
        idx[out] = idx[~out][j]

    grid = pd.DataFrame({
        "lon": lon, "lat": lat, "pop": values,
        "tambon": tam["tambon"].to_numpy()[idx],
    })
    grid.to_parquet(POPGRID, index=False)

    w = grid.assign(wx=grid["lon"] * grid["pop"], wy=grid["lat"] * grid["pop"]) \
        .groupby("tambon")[["pop", "wx", "wy"]].sum()
    tam = tam.merge(w, left_on="tambon", right_index=True, how="left")
    tam["pop"] = tam["pop"].fillna(0.0)
    rep = tam.geometry.representative_point()
    has_pop = tam["pop"] > 0
    tam["c_lon"] = np.where(has_pop, tam["wx"] / tam["pop"].where(has_pop), rep.x)
    tam["c_lat"] = np.where(has_pop, tam["wy"] / tam["pop"].where(has_pop), rep.y)
    tam = tam.drop(columns=["wx", "wy"])
    tam.to_parquet(TAMBONS, index=False)

    print(f"Saved: {POPGRID} ({len(grid):,} populated cells, {grid['pop'].sum():,.0f} people, "
          f"{int(out.sum()):,} cells outside polygons reassigned)")
    print(f"Saved: {TAMBONS} ({len(tam):,} tambons, {int((~has_pop).sum())} with no population)")


if __name__ == "__main__":
    main()
