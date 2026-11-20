"""
Task A3, step 1: build a road graph of Thailand from the OpenStreetMap extract.

Every OSM node on a kept way is a graph node; consecutive nodes on a way are
joined by an edge whose travel time is its length at the speed assumed for that
road class (SPEED_KMH below, motorised travel). Footpaths count at walking speed
and ferries at 15 km/h, so islands connect to the mainland. One-way rules,
turn costs, traffic and real speed limits are ignored: a first estimate.

Input:
  PC_GIS/data/raw/osm/thailand-latest.osm.pbf
Output:
  data/interim/v2/road_nodes.parquet   node index, lon, lat
  data/interim/v2/road_edges.parquet   u, v (node index), km, minutes
"""
import time

import numpy as np
import osmium
import pandas as pd

from config import EARTH_RADIUS_M, INTERIM, OSM_PBF

# Assumed travel speed by OSM highway class (km/h)
SPEED_KMH = {
    "motorway": 90, "motorway_link": 50, "trunk": 80, "trunk_link": 40,
    "primary": 60, "primary_link": 40, "secondary": 50, "secondary_link": 30,
    "tertiary": 40, "tertiary_link": 30, "unclassified": 30, "road": 30,
    "residential": 25, "living_street": 15, "service": 15, "track": 15,
    "footway": 5, "path": 5, "pedestrian": 5, "steps": 5, "cycleway": 5,
}
FERRY_KMH = 15


def haversine_km(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.radians, (lon1, lat1, lon2, lat2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * EARTH_RADIUS_M * np.arcsin(np.sqrt(a)) / 1000


def main():
    t0 = time.time()
    ids, lons, lats, speeds, starts = [], [], [], [], []
    n_ways = 0
    fp = osmium.FileProcessor(str(OSM_PBF)).with_locations() \
        .with_filter(osmium.filter.EntityFilter(osmium.osm.WAY))
    for w in fp:
        hw = w.tags.get("highway")
        speed = SPEED_KMH.get(hw) if hw else (FERRY_KMH if w.tags.get("route") == "ferry" else None)
        if speed is None or len(w.nodes) < 2:
            continue
        nodes = [n for n in w.nodes if n.location.valid()]
        if len(nodes) < 2:
            continue
        starts.append(len(ids))
        for n in nodes:
            ids.append(n.ref)
            lons.append(n.lon)
            lats.append(n.lat)
        speeds.append(speed)
        n_ways += 1
    print(f"read {n_ways:,} ways, {len(ids):,} node references in {time.time() - t0:.0f}s")

    ids = np.asarray(ids, dtype=np.int64)
    lons = np.asarray(lons)
    lats = np.asarray(lats)
    starts = np.asarray(starts + [len(ids)])
    way_len = np.diff(starts)
    # Edge from reference i to i+1, unless i is the last node of its way
    last = starts[1:] - 1
    keep = np.ones(len(ids) - 1, dtype=bool)
    keep[last[:-1]] = False
    a = np.nonzero(keep)[0]
    b = a + 1
    edge_speed = np.repeat(np.asarray(speeds, dtype=float), way_len - 1)

    uniq, inv = np.unique(ids, return_inverse=True)
    node_lon = np.zeros(len(uniq))
    node_lat = np.zeros(len(uniq))
    node_lon[inv] = lons
    node_lat[inv] = lats
    km = haversine_km(lons[a], lats[a], lons[b], lats[b])
    u, v = inv[a], inv[b]
    edges = pd.DataFrame({"u": np.minimum(u, v), "v": np.maximum(u, v), "km": km,
                          "minutes": km / edge_speed * 60})
    edges = edges[edges["u"] != edges["v"]]
    # Two ways sharing a segment: keep the faster
    edges = edges.sort_values("minutes").drop_duplicates(["u", "v"]).reset_index(drop=True)

    INTERIM.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"lon": node_lon, "lat": node_lat}).to_parquet(INTERIM / "road_nodes.parquet", index=False)
    edges.astype({"u": "int32", "v": "int32", "km": "float32", "minutes": "float32"}) \
        .to_parquet(INTERIM / "road_edges.parquet", index=False)
    print(f"Saved: {INTERIM / 'road_nodes.parquet'} ({len(uniq):,} nodes)")
    print(f"Saved: {INTERIM / 'road_edges.parquet'} ({len(edges):,} edges, {edges['km'].sum():,.0f} km) "
          f"in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
