"""Small geometry helpers shared by the v2 scripts."""
import numpy as np
from scipy.spatial import cKDTree

from config import EARTH_RADIUS_M


def to_xyz(lon, lat):
    """Longitude/latitude in degrees -> points on the unit sphere (n x 3)."""
    lon = np.radians(np.asarray(lon, dtype=float))
    lat = np.radians(np.asarray(lat, dtype=float))
    c = np.cos(lat)
    return np.column_stack([c * np.cos(lon), c * np.sin(lon), np.sin(lat)])


def chord_to_m(chord):
    """Chord length on the unit sphere -> great-circle distance in metres."""
    return 2.0 * EARTH_RADIUS_M * np.arcsin(np.clip(chord / 2.0, 0.0, 1.0))


def nearest(src_lon, src_lat, dst_lon, dst_lat, workers=-1):
    """For each source point, the index of and distance (m) to the nearest destination."""
    tree = cKDTree(to_xyz(dst_lon, dst_lat))
    chord, idx = tree.query(to_xyz(src_lon, src_lat), k=1, workers=workers)
    return idx, chord_to_m(chord)
