"""Shared paths and constants for the v2 pipeline (task F: Python in place of ArcGIS).

Large inputs that are not in git (WorldPop raster, tambon boundaries) stay in
the sibling PC_GIS folder. Set PC_GIS_DIR to point somewhere else.
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = ROOT / "data" / "raw"
INTERIM = ROOT / "data" / "interim" / "v2"
PROCESSED = ROOT / "data" / "processed" / "v2"

PC_GIS = Path(os.environ.get("PC_GIS_DIR", ROOT.parent / "PC_GIS"))
WORLDPOP_TIF = PC_GIS / "Dashboard" / "Data" / "tha_ppp_2020_UNadj_constrained.tif"
ADM3_SHP = PC_GIS / "data" / "raw" / "tha_adm" / "tha_admbnda_adm3_rtsd_20250628.shp"
ADM1_SHP = PC_GIS / "data" / "raw" / "tha_adm" / "tha_admbnda_adm1_rtsd_20220121.shp"
# Downloaded 20 Nov 2026 (sources and licences in README.md)
# Meta Relative Wealth Index, Thailand, 2.4 km tiles, April 2021, CC BY-NC 4.0 (Chi et al. 2022 PNAS)
RWI_CSV = PC_GIS / "data" / "raw" / "rwi" / "tha_relative_wealth_index.csv"
# OpenStreetMap Thailand extract from Geofabrik, data to 2026-11-18, ODbL 1.0
OSM_PBF = PC_GIS / "data" / "raw" / "osm" / "thailand-latest.osm.pbf"

# NHSO registry snapshots: download date -> file
REGISTRY = {
    "2025-11-29": DATA_RAW / "hosmap_2025-11-29.csv",
    "2026-11-20": DATA_RAW / "hosmap_2026-11-20.csv",
}
# Facility list behind the live map (ArcGIS run), used only to check the replica
LIVE_FACILITIES = DATA_RAW / "hospitals_confirmed.csv"
LIVE_COVERAGE = DATA_RAW / "coverage.parquet"
LIVE_COVERAGE_PUBLIC = DATA_RAW / "coverage_public.parquet"

POPGRID = INTERIM / "popgrid.parquet"
TAMBONS = INTERIM / "tambons.parquet"
FACILITIES = PROCESSED / "facilities.csv"

# Service classes kept, by type. NHSO moved the contracted private units from
# R02xx to R03xx codes between the two snapshots (R0211 -> R0311 and so on);
# both codes mean the same type.
TYPE_CODES = {
    "P": ["P"],
    "pharmacy": ["R0211", "R0311"],
    "nurse_clinic": ["R0213", "R0313"],
    "doctor_clinic": ["R0207", "R0307"],
}
TYPE_ORDER = ["P", "pharmacy", "nurse_clinic", "doctor_clinic"]
# Class Z (private clinics that take part in some NHSO services, registry
# definition) is not a primary-care access point (project decision, 20 Nov 2026). About 6,100 units moved
# from the R classes to Z between the two snapshots; they count as having left.

# Units that do not serve the general public: prison health units (project
# decision, 20 Nov 2026). Kept in the facility table, left out of every access measure.
NO_PUBLIC_ACCESS = ["สังกัดกระทรวงยุติธรรม"]
# Registry rows that are not units: hcode -0000 "อยู่ระหว่างจัดหาหน่วยบริการ" (a placeholder
# while a unit is being procured), class P in both snapshots
NOT_A_UNIT = ["-0000"]

# A coordinate is accepted when it lies within this distance of the tambon the
# registry gives for the unit.
MAX_OFF_TAMBON_M = 2000

# Radii (km) for coverage curves: every 0.1 km to 10 km, as in the ArcGIS run
RADII_KM = [round(0.1 * i, 1) for i in range(1, 101)]

# Thailand spans UTM 47N and 48N; 47N keeps distance error under about 0.1%
# for the short distances used here. Nearest-neighbour search uses 3D
# unit-sphere coordinates instead, so it has no projection error at all.
UTM = "EPSG:32647"
EARTH_RADIUS_M = 6_371_008.8
