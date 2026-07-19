"""Shared paths and constants for the PC_MAP data pipeline."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
DASHBOARD_DATA = ROOT / "dashboard" / "data"

COVERAGE_RAW_PARQUET = DATA_RAW / "coverage.parquet"
COVERAGE_PUBLIC_PARQUET = DATA_RAW / "coverage_public.parquet"
HOSPITALS_CSV = DATA_RAW / "hospitals_confirmed.csv"
PROVINCE_POPULATION_JSON = DATA_RAW / "province_population.json"

COVERAGE_PROVINCE_JSON = DATA_PROCESSED / "coverage_province.json"

# Radius steps (km) kept in the published dashboard data.
# The source GIS buffer analysis computes every 0.1km out to 10km (100 steps).
# We publish a coarser, tiered set to keep the browser-facing bundle small
# while staying granular where it matters most (short walking distances):
#   - 0.1km steps from 0.1 -> 1.0km   (10 values)
#   - 0.5km steps from 1.0 -> 5.0km   (8 more values)
#   - 1.0km steps from 5.0 -> 10.0km  (5 more values)
def _build_keep_radii():
    radii = [round(0.1 * i, 1) for i in range(1, 11)]          # 0.1..1.0
    radii += [round(1.0 + 0.5 * i, 1) for i in range(1, 9)]    # 1.5..5.0
    radii += [round(5.0 + 1.0 * i, 1) for i in range(1, 6)]    # 6.0..10.0
    return radii

KEEP_RADII = _build_keep_radii()
KEEP_RADII_SET = set(KEEP_RADII)


def radius_key(km: float) -> str:
    return f"{float(km):.1f}"
