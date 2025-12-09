import json, time
import os
from dotenv import load_dotenv
load_dotenv()
from pathlib import Path
from tqdm import tqdm

from huggingface_hub import HfApi, hf_hub_download

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.features import rasterize
from rasterio.enums import MergeAlg
from rasterio.transform import from_origin

import plotly.express as px
import plotly.graph_objects as go
import pydeck as pdk
import streamlit as st

import base64

# -------------------------
# CONFIG
# -------------------------

# HuggingFace dataset config
HF_REPO_ID = "psitthirat/thailand-primary-care"
HF_REPO_TYPE = "dataset"
HF_PRIMARY_CARE_FILE = "hospitals_confirmed.csv"
HF_SUBDIST_FILE = "rtsd_pat.geojson"
HF_POP_RASTER_FILE = "tha_ppp_2020_UNadj_constrained.tif"
HF_COVERAGE_FILE = "coverage.parquet"
HF_UPDATEDMETA_FILE = "updated_meta.json"

# These will be filled by resolve_data_paths()
PRIMARY_CARE_PATH: Path | None = None
SUBDISTRICT_PATH: Path | None = None
POP_RASTER_PATH: Path | None = None
COVERAGE_PATH: Path | None = None
UPDATEDMETA_PATH: Path | None = None

token = os.getenv("HUGGINGFACE_HUB_TOKEN")
if token is None:
    raise RuntimeError("Please run `huggingface-cli login` or set HUGGINGFACE_HUB_TOKEN")
api = HfApi(token=token)


ROOT = Path(__file__).resolve().parent
CACHE_DIR = ROOT / "cache"
CACHE_DIR.mkdir(exist_ok=True)

LOCAL_COVERAGE_PATH = CACHE_DIR / "coverage.parquet"
LOCAL_META_PATH = CACHE_DIR / "updated_meta.json"

# Radii (km) for precomputation
RADII_KM = RADII_KM = [round(x * 0.1, 1) for x in range(1, 101)]

# Projected CRS for Thailand (meters)
PROJECTED_CRS = "EPSG:32647"

# Logo paths (replace with your real files)
def image_to_base64(img_path):
    with open(img_path, "rb") as img_f:
        data = img_f.read()
        return base64.b64encode(data).decode("utf-8")
    
LOGO1_PATH = "assets/rama-hpsr.png"
LOGO2_PATH = "assets/tcels.png"
LOGO3_PATH = "assets/nhso.png"
LOGO1_LINK = "https://www.ramapolicyhub.com/"
LOGO2_LINK = "https://www.tcels.or.th/"
LOGO3_LINK = "https://www.nhso.go.th/"

# -------------------------
# DATA LOADING
# -------------------------

@st.cache_resource(show_spinner=True)
def resolve_data_paths():
    """
    Download data files from HuggingFace (if not already cached)
    and set global paths.
    """
    
    with st.spinner("Downloading hospital data..."):
        PRIMARY_CARE_PATH = Path(
            hf_hub_download(
                repo_id=HF_REPO_ID,
                filename=HF_PRIMARY_CARE_FILE,
                repo_type=HF_REPO_TYPE,
            )
        )

    with st.spinner("Downloading map data..."):
        SUBDISTRICT_PATH = Path(
            hf_hub_download(
                repo_id=HF_REPO_ID,
                filename=HF_SUBDIST_FILE,
                repo_type=HF_REPO_TYPE,
            )
        )

    with st.spinner("Downloading raster data..."):
        POP_RASTER_PATH = None

    with st.spinner("Downloading coverage data..."):
        COVERAGE_PATH = Path(
            hf_hub_download(
                repo_id=HF_REPO_ID,
                filename=HF_COVERAGE_FILE,
                repo_type=HF_REPO_TYPE,
            )
        )

    with st.spinner("Downloading version data..."):
        UPDATEDMETA_PATH = Path(
            hf_hub_download(
                repo_id=HF_REPO_ID,
                filename=HF_UPDATEDMETA_FILE,
                repo_type=HF_REPO_TYPE,
            )
        )

    return PRIMARY_CARE_PATH, SUBDISTRICT_PATH, POP_RASTER_PATH, COVERAGE_PATH, UPDATEDMETA_PATH

@st.cache_resource(show_spinner=True)
def load_primary_care(path: Path) -> gpd.GeoDataFrame:
    """Load primary care coordinates from CSV (lon/lat) in EPSG:4326."""
    df = pd.read_csv(path)
    if not {"lon", "lat"}.issubset(df.columns):
        raise ValueError("CSV must have columns: lon, lat")

    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df["lon"], df["lat"]),
        crs="EPSG:4326",
    )
    gdf["lon"] = gdf.geometry.x
    gdf["lat"] = gdf.geometry.y
    if "pc_id" not in gdf.columns:
        gdf["pc_id"] = gdf.index
    if "pc_name" not in gdf.columns:
        gdf["pc_name"] = gdf["pc_id"].astype(str)
    return gdf

@st.cache_resource(show_spinner=True)
def load_admin_polygons(path: Path, _target_crs="EPSG:4326"):
    """
    Load subdistrict GeoJSON and rename admin columns.
    Streamlit will NOT hash _target_crs since it starts with underscore.
    """
    gdf = gpd.read_file(path)
    gdf = gdf.to_crs(_target_crs)

    rename_map = {
        "P_NAME_TH": "prov_name",
        "A_NAME_TH": "amp_name",
        "T_NAME_TH": "tam_name",
    }
    gdf = gdf.rename(columns={k: v for k, v in rename_map.items() if k in gdf.columns})

    keep_cols = ["prov_name", "amp_name", "tam_name", "geometry"]
    gdf = gdf[[c for c in keep_cols if c in gdf.columns]]

    return gdf

@st.cache_resource(show_spinner=True)
def load_primary_care_with_admin(pc_path: Path, admin_path: Path) -> gpd.GeoDataFrame:
    """
    Primary care points + attached province/district/subdistrict
    via spatial join with subdistrict polygons.
    """
    _, subdistrict_path, _, _, _ = resolve_data_paths()
    gdf_pc = load_primary_care(pc_path)
    gdf_adm = load_admin_polygons(subdistrict_path, _target_crs=gdf_pc.crs)

    gdf_join = gpd.sjoin(
        gdf_pc,
        gdf_adm,
        how="left",
        predicate="within",
    ).drop(columns=["index_right"])

    # Ensure admin columns exist (may be NaN for some)
    for col in ["prov_name", "amp_name", "tam_name"]:
        if col not in gdf_join.columns:
            gdf_join[col] = None

    return gdf_join

@st.cache_resource(show_spinner=True)
def load_pop_raster_projected(path: Path):
    """
    Load population raster. If it's in EPSG:4326, reproject to PROJECTED_CRS (meters).
    Returns: (arr, transform, crs, nodata)
    """
    src = rasterio.open(path)
    src_crs = src.crs

    if src_crs and src_crs.to_string() == PROJECTED_CRS:
        arr = src.read(1)
        transform = src.transform
        nodata = src.nodata
        return arr, transform, src_crs, nodata

    dst_crs = PROJECTED_CRS
    transform, width, height = calculate_default_transform(
        src_crs, dst_crs, src.width, src.height, *src.bounds
    )

    dst_arr = np.empty((height, width), dtype=src.read(1).dtype)

    reproject(
        source=src.read(1),
        destination=dst_arr,
        src_transform=src.transform,
        src_crs=src_crs,
        dst_transform=transform,
        dst_crs=dst_crs,
        resampling=Resampling.nearest,
    )

    nodata = src.nodata
    return dst_arr, transform, rasterio.crs.CRS.from_string(dst_crs), nodata

# -------------------------
# PRECOMPUTE COVERAGE (with overlap adjustment)
# -------------------------

@st.cache_data(show_spinner=True)
def precompute_coverage(radii_km: tuple[float, ...]) -> pd.DataFrame:
    """
    Precompute population coverage for each PCU and each radius,
    accounting for intersection (no double counting).

    For each radius r:
      - Buffer all PCs in projected CRS
      - Rasterize all buffers to overlap_count raster
      - Each cell's population is divided by overlap_count
      - For each PC, sum adjusted_pop within its own buffer
    """

    # --- load data ---
    primary_care_path, subdistrict_path, pop_raster_path, _, _ = resolve_data_paths()
    gdf_pc_4326 = load_primary_care_with_admin(primary_care_path, subdistrict_path)
    pop_arr, transform, pop_crs, pop_nodata = load_pop_raster_projected(pop_raster_path)

    gdf_pc_proj = gdf_pc_4326.to_crs(pop_crs)
    gdf_pc_proj = gdf_pc_proj[:100]

    # base grid = use pop raster itself
    base_shape = pop_arr.shape
    base_transform = transform

    # --- prepare population array ---
    pop_float = pop_arr.astype("float32")

    if pop_nodata is not None:
        valid_mask = pop_arr != pop_nodata
    else:
        valid_mask = np.ones_like(pop_arr, dtype=bool)

    # set nodata to 0
    pop_float[~valid_mask] = 0.0

    buffer_arrays = {}
    buffer_arrays["pop_total"] = pop_float

    records: list[dict] = []

    # --- loop over radii ---
    for r in radii_km:
        radius_m = r * 1000.0

        # ชื่อ column buffer สำหรับรัศมีนี้
        buffer_col = f"buffer_{int(radius_m)}m"

        # สร้าง buffer geometry ต่อ facility
        gdf_pc_proj[buffer_col] = gdf_pc_proj.geometry.buffer(radius_m)

        # 1) buffer_count raster: จำนวน facility ที่ทับแต่ละ cell
        shapes_all = ((geom, 1) for geom in gdf_pc_proj[buffer_col])
        buffer_count = rasterize(
            shapes_all,
            out_shape=base_shape,
            transform=base_transform,
            fill=0,               # นอก buffer = 0 สำคัญมาก
            all_touched=False,
            merge_alg=MergeAlg.add,
            dtype="int16",
        )
        buffer_arrays[buffer_col] = buffer_count

        # 2) adjusted_pop = pop_total / buffer_count (เฉพาะ cell ที่ buffer_count > 0)
        adjusted = np.zeros_like(pop_float, dtype="float32")
        mask_buf = (buffer_count > 0) & valid_mask
        with np.errstate(divide="ignore", invalid="ignore"):
            adjusted[mask_buf] = pop_float[mask_buf] / buffer_count[mask_buf]

        # เก็บไว้เผื่อใช้ต่อ
        # adj_key = f"pop_adj_{int(radius_m)}m"
        # buffer_arrays[adj_key] = adjusted

        # 3) loop ต่อ facility → mask แล้ว sum adjusted_pop
        for row in gdf_pc_proj.itertuples():
            fac_geom = getattr(row, buffer_col)

            # mask raster ของ facility เดียว
            fac_mask = rasterize(
                [(fac_geom, 1)],
                out_shape=base_shape,
                transform=base_transform,
                fill=0,
                all_touched=False,
                dtype="uint8",
            )

            # sum adjusted_pop เฉพาะที่ fac_mask == 1
            fac_pop = float(adjusted[(fac_mask == 1)].sum())

            records.append(
                {
                    "pc_id": row.pc_id,
                    "pc_name": row.pc_name,
                    "prov_name": getattr(row, "prov_name", None),
                    "amp_name": getattr(row, "amp_name", None),
                    "tam_name": getattr(row, "tam_name", None),
                    "radius_km": float(r),
                    "pop": fac_pop,
                }
            )

    df_cov = pd.DataFrame.from_records(records)
    return df_cov

@st.cache_data(show_spinner=True)
def load_coverage_with_disk_cache(radii_km: tuple[float, ...]) -> pd.DataFrame:
    """
    Load overlap-adjusted coverage from disk if possible.
    Recompute and save if inputs changed.
    """
    # 1) Build current metadata (what inputs we are using now)
    coverage_path = hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=HF_UPDATEDMETA_FILE,
        repo_type=HF_REPO_TYPE,
    )
    
    updatedmeta_path = hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=HF_UPDATEDMETA_FILE,
        repo_type=HF_REPO_TYPE,
    )

    meta_current = json.loads(LOCAL_META_PATH.read_text())
    version = meta_current.get("version")

    # 2) Try reading existing meta + data
    if updatedmeta_path.exists():
        try:
            meta_old = json.loads(updatedmeta_path.read_text())
        except Exception:
            meta_old = None
    else:
        meta_old = None

        # If metadata matches → reuse cached parquet
    if (
        meta_old
        and meta_old.get("version") == version
        ):
        
        print("No new version — using existing coverage.parquet")
        df_cov = pd.read_parquet(coverage_path)
        
    else:
        # 3) Cache miss → run heavy precompute, then save
        print("Find new version — recomputing coverage…")
        df_cov = precompute_coverage(radii_km)

        df_cov.to_parquet(LOCAL_COVERAGE_PATH, index=False)
        
        # 5) Upload new coverage.parquet to HuggingFace
        print("Uploading coverage.parquet to HuggingFace…")

        api.upload_file(
            path_or_fileobj=str(LOCAL_COVERAGE_PATH),
            path_in_repo="coverage.parquet",
            repo_id=HF_REPO_ID,
            repo_type=HF_REPO_TYPE,
        )
        
        meta_current = {
            "version": version,
            "updated_at": time.time()
        }
        LOCAL_META_PATH.write_text(json.dumps(meta_current, indent=2))

        # Upload updated_meta.json
        print("Uploading updated_meta.json…")

        api.upload_file(
            path_or_fileobj=str(LOCAL_META_PATH),
            path_in_repo="updated_meta.json",
            repo_id=HF_REPO_ID,
            repo_type=HF_REPO_TYPE,
        )

        print("Upload complete.")

    return df_cov

@st.cache_resource(show_spinner=False)
def preload():
    """
    Load all base data for the app, tied to a specific coverage revision.
    If coverage_revision changes (on HF), Streamlit reruns this.
    """
    
    # Ensure base vector data paths exist
    primary_care_path = hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=HF_PRIMARY_CARE_FILE,
        repo_type=HF_REPO_TYPE,
    )
    
    subdistrict_path = hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=HF_SUBDIST_FILE,
        repo_type=HF_REPO_TYPE,
    )

    # Load primary care & coverage
    with st.spinner("Creating the dashboard..."):
        gdf_pc_4326 = load_primary_care_with_admin(primary_care_path, subdistrict_path)
        df_cov = load_coverage_with_disk_cache(tuple(RADII_KM))
    
    print("Downloaded database completely.")
    
    return gdf_pc_4326, df_cov

# -------------------------
# VISUAL HELPERS
# -------------------------
def make_map(
    gdf_pc_4326: gpd.GeoDataFrame,
    df_cov: pd.DataFrame,
    current_radius_km: float,
) -> pdk.Deck:
    """Map of all PCs + circles + hover popup with pop within current radius."""
    df_points = gdf_pc_4326.copy()

    # Attach current-radius population
    df_current = df_cov[df_cov["radius_km"] == current_radius_km][["pc_id", "pop"]]
    df_points = df_points.merge(df_current, on="pc_id", how="left")
    df_points["pop"] = df_points["pop"].fillna(0.0)

    # ---------- NEW: color by clinic_type ----------
    def color_from_type(ct: str):
        ct = (ct or "").strip().lower()
        # base RGB by clinic type
        if ct in ("public facility", "public facilities"):
            base = (0, 102, 255)      # blue
        elif "pharm" in ct:
            base = (255, 140, 0)      # orange
        elif "nurse" in ct:
            base = (220, 53, 69)      # red
        elif "doctor" in ct:
            base = (40, 167, 69)      # green
        else:
            base = (128, 128, 128)    # grey fallback

        r, g, b = base
        # 20% opacity ~ 51; border nearly solid
        fill = [r, g, b, 51]
        line = [r, g, b, 220]
        point = [r, g, b, 220]
        return pd.Series(
            {
                "point_color": point,
                "circle_fill_color": fill,
                "circle_line_color": line,
            }
        )

    df_points[["point_color", "circle_fill_color", "circle_line_color"]] = (
        df_points["clinic_type"].apply(color_from_type)
    )
    
    df_points["popup_text"] = df_points.apply(
        lambda r: (
            f"{r['pc_name']} (ID: {r['pc_id']})\n"
            f"Type: {r.get('clinic_type', 'N/A')}\n"
            f"Pop within {current_radius_km} km: {r['pop']:,.0f}"
        ),
        axis=1,
    )
    
    # -----------------------------------------------

    center_lat = df_points["lat"].mean()
    center_lon = df_points["lon"].mean()

    # Small center dots (solid color)
    points_layer = pdk.Layer(
        "ScatterplotLayer",
        data=df_points,
        get_position=["lon", "lat"],
        get_fill_color="point_color",   # <--- updated
        get_radius=50,
        pickable=True,
    )

    # Radius rings (20% opacity fill + colored border)
    df_points["radius_m"] = current_radius_km * 1000.0
    circles_layer = pdk.Layer(
        "ScatterplotLayer",
        data=df_points,
        get_position=["lon", "lat"],
        get_fill_color="circle_fill_color",   # <--- updated
        get_line_color="circle_line_color",   # <--- updated
        get_radius="radius_m",
        stroked=True,
        filled=True,      # keep a light shading
        line_width_min_pixels=1,
        pickable=True,
    )

    view_state = pdk.ViewState(
        longitude=float(center_lon),
        latitude=float(center_lat),
        zoom=9,
        pitch=0,
        bearing=0,
    )

    return pdk.Deck(
        layers=[circles_layer, points_layer],
        initial_view_state=view_state,
        map_style="light",
        tooltip={"text": "{popup_text}"},
    )

# -------------------------
# STREAMLIT APP
# -------------------------
def main():
    st.set_page_config(
        page_title="Primary Care Coverage Dashboard",
        layout="wide",
    )

    if "radius_value" not in st.session_state:
        st.session_state["radius_value"] = 3.0

    st.title("Spatial Coverage of Primary Care in Thailand")

    st.markdown(
        """
        A national analysis of population coverage using Euclidean distance and overlap-adjusted service areas

        This dashboard provides an interactive assessment of primary care accessibility across Thailand.
        Coverage is estimated using a gridded population model (100m-resolution) and distance-based service radii around all primary care facilities.
        """
    )

    # --- Load data tied to this revision ---
    gdf_pc_4326, df_cov = preload()
    
    # Build admin-aggregated tables
    df_total = (
        df_cov.groupby("radius_km")["pop"]
        .sum()
        .reset_index()
        .rename(columns={"pop": "total_pop"})
    )

    df_prov = (
        df_cov.groupby(["radius_km", "prov_name"], dropna=False)["pop"]
        .sum()
        .reset_index()
    )
    df_amp = (
        df_cov.groupby(["radius_km", "prov_name", "amp_name"], dropna=False)["pop"]
        .sum()
        .reset_index()
    )
    df_tam = (
        df_cov.groupby(["radius_km", "prov_name", "amp_name", "tam_name"], dropna=False)[
            "pop"
        ]
        .sum()
        .reset_index()
    )

    # ---------------------
    # TOP ROW: NATIONAL + ADMIN TABS
    # ---------------------
    col_nat, col_admin = st.columns([1, 1])

    # --- National (clickable) ---
    with col_nat:
        st.subheader("Crude coverage in Thailand")
        st.markdown("""
            This figure shows the cumulative number of people located within a given Euclidean radius of any primary care facility. 
            The blue line represents total population coverage, while the red line shows the incremental population gain achieved when expanding the radius by 100 meters.
        """)

        # Ensure sorted by radius
        df_total = df_total.sort_values("radius_km").reset_index(drop=True)

        # Incremental coverage: difference between this radius and previous radius
        df_total["delta_pop"] = df_total["total_pop"].diff()
        # For the first radius, use the same as total (or 0 if you prefer)
        df_total.loc[0, "delta_pop"] = df_total.loc[0, "total_pop"]

        fig_nat = go.Figure()

        # Blue line: cumulative total population
        fig_nat.add_trace(
            go.Scatter(
                x=df_total["radius_km"],
                y=df_total["total_pop"],
                mode="lines+markers",
                name="Cumulative population",
                line=dict(color="lightskyblue"),
            )
        )
        
        # Red line: incremental population per step
        fig_nat.add_trace(
            go.Scatter(
                x=df_total["radius_km"],
                y=df_total["delta_pop"],
                mode="lines+markers",
                name="Incremental coverage (Δ per step)",
                line=dict(color="red"),
                yaxis="y2",
            )
        )
        
        # Layout with secondary y-axis
        fig_nat.update_layout(
            title="Total population coverage VS radius",
            xaxis=dict(title="Radius (km)"),
            yaxis=dict(title="Total population (cumulative)"),
            yaxis2=dict(
                title="Incremental population (Δ between radii)",
                overlaying="y",
                side="right",
                showgrid=False,
            ),
            legend=dict(x=0.02, y=0.98),
            margin=dict(l=40, r=40, t=60, b=40),
        )
        
        st.plotly_chart(fig_nat, width='stretch')
        
        st.markdown("""
        - **Cumulative population**: total population reachable within the selected radius
        - **Incremental coverage (Δ per step)**: additional population newly included when the radius increases by one step
        """)

    # --- Admin tabs ---
    with col_admin:
        st.subheader("Coverage by administrative level")
        st.markdown("""
        This plot compares cumulative population coverage across administrative areas. Users may select one or more provinces, districts, or subdistricts to explore differences in accessibility and expansion patterns.
        """)
        tabs = st.tabs(["Province", "District (Amphoe)", "Subdistrict (Tambon)"])

        # Province
        with tabs[0]:
            st.markdown("**Coverage in selected provinces**")
            dfp = df_prov.dropna(subset=["prov_name"])
            top_prov = (
                dfp[dfp["radius_km"] == max(RADII_KM)]
                .sort_values("pop", ascending=False)["prov_name"]
                .head(10)
                .tolist()
            )
            all_provs = sorted(dfp["prov_name"].unique().tolist())
            default_provs = top_prov if top_prov else all_provs[:5]

            selected_provs = st.multiselect(
                "Select provinces to display",
                options=all_provs,
                default=default_provs,
            )

            show = dfp[dfp["prov_name"].isin(selected_provs)]
            if not show.empty:
                fig_prov = px.line(
                    show,
                    x="radius_km",
                    y="pop",
                    color="prov_name",
                    markers=True,
                    labels={
                        "radius_km": "Radius (km)",
                        "pop": "Population within radius",
                        "prov_name": "Province",
                    },
                )
                st.plotly_chart(fig_prov, width='stretch')
            else:
                st.info("Select at least one province.")

        # Amphoe
        with tabs[1]:
            st.markdown("**Coverage by district (amphoe)**")
            dfa = df_amp.copy()
            dfa["admin_label"] = (
                dfa["prov_name"].fillna("") + " - " + dfa["amp_name"].fillna("")
            )
            dfa = dfa.dropna(subset=["admin_label"])
            top_amp = (
                dfa[dfa["radius_km"] == max(RADII_KM)]
                .sort_values("pop", ascending=False)["admin_label"]
                .head(10)
                .tolist()
            )
            all_amps = sorted(dfa["admin_label"].unique().tolist())
            default_amps = top_amp if top_amp else all_amps[:5]

            selected_amps = st.multiselect(
                "Select districts to display",
                options=all_amps,
                default=default_amps,
            )

            show = dfa[dfa["admin_label"].isin(selected_amps)]
            if not show.empty:
                fig_amp = px.line(
                    show,
                    x="radius_km",
                    y="pop",
                    color="admin_label",
                    markers=True,
                    labels={
                        "radius_km": "Radius (km)",
                        "pop": "Population within radius",
                        "admin_label": "District",
                    },
                )
                st.plotly_chart(fig_amp, width='stretch')
            else:
                st.info("Select at least one district.")

        # Tambon
        with tabs[2]:
            st.markdown("**Coverage by subdistrict (tambon)**")
            dft = df_tam.copy()
            dft["admin_label"] = (
                dft["prov_name"].fillna("")
                + " - "
                + dft["amp_name"].fillna("")
                + " - "
                + dft["tam_name"].fillna("")
            )
            dft = dft.dropna(subset=["admin_label"])
            top_tam = (
                dft[dft["radius_km"] == max(RADII_KM)]
                .sort_values("pop", ascending=False)["admin_label"]
                .head(10)
                .tolist()
            )
            all_tams = sorted(dft["admin_label"].unique().tolist())
            default_tams = top_tam if top_tam else all_tams[:5]

            selected_tams = st.multiselect(
                "Select subdistricts to display",
                options=all_tams,
                default=default_tams,
            )

            show = dft[dft["admin_label"].isin(selected_tams)]
            if not show.empty:
                fig_tam = px.line(
                    show,
                    x="radius_km",
                    y="pop",
                    color="admin_label",
                    markers=True,
                    labels={
                        "radius_km": "Radius (km)",
                        "pop": "Population within radius",
                        "admin_label": "Subdistrict",
                    },
                )
                st.plotly_chart(fig_tam, width='stretch')
            else:
                st.info("Select at least one subdistrict.")

    # ---------------------
    # BOTTOM: MAP + RADIUS CONTROL
    # ---------------------
    st.markdown("""
        <style>
        .big-map .stDeckGlJsonChart {
            height: calc(100vh - 200px) !important;
        }
        </style>
        """, unsafe_allow_html=True)
    
    st.subheader("Facility Map")
    st.markdown("""
    The map displays all primary care facilities overlaid with distance-based service rings. 
    Colors represent facility type (e.g., public primary care center, pharmacy, nurse clinic, doctor clinic). 
    Shading indicates the overlap-adjusted population within the selected radius.
    """)

    map_container = st.container()
    with map_container:
        col_map, col_ctrl = st.columns([4, 1])

        # ---- Right column: radius control & metrics ----
        with col_ctrl:
            st.markdown("**Radius control**")
            min_r, max_r = min(RADII_KM), max(RADII_KM)

            slider_val = st.slider(
                "Radius (km)",
                min_value=float(min_r),
                max_value=float(max_r),
                value=float(st.session_state["radius_value"]),
                step=0.1,
                key="radius_value_slider",
            )
            # Keep session_state in sync with slider
            st.session_state["radius_value"] = slider_val

            # Snap to nearest precomputed radius
            current_radius_km = min(
                RADII_KM, key=lambda r: abs(st.session_state["radius_value"] - r)
            )
            st.caption(f"Using nearest precomputed radius: **{current_radius_km} km**")

            df_current = df_cov[df_cov["radius_km"] == current_radius_km]
            total_pop = df_current["pop"].sum()
            
            st.markdown(f"""
                <div style="padding: 12px; border: 1px solid #FFFFFF; border-radius: 8px; background-color: #FFFFFF;">
                    <div style="font-size: 14px; color: #111;">
                        Total population within {current_radius_km} km<br>
                        (overlap-adjusted)
                    </div>
                    <div style="font-size: 42px; font-weight: bold; color: #111;">
                        {total_pop:,.0f}
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("<hr style='margin: 12px 0; border: 1px solid #555;'>", unsafe_allow_html=True)

            st.markdown("""  
            **Legend (suggested as overlay on the map)**
            - Blue: Public Facility
            - Orange: Pharmacy
            - Red: Nurse Clinic
            - Green: Doctor Clinic
            """)

            with st.expander("Population by facility (current radius)"):
                st.dataframe(
                    df_current[["pc_id", "pc_name", "radius_km", "pop"]]
                    .sort_values("pop", ascending=False)
                    .reset_index(drop=True)
                )

        # ---- Left column: map (uses current_radius_km) ----
        with col_map:
            st.markdown('<div class="big-map">', unsafe_allow_html=True)
            deck = make_map(gdf_pc_4326, df_cov, current_radius_km)
            st.pydeck_chart(deck, width='stretch')
            st.markdown('</div>', unsafe_allow_html=True)


    # ---------------------
    # Logos + Citation
    # ---------------------
    st.markdown("---")
    footnote_container = st.container()
    with footnote_container:
        col_credit, col_logo = st.columns([3, 2])
        
        # --- Left: citation text ---
        with col_credit:

            st.markdown(
                """
                **Methodology Summary**
                - Primary care service areas are approximated using Euclidean distance buffers at 100–10,000 meters.
                - Population distribution is derived from a 100m gridded population raster, projected using EPSG:32647 (UTM Zone 47N).
                - Overlap between facilities is corrected using an overlap-adjusted coverage algorithm that divides shared population proportionally among intersecting buffers.
                - All calculations were precomputed for 100 radius steps to ensure smooth, real-time interaction.
                """
            )
            
            st.markdown("""
            ### **Data Sources**
            - **Primary care facilities**  
            National Health Security Office (NHSO).  
            Accessible via the public service locator portal: [link](https://mishos.nhso.go.th/nhso4/primary_nearby)
            - **Population raster**  
            WorldPop Project — *Global High Resolution Population Denominators (2000–2020), Constrained, 100m.*  
            Thailand 2020 (BSGM) dataset: [link](https://data.worldpop.org/GIS/Population/Global_2000_2020_Constrained/2020/BSGM/THA/)
            - **Administrative boundaries**  
            Thai Election Map dataset by Kittapat Ruangroj.  
            GitHub repository: [link](https://github.com/KittapatR/Thai-ECT-election-map-66)
            """
            )

            st.markdown(
                """
                **Citation**

                Sitthirat P, et al. *Spatial Coverage of Primary Care in Thailand: An Interactive Dashboard for Overlap-Adjusted Accessibility Analysis*. Ramathibodi Health Policy and Systems Research Unit, Mahidol University; 2025.
                
                For questions or collaboration:
                [Ramathibodi Health Policy and Systems Research Unit](https://www.ramapolicyhub.com)
                """
            )
        
        # --- Right: logos ---
        with col_logo:
            logo_cols = st.columns(3)

            logos = [
                (LOGO1_PATH, LOGO1_LINK),
                (LOGO2_PATH, LOGO2_LINK),
                (LOGO3_PATH, LOGO3_LINK),
            ]

            for col, (img_path, link_url) in zip(logo_cols, logos):
                with col:
                    if Path(img_path).exists():
                        img_base64 = image_to_base64(img_path)
                        html = f"""
                        <a href="{link_url}" target="_blank">
                            <img src="data:image/png;base64,{img_base64}" width="150">
                        </a>
                        """
                        st.markdown(html, unsafe_allow_html=True)
                    else:
                        st.write("Logo Not Found")
                        
    # --- Cache Management ---
    st.markdown("### Reload the coverage database")

    if st.button("Reload coverage"):
        with st.spinner("Clearing cache and reloading coverage..."):
            load_coverage_with_disk_cache.clear()
        st.success("Cache cleared. Reloading…")
        st.rerun()

if __name__ == "__main__":
    main()