
"""
config.py

Shared configuration for WW3_Gridgen.

All project-wide paths, coordinate systems, model domains,
and visualization settings are defined here.
"""

from pathlib import Path


# ============================================================
# PROJECT PATHS
# ============================================================

ROOT = Path(__file__).resolve().parent

INPUT_DIR = ROOT / "input" / "raw"
ADD_DIR = INPUT_DIR / "ADD"
IBCSO_DIR = INPUT_DIR / "IBCSO"

OUTPUT_DIR = ROOT / "output"
FIGURE_DIR = OUTPUT_DIR / "figures"
NETCDF_DIR = OUTPUT_DIR / "netcdf"
WW3_DIR = OUTPUT_DIR / "ww3"

# Create output directories if they do not exist.
for directory in (FIGURE_DIR, NETCDF_DIR, WW3_DIR):
    directory.mkdir(parents=True, exist_ok=True)


# ============================================================
# SOURCE FILES
# ============================================================

ADD_LINE_FILE = ADD_DIR / "add_coastline_high_res_line_v7_12.gpkg"
ADD_POLYGON_FILE = ADD_DIR / "add_coastline_high_res_polygon_v7_12.gpkg"
IBCSO_FILE = IBCSO_DIR / "IBCSO_bed.tif"


# ============================================================
# COORDINATE REFERENCE SYSTEMS
# ============================================================

# ADD coastline dataset
ADD_CRS = "EPSG:3031"

# IBCSO v2 bathymetry dataset
IBCSO_CRS = "EPSG:9354"

# Geographic coordinates
GEOGRAPHIC_CRS = "EPSG:4326"


# ============================================================
# MODEL DOMAIN
# ============================================================

# Lützow-Holm Bay
# (west longitude, south latitude, east longitude, north latitude)
LHB_BOUNDS = (30.0, -70.0, 42.0, -66.0)

# Additional margin for visualization, in degrees.
LHB_DISPLAY_MARGIN_DEG = 0.1

# Antarctica-wide display domain in EPSG:3031.
# (xmin, ymin, xmax, ymax), meters.
ANTARCTICA_BOUNDS = (
    -3_300_000,
    -3_300_000,
    3_300_000,
    3_300_000,
)


# ============================================================
# VISUALIZATION SETTINGS
# ============================================================

# Display resolution for the reprojected IBCSO raster.
# These settings affect visualization only, not source data.
ANTARCTICA_DISPLAY_RESOLUTION = 10_000.0  # meters
LHB_DISPLAY_RESOLUTION_DEG = 0.01         # degrees

# Water depth range [m]
DEPTH_VMIN = 0
DEPTH_VMAX = 5000

# Colorbar ticks [m]
DEPTH_TICKS = 1000

# Bathymetry colormap
DEPTH_COLORS = [
    "#8c510a",  # 0 m: brown
    "#bf812d",
    "#dfc27d",
    "#f6e8c3",
    "#c7eae5",
    "#80cdc1",
    "#35978f",
    "#2166ac",
    "#053061",  # Deep ocean: dark blue
]

# Colorbar appearance
DEPTH_COLORBAR_LABEL = "Water depth (m)"
DEPTH_COLORBAR_EXTEND = "max"
DEPTH_COLORBAR_SHRINK = 0.78
DEPTH_COLORBAR_PAD = 0.025

# ADD polygon colors
SURFACE_COLORS = {
    "land": "#777777",
    "ice shelf": "#80c9e8",
    "ice tongue": "#2878b5",
    "rumple": "#e6a23c",
}

FIGURE_DPI = 200

# ============================================================
# WW3 STRUCTURED GRID (v6.07.1, RECT, spherical lon/lat)
# ============================================================
# Grid-point coordinates (not cell centres in the WW3 sense): each grid
# point owns a cell that extends +-0.5 increments around it (gridgen
# convention).  Domain limits come from LHB_BOUNDS.

WW3_GRID_NAME = "LHB_RECT_0p10x0p05"            # file stem, no spaces
WW3_GRID_TITLE = "Lutzow-Holm Bay 0.10x0.05"    # written to ww3_grid.inp (<= 30 chars)

GRID_LON_MIN, GRID_LAT_MIN, GRID_LON_MAX, GRID_LAT_MAX = LHB_BOUNDS
GRID_DLON = 0.10   # degrees  -> 121 points for 30-42E
GRID_DLAT = 0.05   # degrees  ->  81 points for 70-66S

# ADD surface classes treated as LAND (full blocking) in the WW3 grid.
# Ice shelves, ice tongues and ice rumples are land by project decision.
LAND_SURFACES = tuple(SURFACE_COLORS.keys())

# --- Bathymetry / land-sea mask (gridgen "generate_grid" + "land mask") ---
# Each cell is sampled on a GRID_SUBSAMPLE x GRID_SUBSAMPLE sub-grid
# (0.01 x 0.005 deg ~ 0.4 x 0.55 km, comparable to IBCSO's 500 m).
GRID_SUBSAMPLE = 10
BATHY_WET_LIMIT = 0.25     # min fraction of IBCSO ocean (elev < 0) for a wet cell
LAND_CUTOFF = 0.50         # cell -> land if ADD land fraction exceeds this
IBCSO_RESAMPLING = "bilinear"   # IBCSO (EPSG:9354) -> lon/lat sub-grid

# --- WW3 depth conventions ---
WW3_ZLIM = -0.10           # limiting depth [m]: sea if depth < ZLIM (negative down)
WW3_DMIN = 2.50            # minimum water depth [m] used by the model
WW3_DRY_DEPTH = 999.0      # value written for land cells [m] (positive = above datum)
WW3_DEPTH_SCALE = 0.001    # file holds integer millimetres

# --- Sub-grid obstruction (gridgen create_obstr) ---
OBSTR_OFFSET = 1           # 1: use both neighbours (OB-2, recommended), 0: own cell only
OBSTR_SCALE = 100          # file holds integer percent
WW3_FLAGTR = 4             # &MISC FLAGTR (2: cell centres, 4: + continuous ice)
WW3_CICE0 = 0.25
WW3_CICEN = 0.75

# --- ww3_grid.inp ---
WW3_MASK_VARIANT = "nobound"   # "nobound": 0/1 only, "bound": open edges = 2
WW3_SPECTRUM = (1.1, 0.04118, 25, 24, 0.0)  # xfr, f1 [Hz], nk, nth, th0 offset
WW3_FLAGS = "F T T T F T"      # FLDRY FLCX FLCY FLCTH FLCK FLSOU
WW3_TIME_STEPS = None          # None -> estimated from CFL; or (dtmax, dtxy, dtkth, dtmin)
WW3_UNIT_DEPTH, WW3_UNIT_OBSTR, WW3_UNIT_MASK = 40, 50, 60
