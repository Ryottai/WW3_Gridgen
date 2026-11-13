
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