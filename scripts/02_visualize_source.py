
"""
02_visualize_source.py

Visualize ADD coastline data and IBCSO bathymetry
around Lutzow-Holm Bay.

- ADD line: coastline
- ADD polygon: land / ice shelf / ice tongue / rumple
- IBCSO: bathymetry and elevation

All data are read-only.
"""

from pathlib import Path
import math

import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import rasterio

from rasterio.enums import Resampling
from rasterio.transform import from_origin
from rasterio.vrt import WarpedVRT
from rasterio.warp import transform_bounds
from matplotlib.ticker import FuncFormatter


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

ADD_DIR = ROOT / "input" / "raw" / "ADD"
IBCSO_DIR = ROOT / "input" / "raw" / "IBCSO"

OUTPUT_DIR = ROOT / "output" / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LINE_FILE = ADD_DIR / "add_coastline_high_res_line_v7_12.gpkg"
POLYGON_FILE = ADD_DIR / "add_coastline_high_res_polygon_v7_12.gpkg"
IBCSO_FILE = IBCSO_DIR / "IBCSO_bed.tif"


# ============================================================
# DISPLAY SETTINGS
# ============================================================

ADD_CRS = "EPSG:3031"

# Preliminary display extent only.
# (west, south, east, north) in degrees.
LONLAT_BOUNDS = (30.0, -70.0, 42.0, -66.0)

# Output raster resolution for visualization.
# This does not modify the original IBCSO data.
DISPLAY_RESOLUTION = 1000.0  # meters

# ADD polygon colors
SURFACE_COLORS = {
    "land": "#777777",
    "ice shelf": "#80c9e8",
    "ice tongue": "#2878b5",
    "rumple": "#e6a23c",
}

# IBCSO display range
BATHY_VMIN = -5000
BATHY_VMAX = 1500


# ============================================================
# LOAD ADD DATA
# ============================================================

def load_add_data(plot_bounds):

    print("\nLoading ADD polygon data...")

    polygons = gpd.read_file(
        POLYGON_FILE,
        bbox=plot_bounds,
    )

    if polygons.crs is None:
        raise ValueError("ADD polygon CRS is undefined.")

    polygons = polygons.to_crs(ADD_CRS)

    print("Polygon features in display area:", len(polygons))
    print("\nSurface categories in display area:")
    print(polygons["surface"].value_counts(dropna=False))

    print("\nLoading ADD line data...")

    lines = gpd.read_file(
        LINE_FILE,
        bbox=plot_bounds,
    )

    if lines.crs is None:
        raise ValueError("ADD line CRS is undefined.")

    lines = lines.to_crs(ADD_CRS)

    print("Line features in display area:", len(lines))

    return polygons, lines


# ============================================================
# LOAD IBCSO DATA
# ============================================================

def load_ibcso(plot_bounds):

    print("\nLoading IBCSO bathymetry...")

    xmin, ymin, xmax, ymax = plot_bounds

    width = math.ceil(
        (xmax - xmin) / DISPLAY_RESOLUTION
    )
    height = math.ceil(
        (ymax - ymin) / DISPLAY_RESOLUTION
    )

    dst_transform = from_origin(
        xmin,
        ymax,
        DISPLAY_RESOLUTION,
        DISPLAY_RESOLUTION,
    )

    with rasterio.open(IBCSO_FILE) as src:

        if src.crs is None:
            raise ValueError("IBCSO CRS is undefined.")

        print("Original CRS:", src.crs)
        print("Display CRS:", ADD_CRS)
        print("Display resolution:", DISPLAY_RESOLUTION, "m")

        # Reproject only the display area.
        # The original GeoTIFF is not modified.
        with WarpedVRT(
            src,
            crs=ADD_CRS,
            transform=dst_transform,
            width=width,
            height=height,
            resampling=Resampling.bilinear,
            nodata=src.nodata,
        ) as vrt:

            bathy = vrt.read(1, masked=True)

    print("Display raster shape:", bathy.shape)
    print("Valid pixels:", bathy.count())

    return bathy, (xmin, xmax, ymin, ymax)


# ============================================================
# PLOTTING
# ============================================================

def format_km(value, position):
    return f"{value / 1000:.0f}"


def plot_sources(polygons, lines, bathy, raster_extent):

    print("\nCreating figure...")

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(16, 8),
        constrained_layout=True,
    )

    # --------------------------------------------------------
    # Panel 1: IBCSO bathymetry only
    # --------------------------------------------------------

    ax = axes[0]

    image = ax.imshow(
        bathy,
        extent=raster_extent,
        origin="upper",
        cmap="terrain",
        vmin=BATHY_VMIN,
        vmax=BATHY_VMAX,
        interpolation="nearest",
    )

    ax.set_title("IBCSO bathymetry")
    ax.set_xlabel("Easting (km)")
    ax.set_ylabel("Northing (km)")

    fig.colorbar(
        image,
        ax=ax,
        label="Elevation / bathymetry (m)",
        shrink=0.8,
    )

    # --------------------------------------------------------
    # Panel 2: IBCSO + ADD
    # --------------------------------------------------------

    ax = axes[1]

    ax.imshow(
        bathy,
        extent=raster_extent,
        origin="upper",
        cmap="terrain",
        vmin=BATHY_VMIN,
        vmax=BATHY_VMAX,
        interpolation="nearest",
    )

    # Draw ADD polygons by surface category.
    for surface, color in SURFACE_COLORS.items():

        subset = polygons[
            polygons["surface"] == surface
        ]

        if not subset.empty:
            subset.plot(
                ax=ax,
                color=color,
                edgecolor="none",
                alpha=0.85,
            )

    # Draw ADD coastline lines.
    if not lines.empty:
        lines.plot(
            ax=ax,
            color="black",
            linewidth=0.35,
            alpha=0.8,
        )

    # Legend
    legend_handles = [
        mpatches.Patch(
            color=color,
            label=surface,
        )
        for surface, color in SURFACE_COLORS.items()
    ]

    ax.legend(
        handles=legend_handles,
        loc="lower left",
        title="ADD surface",
        fontsize=9,
    )

    ax.set_title("IBCSO + ADD")
    ax.set_xlabel("Easting (km)")
    ax.set_ylabel("Northing (km)")

    # --------------------------------------------------------
    # Common axis formatting
    # --------------------------------------------------------

    for ax in axes:

        ax.xaxis.set_major_formatter(
            FuncFormatter(format_km)
        )

        ax.yaxis.set_major_formatter(
            FuncFormatter(format_km)
        )

        ax.set_aspect("equal")
        ax.grid(
            True,
            linestyle=":",
            linewidth=0.5,
            alpha=0.5,
        )

    fig.suptitle(
        "ADD and IBCSO source data: Lutzow-Holm Bay",
        fontsize=14,
    )

    output_file = OUTPUT_DIR / "02_source_data_lhb.png"

    fig.savefig(
        output_file,
        dpi=200,
        bbox_inches="tight",
    )

    print("\nFigure saved:")
    print(output_file)

    plt.show()


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("=" * 65)
    print("WW3 GRIDGEN: SOURCE DATA VISUALIZATION")
    print("=" * 65)

    # Convert geographic display bounds to ADD CRS.
    plot_bounds = transform_bounds(
        "EPSG:4326",
        ADD_CRS,
        *LONLAT_BOUNDS,
        densify_pts=21,
    )

    # Add a small margin around the display area.
    xmin, ymin, xmax, ymax = plot_bounds
    margin = 10000.0

    plot_bounds = (
        xmin - margin,
        ymin - margin,
        xmax + margin,
        ymax + margin,
    )

    print("\nDisplay bounds in EPSG:3031:")
    print(plot_bounds)

    polygons, lines = load_add_data(plot_bounds)

    bathy, raster_extent = load_ibcso(plot_bounds)

    plot_sources(
        polygons,
        lines,
        bathy,
        raster_extent,
    )

    print("\nVisualization completed.")
