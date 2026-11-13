
"""
02_visualize_source.py

Generate three separate figures for ADD and IBCSO source data.

1. Antarctica-wide ADD + IBCSO
   - South Polar Stereographic projection
   - Latitude / longitude graticules

2. Lützow-Holm Bay IBCSO
   - Geographic coordinates (longitude / latitude)

3. Lützow-Holm Bay ADD + IBCSO
   - Geographic coordinates (longitude / latitude)

All source data are read-only.
Reprojection is performed only for visualization.
"""

from pathlib import Path
import sys
import math

import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import rasterio

from rasterio.enums import Resampling
from rasterio.transform import from_origin
from rasterio.vrt import WarpedVRT
from rasterio.warp import transform_bounds

import cartopy.crs as ccrs
import cartopy.mpl.ticker as cticker

# ------------------------------------------------------------
# Import shared configuration
# ------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import (
    ADD_LINE_FILE,
    ADD_POLYGON_FILE,
    IBCSO_FILE,
    FIGURE_DIR,
    ADD_CRS,
    GEOGRAPHIC_CRS,
    LHB_BOUNDS,
    LHB_DISPLAY_MARGIN_DEG,
    ANTARCTICA_BOUNDS,
    ANTARCTICA_DISPLAY_RESOLUTION,
    LHB_DISPLAY_RESOLUTION_DEG,
    BATHY_VMIN,
    BATHY_VMAX,
    BATHY_CMAP,
    SURFACE_COLORS,
    FIGURE_DPI,
)


# ============================================================
# COMMON FUNCTIONS
# ============================================================

def save_figure(fig, filename):

    output_file = FIGURE_DIR / filename

    fig.savefig(
        output_file,
        dpi=FIGURE_DPI,
        bbox_inches="tight",
    )

    print(f"\nFigure saved: {output_file}")

    plt.close(fig)


def add_surface_legend(ax):

    handles = [
        mpatches.Patch(
            color=color,
            label=surface,
        )
        for surface, color in SURFACE_COLORS.items()
    ]

    ax.legend(
        handles=handles,
        loc="lower left",
        title="ADD surface",
        fontsize=9,
    )


# ============================================================
# LOAD ADD DATA
# ============================================================

def load_add_data(plot_bounds=None, target_crs=ADD_CRS):

    """
    Load ADD polygon and line data.

    Parameters
    ----------
    plot_bounds : tuple or None
        Bounds in ADD_CRS (xmin, ymin, xmax, ymax).
        If None, load the entire dataset.

    target_crs : str
        CRS used for plotting.
    """

    print("\nLoading ADD polygon data...")

    if plot_bounds is None:
        polygons = gpd.read_file(ADD_POLYGON_FILE)
    else:
        polygons = gpd.read_file(
            ADD_POLYGON_FILE,
            bbox=plot_bounds,
        )

    if polygons.crs is None:
        raise ValueError("ADD polygon CRS is undefined.")

    polygons = polygons.to_crs(target_crs)

    print("Polygon features:", len(polygons))
    print("Surface categories:")
    print(polygons["surface"].value_counts(dropna=False))

    print("\nLoading ADD line data...")

    if plot_bounds is None:
        lines = gpd.read_file(
            ADD_LINE_FILE,
            layer="add_coastline_high_res_line_v7_12",
        )
    else:
        lines = gpd.read_file(
            ADD_LINE_FILE,
            bbox=plot_bounds,
            layer="add_coastline_high_res_line_v7_12",
        )

    if lines.crs is None:
        raise ValueError("ADD line CRS is undefined.")

    lines = lines.to_crs(target_crs)

    print("Line features:", len(lines))

    return polygons, lines


# ============================================================
# LOAD AND REPROJECT IBCSO
# ============================================================

def load_ibcso(
    plot_bounds,
    target_crs,
    display_resolution,
):
    """
    Reproject only the requested display area.

    Parameters
    ----------
    plot_bounds : tuple
        Bounds in target_crs:
        (xmin, ymin, xmax, ymax).

    target_crs : str
        Destination CRS.

    display_resolution : float
        Pixel size in target CRS units.
        Meters for EPSG:3031, degrees for EPSG:4326.
    """

    xmin, ymin, xmax, ymax = plot_bounds

    width = math.ceil(
        (xmax - xmin) / display_resolution
    )
    height = math.ceil(
        (ymax - ymin) / display_resolution
    )

    dst_transform = from_origin(
        xmin,
        ymax,
        display_resolution,
        display_resolution,
    )

    print("\nLoading IBCSO bathymetry...")
    print("Source file:", IBCSO_FILE)
    print("Target CRS:", target_crs)
    print("Display resolution:", display_resolution)
    print("Output raster size:", width, "x", height)

    with rasterio.open(IBCSO_FILE) as src:

        if src.crs is None:
            raise ValueError("IBCSO CRS is undefined.")

        print("Original CRS:", src.crs)

        with WarpedVRT(
            src,
            crs=target_crs,
            transform=dst_transform,
            width=width,
            height=height,
            resampling=Resampling.bilinear,
            nodata=src.nodata,
        ) as vrt:

            bathy = vrt.read(1, masked=True)

    print("Valid pixels:", bathy.count())

    raster_extent = (
        xmin,
        xmin + width * display_resolution,
        ymax - height * display_resolution,
        ymax,
    )

    return bathy, raster_extent


# ============================================================
# DRAW IBCSO
# ============================================================

def draw_ibcso(
    ax,
    bathy,
    raster_extent,
):
    image = plt.Axes.imshow(
        ax,
        bathy,
        extent=(
            raster_extent[0],
            raster_extent[1],
            raster_extent[2],
            raster_extent[3],
        ),
        origin="upper",
        cmap=BATHY_CMAP,
        vmin=BATHY_VMIN,
        vmax=BATHY_VMAX,
        interpolation="nearest",
    )

    return image


# ============================================================
# DRAW ADD
# ============================================================

def draw_add(ax, polygons, lines, overview=False):

    # Draw polygon categories.
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
                zorder=2,
            )

    # Draw ADD coastline lines.
    if not lines.empty:
        lines.plot(
            ax=ax,
            color="black",
            linewidth=0.15 if overview else 0.35,
            alpha=0.8,
            zorder=3,
        )

    add_surface_legend(ax)


# ============================================================
# FIGURE 1: ANTARCTICA-WIDE ADD + IBCSO
# ============================================================

def plot_antarctica(polygons, lines):

    print("\nCreating Figure 1: Antarctica-wide ADD + IBCSO")

    bathy, raster_extent = load_ibcso(
        ANTARCTICA_BOUNDS,
        ADD_CRS,
        ANTARCTICA_DISPLAY_RESOLUTION,
    )

    map_crs = ccrs.epsg(3031)

    fig = plt.figure(figsize=(10, 10))

    ax = fig.add_subplot(
        1,
        1,
        1,
        projection=map_crs,
    )

    image = draw_ibcso(
        ax,
        bathy,
        raster_extent,
    )

    # ADD geometries are already in EPSG:3031.
    draw_add(
        ax,
        polygons,
        lines,
        overview=True,
    )

    xmin, ymin, xmax, ymax = ANTARCTICA_BOUNDS

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    ax.set_title(
        "Antarctica: IBCSO + ADD",
        fontsize=14,
    )

    # Latitude / longitude gridlines.
    gl = ax.gridlines(
        crs=ccrs.PlateCarree(),
        draw_labels=True,
        linewidth=0.5,
        color="gray",
        alpha=0.6,
        linestyle=":",
        x_inline=False,
        y_inline=False,
    )

    gl.xlocator = cticker.LongitudeLocator(30)
    gl.ylocator = cticker.LatitudeLocator(10)
    gl.xlabel_style = {"size": 8}
    gl.ylabel_style = {"size": 8}

    fig.colorbar(
        image,
        ax=ax,
        label="Elevation / bathymetry (m)",
        shrink=0.75,
        pad=0.08,
    )

    save_figure(
        fig,
        "01_antarctica_add_ibcso.png",
    )


# ============================================================
# FIGURE 2: LHB IBCSO ONLY
# ============================================================

def plot_lhb_ibcso(
    bathy,
    raster_extent,
    display_bounds,
):

    print("\nCreating Figure 2: LHB IBCSO")

    fig, ax = plt.subplots(
        figsize=(10, 8),
        constrained_layout=True,
    )

    image = draw_ibcso(
        ax,
        bathy,
        raster_extent,
    )

    west, south, east, north = display_bounds

    ax.set_xlim(west, east)
    ax.set_ylim(south, north)

    ax.set_xlabel("Longitude (°E)")
    ax.set_ylabel("Latitude (°S)")

    ax.set_title(
        "IBCSO bathymetry: Lützow-Holm Bay",
        fontsize=14,
    )

    ax.set_aspect("equal", adjustable="box")
    ax.grid(
        True,
        linestyle=":",
        linewidth=0.5,
        alpha=0.5,
    )

    fig.colorbar(
        image,
        ax=ax,
        label="Elevation / bathymetry (m)",
        shrink=0.8,
    )

    save_figure(
        fig,
        "02_lhb_ibcso.png",
    )


# ============================================================
# FIGURE 3: LHB ADD + IBCSO
# ============================================================

def plot_lhb_add_ibcso(
    polygons,
    lines,
    bathy,
    raster_extent,
    display_bounds,
):

    print("\nCreating Figure 3: LHB ADD + IBCSO")

    fig, ax = plt.subplots(
        figsize=(10, 8),
        constrained_layout=True,
    )

    image = draw_ibcso(
        ax,
        bathy,
        raster_extent,
    )

    draw_add(
        ax,
        polygons,
        lines,
    )

    west, south, east, north = display_bounds

    ax.set_xlim(west, east)
    ax.set_ylim(south, north)

    ax.set_xlabel("Longitude (°E)")
    ax.set_ylabel("Latitude (°S)")

    ax.set_title(
        "IBCSO + ADD: Lützow-Holm Bay",
        fontsize=14,
    )

    ax.set_aspect("equal", adjustable="box")
    ax.grid(
        True,
        linestyle=":",
        linewidth=0.5,
        alpha=0.5,
    )

    fig.colorbar(
        image,
        ax=ax,
        label="Elevation / bathymetry (m)",
        shrink=0.8,
    )

    save_figure(
        fig,
        "03_lhb_add_ibcso.png",
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("=" * 65)
    print("WW3 GRIDGEN: SOURCE DATA VISUALIZATION")
    print("=" * 65)

    # --------------------------------------------------------
    # LHB geographic display bounds.
    # --------------------------------------------------------

    west, south, east, north = LHB_BOUNDS
    margin = LHB_DISPLAY_MARGIN_DEG

    display_bounds = (
        west - margin,
        south - margin,
        east + margin,
        north + margin,
    )

    print("\nLHB display bounds (EPSG:4326):")
    print(display_bounds)

    # --------------------------------------------------------
    # Figure 1: Load all ADD data in EPSG:3031.
    # --------------------------------------------------------

    polygons_all, lines_all = load_add_data(
        plot_bounds=None,
        target_crs=ADD_CRS,
    )

    plot_antarctica(
        polygons_all,
        lines_all,
    )

    # --------------------------------------------------------
    # Convert LHB bounds to ADD CRS for spatial filtering.
    # --------------------------------------------------------

    lhb_bounds_add = transform_bounds(
        GEOGRAPHIC_CRS,
        ADD_CRS,
        *display_bounds,
        densify_pts=21,
    )

    # --------------------------------------------------------
    # Load only ADD features in LHB.
    # Reproject them to geographic coordinates.
    # --------------------------------------------------------

    polygons_lhb, lines_lhb = load_add_data(
        plot_bounds=lhb_bounds_add,
        target_crs=GEOGRAPHIC_CRS,
    )

    # --------------------------------------------------------
    # Load LHB IBCSO in geographic coordinates.
    # --------------------------------------------------------

    bathy_lhb, raster_extent_lhb = load_ibcso(
        display_bounds,
        GEOGRAPHIC_CRS,
        LHB_DISPLAY_RESOLUTION_DEG,
    )

    # --------------------------------------------------------
    # Figure 2: IBCSO only.
    # --------------------------------------------------------

    plot_lhb_ibcso(
        bathy_lhb,
        raster_extent_lhb,
        display_bounds,
    )

    # --------------------------------------------------------
    # Figure 3: ADD + IBCSO.
    # --------------------------------------------------------

    plot_lhb_add_ibcso(
        polygons_lhb,
        lines_lhb,
        bathy_lhb,
        raster_extent_lhb,
        display_bounds,
    )

    print("\nAll three figures generated successfully.")
