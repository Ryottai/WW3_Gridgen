
"""
02_visualize_source.py

Generate three separate figures for ADD and IBCSO source data.

1. Antarctica-wide ADD + IBCSO
2. Lutzow-Holm Bay IBCSO bathymetry
3. Lutzow-Holm Bay ADD + IBCSO

All source data are read-only.
"""

from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
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

# Lützow-Holm Bay display extent (longitude, latitude).
LONLAT_BOUNDS = (30.0, -70.0, 42.0, -66.0)

# Antarctica-wide display extent in EPSG:3031 (meters).
ANTARCTICA_BOUNDS = (
    -3_300_000,
    -3_300_000,
    3_300_000,
    3_300_000,
)

# Display raster resolutions.
ANTARCTICA_RESOLUTION = 10_000.0  # 10 km
LHB_RESOLUTION = 1_000.0          # 1 km

# ADD polygon colors.
SURFACE_COLORS = {
    "land": "#777777",
    "ice shelf": "#80c9e8",
    "ice tongue": "#2878b5",
    "rumple": "#e6a23c",
}

# IBCSO display range.
BATHY_VMIN = -5000
BATHY_VMAX = 1500

# Figure settings.
DPI = 200


# ============================================================
# COMMON FUNCTIONS
# ============================================================

def format_km(value, position):
    return f"{value / 1000:.0f}"


def format_axis(ax):

    ax.xaxis.set_major_formatter(
        FuncFormatter(format_km)
    )
    ax.yaxis.set_major_formatter(
        FuncFormatter(format_km)
    )

    ax.set_xlabel("Easting (km)")
    ax.set_ylabel("Northing (km)")

    ax.set_aspect("equal")
    ax.grid(
        True,
        linestyle=":",
        linewidth=0.5,
        alpha=0.5,
    )


def save_figure(fig, filename):

    output_file = OUTPUT_DIR / filename

    fig.savefig(
        output_file,
        dpi=DPI,
        bbox_inches="tight",
    )

    print(f"\nFigure saved: {output_file}")

    return output_file


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

def load_add_data(plot_bounds=None):

    """
    Load ADD polygon and line data.

    Parameters
    ----------
    plot_bounds : tuple or None
        Bounds in EPSG:3031 (xmin, ymin, xmax, ymax).
        If None, load the entire dataset.
    """

    print("\nLoading ADD polygon data...")

    if plot_bounds is None:
        polygons = gpd.read_file(POLYGON_FILE)
    else:
        polygons = gpd.read_file(
            POLYGON_FILE,
            bbox=plot_bounds,
        )

    if polygons.crs is None:
        raise ValueError("ADD polygon CRS is undefined.")

    polygons = polygons.to_crs(ADD_CRS)

    print("Polygon features:", len(polygons))
    print("Surface categories:")
    print(polygons["surface"].value_counts(dropna=False))

    print("\nLoading ADD line data...")

    if plot_bounds is None:
        lines = gpd.read_file(LINE_FILE)
    else:
        lines = gpd.read_file(
            LINE_FILE,
            bbox=plot_bounds,
        )

    if lines.crs is None:
        raise ValueError("ADD line CRS is undefined.")

    lines = lines.to_crs(ADD_CRS)

    print("Line features:", len(lines))

    return polygons, lines


# ============================================================
# LOAD IBCSO DATA
# ============================================================

def load_ibcso(plot_bounds, display_resolution):

    """
    Reproject only the requested display area.

    Original IBCSO GeoTIFF is not modified.

    Parameters
    ----------
    plot_bounds : tuple
        EPSG:3031 bounds (xmin, ymin, xmax, ymax).
    display_resolution : float
        Output display pixel size in meters.
    """

    xmin, ymin, xmax, ymax = plot_bounds

    width = round(
        (xmax - xmin) / display_resolution
    )
    height = round(
        (ymax - ymin) / display_resolution
    )

    dst_transform = from_origin(
        xmin,
        ymax,
        display_resolution,
        display_resolution,
    )

    print("\nLoading IBCSO bathymetry...")
    print("Display CRS:", ADD_CRS)
    print("Display resolution:", display_resolution, "m")
    print("Display raster size:", width, "x", height)

    with rasterio.open(IBCSO_FILE) as src:

        if src.crs is None:
            raise ValueError("IBCSO CRS is undefined.")

        print("Original CRS:", src.crs)

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

    print("Valid pixels:", bathy.count())

    raster_extent = (
        xmin,
        xmax,
        ymin,
        ymax,
    )

    return bathy, raster_extent


# ============================================================
# PLOT IBCSO BACKGROUND
# ============================================================

def draw_ibcso(ax, bathy, raster_extent):

    image = ax.imshow(
        bathy,
        extent=raster_extent,
        origin="upper",
        cmap="terrain",
        vmin=BATHY_VMIN,
        vmax=BATHY_VMAX,
        interpolation="nearest",
    )

    return image


# ============================================================
# PLOT ADD OVERLAY
# ============================================================

def draw_add(ax, polygons, lines, overview=False):

    # Draw polygons by surface category.
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
        ANTARCTICA_RESOLUTION,
    )

    fig, ax = plt.subplots(
        figsize=(10, 10),
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
        overview=True,
    )

    ax.set_xlim(
        ANTARCTICA_BOUNDS[0],
        ANTARCTICA_BOUNDS[2],
    )
    ax.set_ylim(
        ANTARCTICA_BOUNDS[1],
        ANTARCTICA_BOUNDS[3],
    )

    ax.set_title("Antarctica: IBCSO + ADD")

    format_axis(ax)

    fig.colorbar(
        image,
        ax=ax,
        label="Elevation / bathymetry (m)",
        shrink=0.75,
    )

    save_figure(
        fig,
        "01_antarctica_add_ibcso.png",
    )

    plt.close(fig)


# ============================================================
# FIGURE 2: LHB IBCSO ONLY
# ============================================================

def plot_lhb_ibcso(bathy, raster_extent, plot_bounds):

    print("\nCreating Figure 2: LHB IBCSO")

    fig, ax = plt.subplots(
        figsize=(10, 9),
        constrained_layout=True,
    )

    image = draw_ibcso(
        ax,
        bathy,
        raster_extent,
    )

    ax.set_xlim(
        plot_bounds[0],
        plot_bounds[2],
    )
    ax.set_ylim(
        plot_bounds[1],
        plot_bounds[3],
    )

    ax.set_title("IBCSO bathymetry: Lützow-Holm Bay")

    format_axis(ax)

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

    plt.close(fig)


# ============================================================
# FIGURE 3: LHB ADD + IBCSO
# ============================================================

def plot_lhb_add_ibcso(
    polygons,
    lines,
    bathy,
    raster_extent,
    plot_bounds,
):

    print("\nCreating Figure 3: LHB ADD + IBCSO")

    fig, ax = plt.subplots(
        figsize=(10, 9),
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

    ax.set_xlim(
        plot_bounds[0],
        plot_bounds[2],
    )
    ax.set_ylim(
        plot_bounds[1],
        plot_bounds[3],
    )

    ax.set_title("IBCSO + ADD: Lützow-Holm Bay")

    format_axis(ax)

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

    plt.close(fig)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("=" * 65)
    print("WW3 GRIDGEN: SOURCE DATA VISUALIZATION")
    print("=" * 65)

    # --------------------------------------------------------
    # Define LHB display bounds in EPSG:3031.
    # --------------------------------------------------------

    lhb_bounds = transform_bounds(
        "EPSG:4326",
        ADD_CRS,
        *LONLAT_BOUNDS,
        densify_pts=21,
    )

    xmin, ymin, xmax, ymax = lhb_bounds

    margin = 10_000.0

    lhb_bounds = (
        xmin - margin,
        ymin - margin,
        xmax + margin,
        ymax + margin,
    )

    print("\nLHB display bounds in EPSG:3031:")
    print(lhb_bounds)

    # --------------------------------------------------------
    # Load all ADD data for Antarctica overview.
    # --------------------------------------------------------

    polygons_all, lines_all = load_add_data()

    # --------------------------------------------------------
    # Figure 1: Antarctica-wide ADD + IBCSO.
    # --------------------------------------------------------

    plot_antarctica(
        polygons_all,
        lines_all,
    )

    # --------------------------------------------------------
    # Load ADD data within LHB display area.
    # --------------------------------------------------------

    polygons_lhb, lines_lhb = load_add_data(
        plot_bounds=lhb_bounds,
    )

    # --------------------------------------------------------
    # Load LHB IBCSO once and reuse for Figures 2 and 3.
    # --------------------------------------------------------

    bathy_lhb, raster_extent_lhb = load_ibcso(
        lhb_bounds,
        LHB_RESOLUTION,
    )

    # --------------------------------------------------------
    # Figure 2: LHB IBCSO only.
    # --------------------------------------------------------

    plot_lhb_ibcso(
        bathy_lhb,
        raster_extent_lhb,
        lhb_bounds,
    )

    # --------------------------------------------------------
    # Figure 3: LHB ADD + IBCSO.
    # --------------------------------------------------------

    plot_lhb_add_ibcso(
        polygons_lhb,
        lines_lhb,
        bathy_lhb,
        raster_extent_lhb,
        lhb_bounds,
    )

    print("\nAll three figures generated successfully.")
