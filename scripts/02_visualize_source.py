"""
02_visualize_source.py

Visualize IBCSO and ADD source data over:
1. Antarctica-wide overview
2. Lutzow-Holm Bay: IBCSO only
3. Lutzow-Holm Bay: IBCSO + ADD

All plotting coordinates are EPSG:3031.
Latitude/longitude graticules are transformed with pyproj.
Raster reprojection is performed with rasterio only.

Source data are read-only.
"""

from pathlib import Path
import sys
import math

import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import Collection
from matplotlib.lines import Line2D

import rasterio
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from rasterio.warp import reproject, transform_bounds

from pyproj import CRS, Transformer


# ============================================================
# CONFIGURATION
# ============================================================

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
    BATHY_VMIN,
    BATHY_VMAX,
    BATHY_CMAP,
    SURFACE_COLORS,
    FIGURE_DPI,
)


# All plotting and raster output use this CRS.
MAP_CRS = CRS.from_epsg(3031)
GEO_CRS = CRS.from_epsg(4326)

# LHB raster resolution in projected meters.
LHB_RESOLUTION_M = 1000

# Graticule intervals.
ANTARCTICA_LON_INTERVAL = 30
ANTARCTICA_LAT_INTERVAL = 10

LHB_LON_INTERVAL = 2
LHB_LAT_INTERVAL = 1

# Graticule appearance.
GRID_COLOR = "0.35"
GRID_ALPHA = 0.65
GRID_LINEWIDTH = 0.55
GRID_LINESTYLE = ":"

# Output.
OUTPUT_DPI = min(FIGURE_DPI, 150)


# ============================================================
# CRS TRANSFORMERS
# ============================================================

TO_MAP = Transformer.from_crs(
    GEO_CRS,
    MAP_CRS,
    always_xy=True,
)

TO_GEO = Transformer.from_crs(
    MAP_CRS,
    GEO_CRS,
    always_xy=True,
)


# ============================================================
# COMMON UTILITIES
# ============================================================

def ensure_output_directory():
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)


def save_figure(fig, filename):
    ensure_output_directory()

    output_file = FIGURE_DIR / filename

    fig.savefig(
        output_file,
        dpi=OUTPUT_DPI,
        bbox_inches="tight",
        facecolor="white",
    )

    print(f"Figure saved: {output_file}")

    plt.close(fig)


def format_longitude(lon):
    if lon < 0:
        return f"{abs(lon):g}°W"
    if lon > 0:
        return f"{lon:g}°E"
    return "0°"


def format_latitude(lat):
    if lat < 0:
        return f"{abs(lat):g}°S"
    if lat > 0:
        return f"{lat:g}°N"
    return "0°"


# ============================================================
# RASTER REPROJECTION
# ============================================================

def reproject_ibcso(
    bounds,
    resolution,
    label,
):
    """
    Reproject IBCSO to EPSG:3031 using rasterio.warp.reproject.

    Parameters
    ----------
    bounds : tuple
        (xmin, ymin, xmax, ymax) in EPSG:3031 meters.

    resolution : float
        Output pixel size in meters.

    label : str
        Name printed in the log.

    Returns
    -------
    masked_array : numpy.ma.MaskedArray
        Reprojected bathymetry.

    extent : tuple
        (xmin, xmax, ymin, ymax) in EPSG:3031.
    """

    xmin, ymin, xmax, ymax = bounds

    width = math.ceil((xmax - xmin) / resolution)
    height = math.ceil((ymax - ymin) / resolution)

    dst_transform = from_origin(
        xmin,
        ymax,
        resolution,
        resolution,
    )

    print("\nReprojecting IBCSO:", label)
    print("Source:", IBCSO_FILE)
    print("Destination CRS:", MAP_CRS.to_string())
    print("Resolution:", resolution, "m")
    print("Raster size:", width, "x", height)

    with rasterio.open(IBCSO_FILE) as src:

        if src.crs is None:
            raise ValueError("IBCSO source CRS is missing.")

        print("Source CRS:", src.crs)
        print("Source NoData:", src.nodata)

        destination = np.full(
            (height, width),
            np.nan,
            dtype=np.float32,
        )

        reproject(
            source=rasterio.band(src, 1),
            destination=destination,
            src_transform=src.transform,
            src_crs=src.crs,
            src_nodata=src.nodata,
            dst_transform=dst_transform,
            dst_crs=MAP_CRS,
            dst_nodata=np.nan,
            resampling=Resampling.bilinear,
            init_dest_nodata=True,
        )

    valid = np.isfinite(destination)

    if not np.any(valid):
        raise ValueError(
            f"No valid IBCSO pixels after reprojection: {label}"
        )

    raster_extent = (
        xmin,
        xmin + width * resolution,
        ymax - height * resolution,
        ymax,
    )

    print("Valid pixels:", int(valid.sum()))
    print("Minimum:", float(np.nanmin(destination)))
    print("Maximum:", float(np.nanmax(destination)))
    print("Extent:", raster_extent)

    return np.ma.masked_invalid(destination), raster_extent


# ============================================================
# ADD DATA
# ============================================================

def load_add_data(bounds=None):
    """
    Read ADD polygon and line data.

    ADD source data are expected to use EPSG:3031.

    Parameters
    ----------
    bounds : tuple or None
        Optional (xmin, ymin, xmax, ymax) in EPSG:3031.
    """

    print("\nLoading ADD polygon data...")

    polygon_kwargs = {
        "layer": "add_coastline_high_res_polygon_v7_12",
    }

    if bounds is not None:
        polygon_kwargs["bbox"] = bounds

    polygons = gpd.read_file(
        ADD_POLYGON_FILE,
        **polygon_kwargs,
    )

    if polygons.crs is None:
        raise ValueError("ADD polygon CRS is missing.")

    if CRS.from_user_input(polygons.crs) != MAP_CRS:
        polygons = polygons.to_crs(MAP_CRS)

    print("Polygon features:", len(polygons))

    if "surface" in polygons.columns:
        print("Surface categories:")
        print(polygons["surface"].value_counts(dropna=False))

    print("\nLoading ADD line data...")

    line_kwargs = {
        "layer": "add_coastline_high_res_line_v7_12",
    }

    if bounds is not None:
        line_kwargs["bbox"] = bounds

    lines = gpd.read_file(
        ADD_LINE_FILE,
        **line_kwargs,
    )

    if lines.crs is None:
        raise ValueError("ADD line CRS is missing.")

    if CRS.from_user_input(lines.crs) != MAP_CRS:
        lines = lines.to_crs(MAP_CRS)

    print("Line features:", len(lines))

    return polygons, lines


# ============================================================
# RASTER DRAWING
# ============================================================

def draw_bathymetry(ax, bathy, extent):
    """
    Draw an EPSG:3031 raster on ordinary Matplotlib Axes.

    No Cartopy transformation is involved.
    """

    xmin, xmax, ymin, ymax = extent

    image = ax.imshow(
        bathy,
        extent=(xmin, xmax, ymin, ymax),
        origin="upper",
        cmap=BATHY_CMAP,
        vmin=BATHY_VMIN,
        vmax=BATHY_VMAX,
        interpolation="nearest",
        zorder=1,
    )

    return image


# ============================================================
# ADD DRAWING
# ============================================================

def draw_add(
    ax,
    polygons,
    lines,
    linewidth=0.3,
    clip_patch=None,
):
    """
    Draw ADD polygons and coastline lines.

    All geometries are in EPSG:3031.
    """

    if not polygons.empty:

        for surface, color in SURFACE_COLORS.items():

            subset = polygons[
                polygons["surface"] == surface
            ]

            if subset.empty:
                continue

            collection_start = len(ax.collections)

            subset.plot(
                ax=ax,
                color=color,
                edgecolor="none",
                linewidth=0,
                alpha=1.0,
                zorder=3,
            )

            if clip_patch is not None:
                for artist in ax.collections[collection_start:]:
                    artist.set_clip_path(clip_patch)

    if not lines.empty:

        collection_start = len(ax.collections)

        lines.plot(
            ax=ax,
            color="black",
            linewidth=linewidth,
            alpha=0.9,
            zorder=4,
        )

        if clip_patch is not None:
            for artist in ax.collections[collection_start:]:
                artist.set_clip_path(clip_patch)


def add_add_legend(ax):
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
        fontsize=8,
        framealpha=0.9,
    )


# ============================================================
# GEOGRAPHIC GRATICULES
# ============================================================

def project_lonlat(lons, lats):
    """
    Transform longitude/latitude arrays to EPSG:3031.
    """

    x, y = TO_MAP.transform(lons, lats)

    return np.asarray(x), np.asarray(y)


def draw_graticules(
    ax,
    bounds,
    lon_interval,
    lat_interval,
    circular=False,
    circle_radius=None,
):
    """
    Draw longitude and latitude lines in EPSG:3031 coordinates.

    The lines are generated in geographic coordinates and
    transformed using pyproj. No Cartopy is used.
    """

    xmin, ymin, xmax, ymax = bounds

    # Clip all graticules to the circular map boundary if requested.
    circle = None

    if circular:
        circle = mpatches.Circle(
            (0, 0),
            circle_radius,
            transform=ax.transData,
            fill=False,
            edgecolor="black",
            linewidth=0.8,
            zorder=8,
        )

    # --------------------------------------------------------
    # Meridians
    # --------------------------------------------------------

    longitude_values = np.arange(
        -180,
        180,
        lon_interval,
    )

    latitude_samples = np.linspace(
        -89.5,
        -45,
        500,
    )

    for lon in longitude_values:

        lons = np.full_like(latitude_samples, lon)

        x, y = project_lonlat(
            lons,
            latitude_samples,
        )

        valid = np.isfinite(x) & np.isfinite(y)

        line, = ax.plot(
            x[valid],
            y[valid],
            color=GRID_COLOR,
            linewidth=GRID_LINEWIDTH,
            linestyle=GRID_LINESTYLE,
            alpha=GRID_ALPHA,
            zorder=5,
        )

        if circular:
            line.set_clip_path(circle)

        # Longitude labels for the overview are placed near
        # the circular boundary.
        if circular:

            label_x, label_y = project_lonlat(
                np.array([lon]),
                np.array([-59.0]),
            )

            ax.annotate(
                format_longitude(lon),
                (label_x[0], label_y[0]),
                xytext=(0, 0),
                textcoords="offset points",
                ha="center",
                va="center",
                fontsize=7,
                zorder=9,
            )

        else:
            inside = (
                valid
                & (x >= xmin)
                & (x <= xmax)
                & (y >= ymin)
                & (y <= ymax)
            )

            if np.any(inside):

                indices = np.where(inside)[0]
                idx = indices[np.argmin(y[inside])]

                ax.annotate(
                    format_longitude(lon),
                    (x[idx], ymin),
                    xytext=(0, -5),
                    textcoords="offset points",
                    ha="center",
                    va="top",
                    fontsize=8,
                    zorder=7,
                )

    # --------------------------------------------------------
    # Parallels
    # --------------------------------------------------------

    latitude_values = np.arange(
        -85,
        -49,
        lat_interval,
    )

    longitude_samples = np.linspace(
        -180,
        180,
        721,
    )

    for lat in latitude_values:

        lats = np.full_like(longitude_samples, lat)

        x, y = project_lonlat(
            longitude_samples,
            lats,
        )

        valid = np.isfinite(x) & np.isfinite(y)

        line, = ax.plot(
            x[valid],
            y[valid],
            color=GRID_COLOR,
            linewidth=GRID_LINEWIDTH,
            linestyle=GRID_LINESTYLE,
            alpha=GRID_ALPHA,
            zorder=5,
        )

        if circular:
            line.set_clip_path(circle)

        inside = (
            valid
            & (x >= xmin)
            & (x <= xmax)
            & (y >= ymin)
            & (y <= ymax)
        )

        if np.any(inside):

            indices = np.where(inside)[0]
            idx = indices[np.argmin(x[inside])]

            if circular:
                ax.annotate(
                    format_latitude(lat),
                    (x[idx], y[idx]),
                    xytext=(-4, 0),
                    textcoords="offset points",
                    ha="right",
                    va="center",
                    fontsize=7,
                    zorder=9,
                )
            else:
                ax.annotate(
                    format_latitude(lat),
                    (xmin, y[idx]),
                    xytext=(-5, 0),
                    textcoords="offset points",
                    ha="right",
                    va="center",
                    fontsize=8,
                    zorder=7,
                )

    if circular:
        ax.add_patch(circle)


# ============================================================
# MAP FORMATTING
# ============================================================

def configure_map_axes(
    ax,
    bounds,
    circular=False,
    circle_radius=None,
):
    xmin, ymin, xmax, ymax = bounds

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    ax.set_aspect("equal", adjustable="box")

    ax.set_facecolor("white")

    ax.set_xlabel("Longitude / Latitude")
    ax.set_ylabel("")

    ax.tick_params(
        left=False,
        bottom=False,
        labelleft=False,
        labelbottom=False,
    )

    for spine in ax.spines.values():
        spine.set_visible(False)

    if circular and circle_radius is not None:
        ax.set_xlim(-circle_radius, circle_radius)
        ax.set_ylim(-circle_radius, circle_radius)


def add_colorbar(fig, ax, image):
    colorbar = fig.colorbar(
        image,
        ax=ax,
        orientation="vertical",
        shrink=0.78,
        pad=0.025,
    )

    colorbar.set_label(
        "Elevation / Bathymetry (m)",
        fontsize=9,
    )

    colorbar.ax.tick_params(labelsize=8)


# ============================================================
# FIGURE 1: ANTARCTICA OVERVIEW
# ============================================================

def plot_antarctica(polygons, lines):

    print("\nCreating Figure 1: Antarctica overview")

    xmin, ymin, xmax, ymax = ANTARCTICA_BOUNDS

    radius = min(
        abs(xmax - xmin),
        abs(ymax - ymin),
    ) / 2.0

    # Use a square raster covering the circular map.
    overview_bounds = (
        -radius,
        -radius,
        radius,
        radius,
    )

    bathy, raster_extent = reproject_ibcso(
        overview_bounds,
        ANTARCTICA_DISPLAY_RESOLUTION,
        "Antarctica overview",
    )

    fig, ax = plt.subplots(
        figsize=(10, 10),
    )

    configure_map_axes(
        ax,
        overview_bounds,
        circular=True,
        circle_radius=radius,
    )

    image = draw_bathymetry(
        ax,
        bathy,
        raster_extent,
    )

    # Clip bathymetry to the circular map.
    map_circle = mpatches.Circle(
        (0, 0),
        radius,
        transform=ax.transData,
    )

    image.set_clip_path(map_circle)

    # Draw geographic gridlines.
    draw_graticules(
        ax,
        overview_bounds,
        lon_interval=ANTARCTICA_LON_INTERVAL,
        lat_interval=ANTARCTICA_LAT_INTERVAL,
        circular=True,
        circle_radius=radius,
    )

    # Draw ADD.
    draw_add(
        ax,
        polygons,
        lines,
        linewidth=0.12,
        clip_patch=map_circle,
    )

    ax.set_title(
        "Antarctica: IBCSO Bathymetry and ADD",
        fontsize=14,
        pad=15,
    )

    add_add_legend(ax)
    add_colorbar(fig, ax, image)

    save_figure(
        fig,
        "01_antarctica_add_ibcso.png",
    )


# ============================================================
# LHB DISPLAY BOUNDS
# ============================================================

def get_lhb_projected_bounds():

    west, south, east, north = LHB_BOUNDS
    margin = LHB_DISPLAY_MARGIN_DEG

    geographic_bounds = (
        west - margin,
        south - margin,
        east + margin,
        north + margin,
    )

    projected_bounds = transform_bounds(
        GEO_CRS,
        MAP_CRS,
        *geographic_bounds,
        densify_pts=101,
    )

    print("\nLHB geographic bounds:")
    print(geographic_bounds)

    print("\nLHB projected bounds (EPSG:3031):")
    print(projected_bounds)

    return projected_bounds


# ============================================================
# FIGURE 2: LHB IBCSO
# ============================================================

def plot_lhb_ibcso(
    bathy,
    raster_extent,
    display_bounds,
):

    print("\nCreating Figure 2: LHB IBCSO")

    fig, ax = plt.subplots(
        figsize=(11, 7),
    )

    image = draw_bathymetry(
        ax,
        bathy,
        raster_extent,
    )

    configure_map_axes(
        ax,
        display_bounds,
    )

    draw_graticules(
        ax,
        display_bounds,
        lon_interval=LHB_LON_INTERVAL,
        lat_interval=LHB_LAT_INTERVAL,
    )

    ax.set_title(
        "IBCSO Bathymetry: Lützow-Holm Bay",
        fontsize=13,
        pad=12,
    )

    add_colorbar(fig, ax, image)

    save_figure(
        fig,
        "02_lhb_ibcso.png",
    )


# ============================================================
# FIGURE 3: LHB IBCSO + ADD
# ============================================================

def plot_lhb_add_ibcso(
    polygons,
    lines,
    bathy,
    raster_extent,
    display_bounds,
):

    print("\nCreating Figure 3: LHB IBCSO + ADD")

    fig, ax = plt.subplots(
        figsize=(11, 7),
    )

    image = draw_bathymetry(
        ax,
        bathy,
        raster_extent,
    )

    configure_map_axes(
        ax,
        display_bounds,
    )

    draw_graticules(
        ax,
        display_bounds,
        lon_interval=LHB_LON_INTERVAL,
        lat_interval=LHB_LAT_INTERVAL,
    )

    draw_add(
        ax,
        polygons,
        lines,
        linewidth=0.35,
    )

    ax.set_title(
        "IBCSO Bathymetry and ADD: Lützow-Holm Bay",
        fontsize=13,
        pad=12,
    )

    add_add_legend(ax)
    add_colorbar(fig, ax, image)

    save_figure(
        fig,
        "03_lhb_add_ibcso.png",
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 65)
    print("WW3 GRIDGEN: SOURCE DATA VISUALIZATION")
    print("=" * 65)

    ensure_output_directory()

    # --------------------------------------------------------
    # 1. Antarctica-wide ADD
    # --------------------------------------------------------

    polygons_all, lines_all = load_add_data()

    # --------------------------------------------------------
    # 2. Antarctica-wide IBCSO + ADD
    # --------------------------------------------------------

    plot_antarctica(
        polygons_all,
        lines_all,
    )

    # --------------------------------------------------------
    # 3. LHB projected bounds
    # --------------------------------------------------------

    lhb_bounds = get_lhb_projected_bounds()

    # --------------------------------------------------------
    # 4. LHB ADD only
    # --------------------------------------------------------

    polygons_lhb, lines_lhb = load_add_data(
        bounds=lhb_bounds,
    )

    # --------------------------------------------------------
    # 5. LHB IBCSO
    # --------------------------------------------------------

    bathy_lhb, raster_extent_lhb = reproject_ibcso(
        lhb_bounds,
        LHB_RESOLUTION_M,
        "Lützow-Holm Bay",
    )

    # --------------------------------------------------------
    # 6. Figure 2
    # --------------------------------------------------------

    plot_lhb_ibcso(
        bathy_lhb,
        raster_extent_lhb,
        lhb_bounds,
    )

    # --------------------------------------------------------
    # 7. Figure 3
    # --------------------------------------------------------

    plot_lhb_add_ibcso(
        polygons_lhb,
        lines_lhb,
        bathy_lhb,
        raster_extent_lhb,
        lhb_bounds,
    )

    print("\n" + "=" * 65)
    print("All three figures generated successfully.")
    print("=" * 65)


if __name__ == "__main__":
    main()