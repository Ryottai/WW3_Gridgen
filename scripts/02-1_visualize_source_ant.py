"""
02-1_visualize_source_ant.py

Antarctica-wide visualization of IBCSO bathymetry and ADD.
The approved Antarctica plotting implementation is preserved.
All plotting coordinates are EPSG:3031.
"""

from pathlib import Path
import sys
import math

import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap

import rasterio
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from rasterio.warp import reproject

from pyproj import CRS, Transformer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import (
    ADD_LINE_FILE, ADD_POLYGON_FILE, IBCSO_FILE, FIGURE_DIR,
    ADD_CRS, GEOGRAPHIC_CRS, ANTARCTICA_DISPLAY_RESOLUTION,
    SURFACE_COLORS, FIGURE_DPI,
    DEPTH_VMIN, DEPTH_VMAX, DEPTH_TICKS, DEPTH_COLORS,
    DEPTH_COLORBAR_LABEL, DEPTH_COLORBAR_EXTEND,
    DEPTH_COLORBAR_SHRINK, DEPTH_COLORBAR_PAD,
)

MAP_CRS = CRS.from_epsg(3031)
GEO_CRS = CRS.from_epsg(4326)
ANTARCTICA_LON_INTERVAL = 30
ANTARCTICA_LAT_INTERVAL = 10
GRID_COLOR = "0.35"
GRID_ALPHA = 0.65
GRID_LINEWIDTH = 0.55
GRID_LINESTYLE = ":"
DEPTH_CMAP = LinearSegmentedColormap.from_list(
    "shallow_brown_deep_blue", DEPTH_COLORS
)
DEPTH_CMAP.set_bad((1.0, 1.0, 1.0, 0.0))
OUTPUT_DPI = min(FIGURE_DPI, 150)

TO_MAP = Transformer.from_crs(GEO_CRS, MAP_CRS, always_xy=True)


def ensure_output_directory():
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)


def save_figure(fig, filename):
    ensure_output_directory()
    output_file = FIGURE_DIR / filename
    fig.savefig(
        output_file, dpi=OUTPUT_DPI, bbox_inches="tight",
        facecolor="white",
    )
    print(f"Figure saved: {output_file}")
    plt.close(fig)


def reproject_ibcso(bounds, resolution, label):
    """Reproject IBCSO to EPSG:3031 and convert elevation to depth."""
    xmin, ymin, xmax, ymax = bounds
    width = math.ceil((xmax - xmin) / resolution)
    height = math.ceil((ymax - ymin) / resolution)
    dst_transform = from_origin(xmin, ymax, resolution, resolution)

    print("\nReprojecting IBCSO:", label)
    print("Source:", IBCSO_FILE)
    print("Destination CRS:", MAP_CRS.to_string())
    print("Resolution:", resolution, "m")
    print("Raster size:", width, "x", height)

    with rasterio.open(IBCSO_FILE) as src:
        if src.crs is None:
            raise ValueError("IBCSO source CRS is missing.")
        elevation = np.full((height, width), np.nan, dtype=np.float32)
        reproject(
            source=rasterio.band(src, 1),
            destination=elevation,
            src_transform=src.transform,
            src_crs=src.crs,
            src_nodata=src.nodata,
            dst_transform=dst_transform,
            dst_crs=MAP_CRS,
            dst_nodata=np.nan,
            resampling=Resampling.bilinear,
            init_dest_nodata=True,
        )

    valid_ocean = np.isfinite(elevation) & (elevation < 0)
    depth = np.full_like(elevation, np.nan, dtype=np.float32)
    depth[valid_ocean] = -elevation[valid_ocean]
    if not np.any(valid_ocean):
        raise ValueError(f"No valid ocean pixels after reprojection: {label}")

    raster_extent = (
        xmin, xmin + width * resolution,
        ymax - height * resolution, ymax,
    )
    print("Valid ocean pixels:", int(valid_ocean.sum()))
    print("Minimum depth:", float(np.nanmin(depth)))
    print("Maximum depth:", float(np.nanmax(depth)))
    print("Extent:", raster_extent)
    return np.ma.masked_invalid(depth), raster_extent


def load_add_data():
    """Read the complete ADD polygon and line datasets."""
    print("\nLoading ADD polygon data...")
    polygons = gpd.read_file(
        ADD_POLYGON_FILE,
        layer="add_coastline_high_res_polygon_v7_12",
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
    lines = gpd.read_file(
        ADD_LINE_FILE,
        layer="add_coastline_high_res_line_v7_12",
    )
    if lines.crs is None:
        raise ValueError("ADD line CRS is missing.")
    if CRS.from_user_input(lines.crs) != MAP_CRS:
        lines = lines.to_crs(MAP_CRS)
    print("Line features:", len(lines))
    return polygons, lines


def draw_bathymetry(ax, bathy, extent):
    xmin, xmax, ymin, ymax = extent
    return ax.imshow(
        bathy,
        extent=(xmin, xmax, ymin, ymax),
        origin="upper",
        cmap=DEPTH_CMAP,
        vmin=DEPTH_VMIN,
        vmax=DEPTH_VMAX,
        interpolation="nearest",
        zorder=1,
    )


def draw_add(ax, polygons, lines, linewidth=0.3, clip_patch=None):
    if not polygons.empty:
        for surface, color in SURFACE_COLORS.items():
            subset = polygons[polygons["surface"] == surface]
            if subset.empty:
                continue
            collection_start = len(ax.collections)
            subset.plot(
                ax=ax, color=color, edgecolor="none", linewidth=0,
                alpha=1.0, zorder=3,
            )
            if clip_patch is not None:
                for artist in ax.collections[collection_start:]:
                    artist.set_clip_path(clip_patch)

    if not lines.empty:
        collection_start = len(ax.collections)
        lines.plot(
            ax=ax, color="black", linewidth=linewidth,
            alpha=0.9, zorder=4,
        )
        if clip_patch is not None:
            for artist in ax.collections[collection_start:]:
                artist.set_clip_path(clip_patch)


def add_add_legend(ax):
    handles = [
        mpatches.Patch(color=color, label=surface)
        for surface, color in SURFACE_COLORS.items()
    ]
    ax.legend(
        handles=handles, loc="lower left",
        fontsize=8, framealpha=0.9,
    )


def project_lonlat(lons, lats):
    x, y = TO_MAP.transform(lons, lats)
    return np.asarray(x), np.asarray(y)


def draw_graticules(
    ax, bounds, lon_interval, lat_interval,
    circular=False, circle_radius=None, label_latitudes=None,
    geographic_bounds=None,
):
    """Draw the approved Antarctica overview graticules."""
    if not circular:
        raise ValueError("This script only supports the Antarctica overview.")
    if circle_radius is None:
        raise ValueError("circle_radius must be specified for circular maps.")

    longitude_values = np.arange(-180, 180, lon_interval)
    latitude_values = [-60, -70, -80]
    if label_latitudes is None:
        label_latitudes = [-60, -70, -80, -90]
    label_latitudes = [int(lat) for lat in label_latitudes]

    circle = mpatches.Circle(
        (0, 0), circle_radius, transform=ax.transData,
        fill=False, edgecolor="black", linewidth=0.8, zorder=8,
    )

    latitude_samples = np.linspace(-89.999, -60.0, 500)
    for lon in longitude_values:
        lons = np.full_like(latitude_samples, lon)
        x, y = project_lonlat(lons, latitude_samples)
        valid = np.isfinite(x) & np.isfinite(y)
        line, = ax.plot(
            x[valid], y[valid], color=GRID_COLOR,
            linewidth=GRID_LINEWIDTH, linestyle=GRID_LINESTYLE,
            alpha=GRID_ALPHA, zorder=5,
        )
        line.set_clip_path(circle)

        label_x, label_y = project_lonlat(
            np.array([lon]), np.array([-60.0])
        )
        x0, y0 = label_x[0], label_y[0]
        radius = math.hypot(x0, y0)
        if radius > 0:
            offset_distance = circle_radius * 0.025
            x_label = x0 + offset_distance * x0 / radius
            y_label = y0 + offset_distance * y0 / radius
            horizontal_alignment = (
                "left" if x0 > 0.1 * circle_radius
                else "right" if x0 < -0.1 * circle_radius
                else "center"
            )
            vertical_alignment = (
                "bottom" if y0 > 0.1 * circle_radius
                else "top" if y0 < -0.1 * circle_radius
                else "center"
            )
            ax.annotate(
                f"{abs(lon):g}°{'W' if lon < 0 else 'E' if lon > 0 else ''}"
                if lon != 0 else "0°",
                (x_label, y_label),
                ha=horizontal_alignment, va=vertical_alignment,
                fontsize=7, zorder=9,
            )

    longitude_samples = np.linspace(-180, 180, 721)
    for lat in latitude_values:
        lats = np.full_like(longitude_samples, lat)
        x, y = project_lonlat(longitude_samples, lats)
        valid = np.isfinite(x) & np.isfinite(y)
        line, = ax.plot(
            x[valid], y[valid], color=GRID_COLOR,
            linewidth=GRID_LINEWIDTH, linestyle=GRID_LINESTYLE,
            alpha=GRID_ALPHA, zorder=5,
        )
        line.set_clip_path(circle)
        if lat in label_latitudes:
            label_x, label_y = project_lonlat(
                np.array([-180.0]), np.array([lat])
            )
            ax.annotate(
                f"{abs(lat):g}°S", (label_x[0], label_y[0]),
                xytext=(0, 4), textcoords="offset points",
                ha="center", va="bottom", fontsize=7, zorder=9,
            )

    if -90 in label_latitudes:
        ax.plot(
            0, 0, marker="o", markersize=2.5,
            color=GRID_COLOR, zorder=9,
        )
        ax.annotate(
            "90°S", (0, 0), xytext=(6, 5),
            textcoords="offset points", ha="left",
            va="bottom", fontsize=8, zorder=9,
        )
    ax.add_patch(circle)


def configure_map_axes(ax, bounds):
    xmin, ymin, xmax, ymax = bounds
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal", adjustable="box")
    ax.set_facecolor("white")
    ax.set_ylabel("")
    ax.tick_params(
        left=False, bottom=False,
        labelleft=False, labelbottom=False,
    )
    for spine in ax.spines.values():
        spine.set_visible(False)


def add_colorbar(fig, ax, image):
    colorbar = fig.colorbar(
        image, ax=ax, orientation="vertical",
        shrink=DEPTH_COLORBAR_SHRINK, pad=DEPTH_COLORBAR_PAD,
        extend=DEPTH_COLORBAR_EXTEND,
        ticks=np.arange(
            DEPTH_VMIN, DEPTH_VMAX + DEPTH_TICKS, DEPTH_TICKS
        ),
    )
    colorbar.set_label(DEPTH_COLORBAR_LABEL, fontsize=9)
    colorbar.ax.tick_params(labelsize=8)
    colorbar.ax.invert_yaxis()
    return colorbar


def plot_antarctica(polygons, lines):
    print("\nCreating Figure 1: Antarctica overview")

    x60, y60 = TO_MAP.transform(0.0, -60.0)
    radius = math.hypot(x60, y60)
    print("60°S map radius:", radius / 1000, "km")

    raster_bounds = (-radius, -radius, radius, radius)
    padding = 1.08
    axes_bounds = (
        -radius * padding, -radius * padding,
        radius * padding, radius * padding,
    )

    bathy, raster_extent = reproject_ibcso(
        raster_bounds, ANTARCTICA_DISPLAY_RESOLUTION,
        "Antarctica overview",
    )

    fig, ax = plt.subplots(figsize=(10, 10))
    configure_map_axes(ax, axes_bounds)

    image = draw_bathymetry(ax, bathy, raster_extent)
    map_circle = mpatches.Circle(
        (0, 0), radius, transform=ax.transData,
    )
    image.set_clip_path(map_circle)

    draw_graticules(
        ax, axes_bounds, lon_interval=30, lat_interval=10,
        circular=True, circle_radius=radius,
        label_latitudes=[-60, -70, -80],
    )

    draw_add(
        ax, polygons, lines, linewidth=0.12,
        clip_patch=map_circle,
    )

    ax.set_title(
        "Antarctica: IBCSO Bathymetry and ADD",
        fontsize=14, pad=15,
    )
    add_add_legend(ax)
    add_colorbar(fig, ax, image)
    save_figure(fig, "01_antarctica_add_ibcso.png")


def main():
    print("=" * 65)
    print("WW3 GRIDGEN: ANTARCTICA SOURCE DATA VISUALIZATION")
    print("=" * 65)
    ensure_output_directory()
    polygons_all, lines_all = load_add_data()
    plot_antarctica(polygons_all, lines_all)
    print("\n" + "=" * 65)
    print("Antarctica figure generated successfully.")
    print("=" * 65)


if __name__ == "__main__":
    main()
