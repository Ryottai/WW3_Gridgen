"""
02-2_visualize_source_lhb.py

Visualize Lützow-Holm Bay:
1. IBCSO bathymetry only
2. IBCSO bathymetry + ADD

This is the LHB-only script. Its graticule implementation is
currently retained from the original pyproj-based version;
Cartopy conversion can be developed independently here.
All data coordinates are EPSG:3031.
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
from rasterio.warp import reproject, transform_bounds

from pyproj import CRS, Transformer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import (
    ADD_LINE_FILE, ADD_POLYGON_FILE, IBCSO_FILE, FIGURE_DIR,
    GEOGRAPHIC_CRS, LHB_BOUNDS, LHB_DISPLAY_MARGIN_DEG,
    SURFACE_COLORS, FIGURE_DPI,
    DEPTH_VMIN, DEPTH_VMAX, DEPTH_TICKS, DEPTH_COLORS,
    DEPTH_COLORBAR_LABEL, DEPTH_COLORBAR_EXTEND,
    DEPTH_COLORBAR_SHRINK, DEPTH_COLORBAR_PAD,
)

MAP_CRS = CRS.from_epsg(3031)
GEO_CRS = CRS.from_epsg(4326)
LHB_RESOLUTION_M = 1000
LHB_LON_INTERVAL = 2
LHB_LAT_INTERVAL = 1
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


def load_add_data(bounds=None):
    print("\nLoading ADD polygon data...")
    polygon_kwargs = {
        "layer": "add_coastline_high_res_polygon_v7_12",
    }
    if bounds is not None:
        polygon_kwargs["bbox"] = bounds
    polygons = gpd.read_file(ADD_POLYGON_FILE, **polygon_kwargs)
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
    lines = gpd.read_file(ADD_LINE_FILE, **line_kwargs)
    if lines.crs is None:
        raise ValueError("ADD line CRS is missing.")
    if CRS.from_user_input(lines.crs) != MAP_CRS:
        lines = lines.to_crs(MAP_CRS)
    print("Line features:", len(lines))
    return polygons, lines


def draw_bathymetry(ax, bathy, extent):
    xmin, xmax, ymin, ymax = extent
    return ax.imshow(
        bathy, extent=(xmin, xmax, ymin, ymax),
        origin="upper", cmap=DEPTH_CMAP,
        vmin=DEPTH_VMIN, vmax=DEPTH_VMAX,
        interpolation="nearest", zorder=1,
    )


def draw_add(ax, polygons, lines, linewidth=0.3):
    if not polygons.empty:
        for surface, color in SURFACE_COLORS.items():
            subset = polygons[polygons["surface"] == surface]
            if subset.empty:
                continue
            subset.plot(
                ax=ax, color=color, edgecolor="none",
                linewidth=0, alpha=1.0, zorder=3,
            )
    if not lines.empty:
        lines.plot(
            ax=ax, color="black", linewidth=linewidth,
            alpha=0.9, zorder=4,
        )


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


def draw_graticules(ax, bounds, lon_interval, lat_interval,
                    geographic_bounds=None):
    """Original pyproj-based LHB graticules (to be replaced by Cartopy)."""
    if geographic_bounds is None:
        raise ValueError("geographic_bounds is required for LHB graticules.")

    west, south, east, north = geographic_bounds
    longitude_values = np.arange(
        math.ceil(west / lon_interval) * lon_interval,
        east + lon_interval * 0.5,
        lon_interval,
    )
    latitude_values = np.arange(
        math.ceil(south / lat_interval) * lat_interval,
        north + lat_interval * 0.5,
        lat_interval,
    )

    latitude_samples = np.linspace(south, north, 500)
    for lon in longitude_values:
        lons = np.full_like(latitude_samples, lon)
        x, y = project_lonlat(lons, latitude_samples)
        valid = np.isfinite(x) & np.isfinite(y)
        ax.plot(
            x[valid], y[valid], color=GRID_COLOR,
            linewidth=GRID_LINEWIDTH, linestyle=GRID_LINESTYLE,
            alpha=GRID_ALPHA, zorder=5,
        )
        label_x, label_y = project_lonlat(
            np.array([lon]), np.array([north])
        )
        ax.annotate(
            f"{abs(lon):g}°{'W' if lon < 0 else 'E' if lon > 0 else ''}"
            if lon != 0 else "0°",
            (label_x[0], label_y[0]), xytext=(0, 7),
            textcoords="offset points", ha="center",
            va="bottom", fontsize=8, zorder=9,
        )

    longitude_samples = np.linspace(west, east, 721)
    for lat in latitude_values:
        lats = np.full_like(longitude_samples, lat)
        x, y = project_lonlat(longitude_samples, lats)
        valid = np.isfinite(x) & np.isfinite(y)
        ax.plot(
            x[valid], y[valid], color=GRID_COLOR,
            linewidth=GRID_LINEWIDTH, linestyle=GRID_LINESTYLE,
            alpha=GRID_ALPHA, zorder=5,
        )
        label_x, label_y = project_lonlat(
            np.array([west]), np.array([lat])
        )
        ax.annotate(
            f"{abs(lat):g}°S" if lat < 0 else f"{lat:g}°N",
            (label_x[0], label_y[0]), xytext=(-7, 0),
            textcoords="offset points", ha="right",
            va="center", fontsize=8, zorder=9,
        )


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


def get_lhb_projected_bounds():
    west, south, east, north = LHB_BOUNDS
    margin = LHB_DISPLAY_MARGIN_DEG
    geographic_bounds = (
        west - margin, south - margin,
        east + margin, north + margin,
    )
    projected_bounds = transform_bounds(
        GEO_CRS, MAP_CRS, *geographic_bounds, densify_pts=101,
    )
    print("\nLHB geographic bounds:")
    print(geographic_bounds)
    print("\nLHB projected bounds (EPSG:3031):")
    print(projected_bounds)
    return projected_bounds, geographic_bounds


def plot_lhb_ibcso(bathy, raster_extent, display_bounds, geographic_bounds):
    print("\nCreating Figure 2: LHB IBCSO")
    fig, ax = plt.subplots(figsize=(11, 7))
    image = draw_bathymetry(ax, bathy, raster_extent)
    configure_map_axes(ax, display_bounds)
    draw_graticules(
        ax, display_bounds,
        lon_interval=LHB_LON_INTERVAL,
        lat_interval=LHB_LAT_INTERVAL,
        geographic_bounds=geographic_bounds,
    )
    ax.set_title(
        "IBCSO Bathymetry: Lützow-Holm Bay",
        fontsize=13, pad=12,
    )
    add_colorbar(fig, ax, image)
    save_figure(fig, "02_lhb_ibcso.png")


def plot_lhb_add_ibcso(
    polygons, lines, bathy, raster_extent,
    display_bounds, geographic_bounds,
):
    print("\nCreating Figure 3: LHB IBCSO + ADD")
    fig, ax = plt.subplots(figsize=(11, 7))
    image = draw_bathymetry(ax, bathy, raster_extent)
    configure_map_axes(ax, display_bounds)
    draw_graticules(
        ax, display_bounds,
        lon_interval=LHB_LON_INTERVAL,
        lat_interval=LHB_LAT_INTERVAL,
        geographic_bounds=geographic_bounds,
    )
    draw_add(ax, polygons, lines, linewidth=0.35)
    ax.set_title(
        "IBCSO Bathymetry and ADD: Lützow-Holm Bay",
        fontsize=13, pad=12,
    )
    add_add_legend(ax)
    add_colorbar(fig, ax, image)
    save_figure(fig, "03_lhb_add_ibcso.png")


def main():
    print("=" * 65)
    print("WW3 GRIDGEN: LHB SOURCE DATA VISUALIZATION")
    print("=" * 65)
    ensure_output_directory()

    lhb_bounds, lhb_geo_bounds = get_lhb_projected_bounds()
    polygons_lhb, lines_lhb = load_add_data(bounds=lhb_bounds)
    bathy_lhb, raster_extent_lhb = reproject_ibcso(
        lhb_bounds, LHB_RESOLUTION_M, "Lützow-Holm Bay",
    )
    plot_lhb_ibcso(
        bathy_lhb, raster_extent_lhb,
        lhb_bounds, lhb_geo_bounds,
    )
    plot_lhb_add_ibcso(
        polygons_lhb, lines_lhb, bathy_lhb,
        raster_extent_lhb, lhb_bounds, lhb_geo_bounds,
    )
    print("\n" + "=" * 65)
    print("Both LHB figures generated successfully.")
    print("=" * 65)


if __name__ == "__main__":
    main()
