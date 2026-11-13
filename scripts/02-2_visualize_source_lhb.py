"""
02-3_visualize_lhb_polar.py

Visualize Lützow-Holm Bay in a South Polar Stereographic projection.

1. IBCSO bathymetry only
2. IBCSO bathymetry + ADD

Design:
- The map projection is SouthPolarStereo (true-scale latitude 71S, WGS84),
  i.e. identical to EPSG:3031 except that the central meridian is rotated to
  the middle of the study area so the bay appears "north-up".
  Set MAP_CENTRAL_LON = 0.0 to get exactly the standard EPSG:3031 orientation.
- The display window is the lon/lat rectangle itself (a curved "wedge" in the
  polar stereographic plane). It is used as the axes boundary, so nothing
  outside the window is drawn.
- IBCSO and ADD are reprojected into the map CRS, so every artist is plotted
  in native map coordinates (no on-the-fly warping).
- Graticules and their labels are drawn manually along the wedge edges, so
  they do not depend on Cartopy's gridliner label placement.
"""

from pathlib import Path
import sys
import math

import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.path as mpath
from matplotlib.colors import LinearSegmentedColormap

import cartopy.crs as ccrs

import rasterio
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from rasterio.warp import reproject

from pyproj import CRS, Transformer


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import (
    ADD_LINE_FILE, ADD_POLYGON_FILE, IBCSO_FILE, FIGURE_DIR,
    LHB_BOUNDS, LHB_DISPLAY_MARGIN_DEG,
    SURFACE_COLORS, FIGURE_DPI,
    DEPTH_VMIN, DEPTH_VMAX, DEPTH_TICKS, DEPTH_COLORS,
    DEPTH_COLORBAR_LABEL, DEPTH_COLORBAR_EXTEND,
    DEPTH_COLORBAR_SHRINK, DEPTH_COLORBAR_PAD,
)


# ----------------------------------------------------------------------
# Projection setup
# ----------------------------------------------------------------------
TRUE_SCALE_LAT = -71.0

_west, _south, _east, _north = LHB_BOUNDS
MAP_CENTRAL_LON = round(0.5 * (_west + _east), 2)   # 0.0 -> plain EPSG:3031

PROJ = ccrs.SouthPolarStereo(
    central_longitude=MAP_CENTRAL_LON,
    true_scale_latitude=TRUE_SCALE_LAT,
)
MAP_CRS = CRS.from_proj4(PROJ.proj4_init)

# ADD files are assumed to be stored in EPSG:3031 (used for the bbox filter).
CRS_3031 = CRS.from_epsg(3031)
CRS_GEO = CRS.from_epsg(4326)

# ----------------------------------------------------------------------
# Display settings
# ----------------------------------------------------------------------
RESOLUTION_M = 1000
LON_INTERVAL = 2
LAT_INTERVAL = 1

GRID_COLOR = "0.35"
GRID_ALPHA = 0.65
GRID_LINEWIDTH = 0.55
GRID_LINESTYLE = ":"
GRID_LABEL_SIZE = 8
GRID_LABEL_COLOR = "0.2"
LABEL_OFFSET_PT = 4

FIGURE_SIZE = (11, 7)
OUTPUT_DPI = min(FIGURE_DPI, 150)

DEPTH_CMAP = LinearSegmentedColormap.from_list(
    "shallow_brown_deep_blue", DEPTH_COLORS
)
DEPTH_CMAP.set_bad((1.0, 1.0, 1.0, 0.0))


# ----------------------------------------------------------------------
# Utilities
# ----------------------------------------------------------------------
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


def geographic_window():
    """Return (west, east, south, north) including the display margin."""
    west, south, east, north = LHB_BOUNDS
    m = LHB_DISPLAY_MARGIN_DEG
    window = (west - m, east + m, south - m, north + m)
    print("\nGeographic window (west, east, south, north):", window)
    return window


def window_outline_lonlat(window, n=200):
    """Closed outline of the lon/lat rectangle, densified along each edge."""
    west, east, south, north = window
    lons = np.concatenate((
        np.linspace(west, east, n),      # bottom (south edge)
        np.full(n, east),                # right
        np.linspace(east, west, n),      # top (north edge)
        np.full(n, west),                # left
        [west],                          # close
    ))
    lats = np.concatenate((
        np.full(n, south),
        np.linspace(south, north, n),
        np.full(n, north),
        np.linspace(north, south, n),
        [south],
    ))
    return lons, lats


def get_window_geometry(window):
    """
    Project the window outline.

    Returns
    -------
    outline_xy : (xs, ys) in the map CRS
    bounds_map : (xmin, ymin, xmax, ymax) in the map CRS
    bounds_3031 : (xmin, ymin, xmax, ymax) in EPSG:3031 (for ADD bbox filter)
    """
    lons, lats = window_outline_lonlat(window)

    to_map = Transformer.from_crs(CRS_GEO, MAP_CRS, always_xy=True)
    xs, ys = to_map.transform(lons, lats)

    to_3031 = Transformer.from_crs(CRS_GEO, CRS_3031, always_xy=True)
    xs31, ys31 = to_3031.transform(lons, lats)

    if not (np.all(np.isfinite(xs)) and np.all(np.isfinite(ys))):
        raise ValueError("Window outline produced non-finite coordinates.")

    bounds_map = (xs.min(), ys.min(), xs.max(), ys.max())
    pad = 10_000.0  # 10 km margin for the ADD bbox filter
    bounds_3031 = (
        xs31.min() - pad, ys31.min() - pad,
        xs31.max() + pad, ys31.max() + pad,
    )

    print("Central longitude:", MAP_CENTRAL_LON)
    print("Map bounds (xmin, ymin, xmax, ymax):", bounds_map)
    return (xs, ys), bounds_map, bounds_3031


# ----------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------
def reproject_ibcso(bounds_map, resolution, label):
    """Reproject IBCSO into the map CRS over the window's bounding box."""
    xmin, ymin, xmax, ymax = bounds_map

    width = math.ceil((xmax - xmin) / resolution)
    height = math.ceil((ymax - ymin) / resolution)
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid destination raster size: {width} x {height}")

    dst_transform = from_origin(xmin, ymax, resolution, resolution)

    print("\nReprojecting IBCSO:", label)
    print("Raster size:", width, "x", height, f"({resolution} m)")

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
    if not np.any(valid_ocean):
        raise ValueError(
            f"No valid ocean pixels after reprojection: {label}. "
            "Check LHB_BOUNDS and the source raster coverage."
        )

    depth = np.full_like(elevation, np.nan, dtype=np.float32)
    depth[valid_ocean] = -elevation[valid_ocean]

    extent = (
        xmin,
        xmin + width * resolution,
        ymax - height * resolution,
        ymax,
    )
    print("Valid ocean pixels:", int(valid_ocean.sum()))
    print("Depth range:", float(np.nanmin(depth)), "-", float(np.nanmax(depth)))
    return np.ma.masked_invalid(depth), extent


def _read_add_layer(path, layer, bbox_3031):
    gdf = gpd.read_file(path, layer=layer, bbox=bbox_3031)
    if gdf.crs is None:
        raise ValueError(f"{layer}: CRS is missing.")
    if CRS.from_user_input(gdf.crs) != MAP_CRS:
        gdf = gdf.to_crs(MAP_CRS)
    print(f"{layer}: {len(gdf)} features")
    return gdf


def load_add_data(bbox_3031):
    print("\nLoading ADD data...")
    polygons = _read_add_layer(
        ADD_POLYGON_FILE, "add_coastline_high_res_polygon_v7_12", bbox_3031
    )
    lines = _read_add_layer(
        ADD_LINE_FILE, "add_coastline_high_res_line_v7_12", bbox_3031
    )
    if "surface" in polygons.columns:
        print(polygons["surface"].value_counts(dropna=False))
    return polygons, lines


# ----------------------------------------------------------------------
# Axes configuration
# ----------------------------------------------------------------------
def create_axes():
    fig = plt.figure(figsize=FIGURE_SIZE)
    ax = fig.add_subplot(1, 1, 1, projection=PROJ)
    return fig, ax


def configure_axes(ax, outline_xy, bounds_map):
    """
    Fix the view to the bounding box, then clip the axes to the exact
    lon/lat wedge. Must be called BEFORE drawing any artist so that the
    clip path is picked up by everything added afterwards.
    """
    xmin, ymin, xmax, ymax = bounds_map
    # Small pad so the wedge lies strictly inside the view limits.
    pad_x = 0.01 * (xmax - xmin)
    pad_y = 0.01 * (ymax - ymin)
    ax.set_extent(
        (xmin - pad_x, xmax + pad_x, ymin - pad_y, ymax + pad_y),
        crs=PROJ,
    )

    # Cartopy clips the boundary path to ax.viewLim (data coordinates),
    # so the path must be given in map coordinates with transData.
    xs, ys = outline_xy
    ax.set_boundary(
        mpath.Path(np.column_stack((xs, ys))),
        transform=ax.transData,
    )
    ax.set_facecolor("white")


def _fmt_lon(lon):
    lon = ((lon + 180) % 360) - 180
    if np.isclose(lon, 0) or np.isclose(abs(lon), 180):
        return f"{abs(lon):g}°"
    return f"{abs(lon):g}°{'E' if lon > 0 else 'W'}"


def _fmt_lat(lat):
    if np.isclose(lat, 0):
        return "0°"
    return f"{abs(lat):g}°{'N' if lat > 0 else 'S'}"


def _ticks(start, stop, step):
    first = math.ceil(start / step - 1e-9) * step
    ticks = np.arange(first, stop + step * 0.5, step)
    return ticks[(ticks >= start - 1e-9) & (ticks <= stop + 1e-9)]


def add_graticules(ax, window):
    """
    Draw meridians/parallels and place labels where they meet the wedge:
    longitudes on the bottom/top arcs, latitudes on the left/right edges.
    """
    west, east, south, north = window
    to_map = Transformer.from_crs(CRS_GEO, MAP_CRS, always_xy=True)

    line_kw = dict(
        color=GRID_COLOR, alpha=GRID_ALPHA, linewidth=GRID_LINEWIDTH,
        linestyle=GRID_LINESTYLE, zorder=5,
    )
    label_kw = dict(
        fontsize=GRID_LABEL_SIZE, color=GRID_LABEL_COLOR,
        textcoords="offset points", annotation_clip=False,
    )
    off = LABEL_OFFSET_PT

    lon_ticks = _ticks(west, east, LON_INTERVAL)
    lat_ticks = _ticks(south, north, LAT_INTERVAL)

    lat_line = np.linspace(south, north, 200)
    for lon in lon_ticks:
        x, y = to_map.transform(np.full_like(lat_line, lon), lat_line)
        ax.plot(x, y, transform=PROJ, **line_kw)

        xb, yb = to_map.transform(lon, south)
        xt, yt = to_map.transform(lon, north)
        ax.annotate(_fmt_lon(lon), xy=(xb, yb), xytext=(0, -off),
                    ha="center", va="top", **label_kw)
        ax.annotate(_fmt_lon(lon), xy=(xt, yt), xytext=(0, off),
                    ha="center", va="bottom", **label_kw)

    lon_line = np.linspace(west, east, 200)
    for lat in lat_ticks:
        x, y = to_map.transform(lon_line, np.full_like(lon_line, lat))
        ax.plot(x, y, transform=PROJ, **line_kw)

        xl, yl = to_map.transform(west, lat)
        xr, yr = to_map.transform(east, lat)
        ax.annotate(_fmt_lat(lat), xy=(xl, yl), xytext=(-off, 0),
                    ha="right", va="center", **label_kw)
        ax.annotate(_fmt_lat(lat), xy=(xr, yr), xytext=(off, 0),
                    ha="left", va="center", **label_kw)


# ----------------------------------------------------------------------
# Drawing
# ----------------------------------------------------------------------
def draw_bathymetry(ax, bathy, extent):
    return ax.imshow(
        bathy,
        extent=extent,
        origin="upper",
        cmap=DEPTH_CMAP,
        vmin=DEPTH_VMIN,
        vmax=DEPTH_VMAX,
        interpolation="nearest",
        transform=PROJ,
        zorder=1,
    )


def draw_add(ax, polygons, lines, linewidth=0.35):
    if not polygons.empty and "surface" in polygons.columns:
        for surface, color in SURFACE_COLORS.items():
            subset = polygons[polygons["surface"] == surface]
            geoms = [g for g in subset.geometry
                     if g is not None and not g.is_empty]
            if geoms:
                ax.add_geometries(
                    geoms, crs=PROJ,
                    facecolor=color, edgecolor="none", linewidth=0,
                    zorder=3,
                )

    if not lines.empty:
        geoms = [g for g in lines.geometry
                 if g is not None and not g.is_empty]
        if geoms:
            ax.add_geometries(
                geoms, crs=PROJ,
                facecolor="none", edgecolor="black",
                linewidth=linewidth, alpha=0.9, zorder=4,
            )


def add_add_legend(ax):
    handles = [
        mpatches.Patch(color=color, label=surface)
        for surface, color in SURFACE_COLORS.items()
    ]
    ax.legend(handles=handles, loc="lower left", fontsize=8, framealpha=0.9)


def add_colorbar(fig, ax, image):
    cbar = fig.colorbar(
        image,
        ax=ax,
        orientation="vertical",
        shrink=DEPTH_COLORBAR_SHRINK,
        pad=DEPTH_COLORBAR_PAD,
        extend=DEPTH_COLORBAR_EXTEND,
        ticks=np.arange(DEPTH_VMIN, DEPTH_VMAX + DEPTH_TICKS, DEPTH_TICKS),
    )
    cbar.set_label(DEPTH_COLORBAR_LABEL, fontsize=9)
    cbar.ax.tick_params(labelsize=8)
    cbar.ax.invert_yaxis()
    return cbar


def set_title(ax, main):
    sub = (f"South Polar Stereographic  "
           f"(central meridian {MAP_CENTRAL_LON:g}°E, "
           f"true scale {abs(TRUE_SCALE_LAT):g}°S)")
    ax.set_title(f"{main}\n{sub}", fontsize=12, pad=24, linespacing=1.4)


# ----------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------
def plot_ibcso(bathy, extent, window, outline_xy, bounds_map):
    print("\nCreating figure: IBCSO (polar stereographic)")
    fig, ax = create_axes()
    configure_axes(ax, outline_xy, bounds_map)
    image = draw_bathymetry(ax, bathy, extent)
    add_graticules(ax, window)
    set_title(ax, "IBCSO Bathymetry: Lützow-Holm Bay")
    add_colorbar(fig, ax, image)
    save_figure(fig, "02_lhb_ibcso_polar.png")


def plot_add_ibcso(polygons, lines, bathy, extent,
                   window, outline_xy, bounds_map):
    print("\nCreating figure: IBCSO + ADD (polar stereographic)")
    fig, ax = create_axes()
    configure_axes(ax, outline_xy, bounds_map)
    image = draw_bathymetry(ax, bathy, extent)
    draw_add(ax, polygons, lines)
    add_graticules(ax, window)
    set_title(ax, "IBCSO Bathymetry and ADD: Lützow-Holm Bay")
    add_add_legend(ax)
    add_colorbar(fig, ax, image)
    save_figure(fig, "03_lhb_add_ibcso_polar.png")


def main():
    print("=" * 65)
    print("WW3 GRIDGEN: LHB SOURCE DATA (POLAR STEREOGRAPHIC)")
    print("=" * 65)
    ensure_output_directory()

    window = geographic_window()
    outline_xy, bounds_map, bounds_3031 = get_window_geometry(window)

    polygons, lines = load_add_data(bounds_3031)
    bathy, extent = reproject_ibcso(bounds_map, RESOLUTION_M, "Lützow-Holm Bay")

    plot_ibcso(bathy, extent, window, outline_xy, bounds_map)
    plot_add_ibcso(polygons, lines, bathy, extent,
                   window, outline_xy, bounds_map)

    print("\n" + "=" * 65)
    print("Both figures generated successfully.")
    print("=" * 65)


if __name__ == "__main__":
    main()