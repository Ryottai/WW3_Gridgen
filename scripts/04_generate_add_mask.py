
"""
04_generate_add_mask.py

Generate an ADD-based sub-grid land-sea mask for the WW3 RECT grid.

Each WW3 grid cell is sampled on a GRID_SUBSAMPLE x GRID_SUBSAMPLE
sub-grid. ADD land fraction is calculated from the configured
LAND_SURFACES.

Outputs:
    output/netcdf/ww3_add_landmask_lhb.nc
    output/figures/05_lhb_add_land_fraction.png
"""

from pathlib import Path
import sys

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import shapely
import xarray as xr

import cartopy.crs as ccrs
from pyproj import Transformer
from rasterio.warp import transform_bounds
from shapely.ops import unary_union


# ============================================================
# Configuration
# ============================================================

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import (
    ADD_POLYGON_FILE,
    ADD_CRS,
    GEOGRAPHIC_CRS,
    GRID_LON_MIN,
    GRID_LON_MAX,
    GRID_LAT_MIN,
    GRID_LAT_MAX,
    GRID_DLON,
    GRID_DLAT,
    GRID_SUBSAMPLE,
    LAND_SURFACES,
    LAND_CUTOFF,
    NETCDF_DIR,
    FIGURE_DIR,
    FIGURE_DPI,
)


NETCDF_OUT = NETCDF_DIR / "ww3_add_landmask_lhb.nc"
FIGURE_OUT = FIGURE_DIR / "05_lhb_add_land_fraction.png"

READ_BUFFER_M = 10_000.0


# ============================================================
# Grid generation
# ============================================================

def make_rect_grid():
    """Generate the WW3 RECT grid-point coordinates."""

    nlon_float = (GRID_LON_MAX - GRID_LON_MIN) / GRID_DLON
    nlat_float = (GRID_LAT_MAX - GRID_LAT_MIN) / GRID_DLAT

    if not np.isclose(nlon_float, round(nlon_float)):
        raise ValueError("Longitude range is not divisible by GRID_DLON.")

    if not np.isclose(nlat_float, round(nlat_float)):
        raise ValueError("Latitude range is not divisible by GRID_DLAT.")

    nlon = int(round(nlon_float)) + 1
    nlat = int(round(nlat_float)) + 1

    lon = GRID_LON_MIN + np.arange(nlon) * GRID_DLON
    lat = GRID_LAT_MIN + np.arange(nlat) * GRID_DLAT

    lon2d, lat2d = np.meshgrid(lon, lat)

    return lon, lat, lon2d, lat2d


# ============================================================
# Read ADD polygons
# ============================================================

def load_add_polygons():

    add_path = Path(ADD_POLYGON_FILE)

    if not add_path.exists():
        raise FileNotFoundError(
            f"ADD polygon file not found: {add_path}"
        )

    # Include the outer half-cell around the grid-point domain.
    west = GRID_LON_MIN - GRID_DLON / 2
    east = GRID_LON_MAX + GRID_DLON / 2
    south = GRID_LAT_MIN - GRID_DLAT / 2
    north = GRID_LAT_MAX + GRID_DLAT / 2

    bounds = transform_bounds(
        GEOGRAPHIC_CRS,
        ADD_CRS,
        west,
        south,
        east,
        north,
        densify_pts=21,
    )

    xmin, ymin, xmax, ymax = bounds

    read_bbox = (
        xmin - READ_BUFFER_M,
        ymin - READ_BUFFER_M,
        xmax + READ_BUFFER_M,
        ymax + READ_BUFFER_M,
    )

    print("\n=== Reading ADD polygons ===")
    print(f"File: {add_path}")

    add = gpd.read_file(add_path, bbox=read_bbox)

    if add.empty:
        raise RuntimeError(
            "No ADD polygon features were found in the grid domain."
        )

    if add.crs is None:
        raise RuntimeError("ADD polygon CRS is undefined.")

    if "surface" not in add.columns:
        raise KeyError(
            "ADD polygon data do not contain a 'surface' column. "
            f"Available columns: {list(add.columns)}"
        )

    add = add.to_crs(ADD_CRS)

    add = add[
        add.geometry.notna() & ~add.geometry.is_empty
    ].copy()

    add["surface_norm"] = (
        add["surface"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    print(f"Features read: {len(add):,}")
    print("\nAvailable surface classes:")
    print(add["surface_norm"].value_counts().to_string())

    requested = {
        str(surface).strip().lower()
        for surface in LAND_SURFACES
    }

    available = set(add["surface_norm"].unique())
    missing = requested - available

    if missing:
        print(f"\n[WARNING] Requested surfaces not found: {sorted(missing)}")

    selected = add[
        add["surface_norm"].isin(requested)
    ]

    if selected.empty:
        raise RuntimeError(
            "No ADD polygons match LAND_SURFACES in config.py."
        )

    print(f"\nSurfaces treated as land: {sorted(requested)}")
    print(f"Selected features: {len(selected):,}")

    land_union = unary_union(selected.geometry.tolist())

    if land_union.is_empty:
        raise RuntimeError("The merged ADD land geometry is empty.")

    return land_union


# ============================================================
# Sub-grid sampling
# ============================================================

def calculate_land_fraction(lon2d, lat2d, land_union):

    nsub = GRID_SUBSAMPLE

    if nsub < 1:
        raise ValueError("GRID_SUBSAMPLE must be >= 1.")

    # Sub-grid sample locations are placed at the centres
    # of equally spaced subdivisions within each WW3 cell.
    lon_offsets = (
        (np.arange(nsub) + 0.5) / nsub - 0.5
    ) * GRID_DLON

    lat_offsets = (
        (np.arange(nsub) + 0.5) / nsub - 0.5
    ) * GRID_DLAT

    sample_lon = (
        lon2d[:, :, None, None]
        + lon_offsets[None, None, None, :]
    )

    sample_lat = (
        lat2d[:, :, None, None]
        + lat_offsets[None, None, :, None]
    )

    sample_lon, sample_lat = np.broadcast_arrays(
        sample_lon,
        sample_lat,
    )

    print("\n=== Sub-grid sampling ===")
    print(f"Sub-grid per cell : {nsub} x {nsub}")
    print(f"Samples per cell  : {nsub * nsub:,}")
    print(f"Total samples     : {sample_lon.size:,}")

    transformer = Transformer.from_crs(
        GEOGRAPHIC_CRS,
        ADD_CRS,
        always_xy=True,
    )

    sample_x, sample_y = transformer.transform(
        sample_lon.ravel(),
        sample_lat.ravel(),
    )

    points = shapely.points(
        np.asarray(sample_x),
        np.asarray(sample_y),
    )

    if not hasattr(shapely, "covers"):
        raise RuntimeError(
            "This script requires Shapely 2.x."
        )

    is_land = shapely.covers(land_union, points)

    is_land = np.asarray(is_land, dtype=bool).reshape(
        lon2d.shape + (nsub, nsub)
    )

    land_fraction = is_land.mean(axis=(2, 3))

    return land_fraction


# ============================================================
# Save NetCDF
# ============================================================

def save_netcdf(lon, lat, land_fraction, landmask):

    ds = xr.Dataset(
        data_vars={
            "land_fraction": (
                ("latitude", "longitude"),
                land_fraction.astype(np.float32),
                {
                    "long_name": "ADD land fraction within WW3 grid cell",
                    "units": "1",
                    "valid_min": 0.0,
                    "valid_max": 1.0,
                    "comment": (
                        "Fraction calculated from GRID_SUBSAMPLE x "
                        "GRID_SUBSAMPLE point samples. LAND_SURFACES "
                        "from config.py are treated as land."
                    ),
                },
            ),
            "landmask": (
                ("latitude", "longitude"),
                landmask,
                {
                    "long_name": "ADD-based WW3 land-sea mask",
                    "flag_values": np.array([0, 1], dtype=np.int8),
                    "flag_meanings": "land sea",
                    "comment": (
                        "0=land, 1=sea. A cell is classified as land "
                        "when land_fraction > LAND_CUTOFF."
                    ),
                },
            ),
        },
        coords={
            "longitude": (
                "longitude",
                lon,
                {
                    "standard_name": "longitude",
                    "units": "degrees_east",
                },
            ),
            "latitude": (
                "latitude",
                lat,
                {
                    "standard_name": "latitude",
                    "units": "degrees_north",
                },
            ),
        },
        attrs={
            "title": "ADD sub-grid land fraction and mask for WW3 LHB",
            "grid_type": "RECT",
            "grid_spacing": f"{GRID_DLON} x {GRID_DLAT} degrees",
            "sub_grid_sampling": f"{GRID_SUBSAMPLE} x {GRID_SUBSAMPLE}",
            "land_surfaces": ", ".join(LAND_SURFACES),
            "land_cutoff": float(LAND_CUTOFF),
            "mask_convention": "0=land, 1=sea",
            "source": str(ADD_POLYGON_FILE),
        },
    )

    ds.to_netcdf(
        NETCDF_OUT,
        encoding={
            "land_fraction": {
                "zlib": True,
                "complevel": 4,
                "dtype": "float32",
            },
            "landmask": {
                "zlib": True,
                "complevel": 4,
                "dtype": "int8",
            },
        },
    )

    print(f"\nSaved NetCDF: {NETCDF_OUT}")


# ============================================================
# Visualization
# ============================================================

def plot_results(lon, lat, land_fraction, landmask):

    lon_edges = (
        GRID_LON_MIN
        - GRID_DLON / 2
        + np.arange(len(lon) + 1) * GRID_DLON
    )

    lat_edges = (
        GRID_LAT_MIN
        - GRID_DLAT / 2
        + np.arange(len(lat) + 1) * GRID_DLAT
    )

    map_crs = ccrs.Stereographic(
        central_longitude=36.0,
        central_latitude=-90.0,
        true_scale_latitude=-70.0,
    )

    data_crs = ccrs.PlateCarree()

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(15, 7),
        subplot_kw={"projection": map_crs},
        constrained_layout=True,
    )

    for ax in axes:
        ax.set_extent(
            [
                lon_edges[0] - 0.2,
                lon_edges[-1] + 0.2,
                lat_edges[0] - 0.15,
                lat_edges[-1] + 0.15,
            ],
            crs=data_crs,
        )

        gl = ax.gridlines(
            crs=data_crs,
            draw_labels=True,
            linewidth=0.4,
            color="gray",
            alpha=0.5,
            linestyle=":",
        )
        gl.top_labels = False
        gl.right_labels = False

    mesh = axes[0].pcolormesh(
        lon_edges,
        lat_edges,
        land_fraction,
        transform=data_crs,
        shading="flat",
        cmap="Greys",
        vmin=0.0,
        vmax=1.0,
    )

    axes[0].set_title("ADD land fraction")

    cbar = fig.colorbar(
        mesh,
        ax=axes[0],
        orientation="horizontal",
        pad=0.08,
        shrink=0.8,
    )
    cbar.set_label("Land fraction")

    mask_mesh = axes[1].pcolormesh(
        lon_edges,
        lat_edges,
        landmask,
        transform=data_crs,
        shading="flat",
        cmap="Blues_r",
        vmin=0,
        vmax=1,
    )

    axes[1].set_title(
        f"ADD land-sea mask (cutoff = {LAND_CUTOFF:.2f})"
    )

    mask_cbar = fig.colorbar(
        mask_mesh,
        ax=axes[1],
        orientation="horizontal",
        pad=0.08,
        shrink=0.8,
        ticks=[0, 1],
    )
    mask_cbar.ax.set_xticklabels(["Land", "Sea"])

    fig.suptitle(
        "ADD sub-grid land fraction and WW3 land-sea mask\n"
        "Lützow-Holm Bay",
        fontsize=14,
    )

    fig.savefig(
        FIGURE_OUT,
        dpi=FIGURE_DPI,
        bbox_inches="tight",
        facecolor="white",
    )

    plt.close(fig)

    print(f"Saved figure: {FIGURE_OUT}")


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 60)
    print("WW3 GRIDGEN: ADD SUB-GRID LAND MASK")
    print("=" * 60)

    NETCDF_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    lon, lat, lon2d, lat2d = make_rect_grid()

    print("\n=== WW3 RECT grid ===")
    print(f"Grid size : {len(lon)} x {len(lat)}")
    print(f"Total     : {lon2d.size:,} points")
    print(f"Domain    : {lon[0]:.2f}–{lon[-1]:.2f} E")
    print(f"Latitude  : {lat[0]:.2f}–{lat[-1]:.2f} S")

    land_union = load_add_polygons()

    land_fraction = calculate_land_fraction(
        lon2d,
        lat2d,
        land_union,
    )

    landmask = np.where(
        land_fraction > LAND_CUTOFF,
        0,
        1,
    ).astype(np.int8)

    n_land = int(np.count_nonzero(landmask == 0))
    n_sea = int(np.count_nonzero(landmask == 1))

    print("\n=== ADD classification ===")
    print(f"Land cells : {n_land:,}")
    print(f"Sea cells  : {n_sea:,}")
    print(f"Total      : {landmask.size:,}")
    print(f"Land cutoff: {LAND_CUTOFF:.2f}")

    print("\n=== Land fraction statistics ===")
    print(f"Minimum : {land_fraction.min():.3f}")
    print(f"Mean    : {land_fraction.mean():.3f}")
    print(f"Maximum : {land_fraction.max():.3f}")

    if not np.all(
        (land_fraction >= 0.0) & (land_fraction <= 1.0)
    ):
        raise RuntimeError("Invalid land fraction values.")

    save_netcdf(
        lon,
        lat,
        land_fraction,
        landmask,
    )

    plot_results(
        lon,
        lat,
        land_fraction,
        landmask,
    )

    print("\nADD mask generation completed.")


if __name__ == "__main__":
    main()
