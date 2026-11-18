
"""
05_generate_bathymetry.py

Generate IBCSO-based bathymetry statistics on the WW3 RECT grid.

The IBCSO raster is reprojected to a regular geographic sub-grid.
Each WW3 grid cell is represented by GRID_SUBSAMPLE x GRID_SUBSAMPLE
samples.

Assumption:
    IBCSO elevation is negative below sea level.
    Positive bathymetry is calculated as -elevation.

Outputs:
    output/netcdf/ww3_ibcso_bathymetry_lhb.nc
    output/figures/06_lhb_ibcso_bathymetry.png

The original IBCSO raster is read-only.
ADD land mask is not applied in this script.
"""

from pathlib import Path
import sys

import numpy as np
import xarray as xr
import rasterio
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

from rasterio.transform import from_origin
from rasterio.warp import reproject, Resampling


# ============================================================
# Project configuration
# ============================================================

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import (
    IBCSO_FILE,
    GEOGRAPHIC_CRS,
    GRID_LON_MIN,
    GRID_LON_MAX,
    GRID_LAT_MIN,
    GRID_LAT_MAX,
    GRID_DLON,
    GRID_DLAT,
    GRID_SUBSAMPLE,
    IBCSO_RESAMPLING,
    BATHY_WET_LIMIT,
    NETCDF_DIR,
    FIGURE_DIR,
    FIGURE_DPI,
)


# ============================================================
# Output paths
# ============================================================

NETCDF_OUT = NETCDF_DIR / "ww3_ibcso_bathymetry_lhb.nc"
FIGURE_OUT = FIGURE_DIR / "06_lhb_ibcso_bathymetry.png"


# ============================================================
# Grid generation
# ============================================================

def make_rect_grid():
    """Generate the WW3 RECT grid-point coordinates."""

    nlon_float = (
        GRID_LON_MAX - GRID_LON_MIN
    ) / GRID_DLON

    nlat_float = (
        GRID_LAT_MAX - GRID_LAT_MIN
    ) / GRID_DLAT

    if not np.isclose(nlon_float, round(nlon_float)):
        raise ValueError(
            "Longitude range is not divisible by GRID_DLON."
        )

    if not np.isclose(nlat_float, round(nlat_float)):
        raise ValueError(
            "Latitude range is not divisible by GRID_DLAT."
        )

    nlon = int(round(nlon_float)) + 1
    nlat = int(round(nlat_float)) + 1

    lon = GRID_LON_MIN + np.arange(nlon) * GRID_DLON
    lat = GRID_LAT_MIN + np.arange(nlat) * GRID_DLAT

    return lon, lat


# ============================================================
# Resampling configuration
# ============================================================

def get_resampling_method():

    name = str(IBCSO_RESAMPLING).strip().lower()

    if not hasattr(Resampling, name):
        raise ValueError(
            f"Unsupported IBCSO_RESAMPLING: {IBCSO_RESAMPLING}"
        )

    method = getattr(Resampling, name)

    print(f"IBCSO resampling: {name}")

    return method


# ============================================================
# Reproject IBCSO to the sub-grid
# ============================================================

def reproject_ibcso(lon, lat):

    nsub = GRID_SUBSAMPLE

    if nsub < 1:
        raise ValueError("GRID_SUBSAMPLE must be >= 1.")

    if BATHY_WET_LIMIT < 0 or BATHY_WET_LIMIT > 1:
        raise ValueError("BATHY_WET_LIMIT must be between 0 and 1.")

    # Sub-grid sample spacing.
    dx = GRID_DLON / nsub
    dy = GRID_DLAT / nsub

    # Sample positions are located at the centres of subdivisions.
    lon_offsets = (
        (np.arange(nsub) + 0.5) / nsub - 0.5
    ) * GRID_DLON

    lat_offsets = (
        (np.arange(nsub) + 0.5) / nsub - 0.5
    ) * GRID_DLAT

    lon_sample_min = lon[0] + lon_offsets[0]
    lon_sample_max = lon[-1] + lon_offsets[-1]

    lat_sample_min = lat[0] + lat_offsets[0]
    lat_sample_max = lat[-1] + lat_offsets[-1]

    # Raster rows run north to south.
    width = len(lon) * nsub
    height = len(lat) * nsub

    dst_transform = from_origin(
        lon_sample_min - dx / 2,
        lat_sample_max + dy / 2,
        dx,
        dy,
    )

    destination = np.full(
        (height, width),
        np.nan,
        dtype=np.float32,
    )

    raster_path = Path(IBCSO_FILE)

    if not raster_path.exists():
        raise FileNotFoundError(raster_path)

    print("\n=== Reprojecting IBCSO ===")
    print(f"File          : {raster_path}")
    print(f"Target CRS    : {GEOGRAPHIC_CRS}")
    print(f"Sub-grid size : {width} x {height}")
    print(f"Resolution    : {dx:.5f} x {dy:.5f} degrees")

    with rasterio.open(raster_path) as src:

        print(f"Source CRS    : {src.crs}")
        print(f"Source NoData : {src.nodata}")
        print(f"Source scale  : {src.scales[0]}")
        print(f"Source offset : {src.offsets[0]}")

        reproject(
            source=rasterio.band(src, 1),
            destination=destination,
            src_transform=src.transform,
            src_crs=src.crs,
            src_nodata=src.nodata,
            dst_transform=dst_transform,
            dst_crs=GEOGRAPHIC_CRS,
            dst_nodata=np.nan,
            resampling=get_resampling_method(),
            init_dest_nodata=True,
        )

        # Apply raster scale and offset to obtain physical elevations.
        destination = (
            destination * src.scales[0]
            + src.offsets[0]
        )

    # Convert raster row order (north to south) to grid order
    # (south to north). Sub-grid order within each cell does not
    # affect the statistics calculated below.
    subgrid = destination.reshape(
        len(lat),
        nsub,
        len(lon),
        nsub,
    )[::-1, :, :, :].transpose(0, 2, 1, 3)

    print("\nReprojection completed.")
    print(f"Sub-grid array shape: {subgrid.shape}")

    return subgrid


# ============================================================
# Calculate bathymetry statistics
# ============================================================

def calculate_bathymetry(subgrid):

    nsub = GRID_SUBSAMPLE
    total_samples = nsub * nsub

    valid = np.isfinite(subgrid)

    # Assumption: negative elevation represents the seafloor.
    wet = valid & (subgrid < 0.0)

    valid_count = valid.sum(axis=(2, 3))
    wet_count = wet.sum(axis=(2, 3))

    valid_fraction = valid_count / total_samples
    wet_fraction = wet_count / total_samples

    # Convert negative elevation to positive-downward depth.
    depth_samples = np.where(
        wet,
        -subgrid,
        np.nan,
    )

    with np.errstate(invalid="ignore"):
        mean_depth = np.nanmean(
            depth_samples,
            axis=(2, 3),
        )

    # Retain bathymetry only where the wet fraction reaches
    # the configured threshold.
    bathymetry = np.where(
        wet_fraction >= BATHY_WET_LIMIT,
        mean_depth,
        np.nan,
    )

    # Avoid retaining a depth where no wet samples exist.
    bathymetry[wet_count == 0] = np.nan

    print("\n=== IBCSO sub-grid statistics ===")
    print(f"Samples per cell : {total_samples}")
    print(f"Valid cells      : {np.count_nonzero(valid_count > 0):,}")
    print(f"Cells with water : {np.count_nonzero(wet_count > 0):,}")
    print(
        f"Cells passing wet limit: "
        f"{np.count_nonzero(np.isfinite(bathymetry)):,}"
    )

    print("\n=== Fraction statistics ===")
    print(f"Valid fraction mean : {np.nanmean(valid_fraction):.3f}")
    print(f"Wet fraction mean   : {np.nanmean(wet_fraction):.3f}")
    print(f"Wet limit           : {BATHY_WET_LIMIT:.2f}")

    if np.any(np.isfinite(bathymetry)):
        print("\n=== Bathymetry statistics ===")
        print(f"Minimum : {np.nanmin(bathymetry):.2f} m")
        print(f"Maximum : {np.nanmax(bathymetry):.2f} m")
        print(f"Mean    : {np.nanmean(bathymetry):.2f} m")
        print(f"Median  : {np.nanmedian(bathymetry):.2f} m")
    else:
        raise RuntimeError(
            "No cells passed BATHY_WET_LIMIT. "
            "Check IBCSO elevation convention and reprojection."
        )

    return bathymetry, valid_fraction, wet_fraction


# ============================================================
# Save NetCDF
# ============================================================

def save_netcdf(
    lon,
    lat,
    bathymetry,
    valid_fraction,
    wet_fraction,
):

    ds = xr.Dataset(
        data_vars={
            "bathymetry": (
                ("latitude", "longitude"),
                bathymetry.astype(np.float32),
                {
                    "long_name": "Mean IBCSO bathymetry",
                    "units": "m",
                    "positive": "down",
                    "comment": (
                        "Mean positive-downward depth of valid "
                        "sub-grid samples classified as water."
                    ),
                },
            ),
            "valid_fraction": (
                ("latitude", "longitude"),
                valid_fraction.astype(np.float32),
                {
                    "long_name": "Fraction of valid IBCSO samples",
                    "units": "1",
                    "valid_min": 0.0,
                    "valid_max": 1.0,
                },
            ),
            "wet_fraction": (
                ("latitude", "longitude"),
                wet_fraction.astype(np.float32),
                {
                    "long_name": "Fraction of sub-grid water samples",
                    "units": "1",
                    "valid_min": 0.0,
                    "valid_max": 1.0,
                    "comment": (
                        "Fraction of all sub-grid samples with "
                        "valid negative IBCSO elevation."
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
            "title": "IBCSO bathymetry statistics for WW3 LHB",
            "grid_type": "RECT",
            "grid_spacing": (
                f"{GRID_DLON} x {GRID_DLAT} degrees"
            ),
            "sub_grid_sampling": f"{GRID_SUBSAMPLE} x {GRID_SUBSAMPLE}",
            "resampling_method": str(IBCSO_RESAMPLING),
            "wet_fraction_threshold": float(BATHY_WET_LIMIT),
            "bathymetry_convention": "positive downward",
            "source": str(IBCSO_FILE),
            "note": (
                "ADD land mask has not been applied. "
                "Negative IBCSO elevation is interpreted as water."
            ),
        },
    )

    ds.to_netcdf(
        NETCDF_OUT,
        encoding={
            "bathymetry": {
                "zlib": True,
                "complevel": 4,
                "dtype": "float32",
                "_FillValue": np.float32(np.nan),
            },
            "valid_fraction": {
                "zlib": True,
                "complevel": 4,
                "dtype": "float32",
            },
            "wet_fraction": {
                "zlib": True,
                "complevel": 4,
                "dtype": "float32",
            },
        },
    )

    print(f"\nSaved NetCDF: {NETCDF_OUT}")


# ============================================================
# Visualization
# ============================================================

def plot_results(
    lon,
    lat,
    bathymetry,
    valid_fraction,
    wet_fraction,
):

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

    depth_mesh = axes[0].pcolormesh(
        lon_edges,
        lat_edges,
        np.ma.masked_invalid(bathymetry),
        transform=data_crs,
        shading="flat",
        cmap="viridis",
    )

    axes[0].set_title("IBCSO bathymetry")

    depth_cbar = fig.colorbar(
        depth_mesh,
        ax=axes[0],
        orientation="horizontal",
        pad=0.08,
        shrink=0.8,
    )
    depth_cbar.set_label("Depth (m, positive downward)")

    wet_mesh = axes[1].pcolormesh(
        lon_edges,
        lat_edges,
        wet_fraction,
        transform=data_crs,
        shading="flat",
        cmap="Blues",
        vmin=0.0,
        vmax=1.0,
    )

    axes[1].set_title("IBCSO wet fraction")

    wet_cbar = fig.colorbar(
        wet_mesh,
        ax=axes[1],
        orientation="horizontal",
        pad=0.08,
        shrink=0.8,
    )
    wet_cbar.set_label("Wet fraction")

    fig.suptitle(
        "IBCSO bathymetry and wet fraction\n"
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
    print("WW3 GRIDGEN: IBCSO BATHYMETRY GENERATION")
    print("=" * 60)

    NETCDF_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    lon, lat = make_rect_grid()

    print("\n=== WW3 RECT grid ===")
    print(f"Grid size : {len(lon)} x {len(lat)}")
    print(f"Total     : {len(lon) * len(lat):,}")
    print(
        f"Domain    : {lon[0]:.2f}–{lon[-1]:.2f} E, "
        f"{lat[0]:.2f}–{lat[-1]:.2f} S"
    )

    subgrid = reproject_ibcso(lon, lat)

    bathymetry, valid_fraction, wet_fraction = (
        calculate_bathymetry(subgrid)
    )

    save_netcdf(
        lon,
        lat,
        bathymetry,
        valid_fraction,
        wet_fraction,
    )

    plot_results(
        lon,
        lat,
        bathymetry,
        valid_fraction,
        wet_fraction,
    )

    print("\nIBCSO bathymetry generation completed.")


if __name__ == "__main__":
    main()
