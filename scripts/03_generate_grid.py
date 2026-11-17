
"""
03_generate_grid.py

Generate and visualize the WW3 RECT grid for Lützow-Holm Bay.

Outputs:
    output/netcdf/ww3_rect_grid_lhb.nc
    output/figures/04_lhb_rect_grid.png

Grid coordinates are geographic longitude/latitude.
The visualization uses South Polar Stereographic projection.
"""

from pathlib import Path
import sys

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

# ------------------------------------------------------------
# Project configuration
# ------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import (
    GRID_LON_MIN,
    GRID_LON_MAX,
    GRID_LAT_MIN,
    GRID_LAT_MAX,
    GRID_DLON,
    GRID_DLAT,
    WW3_GRID_NAME,
    WW3_GRID_TITLE,
    NETCDF_DIR,
    FIGURE_DIR,
)

# ------------------------------------------------------------
# Output paths
# ------------------------------------------------------------

NETCDF_OUT = NETCDF_DIR / "ww3_rect_grid_lhb.nc"
FIGURE_OUT = FIGURE_DIR / "04_lhb_rect_grid.png"

# ------------------------------------------------------------
# Grid generation
# ------------------------------------------------------------

def make_rect_grid():
    """Generate regular geographic RECT grid-point coordinates."""

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


# ------------------------------------------------------------
# Save grid NetCDF
# ------------------------------------------------------------

def save_grid(lon, lat):
    """Save the 1-D RECT grid coordinates."""

    ds = xr.Dataset(
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
            "title": WW3_GRID_TITLE,
            "grid_name": WW3_GRID_NAME,
            "grid_type": "RECT",
            "coordinate_system": "Geographic longitude/latitude",
            "coordinate_convention": "Grid-point coordinates",
            "longitude_spacing_degrees": GRID_DLON,
            "latitude_spacing_degrees": GRID_DLAT,
        },
    )

    ds.to_netcdf(
        NETCDF_OUT,
        encoding={
            "longitude": {"dtype": "float64"},
            "latitude": {"dtype": "float64"},
        },
    )

    print(f"Saved NetCDF: {NETCDF_OUT}")


# ------------------------------------------------------------
# Visualization
# ------------------------------------------------------------

def plot_grid(lon, lat, lon2d, lat2d):
    """Plot the RECT grid in South Polar Stereographic projection."""

    map_crs = ccrs.Stereographic(
        central_longitude=36.0,
        central_latitude=-90.0,
        true_scale_latitude=-70.0,
    )

    data_crs = ccrs.PlateCarree()

    fig = plt.figure(figsize=(11, 9))
    ax = fig.add_subplot(1, 1, 1, projection=map_crs)

    # Display window with a small margin.
    ax.set_extent(
        [
            GRID_LON_MIN - 0.3,
            GRID_LON_MAX + 0.3,
            GRID_LAT_MIN - 0.2,
            GRID_LAT_MAX + 0.2,
        ],
        crs=data_crs,
    )

    # Draw grid lines at every fifth grid interval.
    for j in range(0, len(lat), 5):
        ax.plot(
            lon,
            np.full_like(lon, lat[j]),
            color="steelblue",
            linewidth=0.45,
            alpha=0.8,
            transform=data_crs,
            zorder=2,
        )

    for i in range(0, len(lon), 5):
        ax.plot(
            np.full_like(lat, lon[i]),
            lat,
            color="steelblue",
            linewidth=0.45,
            alpha=0.8,
            transform=data_crs,
            zorder=2,
        )

    # Plot all grid points.
    ax.scatter(
        lon2d,
        lat2d,
        s=1.0,
        color="black",
        linewidths=0,
        transform=data_crs,
        zorder=3,
    )

    # Geographic graticules.
    gl = ax.gridlines(
        crs=data_crs,
        draw_labels=True,
        linewidth=0.5,
        color="gray",
        alpha=0.6,
        linestyle=":",
    )

    gl.top_labels = True
    gl.bottom_labels = True
    gl.left_labels = True
    gl.right_labels = True

    ax.set_title(
        f"WW3 RECT Grid: Lützow-Holm Bay\n"
        f"{len(lon)} × {len(lat)} points, "
        f"Δlon={GRID_DLON}°, Δlat={GRID_DLAT}°",
        fontsize=13,
        pad=12,
    )

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    fig.savefig(
        FIGURE_OUT,
        dpi=200,
        bbox_inches="tight",
        facecolor="white",
    )

    plt.close(fig)

    print(f"Saved figure: {FIGURE_OUT}")


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():

    print("=" * 60)
    print("WW3 GRIDGEN: RECT GRID GENERATION")
    print("=" * 60)

    NETCDF_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    lon, lat, lon2d, lat2d = make_rect_grid()

    print("\nGrid configuration")
    print(f"Longitude : {lon[0]:.3f} to {lon[-1]:.3f} E")
    print(f"Latitude  : {lat[0]:.3f} to {lat[-1]:.3f} S")
    print(f"Spacing   : {GRID_DLON} x {GRID_DLAT} degrees")
    print(f"Grid size : {len(lon)} x {len(lat)}")
    print(f"Total     : {lon2d.size:,} points")

    # Coordinate validation.
    assert np.isclose(lon[0], GRID_LON_MIN)
    assert np.isclose(lon[-1], GRID_LON_MAX)
    assert np.isclose(lat[0], GRID_LAT_MIN)
    assert np.isclose(lat[-1], GRID_LAT_MAX)

    save_grid(lon, lat)
    plot_grid(lon, lat, lon2d, lat2d)

    print("\nGrid generation completed.")


if __name__ == "__main__":
    main()
