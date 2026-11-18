
"""
06_export_ww3_input.py

Export WW3 6.07.1 input files from the generated LHB NetCDF datasets.

Usage:
    python scripts/06_export_ww3_input.py --experiment lhb_control

Input:
    output/netcdf/ww3_rect_grid_lhb.nc
    output/netcdf/ww3_add_landmask_lhb.nc
    output/netcdf/ww3_ibcso_bathymetry_lhb.nc

Output:
    output/ww3/<experiment>/
        ww3_grid.inp
        bottom.inp
        obstr.inp
        mask.inp

The source NetCDF files are read-only.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import xarray as xr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import config as cfg


# ============================================================
# DEFAULT INPUT FILES
# ============================================================

GRID_FILE = cfg.NETCDF_DIR / "ww3_rect_grid_lhb.nc"
LANDMASK_FILE = cfg.NETCDF_DIR / "ww3_add_landmask_lhb.nc"
BATHY_FILE = cfg.NETCDF_DIR / "ww3_ibcso_bathymetry_lhb.nc"


# ============================================================
# DATASET HELPERS
# ============================================================

def find_variable(ds, candidates, description):
    """Find the first matching variable in a NetCDF dataset."""

    for name in candidates:
        if name in ds.variables:
            return ds[name].values, name

    available = list(ds.data_vars)
    raise KeyError(
        f"Could not find {description}.\n"
        f"Candidates: {candidates}\n"
        f"Available data variables: {available}"
    )


def find_coordinate(ds, candidates, description):
    """Find a coordinate variable."""

    for name in candidates:
        if name in ds.coords or name in ds.variables:
            return np.asarray(ds[name].values)

    raise KeyError(
        f"Could not find {description}. "
        f"Available coordinates: {list(ds.coords)}"
    )


def load_grid(path):
    """Load and validate the rectilinear grid."""

    if not path.exists():
        raise FileNotFoundError(path)

    with xr.open_dataset(path) as ds:
        lon = find_coordinate(
            ds, ("longitude", "lon", "x"), "longitude"
        )
        lat = find_coordinate(
            ds, ("latitude", "lat", "y"), "latitude"
        )

    lon = np.asarray(lon, dtype=np.float64).squeeze()
    lat = np.asarray(lat, dtype=np.float64).squeeze()

    if lon.ndim != 1 or lat.ndim != 1:
        raise ValueError(
            "This exporter currently supports 1-D RECT coordinates only."
        )

    if not np.all(np.diff(lon) > 0):
        raise ValueError("Longitude must increase from west to east.")

    if not (
        np.all(np.diff(lat) > 0)
        or np.all(np.diff(lat) < 0)
    ):
        raise ValueError("Latitude must be strictly monotonic.")

    return lon, lat


def load_land_fraction(path):
    """Load ADD-derived land fraction."""

    if not path.exists():
        raise FileNotFoundError(path)

    with xr.open_dataset(path) as ds:
        values, name = find_variable(
            ds,
            (
                "land_fraction",
                "add_land_fraction",
                "fraction_land",
                "land_frac",
            ),
            "ADD land fraction",
        )

    values = np.asarray(values, dtype=np.float64).squeeze()

    if values.ndim != 2:
        raise ValueError(
            f"Land fraction must be 2-D. Got shape {values.shape}"
        )

    if not np.all(np.isfinite(values)):
        raise ValueError("Land fraction contains NaN or infinite values.")

    if np.nanmin(values) < 0 or np.nanmax(values) > 1:
        raise ValueError(
            "Land fraction must be in the range [0, 1]."
        )

    print(f"Land fraction variable: {name}")

    return values


def load_bathymetry(path):
    """Load positive-downward bathymetry in metres."""

    if not path.exists():
        raise FileNotFoundError(path)

    with xr.open_dataset(path) as ds:
        values, name = find_variable(
            ds,
            (
                "mean_depth",
                "bathymetry",
                "water_depth",
                "depth",
                "ibcso_depth",
                "depth_mean",
                "mean_bathymetry",
            ),
            "positive-downward bathymetry",
        )

    values = np.asarray(values, dtype=np.float64).squeeze()

    if values.ndim != 2:
        raise ValueError(
            f"Bathymetry must be 2-D. Got shape {values.shape}"
        )

    print(f"Bathymetry variable: {name}")

    return values


# ============================================================
# ARRAY VALIDATION AND ORIENTATION
# ============================================================

def validate_array_shapes(lon, lat, land_fraction, depth):

    expected = (lat.size, lon.size)

    if land_fraction.shape != expected:
        raise ValueError(
            f"Land fraction shape {land_fraction.shape}; "
            f"expected {expected}."
        )

    if depth.shape != expected:
        raise ValueError(
            f"Bathymetry shape {depth.shape}; "
            f"expected {expected}."
        )


def orient_north_to_south(lat, *arrays):
    """
    WW3 IDLA=3 reads rows from north to south.

    The generated NetCDF arrays are expected to use ascending latitude.
    Flip all arrays vertically when necessary.
    """

    if lat[0] < lat[-1]:
        return tuple(array[::-1, :] for array in arrays)

    return arrays


# ============================================================
# WW3 INPUT ARRAYS
# ============================================================

def make_ww3_arrays(
    land_fraction,
    depth,
    land_cutoff,
    dry_depth,
    obstruction_scale,
):
    """
    Convert ADD land fraction and IBCSO depth to WW3 conventions.

    WW3 bottom convention:
        negative : below mean sea level
        positive : above mean sea level / land

    Obstruction input:
        transparency = 1 - land_fraction
        stored as integer percentage.
    """

    land = land_fraction >= land_cutoff
    sea = ~land

    invalid_sea = sea & (
        ~np.isfinite(depth) | (depth <= 0)
    )

    if np.any(invalid_sea):
        count = int(np.count_nonzero(invalid_sea))
        raise ValueError(
            f"{count} sea grid points have missing or invalid "
            "bathymetry. Resolve these before exporting."
        )

    # Convert positive-downward depth to negative elevation.
    # Store integer millimetres; WW3 applies scale factor 0.001.
    bottom = np.empty(depth.shape, dtype=np.int64)

    bottom[sea] = np.rint(
        -depth[sea] / cfg.WW3_DEPTH_SCALE
    ).astype(np.int64)

    bottom[land] = np.rint(
        dry_depth / cfg.WW3_DEPTH_SCALE
    ).astype(np.int64)

    # Cell-centred transparency for FLAGTR=4.
    transparency = 1.0 - land_fraction
    obstruction = np.rint(
        transparency * obstruction_scale
    ).astype(np.int64)

    # WW3 mask convention: 0 = land, 1 = sea.
    mask = sea.astype(np.int32)

    if cfg.WW3_MASK_VARIANT == "bound":
        mask[0, sea[0, :]] = 2
        mask[-1, sea[-1, :]] = 2
        mask[sea[:, 0], 0] = 2
        mask[sea[:, -1], -1] = 2
    elif cfg.WW3_MASK_VARIANT != "nobound":
        raise ValueError(
            f"Unknown WW3_MASK_VARIANT: {cfg.WW3_MASK_VARIANT}"
        )

    return bottom, obstruction, mask, land, sea


# ============================================================
# ASCII OUTPUT
# ============================================================

def write_integer_grid(path, array):
    """Write a 2-D integer array as free-format ASCII."""

    with path.open("w", encoding="ascii", newline="\n") as f:
        np.savetxt(
            f,
            array,
            fmt="%d",
            delimiter=" ",
        )


def write_ww3_grid_inp(
    path,
    experiment_name,
    lon,
    lat,
    bottom_file,
    obstruction_file,
    mask_file,
    settings,
):
    """Write the positional ww3_grid.inp file."""

    nx = lon.size
    ny = lat.size

    dlon = float(np.median(np.diff(lon)))
    dlat = float(np.median(np.abs(np.diff(lat))))

    title = settings["WW3_GRID_TITLE"]
    if len(title) > 30:
        raise ValueError(
            "WW3_GRID_TITLE must be 30 characters or fewer."
        )

    spectrum = settings["WW3_SPECTRUM"]
    flags = settings["WW3_FLAGS"]
    time_steps = settings["WW3_TIME_STEPS"]

    if time_steps is None:
        # Conservative provisional values for the current LHB grid.
        # Revisit after the first ww3_grid validation.
        time_steps = (850, 170, 850, 30)

    if len(time_steps) != 4:
        raise ValueError(
            "WW3_TIME_STEPS must contain four values."
        )

    zlim = settings["WW3_ZLIM"]
    dmin = settings["WW3_DMIN"]
    depth_unit = settings["WW3_UNIT_DEPTH"]
    obstr_unit = settings["WW3_UNIT_OBSTR"]
    mask_unit = settings["WW3_UNIT_MASK"]

    depth_scale = settings["WW3_DEPTH_SCALE"]
    obstr_scale = 1.0 / settings["OBSTR_SCALE"]

    lines = [
        "$ ------------------------------------------------------------",
        f"$ WW3 grid input generated for experiment: {experiment_name}",
        "$ Generated by 06_export_ww3_input.py",
        "$ ------------------------------------------------------------",
        f"'{title}'",
        "",
        " ".join(str(v) for v in spectrum),
        flags,
        " ".join(str(v) for v in time_steps),
        "",
        (
            f"&MISC CICE0 = {settings['WW3_CICE0']}, "
            f"CICEN = {settings['WW3_CICEN']}, "
            f"FLAGTR = {settings['WW3_FLAGTR']} /"
        ),
        "END OF NAMELISTS",
        "",
        "'RECT' T 'NONE'",
        f"{nx} {ny}",
        f"{dlon:.10f} {dlat:.10f} 1.0",
        f"{lon[0]:.10f} {np.min(lat):.10f} 1.0",
        (
            f"{zlim} {dmin} {depth_unit} "
            f"{depth_scale} 3 1 '(....)' "
            f"'NAME' '{bottom_file}'"
        ),
        (
            f"{obstr_unit} {obstr_scale} 3 1 '(....)' "
            f"'NAME' '{obstruction_file}'"
        ),
        (
            f"{mask_unit} 3 1 '(....)' "
            f"'NAME' '{mask_file}'"
        ),
    ]

    path.write_text(
        "\n".join(lines) + "\n",
        encoding="ascii",
    )


# ============================================================
# EXPERIMENT SETTINGS
# ============================================================

def load_experiment_settings(experiment_dir):
    """
    Load optional experiment.py.

    Example:
        SETTINGS = {
            "LAND_CUTOFF": 0.50,
            "WW3_DMIN": 2.5,
        }
    """

    settings = {
        name: getattr(cfg, name)
        for name in (
            "WW3_GRID_TITLE",
            "WW3_SPECTRUM",
            "WW3_FLAGS",
            "WW3_TIME_STEPS",
            "WW3_ZLIM",
            "WW3_DMIN",
            "WW3_DRY_DEPTH",
            "WW3_DEPTH_SCALE",
            "WW3_FLAGTR",
            "WW3_CICE0",
            "WW3_CICEN",
            "WW3_UNIT_DEPTH",
            "WW3_UNIT_OBSTR",
            "WW3_UNIT_MASK",
            "WW3_MASK_VARIANT",
            "LAND_CUTOFF",
            "OBSTR_SCALE",
        )
    }

    experiment_file = experiment_dir / "experiment.py"

    if experiment_file.exists():
        spec = importlib.util.spec_from_file_location(
            "experiment_settings",
            experiment_file,
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        overrides = getattr(module, "SETTINGS", {})
        settings.update(overrides)

    return settings


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment",
        default="lhb_control",
        help="Experiment directory name.",
    )
    args = parser.parse_args()

    experiment_name = args.experiment
    experiment_dir = ROOT / "experiments" / experiment_name
    output_dir = ROOT / "output" / "ww3" / experiment_name

    output_dir.mkdir(parents=True, exist_ok=True)

    settings = load_experiment_settings(experiment_dir)

    print("=" * 65)
    print("WW3 INPUT EXPORT")
    print("=" * 65)
    print(f"Experiment : {experiment_name}")
    print(f"Output     : {output_dir}")

    # Load shared generated datasets.
    lon, lat = load_grid(GRID_FILE)
    land_fraction = load_land_fraction(LANDMASK_FILE)
    depth = load_bathymetry(BATHY_FILE)

    validate_array_shapes(
        lon, lat, land_fraction, depth
    )

    # Convert to WW3 conventions.
    bottom, obstruction, mask, land, sea = make_ww3_arrays(
        land_fraction=land_fraction,
        depth=depth,
        land_cutoff=settings["LAND_CUTOFF"],
        dry_depth=settings["WW3_DRY_DEPTH"],
        obstruction_scale=settings["OBSTR_SCALE"],
    )

    # WW3 IDLA=3: north-to-south row order.
    bottom, obstruction, mask = orient_north_to_south(
        lat, bottom, obstruction, mask
    )

    # Write data files.
    bottom_path = output_dir / "bottom.inp"
    obstruction_path = output_dir / "obstr.inp"
    mask_path = output_dir / "mask.inp"
    grid_path = output_dir / "ww3_grid.inp"

    write_integer_grid(bottom_path, bottom)
    write_integer_grid(obstruction_path, obstruction)
    write_integer_grid(mask_path, mask)

    write_ww3_grid_inp(
        path=grid_path,
        experiment_name=experiment_name,
        lon=lon,
        lat=lat,
        bottom_file=bottom_path.name,
        obstruction_file=obstruction_path.name,
        mask_file=mask_path.name,
        settings=settings,
    )

    # Summary.
    print("\n--- Grid ---")
    print(f"NX x NY       : {lon.size} x {lat.size}")
    print(f"Longitude     : {lon.min():.4f} to {lon.max():.4f}")
    print(f"Latitude      : {lat.min():.4f} to {lat.max():.4f}")
    print(f"Grid points   : {lon.size * lat.size:,}")

    print("\n--- Land / sea ---")
    print(f"Land points   : {np.count_nonzero(land):,}")
    print(f"Sea points    : {np.count_nonzero(sea):,}")

    print("\n--- Bathymetry ---")
    print(
        f"Sea depth min : {np.nanmin(depth[sea]):.2f} m"
    )
    print(
        f"Sea depth max : {np.nanmax(depth[sea]):.2f} m"
    )

    print("\n--- Output files ---")
    for path in (
        grid_path,
        bottom_path,
        obstruction_path,
        mask_path,
    ):
        print(f"{path.name:16s}: {path.stat().st_size:,} bytes")

    print("\nExport completed.")


if __name__ == "__main__":
    main()
