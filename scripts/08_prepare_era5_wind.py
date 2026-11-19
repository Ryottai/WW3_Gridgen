
from pathlib import Path

import xarray as xr


PROJECT_DIR = Path(__file__).resolve().parents[1]
INPUT_FILE = (
    PROJECT_DIR
    / "input"
    / "raw"
    / "ERA5"
    / "era5_lhb_wind_20230301_20230501.nc"
)
OUTPUT_FILE = (
    PROJECT_DIR
    / "output"
    / "ww3"
    / "lhb_control" 
    / "era5_lhb_wind_20230301_20230501_ww3.nc"
)


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(INPUT_FILE)

    if OUTPUT_FILE.exists():
        raise FileExistsError(
            f"Output already exists: {OUTPUT_FILE}"
        )

    with xr.open_dataset(INPUT_FILE) as source:
        # WW3 expects the time dimension and coordinate to be named "time".
        if "valid_time" not in source.dims:
            raise ValueError("Expected dimension 'valid_time'.")

        ds = source.rename({"valid_time": "time"})

        # These ERA5 auxiliary coordinates are not needed by WW3.
        ds = ds.drop_vars(
            ["number", "expver"],
            errors="ignore",
        )

        # Explicitly define CF-compliant time encoding.
        ds["time"].encoding["units"] = (
            "hours since 2023-03-01 00:00:00"
        )
        ds["time"].encoding["calendar"] = "standard"

        # Keep only the two wind components and their coordinates.
        ds = ds[["u10", "v10"]]

        OUTPUT_FILE.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        ds.to_netcdf(
            OUTPUT_FILE,
            format="NETCDF4",
        )

    print(f"Created: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
