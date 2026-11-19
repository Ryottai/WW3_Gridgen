from pathlib import Path

import xarray as xr


PROJECT_DIR = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    PROJECT_DIR
    / "input/raw/ERA5"
    / "era5_lhb_wind_20230301_20230501.nc"
)

OUTPUT_FILE = (
    PROJECT_DIR
    / "output/ww3/lhb_control"
    / "era5_lhb_wind_20230301_20230501_ww3.nc"
)


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(INPUT_FILE)

    if OUTPUT_FILE.exists():
        raise FileExistsError(
            f"Output already exists: {OUTPUT_FILE}"
        )

    with xr.open_dataset(INPUT_FILE) as src:

        # Construct a clean dataset from only required variables.
        ds = xr.Dataset(
            data_vars={
                "u10": (
                    ("time", "latitude", "longitude"),
                    src["u10"].values,
                    {
                        "long_name": "10 metre U wind component",
                        "units": "m s-1",
                    },
                ),
                "v10": (
                    ("time", "latitude", "longitude"),
                    src["v10"].values,
                    {
                        "long_name": "10 metre V wind component",
                        "units": "m s-1",
                    },
                ),
            },
            coords={
                "time": src["valid_time"].values,
                "latitude": src["latitude"].values,
                "longitude": src["longitude"].values,
            },
        )

    # Reverse latitude if it is descending.
    if ds["latitude"].values[0] > ds["latitude"].values[-1]:
        ds = ds.sortby("latitude")

    print(
        "Latitude order:",
        ds["latitude"].values[0],
        "to",
        ds["latitude"].values[-1],
    )

    # CF-compliant coordinate attributes.
    ds["time"].attrs.update({
        "standard_name": "time",
        "long_name": "time",
    })
    ds["latitude"].attrs.update({
        "standard_name": "latitude",
        "long_name": "latitude",
        "units": "degrees_north",
    })
    ds["longitude"].attrs.update({
        "standard_name": "longitude",
        "long_name": "longitude",
        "units": "degrees_east",
    })

    # Time encoding.
    ds["time"].encoding.update({
        "units": "hours since 2023-03-01 00:00:00",
        "calendar": "standard",
        "dtype": "int64",
    })

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    ds.to_netcdf(
        OUTPUT_FILE,
        format="NETCDF4",
    )

    print(f"Created: {OUTPUT_FILE}")
    print(ds)


if __name__ == "__main__":
    main()