from pathlib import Path
import cdsapi


# ============================================================
# Configuration
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parents[1]

OUTPUT_DIR = PROJECT_DIR / "input" / "raw" / "ERA5"
OUTPUT_FILE = OUTPUT_DIR / "era5_lhb_wind_20230301_20230501.nc"

DATASET = "reanalysis-era5-single-levels"

REQUEST = {
    "product_type": ["reanalysis"],
    "variable": [
        "10m_u_component_of_wind",
        "10m_v_component_of_wind",
    ],
    "date": "2023-03-01/2023-05-01",
    "time": [
        f"{hour:02d}:00"
        for hour in range(24)
    ],
    # North, West, South, East
    "area": [-64, 28, -72, 44],
    "data_format": "netcdf",
}


# ============================================================
# Download
# ============================================================

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if OUTPUT_FILE.exists():
        raise FileExistsError(
            f"Output already exists: {OUTPUT_FILE}\n"
            "Rename or remove it before downloading again."
        )

    print("=" * 65)
    print("ERA5 WIND DATA DOWNLOAD")
    print("=" * 65)
    print(f"Dataset : {DATASET}")
    print(f"Output  : {OUTPUT_FILE}")
    print(f"Period  : 2023-03-01 to 2023-05-01")
    print(f"Area    : 28-44 E, 72-64 S")
    print("Variables: u10, v10")
    print("=" * 65)

    client = cdsapi.Client()

    client.retrieve(
        DATASET,
        REQUEST,
        str(OUTPUT_FILE),
    )

    print("\nDownload completed.")
    print(f"File: {OUTPUT_FILE}")
    print(f"Size: {OUTPUT_FILE.stat().st_size / 1024**2:.2f} MB")


if __name__ == "__main__":
    main()
