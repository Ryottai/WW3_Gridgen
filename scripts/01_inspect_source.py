
"""
01_inspect_source.py

Inspect source datasets for WW3_LHB_Gridgen.
- ADD GeoPackage
- IBCSO bathymetry GeoTIFF

Original files are read-only.
"""

from pathlib import Path

import numpy as np
import geopandas as gpd
import rasterio


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

ADD_DIR = ROOT / "input" / "raw" / "ADD"
IBCSO_DIR = ROOT / "input" / "raw" / "IBCSO"


# ============================================================
# ADD INSPECTION
# ============================================================

def inspect_add():

    print("\n" + "=" * 65)
    print("ADD: GEOPACKAGE")
    print("=" * 65)

    gpkg_files = sorted(ADD_DIR.glob("*.gpkg"))

    if not gpkg_files:
        print(f"No GeoPackage found in: {ADD_DIR}")
        return

    for gpkg in gpkg_files:

        print(f"\nFile: {gpkg.name}")

        layers = gpd.list_layers(gpkg)

        print("\nAvailable layers:")
        print(layers.to_string(index=False))

        for _, layer_info in layers.iterrows():
            layer_name = layer_info["name"]
            geometry_type = layer_info["geometry_type"]

            # Skip non-spatial tables such as layer_styles
            if not isinstance(geometry_type, str):
                print(f"\nSkipping non-spatial table: {layer_name}")
                continue

            print(f"\n{'-' * 50}")
            print(f"Layer: {layer_name}")

            gdf = gpd.read_file(gpkg, layer=layer_name, rows=1)

            print("CRS:", gdf.crs)
            print("Geometry type:", gdf.geom_type.unique())
            print("Columns:", list(gdf.columns))


# ============================================================
# IBCSO INSPECTION
# ============================================================

def inspect_ibcso():

    print("\n" + "=" * 65)
    print("IBCSO: GEOTIFF")
    print("=" * 65)

    tif_files = sorted(IBCSO_DIR.glob("*.tif"))

    if not tif_files:
        tif_files = sorted(IBCSO_DIR.glob("*.tiff"))

    if not tif_files:
        print(f"No GeoTIFF found in: {IBCSO_DIR}")
        return

    for tif in tif_files:

        print(f"\nFile: {tif.name}")

        with rasterio.open(tif) as src:

            print("\n--- Raster metadata ---")
            print("CRS:", src.crs)
            print("Width:", src.width)
            print("Height:", src.height)
            print("Band count:", src.count)
            print("Resolution:", src.res)
            print("Bounds:", src.bounds)
            print("Transform:", src.transform)
            print("NoData:", src.nodata)
            print("Data type:", src.dtypes)
            print("Band descriptions:", src.descriptions)
            print("Units:", src.units)

            print("\n--- Dataset tags ---")
            for key, value in src.tags().items():
                print(f"{key}: {value}")

            print("\n--- Approximate raster statistics ---")

            # Read a reduced-resolution sample to avoid
            # loading a potentially large raster into memory.
            sample_height = min(src.height, 1500)
            sample_width = min(src.width, 1500)

            sample = src.read(
                1,
                out_shape=(sample_height, sample_width),
                masked=True
            )

            values = sample.compressed()

            if values.size > 0:
                print("Valid sample count:", values.size)
                print("Minimum:", float(np.min(values)))
                print("Maximum:", float(np.max(values)))
                print("Mean:", float(np.mean(values)))
                print("Median:", float(np.median(values)))
            else:
                print("No valid raster values found.")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    inspect_add()
    inspect_ibcso()

    print("\nInspection completed.")
