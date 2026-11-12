## Required External Data

The raw geospatial datasets used for grid generation are not included in this repository because of their large file sizes.

Download the following datasets and place them in the corresponding directories.

### 1. Antarctic Digital Database (ADD)

- **Source:** [SCAR Antarctic Digital Database](https://www.bas.ac.uk/project/add/)
- **Download:** [ADD Data Catalogue](https://data.bas.ac.uk/items/e74543c0-4c4e-4b41-aa33-5bb2f67df389/)
- **Required datasets:**
  - High-resolution coastline polylines
  - High-resolution coastline polygons
- **Format:** GeoPackage (`.gpkg`)

Place the downloaded files in:

```text
data/raw/ADD/
```

### 2. International Bathymetric Chart of the Southern Ocean (IBCSO)

- **Source:** [IBCSO](https://ibcso.org/)
- **Download:** [IBCSO 2025 Annual Release](https://ibcso.org/ibcso-2025-annual-release/)
- **Required dataset:** Bed Surface Grid (`IBCSO_bed.tif`)
- **Format:** GeoTIFF (`.tif`)

Place the downloaded file in:

```text
data/raw/IBCSO/
```

### Directory Structure

After downloading the datasets, the directory structure should be:

```text
data/
└── raw/
    ├── ADD/
    │   ├── add_coastline_high_res_line_v7_12.gpkg
    │   └── add_coastline_high_res_polygon_v7_12.gpkg
    └── IBCSO/
        └── IBCSO_bed.tif
```

**Note:** These raw datasets are excluded from Git version control via `.gitignore`. Users must download them separately before running the grid-generation scripts.