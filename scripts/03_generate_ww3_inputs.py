"""
03_generate_ww3_inputs.py

Generate WAVEWATCH III v6.07.1 structured-grid inputs from IBCSO + ADD only:

    <name>.depth_ascii   bathymetry   (integer mm, negative below datum)
    <name>.obstr_lev1    sub-grid obstruction Sx then Sy (integer percent)
    <name>.mask_nobound  land-sea mask, 0/1
    <name>.mask          land-sea mask, 0/1/2 (open edges = active boundary)
    <name>.meta          'Define grid' block of ww3_grid.inp
    ww3_grid.inp         complete grid preprocessor input

All files are written to config.WW3_DIR.  Settings live in config.py.
"""

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import config as cfg
import ww3_gridgen as gg


def main():
    print("=" * 65)
    print("WW3 GRIDGEN: GENERATE WW3 INPUT FILES (IBCSO + ADD)")
    print("=" * 65)

    grid = gg.GridSpec.from_extent(
        cfg.WW3_GRID_NAME,
        cfg.GRID_LON_MIN, cfg.GRID_LON_MAX, cfg.GRID_DLON,
        cfg.GRID_LAT_MIN, cfg.GRID_LAT_MAX, cfg.GRID_DLAT,
    )
    print(grid.describe())

    # ---- source data on the grid window ---------------------------------
    land_polys = gg.load_land_polygons(
        grid, cfg.ADD_POLYGON_FILE, "add_coastline_high_res_polygon_v7_12",
        cfg.LAND_SURFACES, cfg.ADD_CRS,
    )
    elev_fine = gg.fine_bathymetry(
        grid, cfg.IBCSO_FILE, cfg.GRID_SUBSAMPLE, cfg.IBCSO_RESAMPLING
    )
    land_fine = gg.fine_land(grid, land_polys, cfg.GRID_SUBSAMPLE)

    # ---- depth + land-sea mask ------------------------------------------
    depth, wet, _ = gg.build_depth_and_mask(
        grid, elev_fine, land_fine, cfg.GRID_SUBSAMPLE,
        cfg.BATHY_WET_LIMIT, cfg.LAND_CUTOFF,
    )
    wet = gg.remove_isolated_water(wet)
    mask_nobound, mask_bound = gg.make_mask_variants(wet)

    # ---- sub-grid obstruction -------------------------------------------
    sx, sy = gg.compute_obstruction(grid, land_polys, wet, cfg.OBSTR_OFFSET)

    # ---- write files ----------------------------------------------------
    out = Path(cfg.WW3_DIR)
    out.mkdir(parents=True, exist_ok=True)
    stem = grid.name

    files = {
        "depth": f"{stem}.depth_ascii",
        "obstr": f"{stem}.obstr_lev1",
        "mask": (f"{stem}.mask_nobound" if cfg.WW3_MASK_VARIANT == "nobound"
                 else f"{stem}.mask"),
    }

    gg.write_depth_ascii(out / files["depth"], depth, wet,
                         cfg.WW3_DEPTH_SCALE, cfg.WW3_DRY_DEPTH)
    gg.write_obstruction(out / files["obstr"], sx, sy, cfg.OBSTR_SCALE)
    gg.write_mask(out / f"{stem}.mask_nobound", mask_nobound)
    gg.write_mask(out / f"{stem}.mask", mask_bound)

    time_steps = cfg.WW3_TIME_STEPS or gg.suggest_time_steps(
        grid, cfg.WW3_SPECTRUM[1]
    )
    ww3_cfg = {
        "title": cfg.WW3_GRID_TITLE,
        "zlim": cfg.WW3_ZLIM, "dmin": cfg.WW3_DMIN,
        "depth_scale": cfg.WW3_DEPTH_SCALE, "obstr_scale": cfg.OBSTR_SCALE,
        "unit_depth": cfg.WW3_UNIT_DEPTH, "unit_obstr": cfg.WW3_UNIT_OBSTR,
        "unit_mask": cfg.WW3_UNIT_MASK,
        "spectrum": cfg.WW3_SPECTRUM, "flags": cfg.WW3_FLAGS,
        "time_steps": time_steps,
        "flagtr": cfg.WW3_FLAGTR, "cice0": cfg.WW3_CICE0, "cicen": cfg.WW3_CICEN,
    }
    gg.write_meta(out / f"{stem}.meta", grid, files, ww3_cfg)
    gg.write_ww3_grid_inp(out / "ww3_grid.inp", grid, files, ww3_cfg)

    # ---- summary --------------------------------------------------------
    wet_depth = depth[wet]
    print("\n" + "-" * 65)
    print("Output directory:", out)
    for name in (files["depth"], files["obstr"], f"{stem}.mask_nobound",
                 f"{stem}.mask", f"{stem}.meta", "ww3_grid.inp"):
        print("  ", name)
    print(f"Wet cells: {int(wet.sum())} / {wet.size}  "
          f"(mask in ww3_grid.inp: {files['mask']})")
    print(f"Depth of wet cells: {wet_depth.min():.1f} .. {wet_depth.max():.1f} m "
          f"(negative down)")
    print(f"Time steps (dtmax, dtxy, dtkth, dtmin): {time_steps}")
    print("-" * 65)
    print("Run ww3_grid in the output directory:  cd", out, "&& ww3_grid")


if __name__ == "__main__":
    main()
