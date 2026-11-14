"""
04_check_ww3_inputs.py

Read back the files written by 03_generate_ww3_inputs.py exactly as ww3_grid
will (integers, scale factors, IDLA=1 = south row first), run consistency
checks, and plot depth / mask / Sx / Sy with the ADD land outline.
"""

from pathlib import Path
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import config as cfg
import ww3_gridgen as gg


def main():
    grid = gg.GridSpec.from_extent(
        cfg.WW3_GRID_NAME,
        cfg.GRID_LON_MIN, cfg.GRID_LON_MAX, cfg.GRID_DLON,
        cfg.GRID_LAT_MIN, cfg.GRID_LAT_MAX, cfg.GRID_DLAT,
    )
    ny, nx = grid.ny, grid.nx
    out = Path(cfg.WW3_DIR)
    stem = grid.name

    # ---- read back as WW3 does ------------------------------------------
    depth_i, = gg.read_ww3_ascii(out / f"{stem}.depth_ascii", ny, nx)
    sx_i, sy_i = gg.read_ww3_ascii(out / f"{stem}.obstr_lev1", ny, nx, blocks=2)
    mask_nb, = gg.read_ww3_ascii(out / f"{stem}.mask_nobound", ny, nx)
    mask_b, = gg.read_ww3_ascii(out / f"{stem}.mask", ny, nx)

    depth = depth_i * cfg.WW3_DEPTH_SCALE          # metres, negative down
    sx = sx_i / cfg.OBSTR_SCALE
    sy = sy_i / cfg.OBSTR_SCALE

    # WW3 rule: point is sea when depth < ZLIM
    sea_by_depth = depth < cfg.WW3_ZLIM

    # ---- consistency checks ---------------------------------------------
    problems = []

    def check(ok, text):
        print(("  OK   " if ok else "  FAIL ") + text)
        if not ok:
            problems.append(text)

    print("\nConsistency checks")
    check(depth.shape == (ny, nx), f"depth shape {depth.shape} == ({ny}, {nx})")
    check(set(np.unique(mask_nb)) <= {0, 1}, "mask_nobound values in {0,1}")
    check(set(np.unique(mask_b)) <= {0, 1, 2}, "mask values in {0,1,2}")
    check(np.array_equal(mask_nb > 0, mask_b > 0), "wet cells identical in both masks")
    check(np.array_equal(sea_by_depth, mask_nb == 1),
          "depth < ZLIM  <=>  mask == 1 (depth file and mask agree)")
    check(np.all((sx >= 0) & (sx <= 1)) and np.all((sy >= 0) & (sy <= 1)),
          "obstruction within [0, 1]")
    check(np.all(sx[mask_nb == 0] == 0) and np.all(sy[mask_nb == 0] == 0),
          "obstruction is 0 on land cells")
    wet_depth = depth[mask_nb == 1]
    check(wet_depth.max() < cfg.WW3_ZLIM, "all wet depths below ZLIM")
    n_shallow = int((wet_depth > -cfg.WW3_DMIN).sum())
    print(f"  info {n_shallow} wet cells shallower than DMIN={cfg.WW3_DMIN} m "
          "(WW3 raises them to DMIN)")
    print(f"  info Sx>0: {int((sx > 0).sum())} cells, Sy>0: {int((sy > 0).sum())} cells")

    # ---- plot -----------------------------------------------------------
    land_polys = gg.load_land_polygons(
        grid, cfg.ADD_POLYGON_FILE, "add_coastline_high_res_polygon_v7_12",
        cfg.LAND_SURFACES, cfg.ADD_CRS,
    )

    west, east, south, north = grid.window
    lon_edges = np.linspace(west, east, nx + 1)
    lat_edges = np.linspace(south, north, ny + 1)
    aspect = 1.0 / np.cos(np.radians(0.5 * (south + north)))

    depth_cmap = LinearSegmentedColormap.from_list("d", cfg.DEPTH_COLORS)
    shown = np.ma.masked_where(mask_nb == 0, -depth)
    mask_cmap = ListedColormap(["#777777", "#9ecae1", "#fdae6b"])
    obs_cmap = plt.get_cmap("magma_r").copy()

    fig, axes = plt.subplots(2, 2, figsize=(14, 8.5))
    panels = [
        (axes[0, 0], shown, dict(cmap=depth_cmap, vmin=cfg.DEPTH_VMIN,
                                 vmax=cfg.DEPTH_VMAX), "Depth (m)"),
        (axes[0, 1], mask_b, dict(cmap=mask_cmap, vmin=-0.5, vmax=2.5),
         "Mask (0 land, 1 sea, 2 open boundary)"),
        (axes[1, 0], np.ma.masked_where(mask_nb == 0, sx),
         dict(cmap=obs_cmap, vmin=0, vmax=1), "Sx obstruction"),
        (axes[1, 1], np.ma.masked_where(mask_nb == 0, sy),
         dict(cmap=obs_cmap, vmin=0, vmax=1), "Sy obstruction"),
    ]
    for ax, data, kw, title in panels:
        ax.set_facecolor("#777777")
        mesh = ax.pcolormesh(lon_edges, lat_edges, data, shading="flat", **kw)
        for poly in land_polys:
            ax.plot(*poly.exterior.xy, color="k", lw=0.5)
            for ring in poly.interiors:
                ax.plot(*ring.xy, color="k", lw=0.5)
        ax.set_xlim(west, east)
        ax.set_ylim(south, north)
        ax.set_aspect(aspect)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Longitude (deg E)")
        ax.set_ylabel("Latitude (deg)")
        fig.colorbar(mesh, ax=ax, shrink=0.8, pad=0.02)

    fig.suptitle(f"WW3 inputs read back: {stem}", fontsize=13)
    fig.tight_layout()
    fig_file = Path(cfg.FIGURE_DIR) / "04_ww3_inputs_check.png"
    fig.savefig(fig_file, dpi=min(cfg.FIGURE_DPI, 150), bbox_inches="tight",
                facecolor="white")
    print("\nFigure saved:", fig_file)

    if problems:
        print(f"\n{len(problems)} check(s) FAILED.")
        sys.exit(1)
    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
