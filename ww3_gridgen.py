"""
ww3_gridgen.py

Pure-Python re-implementation of the NOAA/EMC gridgen workflow for a
WAVEWATCH III v6.07.1 structured spherical (RECT) grid, using

    - IBCSO bed elevation  (bathymetry)
    - ADD coastline polygons (land / ice shelf / ice tongue / rumple = land)

Pipeline (same order as gridgen):
    1. depth + initial wet/dry     (gridgen: generate_grid, land mask module)
    2. remove isolated water bodies (gridgen: remove_lake)
    3. optional open-boundary flag  (gridgen: mask with active boundaries)
    4. sub-grid obstruction Sx, Sy  (gridgen: create_obstr)
    5. write WW3 files              (*.depth_ascii, *.obstr_lev1, *.mask, ww3_grid.inp)

Conventions
    - Arrays are (ny, nx) with row 0 = SOUTHERNMOST latitude (WW3 IDLA=1,
      "read line-by-line bottom to top").
    - Depth is elevation in metres, negative below sea level (WW3/gridgen).
    - A "cell" is the box of +-0.5 increments around a grid point.
"""

from dataclasses import dataclass
import math

import numpy as np
import geopandas as gpd
import shapely
from shapely.geometry import box, LineString, MultiPolygon, Polygon
from shapely.ops import unary_union
from shapely.strtree import STRtree
from scipy import ndimage

import rasterio
from rasterio import features
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from rasterio.warp import reproject

from pyproj import CRS, Transformer


CRS_GEO = CRS.from_epsg(4326)


# ----------------------------------------------------------------------
# Grid definition
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class GridSpec:
    name: str
    lon0: float
    lat0: float
    dlon: float
    dlat: float
    nx: int
    ny: int

    @classmethod
    def from_extent(cls, name, lon_min, lon_max, dlon, lat_min, lat_max, dlat):
        nx_f = (lon_max - lon_min) / dlon + 1.0
        ny_f = (lat_max - lat_min) / dlat + 1.0
        nx, ny = int(round(nx_f)), int(round(ny_f))
        if abs(nx - nx_f) > 1e-6 or abs(ny - ny_f) > 1e-6:
            raise ValueError(
                "Domain size is not an integer multiple of the increment: "
                f"nx={nx_f}, ny={ny_f}"
            )
        return cls(name, lon_min, lat_min, dlon, dlat, nx, ny)

    @property
    def lon(self):
        return self.lon0 + self.dlon * np.arange(self.nx)

    @property
    def lat(self):
        return self.lat0 + self.dlat * np.arange(self.ny)

    @property
    def window(self):
        """Outer cell edges (west, east, south, north)."""
        return (
            self.lon0 - 0.5 * self.dlon,
            self.lon[-1] + 0.5 * self.dlon,
            self.lat0 - 0.5 * self.dlat,
            self.lat[-1] + 0.5 * self.dlat,
        )

    def describe(self):
        w, e, s, n = self.window
        return (
            f"{self.name}: {self.nx} x {self.ny} = {self.nx * self.ny} points, "
            f"lon {self.lon0:g}..{self.lon[-1]:g} (d={self.dlon:g}), "
            f"lat {self.lat0:g}..{self.lat[-1]:g} (d={self.dlat:g}); "
            f"cell window W{w:g} E{e:g} S{s:g} N{n:g}"
        )


# ----------------------------------------------------------------------
# Reading source data
# ----------------------------------------------------------------------
def _window_bbox_in_crs(grid, crs, pad):
    """Bounding box (xmin, ymin, xmax, ymax) of the lon/lat window in `crs`."""
    west, east, south, north = grid.window
    n = 200
    lons = np.concatenate((
        np.linspace(west, east, n), np.linspace(west, east, n),
        np.full(n, west), np.full(n, east),
    ))
    lats = np.concatenate((
        np.full(n, south), np.full(n, north),
        np.linspace(south, north, n), np.linspace(south, north, n),
    ))
    xs, ys = Transformer.from_crs(CRS_GEO, crs, always_xy=True).transform(lons, lats)
    return (xs.min() - pad, ys.min() - pad, xs.max() + pad, ys.max() + pad)


def load_land_polygons(grid, polygon_file, layer, land_surfaces, add_crs,
                       pad_m=20_000.0, segmentize_m=1_000.0):
    """
    Read ADD polygons covering the grid window, keep the land-like classes,
    dissolve them into one land geometry and return its polygons in lon/lat.

    Dissolving removes internal boundaries (e.g. rock / ice-shelf contacts)
    that are not wave-relevant coastlines.
    """
    add_crs = CRS.from_user_input(add_crs)
    bbox = _window_bbox_in_crs(grid, add_crs, pad_m)

    print("\nLoading ADD polygons:", polygon_file.name)
    gdf = gpd.read_file(polygon_file, layer=layer, bbox=bbox)
    if gdf.crs is None:
        raise ValueError("ADD polygon CRS is missing.")
    if CRS.from_user_input(gdf.crs) != add_crs:
        gdf = gdf.to_crs(add_crs)
    if "surface" not in gdf.columns:
        raise ValueError("ADD polygon layer has no 'surface' column.")

    print("  features in window:", len(gdf))
    print("  surface classes:", dict(gdf["surface"].value_counts()))

    wanted = {s.lower() for s in land_surfaces}
    gdf = gdf[gdf["surface"].astype(str).str.lower().isin(wanted)]
    print("  kept as land:", sorted(wanted), "->", len(gdf), "features")
    if gdf.empty:
        raise ValueError("No ADD land polygons found in the grid window.")

    # The Antarctic polygon has millions of vertices: clip to the window with
    # the fast rectangle clipper FIRST, then repair the (small) result.
    geoms = [shapely.clip_by_rect(g, *bbox) for g in gdf.geometry.values]
    geoms = [shapely.make_valid(g) for g in geoms if not g.is_empty]
    geoms = [g for g in geoms if not g.is_empty]
    land = unary_union(geoms)

    # Densify before reprojection so that straight clip edges follow the
    # true curved lon/lat window instead of cutting across it.
    land = shapely.segmentize(land, segmentize_m)
    land = gpd.GeoSeries([land], crs=add_crs).to_crs(CRS_GEO).iloc[0]
    land = shapely.make_valid(land)

    polys = []
    for g in getattr(land, "geoms", [land]):
        if isinstance(g, Polygon) and not g.is_empty:
            polys.append(g)
        elif isinstance(g, MultiPolygon):
            polys.extend(p for p in g.geoms if not p.is_empty)
    print("  land polygons after dissolve:", len(polys))
    return polys


def _fine_transform(grid, n_sub):
    west, east, south, north = grid.window
    return from_origin(west, north, grid.dlon / n_sub, grid.dlat / n_sub)


def fine_bathymetry(grid, ibcso_file, n_sub, resampling="bilinear"):
    """IBCSO elevation resampled to the lon/lat sub-grid (rows north->south)."""
    height, width = grid.ny * n_sub, grid.nx * n_sub
    dst = np.full((height, width), np.nan, dtype=np.float32)

    print("\nResampling IBCSO to lon/lat sub-grid:", width, "x", height)
    with rasterio.open(ibcso_file) as src:
        if src.crs is None:
            raise ValueError("IBCSO CRS is missing.")
        reproject(
            source=rasterio.band(src, 1),
            destination=dst,
            src_transform=src.transform,
            src_crs=src.crs,
            src_nodata=src.nodata,
            dst_transform=_fine_transform(grid, n_sub),
            dst_crs=CRS_GEO,
            dst_nodata=np.nan,
            resampling=getattr(Resampling, resampling),
            init_dest_nodata=True,
        )

    finite = np.isfinite(dst)
    print(f"  valid fraction: {finite.mean():.3f}")
    if finite.any():
        print(f"  elevation range: {np.nanmin(dst):.1f} .. {np.nanmax(dst):.1f} m")
    return dst


def fine_land(grid, land_polys, n_sub):
    """ADD land rasterised on the same sub-grid (rows north->south)."""
    shapes = ((g, 1) for g in land_polys)
    return features.rasterize(
        shapes,
        out_shape=(grid.ny * n_sub, grid.nx * n_sub),
        transform=_fine_transform(grid, n_sub),
        fill=0,
        dtype="uint8",
    ).astype(bool)


# ----------------------------------------------------------------------
# Depth and land-sea mask
# ----------------------------------------------------------------------
def _blocks(fine, grid, n_sub):
    """(ny*n, nx*n) rows north->south  ->  (ny, n, nx, n) with j=0 southernmost."""
    return fine[::-1, :].reshape(grid.ny, n_sub, grid.nx, n_sub)


def build_depth_and_mask(grid, elev_fine, land_fine, n_sub,
                         wet_limit, land_cutoff):
    """
    gridgen generate_grid + land mask module.

    A sub-grid pixel is OCEAN when IBCSO elevation < 0.
    A cell is wet when
        (ocean fraction >= wet_limit)  and  (ADD land fraction <= land_cutoff).
    Cell depth is the mean elevation of ocean pixels that are NOT under ADD
    land (so sub-ice-shelf bed is not mixed into open-water depth).
    """
    e = _blocks(elev_fine, grid, n_sub)
    land = _blocks(land_fine, grid, n_sub)

    ocean = np.isfinite(e) & (e < 0.0)
    frac_ocean = ocean.mean(axis=(1, 3))
    frac_land = land.mean(axis=(1, 3))

    open_px = ocean & ~land
    count = open_px.sum(axis=(1, 3))
    total = np.where(open_px, e, 0.0).sum(axis=(1, 3))
    depth = np.where(count > 0, total / np.maximum(count, 1), np.nan)

    wet = (frac_ocean >= wet_limit) & (frac_land <= land_cutoff) & (count > 0)

    print("\nInitial land-sea mask:")
    print(f"  wet cells: {int(wet.sum())} / {wet.size}")
    print(f"  cells made dry by ADD land (> {land_cutoff}): "
          f"{int(((frac_ocean >= wet_limit) & (frac_land > land_cutoff)).sum())}")
    return depth.astype(np.float64), wet, frac_land


def remove_isolated_water(wet):
    """
    gridgen remove_lake, adapted: keep only water bodies (4-connected) that
    touch the outer edge of the grid, i.e. that connect to the open ocean.
    Enclosed bays/lakes/cavities are set dry.
    """
    labels, n = ndimage.label(wet)          # default structure: 4-connectivity
    if n == 0:
        raise ValueError("No wet cells in the grid.")

    edge = np.zeros_like(wet, dtype=bool)
    edge[0, :] = edge[-1, :] = True
    edge[:, 0] = edge[:, -1] = True
    keep = np.unique(labels[edge & wet])
    keep = keep[keep > 0]
    if keep.size == 0:                       # fall back: largest body
        sizes = ndimage.sum(wet, labels, index=np.arange(1, n + 1))
        keep = np.array([1 + int(np.argmax(sizes))])
        print("  WARNING: no water body touches the grid edge; kept the largest.")

    cleaned = np.isin(labels, keep)
    print(f"\nWater bodies found: {n}; kept {keep.size} "
          f"({int(cleaned.sum())} cells); removed {int(wet.sum() - cleaned.sum())} cells")
    return cleaned


def make_mask_variants(wet):
    """Return (mask_nobound, mask_bound): 0 land, 1 sea, 2 active boundary."""
    nobound = wet.astype(np.int8)
    bound = nobound.copy()
    edge = np.zeros_like(wet, dtype=bool)
    edge[0, :] = edge[-1, :] = True
    edge[:, 0] = edge[:, -1] = True
    bound[edge & wet] = 2
    return nobound, bound


# ----------------------------------------------------------------------
# Sub-grid obstruction  (port of gridgen create_obstr, rectilinear case)
# ----------------------------------------------------------------------
def _merge_segments(segs):
    """Join overlapping [lo, hi] segments (closed intervals)."""
    out = []
    for lo, hi in sorted(segs):
        if out and lo <= out[-1][1]:
            out[-1][1] = max(out[-1][1], hi)
        else:
            out.append([lo, hi])
    return out


def _combine_with_neighbours(own, left, right):
    """
    Combine a cell's net segments with those of its neighbours exactly as
    gridgen does: own segments completely shadowed by a neighbour segment are
    dropped (all dropped -> no obstruction); neighbour segments that are
    completely inside an own segment are ignored; the rest are merged and
    summed.
    """
    base = [list(s) for s in own]
    for nb in (left, right):
        if not nb:
            continue
        keep = [s for s in base
                if not any(nlo <= s[0] and nhi >= s[1] for nlo, nhi in nb)]
        if not keep:
            return 0.0
        base = keep
        add = [list(n) for n in nb
               if not any(s[0] <= n[0] and s[1] >= n[1] for s in base)]
        base = base + add
    merged = _merge_segments(base)
    return float(sum(hi - lo for lo, hi in merged))


def compute_obstruction(grid, land_polys, wet, offset=1):
    """
    Sx: fraction of the cell HEIGHT blocked for x-propagation.
    Sy: fraction of the cell WIDTH  blocked for y-propagation.

    Boundaries are the rings (outer + holes) of the dissolved ADD land
    polygons.  For every wet cell and every ring, the part of the ring inside
    the cell gives a N-S extent (-> Sx) and an E-W extent (-> Sy).  Then, as in
    gridgen: (1) move segments of one boundary shared by adjacent cells to the
    cell with the larger segment, (2) merge overlapping segments, (3) include
    neighbours (offset=1), (4) zero obstruction next to dry cells.
    """
    ny, nx = wet.shape
    lon, lat = grid.lon, grid.lat
    dlon, dlat = grid.dlon, grid.dlat

    rings = []
    for poly in land_polys:
        rings.append(LineString(poly.exterior.coords))
        rings.extend(LineString(r.coords) for r in poly.interiors)
    rings = [r for r in rings if r.length > 0]
    for r in rings:
        shapely.prepare(r)
    tree = STRtree(rings)

    # --- step 1: per-cell, per-boundary extents (normalised 0..1 in the cell)
    seg_y = {}   # (j, i) -> {ring_id: [lo, hi]}   N-S extent  -> Sx
    seg_x = {}   # (j, i) -> {ring_id: [lo, hi]}   E-W extent  -> Sy
    jj, ii = np.nonzero(wet)
    for j, i in zip(jj, ii):
        x0 = lon[i] - 0.5 * dlon
        y0 = lat[j] - 0.5 * dlat
        x1, y1 = x0 + dlon, y0 + dlat
        cell = box(x0, y0, x1, y1)
        for k in tree.query(cell):
            if not shapely.intersects(rings[k], cell):
                continue
            part = shapely.clip_by_rect(rings[k], x0, y0, x1, y1)
            if part.is_empty:
                continue
            bx0, by0, bx1, by1 = part.bounds
            ylo, yhi = (by0 - y0) / dlat, (by1 - y0) / dlat
            xlo, xhi = (bx0 - x0) / dlon, (bx1 - x0) / dlon
            seg_y.setdefault((j, i), {})[k] = [min(max(ylo, 0.0), 1.0),
                                               min(max(yhi, 0.0), 1.0)]
            seg_x.setdefault((j, i), {})[k] = [min(max(xlo, 0.0), 1.0),
                                               min(max(xhi, 0.0), 1.0)]

    n_bnd_cells = len(seg_y)
    print(f"\nObstruction: {n_bnd_cells} wet cells contain land boundaries")

    # --- step 2: move shared boundary segments to the cell with the larger one
    def transfer(a, b):
        for k in set(a) & set(b):
            la, lb = a[k][1] - a[k][0], b[k][1] - b[k][0]
            hull = [min(a[k][0], b[k][0]), max(a[k][1], b[k][1])]
            if la >= lb:
                a[k] = hull
                del b[k]
            else:
                b[k] = hull
                del a[k]

    for (j, i) in sorted(seg_y, key=lambda c: (c[1], c[0])):   # column-major
        if (j, i + 1) in seg_y:
            transfer(seg_y[(j, i)], seg_y[(j, i + 1)])
        if (j + 1, i) in seg_x:
            transfer(seg_x[(j, i)], seg_x[(j + 1, i)])

    # --- step 3: net (non-overlapping) segments per cell
    net_y = {c: _merge_segments(list(d.values())) for c, d in seg_y.items() if d}
    net_x = {c: _merge_segments(list(d.values())) for c, d in seg_x.items() if d}

    # --- step 4: build Sx, Sy including neighbour information
    sx = np.zeros((ny, nx))
    sy = np.zeros((ny, nx))
    use_nb = bool(offset)
    for (j, i), own in net_y.items():
        left = net_y.get((j, i - 1), []) if use_nb else []
        right = net_y.get((j, i + 1), []) if use_nb else []
        sx[j, i] = _combine_with_neighbours(own, left, right)
    for (j, i), own in net_x.items():
        below = net_x.get((j - 1, i), []) if use_nb else []
        above = net_x.get((j + 1, i), []) if use_nb else []
        sy[j, i] = _combine_with_neighbours(own, below, above)

    # --- step 5: zero where the neighbour along that axis is dry
    dry = ~wet
    for j, i in zip(*np.nonzero(sx > 0)):
        if (i > 0 and dry[j, i - 1]) or (i < nx - 1 and dry[j, i + 1]):
            sx[j, i] = 0.0
    for j, i in zip(*np.nonzero(sy > 0)):
        if (j > 0 and dry[j - 1, i]) or (j < ny - 1 and dry[j + 1, i]):
            sy[j, i] = 0.0

    sx = np.clip(sx, 0.0, 1.0)
    sy = np.clip(sy, 0.0, 1.0)
    sx[dry] = 0.0
    sy[dry] = 0.0
    print(f"  Sx > 0 in {int((sx > 0).sum())} cells (max {sx.max():.2f}); "
          f"Sy > 0 in {int((sy > 0).sum())} cells (max {sy.max():.2f})")
    return sx, sy


# ----------------------------------------------------------------------
# WW3 file writers
# ----------------------------------------------------------------------
def _write_rows(handle, arr, fmt):
    for row in arr:                      # row 0 = south  (IDLA = 1)
        handle.write(" ".join(fmt % v for v in row) + "\n")


def write_depth_ascii(path, depth, wet, depth_scale, dry_depth):
    """Integer file; value * depth_scale = metres, negative below datum."""
    field = np.where(wet, depth, dry_depth)
    ints = np.rint(field / depth_scale).astype(np.int64)
    with open(path, "w") as f:
        _write_rows(f, ints, "%d")
    return path


def write_obstruction(path, sx, sy, scale):
    """Integer file with Sx block then Sy block (value / scale = fraction)."""
    with open(path, "w") as f:
        _write_rows(f, np.rint(sx * scale).astype(np.int64), "%d")
        _write_rows(f, np.rint(sy * scale).astype(np.int64), "%d")
    return path


def write_mask(path, mask):
    with open(path, "w") as f:
        _write_rows(f, mask.astype(np.int64), "%d")
    return path


def suggest_time_steps(grid, f1_hz):
    """
    Rough CFL-based defaults (dtmax, dtxy, dtkth, dtmin) in seconds, using the
    deep-water group speed of the lowest frequency.  Review before real runs.
    """
    g = 9.81
    cg_max = g / (4.0 * math.pi * f1_hz)
    lat_edge = max(abs(grid.window[2]), abs(grid.window[3]))
    dx = grid.dlon * 111_320.0 * math.cos(math.radians(lat_edge))
    dy = grid.dlat * 110_574.0
    dtxy = max(10.0, math.floor(min(dx, dy) / cg_max / 10.0) * 10.0)
    return 3.0 * dtxy, dtxy, 1.5 * dtxy, 10.0


def grid_definition_block(grid, files, cfg):
    """The 'Define grid' records, i.e. what gridgen writes to the .meta file."""
    z = cfg
    lines = [
        "'RECT' T 'NONE'",
        f"{grid.nx} {grid.ny}",
        f"{grid.dlon:.6f} {grid.dlat:.6f} 1.00",
        f"{grid.lon0:.6f} {grid.lat0:.6f} 1.00",
        "$ Bottom bathymetry",
        f"{z['zlim']:.2f} {z['dmin']:.2f} {z['unit_depth']} {z['depth_scale']:.6f} "
        f"1 1 '(....)' 'NAME' '{files['depth']}'",
        "$ Sub-grid information",
        f"{z['unit_obstr']} {1.0 / z['obstr_scale']:.6f} 1 1 '(....)' 'NAME' '{files['obstr']}'",
        "$ Mask information",
        f"{z['unit_mask']} 1 1 '(....)' 'NAME' '{files['mask']}'",
    ]
    return "\n".join(lines) + "\n"


def write_meta(path, grid, files, cfg):
    with open(path, "w") as f:
        f.write(grid_definition_block(grid, files, cfg))
    return path


def write_ww3_grid_inp(path, grid, files, cfg):
    """Complete ww3_grid.inp (v6.07 free-format input, not namelist .nml)."""
    xfr, f1, nk, nth, th0 = cfg["spectrum"]
    dtmax, dtxy, dtkth, dtmin = cfg["time_steps"]
    title = cfg["title"][:30]

    text = f"""$ -------------------------------------------------------------------- $
$ WAVEWATCH III Grid preprocessor input file                           $
$ Generated by ww3_gridgen.py  (IBCSO + ADD)                           $
$ -------------------------------------------------------------------- $
$ Grid name (C*30, in quotes)
  '{title}'
$
$ Frequency increment factor, first frequency (Hz), number of frequencies,
$ number of directions, relative offset of first direction
  {xfr:g} {f1:g} {nk} {nth} {th0:g}
$
$ Model flags: FLDRY FLCX FLCY FLCTH FLCK FLSOU
  {cfg['flags']}
$
$ Time steps (s): max global, max CFL x-y, max CFL k-theta, min source term
$ (CFL-based estimate: review before production runs)
  {dtmax:g}. {dtxy:g}. {dtkth:g}. {dtmin:g}.
$
$ Namelists (source terms etc. must match the switch file: add as needed)
  &MISC CICE0 = {cfg['cice0']:g}, CICEN = {cfg['cicen']:g}, FLAGTR = {cfg['flagtr']} /
$
END OF NAMELISTS
$
$ Define grid -------------------------------------------------------- $
{grid_definition_block(grid, files, cfg)}$
$ Output boundary points (none)
  0. 0. 0. 0. 0
$
$ -------------------------------------------------------------------- $
$ End of input file                                                    $
$ -------------------------------------------------------------------- $
"""
    with open(path, "w") as f:
        f.write(text)
    return path


# ----------------------------------------------------------------------
# Reading back (used by the checker)
# ----------------------------------------------------------------------
def read_ww3_ascii(path, ny, nx, blocks=1):
    data = np.loadtxt(path, dtype=np.int64)
    if data.shape != (blocks * ny, nx):
        raise ValueError(f"{path.name}: expected {(blocks * ny, nx)}, got {data.shape}")
    return [data[b * ny:(b + 1) * ny] for b in range(blocks)]
