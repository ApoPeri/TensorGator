"""
Fused CUDA kernels for the coverage / visibility pipeline.

Profiling the coverage_map example (10 sats, 14400 steps, 2701 ground points)
put 74% of the runtime in calculate_max_gaps, a pure python double loop, and
another chunk in moving the P x T visibility array back to the host. Both
disappear if the visibility test and the gap scan run in one kernel: the
P x T array is never materialised anywhere, and only P floats come back.

Kernels here:
  visibility_cuda        upgraded standalone visibility (uint8, no asin/sqrt/div)
  coverage_max_gaps_cuda fused visibility + longest-gap scan, returns (P,) only
  coverage_report        whole pipeline, positions stay on the device

The visibility predicate is unchanged but evaluated without transcendentals:

    elevation >= min_el
      <=> dot(d, g_hat) / |d| >= sin(min_el)
      <=> signsq(dot) >= signsq(sin(min_el)) * |d|^2      (x|x| is monotonic)

which needs no asin, no sqrt and no divide, and stays exact for negative
minimum elevations.
"""

import math

import numpy as np
from numba import cuda, float32, float64, int32

_TPB_GAP = 256          # threads per block for the gap reduction

_cache = {}


def _visible_fn(dtype):
    """Shared device predicate: elevation >= min_el, no asin/sqrt/divide."""
    key = ('visible', np.dtype(dtype).name)
    if key in _cache:
        return _cache[key]

    @cuda.jit(device=True, inline=True)
    def _visible(px, py, pz, gx, gy, gz, ux, uy, uz, s2):
        dx = px - gx
        dy = py - gy
        dz = pz - gz
        dot = dx * ux + dy * uy + dz * uz
        d2 = dx * dx + dy * dy + dz * dz
        return dot * abs(dot) >= s2 * d2

    _cache[key] = _visible
    return _visible


def _make_kernels(dtype):
    key = np.dtype(dtype).name
    if key in _cache:
        return _cache[key]
    flt = float32 if key == 'float32' else float64
    _visible = _visible_fn(dtype)

    @cuda.jit(fastmath=True)
    def visibility_kernel(pos, g_pos, g_unit, s2, visibility):
        t_idx, p_idx = cuda.grid(2)
        if p_idx >= g_pos.shape[0] or t_idx >= pos.shape[1]:
            return
        gx = g_pos[p_idx, 0]; gy = g_pos[p_idx, 1]; gz = g_pos[p_idx, 2]
        ux = g_unit[p_idx, 0]; uy = g_unit[p_idx, 1]; uz = g_unit[p_idx, 2]
        vis = np.uint8(0)
        for s in range(pos.shape[0]):
            if _visible(pos[s, t_idx, 0], pos[s, t_idx, 1], pos[s, t_idx, 2],
                        gx, gy, gz, ux, uy, uz, s2):
                vis = np.uint8(1)
                break
        visibility[p_idx, t_idx] = vis

    @cuda.jit(fastmath=True)
    def gap_kernel(pos, g_pos, g_unit, s2, best_out, count_out):
        """
        One block per ground point. Each thread scans a contiguous slice of the
        timeline, producing (leading gap, trailing gap, longest interior gap,
        slice length); a tree reduction merges adjacent slices.
        """
        p_idx = cuda.blockIdx.x
        tid = cuda.threadIdx.x
        num_times = pos.shape[1]
        num_sats = pos.shape[0]

        sh_pre = cuda.shared.array(_TPB_GAP, int32)
        sh_suf = cuda.shared.array(_TPB_GAP, int32)
        sh_best = cuda.shared.array(_TPB_GAP, int32)
        sh_len = cuda.shared.array(_TPB_GAP, int32)
        sh_cnt = cuda.shared.array(_TPB_GAP, int32)

        gx = g_pos[p_idx, 0]; gy = g_pos[p_idx, 1]; gz = g_pos[p_idx, 2]
        ux = g_unit[p_idx, 0]; uy = g_unit[p_idx, 1]; uz = g_unit[p_idx, 2]

        per = (num_times + _TPB_GAP - 1) // _TPB_GAP
        t0 = tid * per
        t1 = min(t0 + per, num_times)
        if t1 < t0:
            t1 = t0

        cur = int32(0)
        best = int32(0)
        prefix = int32(0)
        nvis = int32(0)
        seen = False
        for t in range(t0, t1):
            vis = False
            for s in range(num_sats):
                if _visible(pos[s, t, 0], pos[s, t, 1], pos[s, t, 2],
                            gx, gy, gz, ux, uy, uz, s2):
                    vis = True
                    break
            if vis:
                nvis += 1
                if not seen:
                    prefix = cur
                    seen = True
                if cur > best:
                    best = cur
                cur = int32(0)
            else:
                cur += 1
        if cur > best:
            best = cur
        if not seen:
            prefix = cur

        sh_pre[tid] = prefix
        sh_suf[tid] = cur
        sh_best[tid] = best
        sh_len[tid] = t1 - t0
        sh_cnt[tid] = nvis
        cuda.syncthreads()

        # merge adjacent slices: stride doubling keeps the segments in order,
        # which a standard halving reduction would not (the merge is not
        # commutative).
        stride = 1
        while stride < _TPB_GAP:
            idx = tid * stride * 2
            if idx + stride < _TPB_GAP:
                a = idx
                b = idx + stride
                nb = sh_best[a]
                if sh_best[b] > nb:
                    nb = sh_best[b]
                joined = sh_suf[a] + sh_pre[b]
                if joined > nb:
                    nb = joined
                npre = sh_pre[a] + (sh_pre[b] if sh_pre[a] == sh_len[a] else int32(0))
                nsuf = sh_suf[b] + (sh_suf[a] if sh_suf[b] == sh_len[b] else int32(0))
                sh_best[a] = nb
                sh_pre[a] = npre
                sh_suf[a] = nsuf
                sh_len[a] = sh_len[a] + sh_len[b]
                sh_cnt[a] = sh_cnt[a] + sh_cnt[b]
            cuda.syncthreads()
            stride *= 2

        if tid == 0:
            best_out[p_idx] = sh_best[0]
            count_out[p_idx] = sh_cnt[0]

    _cache[key] = (visibility_kernel, gap_kernel)
    return _cache[key]


def _ground_arrays(ground_points, dtype):
    g = np.ascontiguousarray(ground_points, dtype=dtype)
    if g.ndim != 2 or g.shape[1] != 3:
        raise ValueError("ground_points must have shape (P, 3) in ECEF metres")
    mag = np.linalg.norm(g.astype(np.float64), axis=1, keepdims=True)
    if not np.all(mag > 0):
        raise ValueError("ground_points must be non-zero ECEF vectors")
    return cuda.to_device(g), cuda.to_device((g / mag).astype(dtype))


def _as_device(positions):
    if cuda.is_cuda_array(positions):
        return cuda.as_cuda_array(positions)
    return cuda.to_device(np.ascontiguousarray(positions))


def visibility_cuda(positions, ground_points, min_elevation, out='host'):
    """
    Boolean visibility, shape (P, num_times). Same predicate as
    visibility.calculate_visibility_cuda but without the transcendentals, and
    with a uint8 output instead of int32.

    positions may be a numpy array or a device array (from Propagator with
    out='device'), in which case nothing is copied to the GPU.
    """
    d_pos = _as_device(positions)
    dtype = d_pos.dtype.type
    vis_k, _ = _make_kernels(dtype)
    d_g, d_u = _ground_arrays(ground_points, dtype)
    P, T = d_g.shape[0], d_pos.shape[1]
    d_vis = cuda.device_array((P, T), np.uint8)
    s = math.sin(min_elevation)
    tpb = (64, 4)
    vis_k[((T + tpb[0] - 1) // tpb[0], (P + tpb[1] - 1) // tpb[1]), tpb](
        d_pos, d_g, d_u, dtype(s * abs(s)), d_vis)
    if out == 'device':
        return d_vis
    return d_vis.copy_to_host().astype(bool)


def coverage_max_gaps_cuda(positions, ground_points, min_elevation, time_step,
                           return_counts=False):
    """
    Longest coverage gap per ground point, in seconds.

    Equivalent to calculate_max_gaps(calculate_visibility_cuda(...), time_step)
    but the P x T visibility array is never built: each block scans one ground
    point's timeline and reduces it to a single number.

    return_counts=True also gives the number of visible timesteps per point.
    """
    d_pos = _as_device(positions)
    dtype = d_pos.dtype.type
    _, gap_k = _make_kernels(dtype)
    d_g, d_u = _ground_arrays(ground_points, dtype)
    P = d_g.shape[0]
    d_best = cuda.device_array(P, np.int32)
    d_cnt = cuda.device_array(P, np.int32)
    s = math.sin(min_elevation)
    gap_k[P, _TPB_GAP](d_pos, d_g, d_u, dtype(s * abs(s)), d_best, d_cnt)
    gaps = d_best.copy_to_host().astype(np.float64) * time_step
    if return_counts:
        return gaps, d_cnt.copy_to_host()
    return gaps


def coverage_report(satellite_elements, times, ground_points, min_elevation,
                    dtype=np.float32, propagator=None):
    """
    Whole coverage pipeline with nothing but the final statistics crossing PCIe.

    Returns (max_gaps_seconds, visible_fraction), both shape (P,).
    """
    from .prop_cuda_fast import Propagator

    times = np.asarray(times, dtype=np.float64)
    p = propagator or Propagator(len(satellite_elements), len(times), dtype=dtype)
    d_pos = p.run(satellite_elements, times, return_frame='ecef', out='device')
    step = float(times[1] - times[0]) if len(times) > 1 else 0.0
    gaps, counts = coverage_max_gaps_cuda(d_pos, ground_points, min_elevation,
                                          step, return_counts=True)
    return gaps, counts / float(len(times))


# ---------------------------------------------------------------------------
# Spherical-cap culling
#
# The brute force kernels above are O(P*T*S): every ground point is tested
# against every satellite at every step. But a satellite at radius r is only
# visible, at elevation >= el, from ground points within a central angle
#
#     lambda_max = arccos( (Rg / r) * cos(el) ) - el
#
# of its sub-satellite point. For a 550 km LEO at 10 degrees that cap is ~15
# degrees, about 1.7% of the sphere. Binning the ground points by latitude and
# longitude lets each satellite touch only the cells its cap overlaps.
#
# The cap is computed with the SMALLEST ground radius present, which makes
# lambda_max the largest and the cull conservative: it can never drop a point
# that the exact test would have accepted. The exact test still runs on every
# surviving point, so results are identical to the brute force kernels.
# ---------------------------------------------------------------------------

class GroundIndex:
    """Lat/lon bin index over a fixed set of ground points, built once."""

    def __init__(self, ground_points, n_lat=36, n_lon=72, dtype=np.float32):
        # Cast first, then derive magnitudes from the cast values, exactly as
        # _ground_arrays does. Deriving the unit vectors from the float64 input
        # instead differs in the last float32 ulp, which flips cells that sit
        # within ~1e-5 degrees of the elevation threshold.
        g = np.ascontiguousarray(ground_points, dtype=dtype)
        if g.ndim != 2 or g.shape[1] != 3:
            raise ValueError("ground_points must have shape (P, 3) in ECEF metres")
        radius = np.linalg.norm(g.astype(np.float64), axis=1)
        if not np.all(radius > 0):
            raise ValueError("ground_points must be non-zero ECEF vectors")
        unit = (g / radius[:, None]).astype(dtype)

        self.n_lat = int(n_lat)
        self.n_lon = int(n_lon)
        self.num_points = g.shape[0]
        self.min_radius = float(radius.min())

        lat = np.arcsin(np.clip(g[:, 2].astype(np.float64) / radius, -1.0, 1.0))
        lon = np.arctan2(g[:, 1].astype(np.float64), g[:, 0].astype(np.float64))
        lat_bin = np.clip(((lat + math.pi / 2) / math.pi * self.n_lat).astype(np.int64),
                          0, self.n_lat - 1)
        lon_bin = np.clip(((lon + math.pi) / (2 * math.pi) * self.n_lon).astype(np.int64),
                          0, self.n_lon - 1)

        cell = lat_bin * self.n_lon + lon_bin
        order = np.argsort(cell, kind='stable')
        counts = np.bincount(cell, minlength=self.n_lat * self.n_lon)
        starts = np.concatenate([[0], np.cumsum(counts)]).astype(np.int32)

        self.order = order
        self.d_start = cuda.to_device(starts)
        self.d_pos = cuda.to_device(np.ascontiguousarray(g[order]))
        self.d_unit = cuda.to_device(np.ascontiguousarray(unit[order]))
        self.d_index = cuda.to_device(np.ascontiguousarray(order, dtype=np.int32))
        self.dtype = dtype


def _make_culled_kernels(dtype):
    key = ('culled', np.dtype(dtype).name)
    if key in _cache:
        return _cache[key]
    flt = float32 if np.dtype(dtype).name == 'float32' else float64
    _visible = _visible_fn(dtype)

    @cuda.jit
    def fill_zero(arr):
        i = cuda.grid(1)
        if i < arr.size:
            arr[i] = 0

    @cuda.jit(fastmath=True)
    def scatter_kernel(pos, g_pos, g_unit, g_start, g_index, s2, sin_el,
                       min_radius, n_lat, n_lon, visibility):
        t_idx, s_idx = cuda.grid(2)
        if s_idx >= pos.shape[0] or t_idx >= pos.shape[1]:
            return

        px = pos[s_idx, t_idx, 0]
        py = pos[s_idx, t_idx, 1]
        pz = pos[s_idx, t_idx, 2]
        r = math.sqrt(px * px + py * py + pz * pz)
        if r <= min_radius:
            return

        # cap half-angle for the most favourable (smallest) ground radius
        c = (min_radius / r) * math.sqrt(max(flt(0.0), flt(1.0) - sin_el * sin_el))
        if c > flt(1.0):
            return                                   # nothing can see it
        lam = math.acos(c) - math.asin(sin_el)
        if lam <= flt(0.0):
            return

        sat_lat = math.asin(max(flt(-1.0), min(flt(1.0), pz / r)))
        sat_lon = math.atan2(py, px)

        b_lo = int(math.floor((sat_lat - lam + flt(math.pi / 2))
                              / flt(math.pi) * n_lat)) - 1
        b_hi = int(math.floor((sat_lat + lam + flt(math.pi / 2))
                              / flt(math.pi) * n_lat)) + 1
        if b_lo < 0:
            b_lo = 0
        if b_hi > n_lat - 1:
            b_hi = n_lat - 1

        cos_lam = math.cos(lam)
        sin_slat = math.sin(sat_lat)
        cos_slat = math.cos(sat_lat)
        # A cap wider than a hemisphere reaches every longitude somewhere; the
        # interior-extremum argument below also assumes cos(lambda) > 0.
        wide = cos_lam <= flt(1e-6) or cos_slat <= flt(1e-6)

        for lb in range(b_lo, b_hi + 1):
            lat_a = (flt(lb) / n_lat) * flt(math.pi) - flt(math.pi / 2)
            lat_b = (flt(lb + 1) / n_lat) * flt(math.pi) - flt(math.pi / 2)

            # Widest |dlon| over this latitude band. cos(dlon) = v(latp) with
            #   v = (cos L - sin(phi_s) sin(latp)) / (cos(phi_s) cos(latp))
            # and dv/dlatp vanishes at sin(latp) = sin(phi_s)/cos(L), which is
            # an interior minimum of v (so a maximum of dlon). Checking only the
            # band edges under-estimates the span and drops real detections.
            take_all = wide
            best = flt(2.0)
            if not take_all:
                da = cos_slat * math.cos(lat_a)
                db = cos_slat * math.cos(lat_b)
                if da <= flt(1e-9) or db <= flt(1e-9):
                    take_all = True
                else:
                    va = (cos_lam - sin_slat * math.sin(lat_a)) / da
                    vb = (cos_lam - sin_slat * math.sin(lat_b)) / db
                    best = va if va < vb else vb
                    tstar = sin_slat / cos_lam
                    if tstar > flt(-1.0) and tstar < flt(1.0):
                        ps = math.asin(tstar)
                        if ps > lat_a and ps < lat_b:
                            ds = cos_slat * math.cos(ps)
                            if ds <= flt(1e-9):
                                take_all = True
                            else:
                                vs = (cos_lam - sin_slat * math.sin(ps)) / ds
                                if vs < best:
                                    best = vs

            if not take_all and best > flt(1.0):
                continue                             # cap misses this band
            lo_cell = lb * n_lon
            if take_all or best <= flt(-1.0):
                start_lon = 0
                n_cells = n_lon
            else:
                dlon = math.acos(best)
                # floor, not truncation: the argument goes negative near lon=-pi.
                # One bin of margin absorbs float32 rounding at the edges.
                lo = int(math.floor((sat_lon - dlon + flt(math.pi))
                                    / flt(2 * math.pi) * n_lon)) - 1
                hi = int(math.floor((sat_lon + dlon + flt(math.pi))
                                    / flt(2 * math.pi) * n_lon)) + 1
                n_cells = hi - lo + 1
                if n_cells > n_lon:
                    n_cells = n_lon
                start_lon = lo

            for j in range(n_cells):
                lonb = (start_lon + j) % n_lon
                if lonb < 0:
                    lonb += n_lon
                cell = lo_cell + lonb
                for q in range(g_start[cell], g_start[cell + 1]):
                    if _visible(px, py, pz, g_pos[q, 0], g_pos[q, 1], g_pos[q, 2],
                                g_unit[q, 0], g_unit[q, 1], g_unit[q, 2], s2):
                        visibility[g_index[q], t_idx] = np.uint8(1)

    _cache[key] = (fill_zero, scatter_kernel)
    return _cache[key]


def visibility_cuda_culled(positions, index, min_elevation, out='host'):
    """
    Visibility via spherical-cap culling. Identical results to visibility_cuda,
    but each satellite only visits the ground-point cells its cap overlaps.

    index: a GroundIndex built once over the ground points.
    """
    if not isinstance(index, GroundIndex):
        index = GroundIndex(index)
    d_pos = _as_device(positions)
    dtype = d_pos.dtype.type
    fill, scatter = _make_culled_kernels(dtype)
    S, T = d_pos.shape[0], d_pos.shape[1]
    d_vis = cuda.device_array((index.num_points, T), np.uint8)
    flat = d_vis.reshape(index.num_points * T)
    fill[(flat.size + 255) // 256, 256](flat)
    s = math.sin(min_elevation)
    tpb = (64, 4)
    scatter[((T + tpb[0] - 1) // tpb[0], (S + tpb[1] - 1) // tpb[1]), tpb](
        d_pos, index.d_pos, index.d_unit, index.d_start, index.d_index,
        dtype(s * abs(s)), dtype(s), dtype(index.min_radius),
        index.n_lat, index.n_lon, d_vis)
    if out == 'device':
        return d_vis
    return d_vis.copy_to_host().astype(bool)


@cuda.jit
def _gap_from_vis_kernel(vis, best_out, count_out):
    """Longest run of zeros per row of a (P, T) uint8 visibility array."""
    p_idx = cuda.blockIdx.x
    tid = cuda.threadIdx.x
    num_times = vis.shape[1]

    sh_pre = cuda.shared.array(_TPB_GAP, int32)
    sh_suf = cuda.shared.array(_TPB_GAP, int32)
    sh_best = cuda.shared.array(_TPB_GAP, int32)
    sh_len = cuda.shared.array(_TPB_GAP, int32)
    sh_cnt = cuda.shared.array(_TPB_GAP, int32)

    per = (num_times + _TPB_GAP - 1) // _TPB_GAP
    t0 = tid * per
    t1 = min(t0 + per, num_times)
    if t1 < t0:
        t1 = t0

    cur = int32(0)
    best = int32(0)
    prefix = int32(0)
    nvis = int32(0)
    seen = False
    for t in range(t0, t1):
        if vis[p_idx, t] != 0:
            nvis += 1
            if not seen:
                prefix = cur
                seen = True
            if cur > best:
                best = cur
            cur = int32(0)
        else:
            cur += 1
    if cur > best:
        best = cur
    if not seen:
        prefix = cur

    sh_pre[tid] = prefix
    sh_suf[tid] = cur
    sh_best[tid] = best
    sh_len[tid] = t1 - t0
    sh_cnt[tid] = nvis
    cuda.syncthreads()

    stride = 1
    while stride < _TPB_GAP:
        idx = tid * stride * 2
        if idx + stride < _TPB_GAP:
            a = idx
            b = idx + stride
            nb = sh_best[a]
            if sh_best[b] > nb:
                nb = sh_best[b]
            joined = sh_suf[a] + sh_pre[b]
            if joined > nb:
                nb = joined
            npre = sh_pre[a] + (sh_pre[b] if sh_pre[a] == sh_len[a] else int32(0))
            nsuf = sh_suf[b] + (sh_suf[a] if sh_suf[b] == sh_len[b] else int32(0))
            sh_best[a] = nb
            sh_pre[a] = npre
            sh_suf[a] = nsuf
            sh_len[a] = sh_len[a] + sh_len[b]
            sh_cnt[a] = sh_cnt[a] + sh_cnt[b]
        cuda.syncthreads()
        stride *= 2

    if tid == 0:
        best_out[p_idx] = sh_best[0]
        count_out[p_idx] = sh_cnt[0]


def max_gaps_from_visibility(d_visibility, time_step, return_counts=False):
    """
    Longest gap per ground point from an existing (P, T) device visibility
    array, so culled visibility can feed the coverage statistics.
    """
    d_vis = _as_device(d_visibility)
    P = d_vis.shape[0]
    d_best = cuda.device_array(P, np.int32)
    d_cnt = cuda.device_array(P, np.int32)
    _gap_from_vis_kernel[P, _TPB_GAP](d_vis, d_best, d_cnt)
    gaps = d_best.copy_to_host().astype(np.float64) * time_step
    if return_counts:
        return gaps, d_cnt.copy_to_host()
    return gaps


def coverage_max_gaps_culled(positions, index, min_elevation, time_step,
                             return_counts=False):
    """
    Culled equivalent of coverage_max_gaps_cuda. Builds the visibility array on
    the device (never on the host) and reduces it. Faster than the fused kernel
    when the elevation mask is high; see visibility_cuda_culled for the
    measured crossover.
    """
    d_vis = visibility_cuda_culled(positions, index, min_elevation, out='device')
    return max_gaps_from_visibility(d_vis, time_step, return_counts=return_counts)
