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


def _make_kernels(dtype):
    key = np.dtype(dtype).name
    if key in _cache:
        return _cache[key]
    flt = float32 if key == 'float32' else float64

    @cuda.jit(device=True, inline=True)
    def _visible(px, py, pz, gx, gy, gz, ux, uy, uz, s2):
        dx = px - gx
        dy = py - gy
        dz = pz - gz
        dot = dx * ux + dy * uy + dz * uz
        d2 = dx * dx + dy * dy + dz * dz
        return dot * abs(dot) >= s2 * d2

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
