"""
Optimised CUDA propagator (drop-in alternative to prop_cuda.propagate_constellation_cuda).

Key differences from prop_cuda.py:
  1. All device math is genuinely single precision (the original promotes every
     expression to float64 because mu/j2/re arrive as Python floats).
  2. The ECI->ECEF rotation is folded into the RAAN angle inside the kernel, so
     the host-side per-timestep rotation pass disappears entirely.
  3. Per-satellite invariants (mean motion, J2 rates, sin/cos of inclination...)
     are computed once on the host instead of once per (satellite, timestep).
  4. Kepler's equation is solved with a Halley iteration seeded by a 2nd-order
     series instead of a 10-node contour integral.
  5. Thread x-index runs along time so warps read satellite data by broadcast
     and write consecutive positions.
  6. Optional satellite-chunked, stream-overlapped, pinned-memory device->host
     transfer, or a device-resident return for downstream GPU work.
"""

import math

import numpy as np
from numba import cuda, float32, float64
from numba.cuda import libdevice

from .constants import MU, J2, RE
from .coord_conv import calculate_gmst_from_seconds

_TWO_PI = 6.283185307179586
_INV_TWO_PI = 1.0 / _TWO_PI

# 2*pi split into two float32 pieces so the range reduction keeps full
# precision even when the accumulated mean anomaly is ~100 rad.
_TWO_PI_HI_F32 = float(np.float32(_TWO_PI))
_TWO_PI_LO_F32 = float(np.float32(_TWO_PI - _TWO_PI_HI_F32))

_kernel_cache = {}


def _make_kernel(dtype):
    """Build the propagation kernel for a given floating point type."""
    key = np.dtype(dtype).name
    if key in _kernel_cache:
        return _kernel_cache[key]

    if key == 'float32':
        flt, fma, rint = float32, libdevice.fmaf, libdevice.rintf
        two_pi_hi, two_pi_lo = _TWO_PI_HI_F32, _TWO_PI_LO_F32
    else:
        flt, fma, rint = float64, libdevice.fma, libdevice.rint
        two_pi_hi, two_pi_lo = _TWO_PI, 0.0

    # fastmath is deliberately OFF here: the fma-based residual is what keeps
    # the mean anomaly accurate, and unsafe algebra would fold it away.
    @cuda.jit(device=True, inline=False, fastmath=False)
    def _phase(M0, n, t):
        p = n * t
        q = fma(n, t, -p)                      # exact low bits of n*t
        k = rint(p * flt(_INV_TWO_PI))
        r = (p - k * flt(two_pi_hi)) - k * flt(two_pi_lo)
        return r + q + M0

    @cuda.jit(device=True, inline=True, fastmath=True)
    def _kepler_sincos(M, e):
        """Return (sin E, cos E) for E - e sin E = M."""
        sM = math.sin(M)
        cM = math.cos(M)
        if e < flt(1e-7):                      # warp-uniform: same sat per warp
            return sM, cM

        E = M + e * sM * (flt(1.0) + e * cM)   # 2nd order series seed
        sE = math.sin(E)
        cE = math.cos(E)
        for _ in range(2):
            f = E - e * sE - M
            fp = flt(1.0) - e * cE
            d = f / fp
            d = f / (fp - flt(0.5) * d * e * sE)   # Halley
            E = E - d
            sE = math.sin(E)
            cE = math.cos(E)
        # one more Halley step, propagated onto sin/cos by first order Taylor
        f = E - e * sE - M
        fp = flt(1.0) - e * cE
        d = f / fp
        d = f / (fp - flt(0.5) * d * e * sE)
        return sE - d * cE, cE + d * sE

    @cuda.jit(fastmath=True)
    def kernel(inv, times, gmst, positions):
        """inv: (num_sats, 12) precomputed invariants. gmst: (num_times,)."""
        t_idx, s_idx0 = cuda.grid(2)
        num_times = times.shape[0]
        num_sats = inv.shape[0]
        stride_s = cuda.gridDim.y * cuda.blockDim.y

        if t_idx >= num_times:
            return

        tk = times[t_idx]
        g = gmst[t_idx]

        for s_idx in range(s_idx0, num_sats, stride_s):
            a = inv[s_idx, 0]
            e = inv[s_idx, 1]
            n_tot = inv[s_idx, 2]
            draan = inv[s_idx, 3]
            dargp = inv[s_idx, 4]
            M0 = inv[s_idx, 5]
            raan0 = inv[s_idx, 6]
            argp0 = inv[s_idx, 7]
            cos_i = inv[s_idx, 8]
            sin_i = inv[s_idx, 9]
            b = inv[s_idx, 10]
            t = tk - inv[s_idx, 11]

            M = _phase(M0, n_tot, t)
            sE, cE = _kepler_sincos(M, e)

            x_orb = a * (cE - e)               # == r*cos(nu), no atan2 needed
            y_orb = b * sE                     # == r*sin(nu)

            # Folding -gmst into RAAN makes the ECEF rotation free.
            raan_t = raan0 + draan * t - g
            w_t = argp0 + dargp * t
            sin_w = math.sin(w_t)
            cos_w = math.cos(w_t)
            sin_r = math.sin(raan_t)
            cos_r = math.cos(raan_t)

            positions[s_idx, t_idx, 0] = ((cos_r * cos_w - sin_r * sin_w * cos_i) * x_orb
                                          + (-cos_r * sin_w - sin_r * cos_w * cos_i) * y_orb)
            positions[s_idx, t_idx, 1] = ((sin_r * cos_w + cos_r * sin_w * cos_i) * x_orb
                                          + (-sin_r * sin_w + cos_r * cos_w * cos_i) * y_orb)
            positions[s_idx, t_idx, 2] = (sin_w * sin_i) * x_orb + (cos_w * sin_i) * y_orb

    _kernel_cache[key] = kernel
    return kernel


def _invariants(elements, epochs, dtype):
    """Per-satellite quantities the original kernel recomputed for every timestep."""
    el = np.asarray(elements, dtype=np.float64)
    a, e, inc, raan, argp, M0 = (el[:, k] for k in range(6))

    n0 = np.sqrt(MU / a ** 3)
    j2_scale = (n0 * RE ** 2 * J2) / (a ** 2 * (1.0 - e ** 2) ** 2)
    sin_i, cos_i = np.sin(inc), np.cos(inc)

    draan = -1.5 * j2_scale * cos_i
    dargp = 0.75 * j2_scale * (4.0 - 5.0 * sin_i ** 2)
    dM = 0.75 * j2_scale * np.sqrt(1.0 - e ** 2) * (2.0 - 3.0 * sin_i ** 2)

    inv = np.empty((el.shape[0], 12), dtype=dtype)
    inv[:, 0] = a
    inv[:, 1] = e
    inv[:, 2] = n0 + dM
    inv[:, 3] = draan
    inv[:, 4] = dargp
    inv[:, 5] = np.mod(M0, _TWO_PI)
    inv[:, 6] = np.mod(raan, _TWO_PI)
    inv[:, 7] = np.mod(argp, _TWO_PI)
    inv[:, 8] = cos_i
    inv[:, 9] = sin_i
    inv[:, 10] = a * np.sqrt(1.0 - e ** 2)
    inv[:, 11] = 0.0 if epochs is None else np.asarray(epochs, dtype=np.float64)
    return inv


def propagate_constellation_cuda_fast(satellite_elements, times, return_frame='ecef',
                                      epochs=None, input_type='kepler', dtype=np.float32,
                                      out='host', sat_chunk=None, threads=(64, 4)):
    """
    Propagate a constellation on the GPU.

    out='host'   -> numpy array (num_sats, num_times, 3)
    out='device' -> numba device array, no transfer (use for GPU post-processing)
    sat_chunk    -> satellites per batch; bounds device memory and overlaps the
                    device->host copy of one batch with the kernel of the next.
    """
    if input_type.lower() != 'kepler':
        raise ValueError("only 'kepler' input_type is supported")

    dtype = np.dtype(dtype).type
    kernel = _make_kernel(dtype)

    num_sats = len(satellite_elements)
    times = np.asarray(times, dtype=np.float64)
    num_times = len(times)

    times_seconds = times - times[0] if epochs is None else times
    gmst = np.array([calculate_gmst_from_seconds(s) for s in times_seconds], dtype=dtype)
    if return_frame.lower() != 'ecef':
        gmst[:] = 0.0

    inv = _invariants(satellite_elements, epochs, dtype)
    d_times = cuda.to_device(np.asarray(times_seconds, dtype=dtype))
    d_gmst = cuda.to_device(gmst)

    tpb = (int(threads[0]), int(threads[1]))
    grid_x = (num_times + tpb[0] - 1) // tpb[0]

    if out == 'device' or sat_chunk is None:
        d_inv = cuda.to_device(inv)
        d_pos = cuda.device_array((num_sats, num_times, 3), dtype=dtype)
        grid_y = min(65535, (num_sats + tpb[1] - 1) // tpb[1])
        kernel[(grid_x, grid_y), tpb](d_inv, d_times, d_gmst, d_pos)
        if out == 'device':
            return d_pos
        return d_pos.copy_to_host()

    # Chunked + double buffered: kernel of chunk i+1 overlaps the copy of chunk i.
    positions = cuda.pinned_array((num_sats, num_times, 3), dtype=dtype)
    streams = [cuda.stream(), cuda.stream()]
    bufs = [None, None]
    for j, s_start in enumerate(range(0, num_sats, sat_chunk)):
        s_end = min(s_start + sat_chunk, num_sats)
        n = s_end - s_start
        st = streams[j % 2]
        d_inv = cuda.to_device(inv[s_start:s_end], stream=st)
        if bufs[j % 2] is None or bufs[j % 2].shape[0] != n:
            bufs[j % 2] = cuda.device_array((n, num_times, 3), dtype=dtype, stream=st)
        d_pos = bufs[j % 2]
        grid_y = min(65535, (n + tpb[1] - 1) // tpb[1])
        kernel[(grid_x, grid_y), tpb, st](d_inv, d_times, d_gmst, d_pos)
        d_pos.copy_to_host(positions[s_start:s_end], stream=st)
    cuda.synchronize()
    return np.asarray(positions)
