"""
Optimised CUDA propagator.

This is the implementation behind ``backend='cuda'``; the original kernel is
kept as ``backend='cuda_legacy'`` (prop_cuda.py) for comparison.

Design notes, each driven by a measurement (see bench/profile_cuda.py):

  * All device math is genuinely single precision. The original promotes every
    expression to float64 -- not because of the mu/j2/re arguments alone, but
    because every untyped literal (3/2, 0.5, 2.0*pi, 1e-14) is a Python float
    and numba promotes on contact. Passing np.float32 scalars is NOT enough;
    every constant has to be built from the target type.
  * The ECI->ECEF rotation is folded into RAAN: Rz(-g)Rz(raan) == Rz(raan-g).
    That deletes the host-side per-timestep rotation pass entirely.
  * Per-satellite invariants are computed once by a small float64 prep kernel
    rather than once per (satellite, timestep) in float32.
  * Kepler's equation uses a series-seeded Halley iteration, and the true
    anomaly is skipped: x = a(cosE - e), y = a*sqrt(1-e^2)*sinE is exact.
  * The mean motion is carried as a hi/lo float32 pair and the mean anomaly is
    range-reduced with a Cody-Waite split. The kernel runs at ~100% of the
    achievable pure-store rate, so this extra arithmetic is free, and it buys
    ~3 orders of magnitude of long-arc accuracy (30-day max error 22.9 km -> 27 m).
  * Thread x-index runs along time, so satellite data is read by warp broadcast
    and stores are near-contiguous.
  * Propagator reuses its device buffers. Allocating a fresh 3 GB device buffer
    costs ~77 ms per call on WDDM, 13x the ~6 ms kernel.
  * What is left is the device->host copy. Into a freshly allocated numpy array
    most of that is first-touch page faulting, not PCIe; a reused pinned buffer
    cuts 500k x 500 from ~850 ms to ~145 ms.
"""

import math

import numpy as np
from numba import cuda, float32, float64
from numba.cuda import libdevice

from .constants import MU, J2, RE
from .coord_conv import gmst_from_seconds

_TWO_PI = 6.283185307179586
_INV_TWO_PI = 1.0 / _TWO_PI

# 2*pi split into two float32 pieces so the range reduction stays accurate
# even when the accumulated mean anomaly reaches hundreds of radians.
_TWO_PI_HI_F32 = float(np.float32(_TWO_PI))
_TWO_PI_LO_F32 = float(np.float32(_TWO_PI - _TWO_PI_HI_F32))

# inv columns
_A, _E, _NHI, _NLO, _DRAAN, _DARGP, _M0, _RAAN0, _ARGP0, _COSI, _SINI, _B = range(12)
_NINV = 12

_kernel_cache = {}


# gmst_from_seconds lives in coord_conv (no CUDA dependency); re-exported here
# because this module is where callers of the fast path look for it.


def _make_kernels(dtype, fastmath=True):
    """
    Build (prep, propagate) kernels for a given floating point type.

    fastmath=False swaps the approximate sin/cos (sin.approx.f32, ~2^-20
    relative) for the accurate libdevice ones. Measured at N=500k T=500:
    error 4.0 -> 1.8 m mean, 23 -> 12 m max, but the kernel goes 4.7 -> 14.6 ms
    because the approximate trig is precisely what lets it reach the store
    bandwidth ceiling. Worth it when results are copied back to the host (the
    extra 10 ms hides behind a ~145 ms transfer); not worth it for
    device-resident work, where it is a straight 3x on the only cost there is.
    """
    key = (np.dtype(dtype).name, bool(fastmath))
    if key in _kernel_cache:
        return _kernel_cache[key]

    if key[0] == 'float32':
        flt, fma, rint = float32, libdevice.fmaf, libdevice.rintf
        two_pi_hi, two_pi_lo = _TWO_PI_HI_F32, _TWO_PI_LO_F32
        split = True
    elif key[0] == 'float64':
        flt, fma, rint = float64, libdevice.fma, libdevice.rint
        two_pi_hi, two_pi_lo = _TWO_PI, 0.0
        split = False
    else:
        raise ValueError("dtype must be float32 or float64")

    # Runs in float64 regardless of the output type: these are per-satellite
    # constants whose error is amplified by the full propagation arc. The
    # satellite epoch is absorbed into the angles here, so the kernel never
    # subtracts two large times in float32 (that cost ~600 m of accuracy).
    @cuda.jit
    def prep(elements, epochs, inv):
        i = cuda.grid(1)
        if i >= elements.shape[0]:
            return
        a = elements[i, 0]
        e = elements[i, 1]
        inc = elements[i, 2]

        n0 = math.sqrt(MU / (a * a * a))
        om = 1.0 - e * e
        j2s = (n0 * RE * RE * J2) / (a * a * om * om)
        si = math.sin(inc)
        ci = math.cos(inc)
        n_tot = n0 + 0.75 * j2s * math.sqrt(om) * (2.0 - 3.0 * si * si)
        draan = -1.5 * j2s * ci
        dargp = 0.75 * j2s * (4.0 - 5.0 * si * si)

        # angles rewound to the common time origin: theta(t) = theta0' + rate*t
        ep = epochs[i]
        M0 = (elements[i, 5] - n_tot * ep) % _TWO_PI
        raan0 = (elements[i, 3] - draan * ep) % _TWO_PI
        argp0 = (elements[i, 4] - dargp * ep) % _TWO_PI

        inv[i, _A] = a
        inv[i, _E] = e
        if split:
            nhi = float32(n_tot)
            inv[i, _NHI] = nhi
            inv[i, _NLO] = float32(n_tot - nhi)
        else:
            inv[i, _NHI] = n_tot
            inv[i, _NLO] = 0.0
        inv[i, _DRAAN] = draan
        inv[i, _DARGP] = dargp
        inv[i, _M0] = M0
        inv[i, _RAAN0] = raan0
        inv[i, _ARGP0] = argp0
        inv[i, _COSI] = ci
        inv[i, _SINI] = si
        inv[i, _B] = a * math.sqrt(om)

    # fastmath is deliberately OFF here: the fma residual is what keeps the mean
    # anomaly accurate, and unsafe algebra would fold it away.
    @cuda.jit(device=True, inline=False, fastmath=False)
    def _phase(M0, n_hi, n_lo, t):
        p = n_hi * t
        q = fma(n_hi, t, -p) + n_lo * t        # exact low bits + lo term
        k = rint(p * flt(_INV_TWO_PI))
        r = (p - k * flt(two_pi_hi)) - k * flt(two_pi_lo)
        return r + q + M0

    @cuda.jit(device=True, inline=True, fastmath=fastmath)
    def _kepler_sincos(M, e):
        """Return (sin E, cos E) solving E - e sin E = M."""
        sM = math.sin(M)
        cM = math.cos(M)
        if e < flt(1e-7):                      # warp-uniform: one sat per warp
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
        # final Halley step folded onto sin/cos by first order Taylor
        f = E - e * sE - M
        fp = flt(1.0) - e * cE
        d = f / fp
        d = f / (fp - flt(0.5) * d * e * sE)
        return sE - d * cE, cE + d * sE

    @cuda.jit(fastmath=fastmath)
    def propagate(inv, times, gmst, positions):
        t_idx, s_idx0 = cuda.grid(2)
        if t_idx >= times.shape[0]:
            return
        num_sats = inv.shape[0]
        stride_s = cuda.gridDim.y * cuda.blockDim.y

        tk = times[t_idx]
        g = gmst[t_idx]

        for s_idx in range(s_idx0, num_sats, stride_s):
            a = inv[s_idx, _A]
            e = inv[s_idx, _E]
            t = tk

            M = _phase(inv[s_idx, _M0], inv[s_idx, _NHI], inv[s_idx, _NLO], t)
            sE, cE = _kepler_sincos(M, e)

            x_orb = a * (cE - e)                       # r*cos(nu), no atan2
            y_orb = inv[s_idx, _B] * sE                # r*sin(nu)

            # Folding -gmst into RAAN makes the ECEF rotation free.
            raan_t = inv[s_idx, _RAAN0] + inv[s_idx, _DRAAN] * t - g
            w_t = inv[s_idx, _ARGP0] + inv[s_idx, _DARGP] * t
            cos_i = inv[s_idx, _COSI]
            sin_i = inv[s_idx, _SINI]
            sin_w = math.sin(w_t)
            cos_w = math.cos(w_t)
            sin_r = math.sin(raan_t)
            cos_r = math.cos(raan_t)

            positions[s_idx, t_idx, 0] = ((cos_r * cos_w - sin_r * sin_w * cos_i) * x_orb
                                          + (-cos_r * sin_w - sin_r * cos_w * cos_i) * y_orb)
            positions[s_idx, t_idx, 1] = ((sin_r * cos_w + cos_r * sin_w * cos_i) * x_orb
                                          + (-sin_r * sin_w + cos_r * cos_w * cos_i) * y_orb)
            positions[s_idx, t_idx, 2] = (sin_w * sin_i) * x_orb + (cos_w * sin_i) * y_orb

    _kernel_cache[key] = (prep, propagate)
    return prep, propagate


class Propagator:
    """
    Reusable propagator. Holds its device buffers, so repeated calls of the
    same shape avoid the (large) cost of allocating the output buffer.

        prop = Propagator(num_sats, num_times)
        for epoch in epochs:
            pos = prop.run(elements, times, out='device')   # stays on the GPU

    ``pinned=True`` also allocates a page-locked host buffer; ``out='pinned'``
    then returns a numpy view of it at full PCIe rate. That buffer is reused,
    so each call overwrites the array returned by the previous one.
    """

    def __init__(self, num_sats, num_times, dtype=np.float32, pinned=False,
                 threads=(64, 4), fastmath=True):
        self.dtype = np.dtype(dtype).type
        self.num_sats = int(num_sats)
        self.num_times = int(num_times)
        self.threads = (int(threads[0]), int(threads[1]))
        self._prep, self._prop = _make_kernels(self.dtype, fastmath)

        self.d_elements = cuda.device_array((self.num_sats, 6), np.float64)
        self.d_epochs = cuda.device_array(self.num_sats, np.float64)
        self.d_inv = cuda.device_array((self.num_sats, _NINV), self.dtype)
        self.d_times = cuda.device_array(self.num_times, self.dtype)
        self.d_gmst = cuda.device_array(self.num_times, self.dtype)
        self.d_pos = cuda.device_array((self.num_sats, self.num_times, 3), self.dtype)
        self.h_pinned = (cuda.pinned_array((self.num_sats, self.num_times, 3), self.dtype)
                         if pinned else None)
        if self.h_pinned is not None:
            self.h_pinned[:] = 0        # fault the pages in once, not per call

    def run(self, satellite_elements, times, return_frame='ecef', epochs=None,
            input_type='kepler', out=None):
        """
        out=None       -> new numpy array
        out='device'   -> the internal device array (no transfer)
        out='pinned'   -> numpy view of the internal pinned buffer (reused!)
        out=ndarray    -> written in place
        """
        if input_type.lower() != 'kepler':
            raise ValueError("only 'kepler' input_type is supported")

        el = np.ascontiguousarray(satellite_elements, dtype=np.float64)
        if el.shape != (self.num_sats, 6):
            raise ValueError(f"expected elements of shape {(self.num_sats, 6)}, got {el.shape}")
        times = np.asarray(times, dtype=np.float64)
        if len(times) != self.num_times:
            raise ValueError(f"expected {self.num_times} times, got {len(times)}")

        # Propagate on a time axis relative to times[0] and rewind the epochs to
        # the same origin, so no large absolute time is ever stored in float32.
        # GMST still uses the absolute times when epochs are supplied.
        t_ref = times[0]
        times_rel = times - t_ref
        ep = (np.zeros(self.num_sats, np.float64) if epochs is None
              else np.ascontiguousarray(epochs, dtype=np.float64) - t_ref)
        gmst_src = times_rel if epochs is None else times
        gmst = (gmst_from_seconds(gmst_src) if return_frame.lower() == 'ecef'
                else np.zeros(self.num_times))

        self.d_elements.copy_to_device(el)
        self.d_epochs.copy_to_device(ep)
        self.d_times.copy_to_device(times_rel.astype(self.dtype))
        self.d_gmst.copy_to_device(gmst.astype(self.dtype))

        self._prep[(self.num_sats + 127) // 128, 128](
            self.d_elements, self.d_epochs, self.d_inv)

        tpb = self.threads
        grid = ((self.num_times + tpb[0] - 1) // tpb[0],
                min(65535, (self.num_sats + tpb[1] - 1) // tpb[1]))
        self._prop[grid, tpb](self.d_inv, self.d_times, self.d_gmst, self.d_pos)

        if out is None:
            return self.d_pos.copy_to_host()
        if isinstance(out, str):
            if out == 'device':
                return self.d_pos
            if out == 'pinned':
                if self.h_pinned is None:
                    raise ValueError("Propagator was created with pinned=False")
                self.d_pos.copy_to_host(self.h_pinned)
                return np.asarray(self.h_pinned)
            raise ValueError(f"unknown out mode: {out!r}")
        self.d_pos.copy_to_host(out)
        return out


def propagate_constellation_cuda_fast(satellite_elements, times, return_frame='ecef',
                                      epochs=None, input_type='kepler',
                                      dtype=np.float32, out=None, threads=(64, 4),
                                      fastmath=True):
    """One-shot propagation. For repeated calls use Propagator, which reuses buffers."""
    p = Propagator(len(satellite_elements), len(times), dtype=dtype,
                   pinned=(out == 'pinned'), threads=threads, fastmath=fastmath)
    return p.run(satellite_elements, times, return_frame=return_frame, epochs=epochs,
                 input_type=input_type, out=out)
