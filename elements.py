"""
Shared orbital-element preprocessing.

Every backend needs the same per-satellite constants (mean motion, J2 secular
rates, sin/cos of inclination...). These are computed once, in float64, because
their error is amplified by the whole propagation arc: rounding the semi-major
axis to float32 before forming n0 = sqrt(mu/a^3) is worth ~70 m/day on its own.

The CUDA backend runs the identical formulas in a float64 device kernel
(prop_cuda_fast._make_kernels.prep); bench/profile_cuda.py checks the two agree.
"""

import numpy as np

from .constants import MU, J2, RE

TWO_PI = 6.283185307179586


def orbital_invariants(satellite_elements, epochs=None, t_ref=0.0):
    """
    Per-satellite constants for the analytic J2 model, all float64.

    Args:
        satellite_elements: (N, 6) Keplerian elements [a, e, i, raan, argp, M0]
        epochs: optional (N,) epoch seconds per satellite
        t_ref: time origin the propagation axis is measured from

    Returns:
        dict of (N,) float64 arrays. The angles are rewound to t_ref, so a
        propagator only needs theta(t) = theta0 + rate * (t - t_ref) and never
        subtracts two large absolute times in reduced precision.
    """
    el = np.ascontiguousarray(satellite_elements, dtype=np.float64)
    if el.ndim != 2 or el.shape[1] != 6:
        raise ValueError(f"expected elements of shape (N, 6), got {el.shape}")
    a, e, inc, raan, argp, M0 = (el[:, k] for k in range(6))
    if np.any(a <= 0):
        raise ValueError("semi-major axis must be positive")
    if np.any((e < 0) | (e >= 1)):
        raise ValueError("eccentricity must be in [0, 1)")

    one_minus_e2 = 1.0 - e * e
    n0 = np.sqrt(MU / a ** 3)
    j2s = (n0 * RE ** 2 * J2) / (a ** 2 * one_minus_e2 ** 2)
    sin_i = np.sin(inc)
    cos_i = np.cos(inc)

    n_tot = n0 + 0.75 * j2s * np.sqrt(one_minus_e2) * (2.0 - 3.0 * sin_i ** 2)
    draan = -1.5 * j2s * cos_i
    dargp = 0.75 * j2s * (4.0 - 5.0 * sin_i ** 2)

    ep = np.zeros_like(a) if epochs is None else np.asarray(epochs, np.float64) - t_ref

    return {
        'a': a,
        'e': e,
        'n_tot': n_tot,
        'draan': draan,
        'dargp': dargp,
        'M0': np.mod(M0 - n_tot * ep, TWO_PI),
        'raan0': np.mod(raan - draan * ep, TWO_PI),
        'argp0': np.mod(argp - dargp * ep, TWO_PI),
        'cos_i': cos_i,
        'sin_i': sin_i,
        'b': a * np.sqrt(one_minus_e2),
    }


def split_float32(x):
    """Split a float64 array into a float32 hi part and its float32 remainder."""
    hi = np.asarray(x, np.float64).astype(np.float32)
    lo = (np.asarray(x, np.float64) - hi.astype(np.float64)).astype(np.float32)
    return hi, lo
