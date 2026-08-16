"""
Coordinate conversion utilities for the HyperGator package.

This module provides functions for converting between different coordinate frames:
- Earth-Centered Inertial (ECI)
- Earth-Centered Earth-Fixed (ECEF)
- Latitude/Longitude/Altitude (LLA)
- Ground station coordinates

It also includes utilities for calculating Greenwich Mean Sidereal Time (GMST).
"""

import numpy as np
import math
from datetime import datetime, timedelta
from numba import njit, vectorize, prange, jit
from .constants import RE, EARTH_ROTATION_RATE

def calculate_gmst(epoch):
    """
    Calculate Greenwich Mean Sidereal Time (GMST) for a given epoch.

    Args:
        epoch: datetime object representing the epoch (UTC/TT)

    Returns:
        GMST in radians, normalized to [0, 2π]
    """
    # 1) Compute Julian centuries since J2000.0
    j2000 = datetime(2000, 1, 1, 12, 0, 0)  # J2000.0 is 2000-01-01 12:00 TT
    delta = epoch - j2000
    days_since_j2000 = delta.total_seconds() / 86400.0
    T = days_since_j2000 / 36525.0

    # 2) GMST at 0h UT (in degrees)
    gmst0 = (
        100.46061837
        + 36000.770053608 * T
        + 0.000387933 * T**2
        - (T**3) / 38710000.0
    )

    # 3) Time-of-day fraction (seconds since last 00:00 UTC)
    midnight = epoch.replace(hour=0, minute=0, second=0, microsecond=0)
    tod_seconds = (epoch - midnight).total_seconds()

    # 4) Add true sidereal rotation since 0h UT:
    #    Earth rotates ~360.98564736629° per mean solar day
    gmst_deg = gmst0 + 360.98564736629 * (tod_seconds / 86400.0)

    gmst_rad = (gmst_deg % 360.0) * (math.pi / 180.0)
    return gmst_rad


from numba import njit

@njit
def calculate_gmst_from_seconds(seconds_since_j2000):
    """
    Calculate GMST from seconds since J2000.0 (2000-01-01 12:00 TT).

    Args:
        seconds_since_j2000: float32/float64 seconds since J2000.0 epoch

    Returns:
        GMST in radians, normalized to [0, 2π]
    """
    # 1) Days and centuries since J2000.0
    days = seconds_since_j2000 / 86400.0
    T = days / 36525.0

    # 2) GMST at 0h UT in degrees
    gmst0 = (
        100.46061837
        + 36000.770053608 * T
        + 0.000387933 * T * T
        - (T**3) / 38710000.0
    )

    # 3) Seconds past last 0h UT
    tod = seconds_since_j2000 % 86400.0

    # 4) Add sidereal rotation
    gmst_deg = gmst0 + 360.98564736629 * (tod / 86400.0)

    # 5) Normalize and convert to radians
    gmst_rad = (gmst_deg % 360.0) * (math.pi / 180.0)
    return gmst_rad

def gmst_from_seconds(seconds):
    """
    Vectorised GMST in radians for an array of seconds since J2000.0.

    Same formula as calculate_gmst_from_seconds (verified bit-identical), but
    evaluated over a whole array at once so callers do not loop in python.
    """
    s = np.asarray(seconds, dtype=np.float64)
    T = (s / 86400.0) / 36525.0
    gmst0 = (100.46061837 + 36000.770053608 * T + 0.000387933 * T * T
             - (T ** 3) / 38710000.0)
    gmst_deg = gmst0 + 360.98564736629 * (np.mod(s, 86400.0) / 86400.0)
    return np.mod(gmst_deg, 360.0) * (math.pi / 180.0)


@njit
def eci_to_ecef(position_eci, gmst):
    """
    Convert ECI coordinates to ECEF coordinates using GMST.
    
    Args:
        position_eci: Position vector [x, y, z] in ECI frame
        gmst: Greenwich Mean Sidereal Time in radians
        
    Returns:
        Position vector [x, y, z] in ECEF frame
    """
    x_eci, y_eci, z_eci = position_eci
    
    cos_gmst = math.cos(gmst)
    sin_gmst = math.sin(gmst)
    
    x_ecef = x_eci * cos_gmst + y_eci * sin_gmst
    y_ecef = -x_eci * sin_gmst + y_eci * cos_gmst
    z_ecef = z_eci
    
    return np.array([x_ecef, y_ecef, z_ecef])

@njit
def ecef_to_eci(position_ecef, gmst):
    """
    Convert ECEF coordinates to ECI coordinates using GMST.
    
    Args:
        position_ecef: Position vector [x, y, z] in ECEF frame
        gmst: Greenwich Mean Sidereal Time in radians
        
    Returns:
        Position vector [x, y, z] in ECI frame
    """
    x_ecef, y_ecef, z_ecef = position_ecef
    
    cos_gmst = math.cos(gmst)
    sin_gmst = math.sin(gmst)
    
    x_eci = x_ecef * cos_gmst - y_ecef * sin_gmst
    y_eci = x_ecef * sin_gmst + y_ecef * cos_gmst
    z_eci = z_ecef
    
    return np.array([x_eci, y_eci, z_eci])

_WGS84_F = 1 / 298.257223563
_WGS84_B = RE * (1 - _WGS84_F)
_WGS84_E2 = 1 - (_WGS84_B * _WGS84_B) / (RE * RE)


@njit(parallel=True, cache=True)
def _ecef_to_lla_kernel(pos, out):
    """pos: (M, 3) ECEF metres -> out: (M, 3) of (lat_deg, lon_deg, alt_m)."""
    a = RE
    e2 = _WGS84_E2
    for i in prange(pos.shape[0]):
        x = pos[i, 0]
        y = pos[i, 1]
        z = pos[i, 2]

        lon = math.atan2(y, x)
        p = math.sqrt(x * x + y * y)
        lat = math.atan2(z, p * (1 - e2))

        h = 0.0
        for _ in range(5):
            sin_lat = math.sin(lat)
            cos_lat = math.cos(lat)
            N = a / math.sqrt(1 - e2 * sin_lat * sin_lat)
            # p/cos(lat) blows up on the polar axis; z/sin(lat) blows up on the
            # equator. Use whichever denominator is larger.
            if p > abs(z):
                h = p / cos_lat - N
            else:
                h = z / sin_lat - N * (1 - e2)
            lat = math.atan2(z, p * (1 - e2 * N / (N + h)))

        sin_lat = math.sin(lat)
        cos_lat = math.cos(lat)
        N = a / math.sqrt(1 - e2 * sin_lat * sin_lat)
        if p > abs(z):
            h = p / cos_lat - N
        else:
            h = z / sin_lat - N * (1 - e2)

        out[i, 0] = math.degrees(lat)
        out[i, 1] = math.degrees(lon)
        out[i, 2] = h


def ecef_to_lla(position_ecef):
    """
    Convert ECEF coordinates to latitude, longitude, and altitude.

    Args:
        position_ecef: Position vector [x, y, z], or an array of shape (..., 3)
                       of position vectors, in ECEF frame (meters)

    Returns:
        Tuple of (latitude, longitude, altitude) in (degrees, degrees, meters).
        Scalars for a single vector, arrays of shape (...) otherwise.

    Vectorised over any leading shape; this used to be a scalar python function
    with an inner loop, so converting a whole ground track meant one python call
    per sample. It is also now safe on the polar axis, where the old
    p/cos(lat) height formula divided by zero.
    """
    arr = np.ascontiguousarray(position_ecef, dtype=np.float64)
    if arr.shape[-1] != 3:
        raise ValueError(f"expected trailing axis of length 3, got {arr.shape}")
    scalar = arr.ndim == 1
    flat = arr.reshape(-1, 3)
    out = np.empty_like(flat)
    _ecef_to_lla_kernel(flat, out)
    if scalar:
        return out[0, 0], out[0, 1], out[0, 2]
    lead = arr.shape[:-1]
    return (out[:, 0].reshape(lead), out[:, 1].reshape(lead), out[:, 2].reshape(lead))

def lla_to_ecef(lat, lon, alt):
    """
    Convert latitude, longitude, and altitude to ECEF coordinates.
    
    Args:
        lat: Latitude in degrees
        lon: Longitude in degrees
        alt: Altitude in meters
        
    Returns:
        Position vector [x, y, z] in ECEF frame (meters)
    """
    lat_rad = math.radians(lat)
    lon_rad = math.radians(lon)
    
    # WGS-84 ellipsoid parameters
    a = RE  # Earth's equatorial radius in meters
    f = 1/298.257223563  # Flattening
    e2 = 2*f - f*f  # Square of eccentricity
    
    N = a / math.sqrt(1 - e2 * math.sin(lat_rad) * math.sin(lat_rad))
    
    x = (N + alt) * math.cos(lat_rad) * math.cos(lon_rad)
    y = (N + alt) * math.cos(lat_rad) * math.sin(lon_rad)
    z = (N * (1 - e2) + alt) * math.sin(lat_rad)
    
    return np.array([x, y, z])

def ground_station_ecef(lat, lon, alt=0):
    """
    Calculate ECEF coordinates of a ground station.
    
    Args:
        lat: Latitude in degrees
        lon: Longitude in degrees
        alt: Altitude in meters (default: 0)
        
    Returns:
        Position vector [x, y, z] in ECEF frame (meters)
    """
    return lla_to_ecef(lat, lon, alt)

def cart_to_lat_lon(x, y, z):
    """
    Convert Cartesian ECEF coordinates to latitude and longitude.

    Args:
        x, y, z: Cartesian coordinates in ECEF frame (meters). Scalars or
                 broadcastable arrays.

    Returns:
        Tuple of (latitude, longitude) in degrees, matching the input shape.
    """
    x, y, z = np.broadcast_arrays(np.asarray(x, np.float64),
                                  np.asarray(y, np.float64),
                                  np.asarray(z, np.float64))
    lat, lon, _ = ecef_to_lla(np.stack([x, y, z], axis=-1))
    return lat, lon


def ground_track(positions_ecef):
    """
    Latitude/longitude ground track for a whole propagation result.

    Args:
        positions_ecef: array of shape (num_sats, num_times, 3) in ECEF metres

    Returns:
        (lat_deg, lon_deg), each of shape (num_sats, num_times)
    """
    lat, lon, _ = ecef_to_lla(np.asarray(positions_ecef, dtype=np.float64))
    return lat, lon

@njit(parallel=True)
def batch_eci_to_ecef(positions_eci, gmst):
    """
    Convert batch of ECI coordinates to ECEF coordinates.
    
    Args:
        positions_eci: Array of shape (N, 3) with ECI positions
        gmst: Greenwich Mean Sidereal Time in radians
        
    Returns:
        Array of shape (N, 3) with ECEF positions
    """
    cos_gmst = np.cos(gmst)
    sin_gmst = np.sin(gmst)
    
    n = positions_eci.shape[0]
    positions_ecef = np.zeros_like(positions_eci)
    
    for i in prange(n):
        x_eci = positions_eci[i, 0]
        y_eci = positions_eci[i, 1]
        z_eci = positions_eci[i, 2]
        
        positions_ecef[i, 0] = x_eci * cos_gmst + y_eci * sin_gmst
        positions_ecef[i, 1] = -x_eci * sin_gmst + y_eci * cos_gmst
        positions_ecef[i, 2] = z_eci
    
    return positions_ecef

@njit(parallel=True, cache=True)
def eci_to_ecef_series(positions_eci, gmst):
    """
    Rotate a whole (num_sats, num_times, 3) ECI series into ECEF in one call.

    Replaces looping over timesteps in python and calling batch_eci_to_ecef per
    step, which measured as 85% of the CPU backend's runtime.

    Args:
        positions_eci: Array of shape (num_sats, num_times, 3)
        gmst: Array of shape (num_times,) with GMST in radians per timestep

    Returns:
        Array of shape (num_sats, num_times, 3) with ECEF positions
    """
    num_sats, num_times = positions_eci.shape[0], positions_eci.shape[1]
    cos_gmst = np.cos(gmst)
    sin_gmst = np.sin(gmst)
    positions_ecef = np.empty_like(positions_eci)

    for i in prange(num_sats):
        for t in range(num_times):
            c = cos_gmst[t]
            s = sin_gmst[t]
            x_eci = positions_eci[i, t, 0]
            y_eci = positions_eci[i, t, 1]
            positions_ecef[i, t, 0] = x_eci * c + y_eci * s
            positions_ecef[i, t, 1] = -x_eci * s + y_eci * c
            positions_ecef[i, t, 2] = positions_eci[i, t, 2]

    return positions_ecef


@njit(parallel=True)
def batch_ecef_to_eci(positions_ecef, gmst):
    """
    Convert batch of ECEF coordinates to ECI coordinates.
    
    Args:
        positions_ecef: Array of shape (N, 3) with ECEF positions
        gmst: Greenwich Mean Sidereal Time in radians
        
    Returns:
        Array of shape (N, 3) with ECI positions
    """
    cos_gmst = np.cos(gmst)
    sin_gmst = np.sin(gmst)
    
    n = positions_ecef.shape[0]
    positions_eci = np.zeros_like(positions_ecef)
    
    for i in prange(n):
        x_ecef = positions_ecef[i, 0]
        y_ecef = positions_ecef[i, 1]
        z_ecef = positions_ecef[i, 2]
        
        positions_eci[i, 0] = x_ecef * cos_gmst - y_ecef * sin_gmst
        positions_eci[i, 1] = x_ecef * sin_gmst + y_ecef * cos_gmst
        positions_eci[i, 2] = z_ecef
    
    return positions_eci
