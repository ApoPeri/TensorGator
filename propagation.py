import numpy as np
from numba import njit, prange
from .constants import MU, RE, J2

def satellite_positions(times, constellation, backend='cpu', return_frame='ecef', epochs=None, input_type='kepler', **kwargs):
    """
    Propagate satellites using the selected backend ('cpu' or 'cuda').
    
    Args:
        times: np.ndarray of times (seconds since J2000 or since reference epoch)
        constellation: np.ndarray of shape (N, 6) with either:
                      - Keplerian elements [a, e, inc, Omega, omega, M0] if input_type='kepler'
                      - Position and velocity vectors [rx, ry, rz, vx, vy, vz] if input_type='rv'
                        where positions are in meters and velocities in m/s
        backend: 'cpu' or 'cuda'
        return_frame: Coordinate frame to return ('ecef' or 'eci')
        epochs: np.ndarray of shape (N,) with epoch times in seconds since J2000
                for each satellite. If None, assumes all satellites use times[0] as epoch.
        input_type: 'kepler' for Keplerian elements or 'rv' for position-velocity vectors
        
    Returns:
        np.ndarray of shape (num_sats, num_times, 3) with satellite positions
        in the specified coordinate frame (ECEF by default)
    """
    if backend == 'cpu':
        # Note: CPU backend currently only supports Keplerian elements
        if input_type.lower() != 'kepler':
            raise ValueError(f"CPU backend only supports 'kepler' input_type, not '{input_type}'")
            
        from .prop_cpu import propagate_constellation_cpu
        from .coord_conv import eci_to_ecef_series

        # Propagate constellation using CPU with J2 perturbation
        positions_eci = propagate_constellation_cpu(constellation, times, epochs=epochs)

        # Transform ECI to ECEF if needed
        if return_frame.lower() == 'ecef':
            # Handle time references for GMST calculation
            if epochs is None:
                # If no epochs provided, use times[0] as reference
                times_seconds = times - times[0]
            else:
                # If epochs provided, use absolute times
                times_seconds = times

            # One vectorised rotation over the whole series rather than a
            # python loop calling batch_eci_to_ecef once per timestep.
            from .coord_conv import gmst_from_seconds
            gmst = gmst_from_seconds(times_seconds)
            return eci_to_ecef_series(positions_eci, gmst)
        else:
            # Return ECI coordinates
            return positions_eci
    elif backend in ('cuda', 'cuda_fast'):
        # 'cuda_fast' is kept as an alias; 'cuda' now uses the optimised kernel.
        # Extra keywords (dtype, out) are forwarded to prop_cuda_fast.
        from .prop_cuda_fast import propagate_constellation_cuda_fast
        return propagate_constellation_cuda_fast(constellation, times, return_frame=return_frame,
                                                 epochs=epochs, input_type=input_type, **kwargs)
    elif backend == 'cuda_legacy':
        from .prop_cuda import propagate_constellation_cuda_legacy
        return propagate_constellation_cuda_legacy(constellation, times, return_frame=return_frame,
                                                   epochs=epochs, input_type=input_type)
    else:
        raise ValueError(f"Unknown backend: {backend}")
