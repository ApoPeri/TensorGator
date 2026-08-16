# TensorGator

TensorGator is a CUDA-accelerated satellite propagation library designed for massively parallel orbital mechanics calculations.

## Performance

TensorGator's CUDA backend is far faster than the CPU backend for large constellations. On a 500,000 satellite × 500 timestep problem (250M positions, RTX 3080, float32, ECEF):

| How results are consumed | Time |
|---|---|
| `backend='cuda'`, new array per call | 1.00 s |
| `Propagator(..., pinned=True)`, `out='pinned'` | 0.13 s |
| `Propagator`, `out='device'` (stays on the GPU) | 0.012 s |

The propagation kernel itself runs in ~5 ms; the remaining time is spent moving results to the host. The CPU backend is available as a fallback, but the same workload is orders of magnitude slower on CPU.

Reproduce with:

```bash
python -m tensorgator.bench.profile_cuda
```

## Features

- **CUDA Acceleration**: Propagate thousands of satellites simultaneously using GPU parallelization
- **J2 Perturbation Model**: Accurate orbital propagation including Earth's oblateness effects
- **Flexible Input Formats**: Support for both Keplerian elements and R,V vectors
- **Coordinate Transformations**: Fast ECI/ECEF conversions with Numba acceleration
- **Satellite Visibility Analysis**: Determine satellite coverage and visibility from the ground
- **Batch Processing**: Memory-efficient handling of large constellations through automatic batching
- **CPU Fall Back**: Support for CPU mode when CUDA is not available

## Installation

```bash
pip install tensorgator
```

Or install from source:

```bash
git clone https://github.com/yourusername/tensorgator.git
cd tensorgator
pip install -e .
```

## Requirements

- Python 3.6+
- CUDA-compatible GPU
- NumPy
- Numba

## Visualization Libraraies
- Matplotlib
- Basemap

## Quick Start

    import numpy as np
    import time
    import matplotlib.pyplot as plt

    import tensorgator as tg
    from tensorgator.prop_cuda import propagate_constellation_cuda
    from numba import config
    config.CUDA_ENABLE_PYNVJITLINK = 1 # Enable CUDA support on Google Colab
    
    def main():
        np.random.seed(21)
        
        RE = tg.RE
        
        num_sats = 10
        constellation = []
        
        for _ in range(num_sats):
            altitude = np.random.uniform(300000, 2000000)
            a = RE + altitude
            e = 0.0
            inc = np.radians(np.random.uniform(20, 98))
            raan = np.radians(np.random.uniform(0, 360))
            argp = np.radians(np.random.uniform(0, 360))
            M0 = np.radians(np.random.uniform(0, 360))
            
            constellation.append([a, e, inc, raan, argp, M0])
        
        constellation = np.array(constellation)
        
        time_step = 60 # seconds
        num_steps = 1440
        times = np.arange(0, num_steps * time_step, time_step)
        
        print(f"Propagating {num_sats} satellites over {num_steps} time steps...")
        start_time = time.time()
        
        positions = propagate_constellation_cuda(constellation, times, return_frame='ecef')
        
        prop_time = time.time() - start_time
        print(f"Propagation completed in {prop_time:.2f} seconds")
        
        # Simple 2D plot
        plt.figure(figsize=(8, 8))
        
        # Draw Earth
        earth_radius_scaled = 1.0
        scale_factor = earth_radius_scaled / RE
        earth_circle = plt.Circle((0, 0), earth_radius_scaled, color='blue', alpha=0.3)
        plt.gca().add_patch(earth_circle)
        
        # Plot orbit trails for 10 satellites
        for i in range(0, min(num_sats, 100), 1):
            x = positions[i, :, 0] * scale_factor
            y = positions[i, :, 1] * scale_factor
            plt.plot(x, y, linewidth=0.8, alpha=0.7)
        
        plt.axis('equal')
        max_alt = np.max(constellation[:, 0]) * scale_factor
        plt.xlim(-max_alt, max_alt)
        plt.ylim(-max_alt, max_alt)
        plt.grid(True, linestyle='--', alpha=0.3)
        plt.title('Satellite Orbits')
        plt.savefig('orbits.png')
        plt.show()

    if __name__ == "__main__":
        main()

## Core Functions

### Propagation

```python
tg.satellite_positions(times, constellation, backend='cpu', return_frame='ecef', epochs=None, input_type='kepler')
```

Propagates satellite positions over time using either CPU or CUDA backend.

Parameters:
- `times`: Array of times (seconds since J2000 or reference epoch)
- `constellation`: Array of satellite elements (Keplerian or position-velocity)
- `backend`: 'cpu', 'cuda' (optimised GPU kernel) or 'cuda_legacy' (original kernel)
- `return_frame`: Coordinate frame to return ('ecef' or 'eci')
- `epochs`: Optional array of epoch times for each satellite
- `input_type`: 'kepler' for Keplerian elements or 'rv' for position-velocity vectors

### Visibility Analysis

```python
tg.calculate_visibility_cuda(satellite_positions, ground_points, min_elevation_rad)
```

Calculates visibility between satellites and ground points.

Parameters:
- `satellite_positions`: Array of satellite positions (ECEF)
- `ground_points`: Array of ground point coordinates in ECEF metres, shape (P, 3)
- `min_elevation`: Minimum elevation angle for visibility (radians)

Returns a boolean array indicating visibility.

## Examples

TensorGator includes several example applications:

These can be uploaded to google colab or run on your own NVIDIA GPU

3d_orbit_animation_google_colab.ipynb
benchmark_google_colab.ipynb
coverage_map_google_colab.ipynb

## Validation

TensorGator has been validated against:
- CPU-based propagator,Beyond (validated to <1m/day precision with float64 dtype)
- Poliastro (~several km/day discrepancies due to difference between integrated force model and tensorgator analytical model)

## Future Roadmap

- **Hardware Acceleration**: Support for acceleration libraries (TensorFlow, Jax) beyond CUDA
- **Additional Mission Simulation**: End-to-end satellite mission simulation capabilities
- **Higher-order Perturbation Models**: Add support for atmospheric drag, solar radiation pressure, and third-body gravity

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 

Additional Terms: If you are using Tensoragator for commercial (or any other) purposes, I'd love to hear from you! Please drop a line at ApogeePerigee@protonmail.com - It helps me keep track of the impact of Tensoragator and motivates me to continue improving it.
