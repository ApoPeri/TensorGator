"""
Profiling / regression harness for the CUDA backend.

    python -m tensorgator.bench.profile_cuda           # everything
    python -m tensorgator.bench.profile_cuda accuracy  # one section

Sections: phases, ceiling, alloc, blocks, accuracy, endtoend, coverage, culling
"""

import gc
import math
import sys
import time

import numpy as np
from numba import cuda, float32

from ..constants import MU, J2, RE
from ..prop_cuda import propagate_constellation_cuda_legacy
from ..prop_cuda_fast import Propagator
from ..coord_conv import gmst_from_seconds


def make(n, emax=0.02, seed=21):
    rng = np.random.default_rng(seed)
    return np.column_stack([
        RE + rng.uniform(300e3, 2000e3, n), rng.uniform(0.0, emax, n),
        np.radians(rng.uniform(20, 98, n)), np.radians(rng.uniform(0, 360, n)),
        np.radians(rng.uniform(0, 360, n)), np.radians(rng.uniform(0, 360, n))])


def best(fn, reps=3):
    fn(); cuda.synchronize()
    ts = []
    for _ in range(reps):
        gc.collect(); cuda.synchronize(); t0 = time.perf_counter()
        x = fn(); cuda.synchronize(); ts.append(time.perf_counter() - t0); del x
    return min(ts)


def reference(elements, times, ecef=True, epochs=None):
    """float64 numpy evaluation of the same analytic J2 model."""
    el = np.asarray(elements, np.float64)
    a, e, inc, raan, argp, M0 = (el[:, k] for k in range(6))
    n0 = np.sqrt(MU / a ** 3)
    om = 1 - e ** 2
    sc = (n0 * RE ** 2 * J2) / (a ** 2 * om ** 2)
    draan = -1.5 * sc * np.cos(inc)
    dargp = 0.75 * sc * (4 - 5 * np.sin(inc) ** 2)
    dM = 0.75 * sc * np.sqrt(om) * (2 - 3 * np.sin(inc) ** 2)
    ts = (times - times[0]) if epochs is None else np.asarray(times, np.float64)
    t = ts[None, :] - (0.0 if epochs is None else np.asarray(epochs, np.float64)[:, None])
    M = (M0[:, None] + (n0 + dM)[:, None] * t) % (2 * np.pi)
    E = M.copy()
    for _ in range(80):
        E = E - (E - e[:, None] * np.sin(E) - M) / (1 - e[:, None] * np.cos(E))
    xo = a[:, None] * (np.cos(E) - e[:, None])
    yo = (a * np.sqrt(om))[:, None] * np.sin(E)
    g = gmst_from_seconds(ts)[None, :] if ecef else 0.0
    rt = raan[:, None] + draan[:, None] * t - g
    wt = argp[:, None] + dargp[:, None] * t
    ci, si = np.cos(inc)[:, None], np.sin(inc)[:, None]
    cw, sw, cr, sr = np.cos(wt), np.sin(wt), np.cos(rt), np.sin(rt)
    out = np.empty(xo.shape + (3,))
    out[..., 0] = (cr * cw - sr * sw * ci) * xo + (-cr * sw - sr * cw * ci) * yo
    out[..., 1] = (sr * cw + cr * sw * ci) * xo + (-sr * sw + cr * cw * ci) * yo
    out[..., 2] = (sw * si) * xo + (cw * si) * yo
    return out


def err(p, ref):
    d = np.linalg.norm(np.asarray(p, np.float64) - ref, axis=-1)
    return d.mean(), d.max()


# --------------------------------------------------------------------------
def phases(n=500000, t=500):
    """Where the time goes once buffers are reused."""
    print(f"\n== phase profile, N={n} T={t}, buffers reused ==")
    els = make(n); times = np.arange(t, dtype=np.float64) * 60.0
    p = Propagator(n, t, pinned=True)
    p.run(els, times, out='device')
    el64 = np.ascontiguousarray(els, np.float64)
    ep = np.zeros(n); gm = np.zeros(t)
    grid = ((t + 63) // 64, min(65535, (n + 3) // 4))

    def show(name, fn):
        print(f"  {name:<26}{best(fn)*1e3:8.2f} ms")

    show("H2D inputs", lambda: (p.d_elements.copy_to_device(el64),
                                p.d_epochs.copy_to_device(ep),
                                p.d_times.copy_to_device(times.astype(p.dtype)),
                                p.d_gmst.copy_to_device(gm.astype(p.dtype))))
    show("prep kernel (float64)", lambda: p._prep[(n + 127)//128, 128](
        p.d_elements, p.d_epochs, p.d_inv))
    show("propagate kernel", lambda: p._prop[grid, (64, 4)](
        p.d_inv, p.d_times, p.d_gmst, p.d_pos))
    show("D2H into pinned buffer", lambda: p.d_pos.copy_to_host(p.h_pinned))
    tot = best(lambda: p.run(els, times, out='pinned'))
    print(f"  {'full run(out=pinned)':<26}{tot*1e3:8.2f} ms"
          f"   -> {n*t/tot/1e9:.2f} G positions/s")


def ceiling(n=500000, t=500):
    """Is the kernel at the achievable store bandwidth?"""
    print(f"\n== store bandwidth ceiling, N={n} T={t} ==")
    nbytes = n * t * 3 * 4
    grid = ((t + 63) // 64, min(65535, (n + 3) // 4))

    @cuda.jit(fastmath=True)
    def store_aos(out):
        ti, si = cuda.grid(2)
        if ti >= out.shape[1]:
            return
        for s in range(si, out.shape[0], cuda.gridDim.y * cuda.blockDim.y):
            out[s, ti, 0] = float32(1.0)
            out[s, ti, 1] = float32(2.0)
            out[s, ti, 2] = float32(3.0)

    p = Propagator(n, t)
    p.run(make(n), np.arange(t, dtype=np.float64) * 60.0, out='device')
    ks = best(lambda: store_aos[grid, (64, 4)](p.d_pos))
    kp = best(lambda: p._prop[grid, (64, 4)](p.d_inv, p.d_times, p.d_gmst, p.d_pos))
    print(f"  pure store, same layout : {ks*1e3:7.2f} ms  {nbytes/ks/1e9:5.0f} GB/s")
    print(f"  propagation kernel      : {kp*1e3:7.2f} ms  {nbytes/kp/1e9:5.0f} GB/s"
          f"   ({100*ks/kp:.0f}% of ceiling)")


def alloc(n=500000, t=500):
    """Cost of allocating the output buffer per call."""
    print(f"\n== output buffer allocation, N={n} T={t} ==")
    els = make(n); times = np.arange(t, dtype=np.float64) * 60.0
    p = Propagator(n, t)
    p.run(els, times, out='device')
    grid = ((t + 63) // 64, min(65535, (n + 3) // 4))
    reuse = best(lambda: p._prop[grid, (64, 4)](p.d_inv, p.d_times, p.d_gmst, p.d_pos))
    ts = []
    for _ in range(3):
        cuda.synchronize(); t0 = time.perf_counter()
        b = cuda.device_array((n, t, 3), np.float32)
        p._prop[grid, (64, 4)](p.d_inv, p.d_times, p.d_gmst, b)
        cuda.synchronize(); ts.append(time.perf_counter() - t0); del b
    print(f"  reused buffer  : {reuse*1e3:7.2f} ms")
    print(f"  fresh cudaMalloc: {min(ts)*1e3:7.2f} ms   ({min(ts)/reuse:.0f}x more)")


def blocks(n=500000, t=500):
    print(f"\n== block shape sweep, N={n} T={t} ==")
    nbytes = n * t * 3 * 4
    p = Propagator(n, t)
    p.run(make(n), np.arange(t, dtype=np.float64) * 60.0, out='device')
    for tpb in [(32, 4), (32, 8), (64, 4), (64, 8), (128, 2), (128, 4), (256, 1)]:
        g = ((t + tpb[0] - 1)//tpb[0], min(65535, (n + tpb[1] - 1)//tpb[1]))
        m = best(lambda: p._prop[g, tpb](p.d_inv, p.d_times, p.d_gmst, p.d_pos))
        print(f"  threads={str(tpb):<9} {m*1e3:7.2f} ms  {nbytes/m/1e9:5.0f} GB/s")


def accuracy():
    print("\n== position error vs float64 analytic reference (metres) ==")
    print(f"  {'case':<30}{'legacy mean':>12}{'legacy max':>12}{'new mean':>10}{'new max':>10}")
    for label, n, nt, dt, emax in [
        ("1000 sats, 1 day,  e<0.001", 1000, 1440, 60, 0.001),
        ("1000 sats, 1 day,  e<0.05", 1000, 1440, 60, 0.05),
        ("1000 sats, 1 day,  e<0.30", 1000, 1440, 60, 0.30),
        ("1000 sats, 7 days, e<0.05", 1000, 10080, 60, 0.05),
        ("1000 sats, 30 days,e<0.05", 1000, 43200, 60, 0.05),
    ]:
        els = make(n, emax); times = np.arange(nt, dtype=np.float64) * dt
        ref = reference(els, times)
        p = Propagator(n, nt)
        lm, lx = err(propagate_constellation_cuda_legacy(els, times, return_frame='ecef'), ref)
        fm, fx = err(p.run(els, times), ref)
        print(f"  {label:<30}{lm:>12.2f}{lx:>12.1f}{fm:>10.2f}{fx:>10.1f}")
        del p
    els = make(1000, 0.3); times = np.arange(1440, dtype=np.float64) * 60
    p = Propagator(1000, 1440, dtype=np.float64)
    print("  %-30s%34s%10.2e" % ("1000 sats, 1 day, float64", "max err (m):",
                                 err(p.run(els, times), reference(els, times))[1]))


def endtoend():
    print("\n== end to end, ECEF, float32 ==")
    print(f"  {'case':<17}{'legacy':>9}{'new one-shot':>14}{'new reuse+pin':>15}{'new device':>12}")
    for n, t in [(10, 1440), (10000, 500), (100000, 500), (500000, 500)]:
        els = make(n); times = np.arange(t, dtype=np.float64) * 60.0
        r = 3 if n <= 100000 else 2
        L = best(lambda: propagate_constellation_cuda_legacy(els, times, return_frame='ecef'), r)
        p1 = Propagator(n, t)
        O = best(lambda: p1.run(els, times), r)
        p = Propagator(n, t, pinned=True)
        P = best(lambda: p.run(els, times, out='pinned'), r)
        D = best(lambda: p.run(els, times, out='device'), r)
        print(f"  N={n:<6} T={t:<5}{L:>8.3f}s{O:>13.3f}s{P:>14.3f}s{D:>11.3f}s"
              f"   ({L/O:.0f}x / {L/P:.0f}x / {L/D:.0f}x)")
        del p, p1


def coverage():
    """The fused visibility + gap pipeline against the original one."""
    from ..visibility import calculate_visibility_cuda_legacy, calculate_max_gaps
    from ..fused import visibility_cuda, coverage_report

    def py_max_gaps(visibility, time_step):
        num_points, num_times = visibility.shape
        out = np.zeros(num_points)
        for q in range(num_points):
            row = visibility[q]; cur = 0; peak = 0
            for t in range(num_times):
                if not row[t]:
                    cur += 1
                else:
                    if cur > peak: peak = cur
                    cur = 0
            if cur > peak: peak = cur
            out[q] = peak * time_step
        return out

    min_el = math.radians(10.0)
    for label, S, T, gstep in [("coverage_map example", 10, 14400, 5),
                               ("dense 2 deg grid", 60, 5000, 2),
                               ("large constellation", 2000, 2000, 5)]:
        rng = np.random.default_rng(21)
        els = np.column_stack([RE + rng.uniform(500e3, 2000e3, S), np.zeros(S),
                               np.radians(rng.uniform(0, 90, S)), np.radians(rng.uniform(0, 360, S)),
                               np.radians(rng.uniform(0, 360, S)), np.radians(rng.uniform(0, 360, S))])
        times = np.arange(T, dtype=np.float64) * 60
        lats = np.arange(-90, 91, gstep); lons = np.arange(-180, 181, gstep)
        la, lo = np.meshgrid(np.radians(lats), np.radians(lons), indexing='ij')
        la = la.ravel(); lo = lo.ravel()
        gp = np.column_stack([RE*np.cos(la)*np.cos(lo), RE*np.cos(la)*np.sin(lo), RE*np.sin(la)])
        P = len(gp)
        print()
        print(f"== coverage: {label}  S={S} T={T} P={P} ({P*T*S/1e9:.2f} G tests) ==")

        p = Propagator(S, T)
        pos = p.run(els, times)
        gpf = gp.astype(np.float32)
        vo = best(lambda: calculate_visibility_cuda_legacy(pos, gpf, min_el))
        vn = best(lambda: visibility_cuda(pos, gpf, min_el))
        vis = visibility_cuda(pos, gpf, min_el)

        t0 = time.perf_counter()
        old = py_max_gaps(calculate_visibility_cuda_legacy(p.run(els, times), gpf, min_el), 60)
        t_old = time.perf_counter() - t0
        gj = best(lambda: calculate_max_gaps(vis, 60))
        t_new = best(lambda: coverage_report(els, times, gp, min_el, propagator=p))
        new = coverage_report(els, times, gp, min_el, propagator=p)[0]

        print(f"  visibility kernel : {vo*1e3:8.1f} ms -> {vn*1e3:8.1f} ms  ({vo/vn:.1f}x)")
        print(f"  max_gaps njit     : {gj*1e3:8.1f} ms")
        print(f"  full pipeline     : {t_old:8.3f} s  -> {t_new:8.3f} s  ({t_old/t_new:.0f}x)"
              f"   identical={np.array_equal(old, new)}")
        del p


def culling():
    """Cap culling vs brute force. The win tracks the elevation mask."""
    from ..fused import visibility_cuda, visibility_cuda_culled, GroundIndex

    T = 1000
    times = np.arange(T, dtype=np.float64) * 60
    lats = np.arange(-90, 91, 1); lons = np.arange(-180, 181, 1)
    la, lo = np.meshgrid(np.radians(lats), np.radians(lons), indexing='ij')
    la = la.ravel(); lo = lo.ravel()
    gp = np.column_stack([RE*np.cos(la)*np.cos(lo), RE*np.cos(la)*np.sin(lo), RE*np.sin(la)])
    index = GroundIndex(gp)
    print()
    print(f"== cap culling, P={len(gp)} T={T} ==")
    print(f"  {'sats':>6}{'min_el':>8}{'vis frac':>10}{'brute':>10}{'culled':>10}{'speedup':>9}{'exact':>8}")
    for S in [100, 400]:
        rng = np.random.default_rng(5)
        els = np.column_stack([RE + rng.uniform(500e3, 600e3, S), np.zeros(S),
                               np.radians(rng.uniform(0, 90, S)), np.radians(rng.uniform(0, 360, S)),
                               np.radians(rng.uniform(0, 360, S)), np.radians(rng.uniform(0, 360, S))])
        p = Propagator(S, T)
        pos = p.run(els, times, out='device')
        for el in [10.0, 25.0, 40.0]:
            me = math.radians(el)
            a = best(lambda: visibility_cuda(pos, gp, me, out='device'))
            b = best(lambda: visibility_cuda_culled(pos, index, me, out='device'))
            ref = visibility_cuda(pos, gp, me)
            ok = np.array_equal(ref, visibility_cuda_culled(pos, index, me))
            print(f"  {S:>6}{el:>8.0f}{ref.mean():>10.3f}{a*1e3:>9.1f}ms{b*1e3:>9.1f}ms"
                  f"{a/b:>8.1f}x{str(ok):>8}")
        del p


SECTIONS = dict(phases=phases, ceiling=ceiling, alloc=alloc, blocks=blocks,
                accuracy=accuracy, endtoend=endtoend, coverage=coverage,
                culling=culling)

if __name__ == '__main__':
    print(f"device: {cuda.get_current_device().name.decode()}")
    want = sys.argv[1:] or list(SECTIONS)
    for w in want:
        if w not in SECTIONS:
            raise SystemExit(f"unknown section {w!r}; pick from {list(SECTIONS)}")
        SECTIONS[w]()
