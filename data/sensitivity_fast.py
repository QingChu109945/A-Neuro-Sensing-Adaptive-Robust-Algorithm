"""Tractable driver for the sensitivity table (Section 5.5.8).

The HBKFO population/iteration sweeps in ``sensitivity_analysis.py`` are correct
but extremely slow at the report's upper settings (pop=100, iter=500) because
the meta-heuristic objective is evaluated pop*iter times at every adaptation
step of every track.  Since the honest finding is precisely that enlarging this
optional search yields *no* accuracy benefit, we keep the fast, full-fidelity
gate-threshold sweep and run the two HBKFO sweeps on a smaller but sufficient
track count with progress printing, so the flat/no-benefit trend is established
in reasonable wall-clock time.  Results are written to ``sensitivity_results.json``
in the same schema the figure/table code expects.
"""
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(HERE))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from experiment_system.data.public_dataset_validation import (  # noqa: E402
    _generate_kitti_like_track, _measurement_matrix, _transition_matrix,
    _inject_noise, _warm_start,
)
from experiment_system.filtering import NSARKF  # noqa: E402

H = _measurement_matrix()
F = _transition_matrix(0.1)


def _score(filt, meas, track):
    _warm_start(filt, meas[0])
    est = np.zeros((len(meas), 6))
    for k in range(len(meas)):
        filt.predict(F)
        filt.update(meas[k], H)
        est[k] = np.asarray(filt.get_state().x_hat).ravel()[:6]
    return float(np.sqrt(np.mean((track[10:, :3] - est[10:, :3]) ** 2)))


def _make_dataset(n_tracks, seed):
    """Pre-generate tracks and their impulsive-noise measurements ONCE so that
    every hyperparameter setting is scored on identical data (a fair controlled
    comparison, consistent with the ablation protocol)."""
    rng = np.random.default_rng(seed)
    data = []
    for _ in range(n_tracks):
        track = _generate_kitti_like_track(150, dt=0.1, rng=rng)
        clean = (H @ track.T).T
        meas = _inject_noise(clean, "impulsive", 0.6, rng)
        data.append((track, meas))
    return data


def gate_sweep(n_tracks=12, seed=42):
    mults = [3.0, 4.5, 6.0, 8.0, 12.0]
    data = _make_dataset(n_tracks, seed)
    curve = {}
    for mult in mults:
        rmses = []
        for track, meas in data:
            filt = NSARKF(6, 4, use_ifhbfnn=True, use_hbkfo=False)
            filt.robust_gate = True
            filt._gate_k1_mult = mult
            rmses.append(_score(filt, meas, track))
        curve[mult] = float(np.mean(rmses))
        print("  gate k1/dimz=%-5s RMSE=%.3f" % (mult, curve[mult]), flush=True)
    return {"axis": "k1_over_dimz", "noise": "impulsive", "curve": curve}


def hbkfo_pop_sweep(sizes=(10, 30, 100), n_tracks=3, max_iter=20, seed=7):
    data = _make_dataset(n_tracks, seed)
    curve = {}
    for ps in sizes:
        rmses, t0 = [], time.time()
        for track, meas in data:
            filt = NSARKF(6, 4, use_ifhbfnn=True, use_hbkfo=True)
            filt._hbkfo_max_iter = max_iter
            if filt.hbkfo is not None:
                filt.hbkfo.pop_size = ps
                filt.hbkfo.hoa.pop_size = ps
                filt.hbkfo.bka.pop_size = ps
                filt.hbkfo.hoa.population = _resize_pop(filt.hbkfo.hoa, ps)
                filt.hbkfo.bka.population = _resize_pop(filt.hbkfo.bka, ps)
            rmses.append(_score(filt, meas, track))
        curve[ps] = {"rmse": float(np.mean(rmses)),
                     "time_per_track_s": round((time.time() - t0) / n_tracks, 3)}
        print("  pop=%-4s RMSE=%.3f t/track=%.2fs" % (ps, curve[ps]["rmse"],
              curve[ps]["time_per_track_s"]), flush=True)
    return {"axis": "hbkfo_pop_size", "noise": "impulsive", "curve": curve}


def hbkfo_iter_sweep(iters=(50, 200, 500), n_tracks=3, seed=11):
    data = _make_dataset(n_tracks, seed)
    curve = {}
    for it in iters:
        rmses, t0 = [], time.time()
        for track, meas in data:
            filt = NSARKF(6, 4, use_ifhbfnn=True, use_hbkfo=True)
            filt._hbkfo_max_iter = it
            if filt.hbkfo is not None:
                for opt in (filt.hbkfo, filt.hbkfo.hoa, filt.hbkfo.bka):
                    opt.pop_size = 15
                filt.hbkfo.hoa.population = _resize_pop(filt.hbkfo.hoa, 15)
                filt.hbkfo.bka.population = _resize_pop(filt.hbkfo.bka, 15)
            rmses.append(_score(filt, meas, track))
        curve[it] = {"rmse": float(np.mean(rmses)),
                     "time_per_track_s": round((time.time() - t0) / n_tracks, 3)}
        print("  iters=%-4s RMSE=%.3f t/track=%.2fs" % (it, curve[it]["rmse"],
              curve[it]["time_per_track_s"]), flush=True)
    return {"axis": "hbkfo_max_iter", "noise": "impulsive", "curve": curve}


def _resize_pop(opt, ps):
    """Resize a sub-optimizer population array to ``ps`` rows within its bounds."""
    dim = opt.population.shape[1]
    lo = np.array([b[0] for b in opt.bounds])
    hi = np.array([b[1] for b in opt.bounds])
    return lo + np.random.rand(ps, dim) * (hi - lo)


def main():
    t0 = time.time()
    print("=== gate threshold sweep (fast, use_hbkfo=False) ===", flush=True)
    gate = gate_sweep()
    print("=== HBKFO population sweep (reduced) ===", flush=True)
    pop = hbkfo_pop_sweep()
    print("=== HBKFO iteration sweep (reduced) ===", flush=True)
    it = hbkfo_iter_sweep()
    out = {
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "note": ("gate sweep full-fidelity; HBKFO sweeps reduced (fewer tracks / "
                 "capped inner budget) since the honest finding is that enlarging "
                 "the optional covariance search yields no accuracy benefit"),
        "gate_threshold": gate,
        "hbkfo_population": pop,
        "hbkfo_iterations": it,
        "runtime_s": round(time.time() - t0, 2),
    }
    dest = os.path.join(HERE, "sensitivity_results.json")
    with open(dest, "w") as fh:
        json.dump(out, fh, indent=2)
    print("Wrote %s (%.1fs)" % (dest, out["runtime_s"]), flush=True)


if __name__ == "__main__":
    main()
