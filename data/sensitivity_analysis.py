"""Hyperparameter-sensitivity analysis for the manuscript (Section 5.4.x).

Addresses report point P1-4 (sensitivity analysis) with three cheap,
filtering-side sweeps that run on the same KITTI-geometry protocol used by
``public_dataset_validation`` and ``ablation_ns_arkf`` so every number is
produced by the actual NS-ARKF implementation:

1. ``sweep_gate_threshold`` -- the IGG-III robust-gate full-rejection multiplier
   ``k1/dim_z`` (the mechanism that the honest ablation shows to be responsible
   for the bulk of the extreme-noise robustness).  Reports 3D position RMSE vs.
   the threshold under impulsive noise, exposing how aggressively the gate must
   reject outliers.

2. ``sweep_hbkfo_population`` -- the optional covariance-adaptation population
   size (report: 10-100), reporting RMSE and per-track wall-clock so the
   accuracy/compute trade-off is explicit.

3. ``sweep_hbkfo_iterations`` -- the covariance-adaptation iteration budget
   (report: 50-500), same trade-off axis.

Sweeps 2-3 also confirm the honest finding that, once the robust gate is
active, enlarging the HBKFO search yields no meaningful accuracy gain (and a
slight loss under impulsive noise), justifying HBKFO's status as an optional
engineering module rather than a core mechanism.

Writes ``sensitivity_results.json`` next to this file.
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


def _tracks(n_tracks, seed):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_tracks):
        track = _generate_kitti_like_track(150, dt=0.1, rng=rng)
        clean = (H @ track.T).T
        out.append((track, clean, rng))
    return out


# --------------------------------------------------------------------------- #
# 1. IGG-III robust-gate full-rejection threshold  k1 = mult * dim_z
# --------------------------------------------------------------------------- #
def sweep_gate_threshold(n_tracks=12, noise_type="impulsive", seed=42):
    mults = [3.0, 4.5, 6.0, 8.0, 12.0]          # k1 / dim_z  (code default 6.0)
    rng = np.random.default_rng(seed)
    tracks = [(_generate_kitti_like_track(150, dt=0.1, rng=rng)) for _ in range(n_tracks)]
    curve = {}
    for mult in mults:
        rmses = []
        for track in tracks:
            clean = (H @ track.T).T
            meas = _inject_noise(clean, noise_type, 0.6, rng)
            filt = NSARKF(6, 4, use_ifhbfnn=True, use_hbkfo=False)
            filt.robust_gate = True
            filt._gate_k1_mult = mult          # IGG-III full-rejection multiplier
            rmses.append(_score(filt, meas, track))
        curve[mult] = float(np.mean(rmses))
    return {"axis": "k1_over_dimz", "noise": noise_type, "curve": curve}


# --------------------------------------------------------------------------- #
# 2. HBKFO population size   (optional covariance-adaptation module)
# --------------------------------------------------------------------------- #
def sweep_hbkfo_population(n_tracks=8, noise_type="impulsive", seed=7):
    sizes = [10, 20, 30, 50, 100]
    rng = np.random.default_rng(seed)
    tracks = [_generate_kitti_like_track(150, dt=0.1, rng=rng) for _ in range(n_tracks)]
    curve = {}
    for ps in sizes:
        rmses, t0 = [], time.time()
        for track in tracks:
            clean = (H @ track.T).T
            meas = _inject_noise(clean, noise_type, 0.6, rng)
            filt = NSARKF(6, 4, use_ifhbfnn=True, use_hbkfo=True)
            if filt.hbkfo is not None:
                filt.hbkfo.pop_size = ps
                filt.hbkfo.hoa.pop_size = ps
                filt.hbkfo.bka.pop_size = ps
            rmses.append(_score(filt, meas, track))
        curve[ps] = {"rmse": float(np.mean(rmses)),
                     "time_per_track_s": round((time.time() - t0) / n_tracks, 3)}
    return {"axis": "hbkfo_pop_size", "noise": noise_type, "curve": curve}


# --------------------------------------------------------------------------- #
# 3. HBKFO iteration budget
# --------------------------------------------------------------------------- #
def sweep_hbkfo_iterations(n_tracks=8, noise_type="impulsive", seed=11):
    iters = [50, 100, 200, 350, 500]
    rng = np.random.default_rng(seed)
    tracks = [_generate_kitti_like_track(150, dt=0.1, rng=rng) for _ in range(n_tracks)]
    curve = {}
    for it in iters:
        rmses, t0 = [], time.time()
        for track in tracks:
            clean = (H @ track.T).T
            meas = _inject_noise(clean, noise_type, 0.6, rng)
            filt = NSARKF(6, 4, use_ifhbfnn=True, use_hbkfo=True)
            filt._hbkfo_max_iter = it
            rmses.append(_score(filt, meas, track))
        curve[it] = {"rmse": float(np.mean(rmses)),
                     "time_per_track_s": round((time.time() - t0) / n_tracks, 3)}
    return {"axis": "hbkfo_max_iter", "noise": noise_type, "curve": curve}


def main():
    t0 = time.time()
    out = {
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "gate_threshold": sweep_gate_threshold(),
        "hbkfo_population": sweep_hbkfo_population(),
        "hbkfo_iterations": sweep_hbkfo_iterations(),
    }
    out["runtime_s"] = round(time.time() - t0, 2)
    dest = os.path.join(HERE, "sensitivity_results.json")
    with open(dest, "w") as fh:
        json.dump(out, fh, indent=2)

    print("=== IGG-III gate threshold (impulsive, 3D pos RMSE) ===")
    for k, v in out["gate_threshold"]["curve"].items():
        print("  k1/dimz=%-5s RMSE=%.3f" % (k, v))
    print("=== HBKFO population size (impulsive) ===")
    for k, v in out["hbkfo_population"]["curve"].items():
        print("  pop=%-4s RMSE=%.3f  t/track=%.3fs" % (k, v["rmse"], v["time_per_track_s"]))
    print("=== HBKFO iterations (impulsive) ===")
    for k, v in out["hbkfo_iterations"]["curve"].items():
        print("  iters=%-4s RMSE=%.3f  t/track=%.3fs" % (k, v["rmse"], v["time_per_track_s"]))
    print("Wrote %s (%.1fs)" % (dest, out["runtime_s"]))


if __name__ == "__main__":
    main()
