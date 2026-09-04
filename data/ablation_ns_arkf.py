"""Fine-grained NS-ARKF ablation for the manuscript ablation table.

Isolates the marginal contribution of each NS-ARKF mechanism on the same
KITTI-geometry protocol used by public_dataset_validation, averaged over the
three extreme-noise families.  The configurations, in cumulative order:

    UIF                 : unknown-input filter only (no IFHBFNN, no robust gate, no HBKFO)
    + IFHBFNN           : add the fuzzy hyper-RBF nonlinear-interference estimator
    + robust gate       : add the IGG-III robust innovation gate (impulse rejection)
    + HBKFO (full)      : add the meta-heuristic online covariance adaptation

Reports 3D position RMSE per noise family and the average, plus the marginal
delta of each added mechanism, so the manuscript can report HBKFO's *honest*
marginal (near-zero / slightly negative) contribution.
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(HERE))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from experiment_system.data.public_dataset_validation import (
    _generate_kitti_like_track, _measurement_matrix, _transition_matrix,
    _inject_noise, _warm_start,
)
from experiment_system.filtering import NSARKF

H = _measurement_matrix()
F = _transition_matrix(0.1)


def run_cfg(use_if, gate, use_hb, meas, track):
    filt = NSARKF(6, 4, use_ifhbfnn=use_if, use_hbkfo=use_hb)
    filt.robust_gate = gate
    _warm_start(filt, meas[0])
    est = np.zeros((len(meas), 6))
    for k in range(len(meas)):
        filt.predict(F)
        filt.update(meas[k], H)
        est[k] = np.asarray(filt.get_state().x_hat).ravel()[:6]
    return float(np.sqrt(np.mean((track[10:, :3] - est[10:, :3]) ** 2)))


# (label, use_ifhbfnn, robust_gate, use_hbkfo)
cfgs = [
    ("UIF", False, False, False),
    ("+ IFHBFNN", True, False, False),
    ("+ robust gate (IGG-III)", True, True, False),
    ("+ HBKFO (full NS-ARKF)", True, True, True),
]
noise_types = ["gaussian", "impulsive", "time_varying"]

rng = np.random.default_rng(42)
acc = {c[0]: {nt: [] for nt in noise_types} for c in cfgs}
for _ in range(12):
    track = _generate_kitti_like_track(150, dt=0.1, rng=rng)
    clean = (H @ track.T).T
    for nt in noise_types:
        meas = _inject_noise(clean, nt, 0.6, rng)
        for name, ui, gate, hb in cfgs:
            acc[name][nt].append(run_cfg(ui, gate, hb, meas, track))

table = {}
for name, *_ in cfgs:
    row = {nt: float(np.mean(acc[name][nt])) for nt in noise_types}
    row["average"] = float(np.mean(list(row.values())))
    table[name] = row

print("=== NS-ARKF fine-grained ablation (3D position RMSE) ===")
prev = None
for name, *_ in cfgs:
    row = table[name]
    delta = "" if prev is None else "  (Delta avg %+.3f)" % (row["average"] - prev)
    print("  %-26s gauss=%.3f imp=%.3f tv=%.3f avg=%.3f%s"
          % (name, row["gaussian"], row["impulsive"], row["time_varying"],
             row["average"], delta))
    prev = row["average"]

with open(os.path.join(HERE, "ablation_results.json"), "w") as fh:
    json.dump(table, fh, indent=2)
print("Wrote ablation_results.json")
