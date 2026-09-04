"""Diagnostic 3: use the real get_state().x_hat accessor (combined NS-ARKF)."""
import os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(HERE))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
from experiment_system.data.public_dataset_validation import (
    _generate_kitti_like_track, _measurement_matrix, _transition_matrix,
    _inject_noise, _run_filter, _meta_heuristic_filter,
)
from experiment_system.filtering import (
    create_ekf_filter, create_rukf_filter, create_aekf_filter,
    create_deepkf_filter, create_ns_arkf_filter,
)
H = _measurement_matrix(); F = _transition_matrix(0.1)
methods = {"EKF":create_ekf_filter,"AEKF":create_aekf_filter,
           "RUKF":create_rukf_filter,"DeepKF":create_deepkf_filter,
           "NS-ARKF":create_ns_arkf_filter}
rng = np.random.default_rng(42)
for nt in ["gaussian","impulsive","time_varying"]:
    accum = {m:[] for m in methods}
    for _ in range(6):
        track = _generate_kitti_like_track(150, dt=0.1, rng=rng)
        clean = (H@track.T).T
        meas = _inject_noise(clean, nt, 0.6, rng)
        for m,fac in methods.items():
            est = _run_filter(fac, meas, H, F)
            accum[m].append(float(np.sqrt(np.mean((track[10:,:3]-est[10:,:3])**2))))
    print(f"=== {nt} ===  " + "  ".join(f"{m}={np.mean(accum[m]):.3f}" for m in methods))
