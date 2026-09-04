"""Diagnostic 4: isolate NS-ARKF components under impulsive noise."""
import os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(HERE))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
from experiment_system.data.public_dataset_validation import (
    _generate_kitti_like_track, _measurement_matrix, _transition_matrix,
    _inject_noise, _warm_start,
)
from experiment_system.filtering import NSARKF, create_rukf_filter

H = _measurement_matrix(); F = _transition_matrix(0.1)

def run_cfg(use_if, use_hb, gate, meas, track):
    filt = NSARKF(6, 4, use_ifhbfnn=use_if, use_hbkfo=use_hb)
    filt.robust_gate = gate
    _warm_start(filt, meas[0])
    est = np.zeros((len(meas),6))
    for k in range(len(meas)):
        filt.predict(F); filt.update(meas[k], H)
        est[k] = np.asarray(filt.get_state().x_hat).ravel()[:6]
    return float(np.sqrt(np.mean((track[10:,:3]-est[10:,:3])**2)))

rng = np.random.default_rng(42)
cfgs = [("UIF-only(no if,no hb)",False,False),
        ("UIF+IFHBFNN",True,False),
        ("UIF+HBKFO",False,True),
        ("full NS-ARKF",True,True)]
for nt in ["impulsive","time_varying","gaussian"]:
    acc = {c[0]:[] for c in cfgs}
    for _ in range(6):
        track = _generate_kitti_like_track(150, dt=0.1, rng=rng)
        clean = (H@track.T).T
        meas = _inject_noise(clean, nt, 0.6, rng)
        for name,ui,hb in cfgs:
            acc[name].append(run_cfg(ui,hb,True,meas,track))
    print(f"=== {nt} ===")
    for name,_,_ in cfgs:
        print(f"    {name:28s} {np.mean(acc[name]):.3f}")
