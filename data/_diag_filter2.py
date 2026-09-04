"""Diagnostic 2: filter behaviour under extreme / heavy-tailed noise."""
import os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(HERE))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from experiment_system.filtering import (
    create_ekf_filter, create_ukf_filter, create_ckf_filter,
    create_aekf_filter, create_rukf_filter, create_deepkf_filter,
    create_ns_arkf_filter,
)
from experiment_system.data_generator import NoiseInjector, NoiseConfig

def make_track(n, dt=0.1, rng=None):
    rng = rng or np.random.default_rng(0)
    x = np.zeros((n, 6))
    x[0] = [rng.uniform(5,40), rng.uniform(-8,8), rng.uniform(-1,1),
            rng.uniform(3,12), rng.uniform(-1.5,1.5), 0.0]
    for k in range(1, n):
        acc = rng.normal(0, 0.4, 3)
        x[k, 3:] = x[k-1, 3:] + acc*dt
        x[k, :3] = x[k-1, :3] + x[k, 3:]*dt
    return x

def get_state(filt):
    if hasattr(filt, "uif"):
        return np.asarray(filt.uif.x_hat).ravel()[:6]
    return np.asarray(filt.x_hat).ravel()[:6]

def run(fac, meas, H, F, track):
    filt = fac(dim_x=6, dim_z=4)
    # warm start on the underlying state container
    x0 = np.zeros(6); x0[:4] = meas[0, :4]
    if hasattr(filt, "uif"):
        filt.uif.x_hat = x0.copy(); filt.uif.P = np.eye(6)*10.0
    else:
        filt.x_hat = x0.copy(); filt.P = np.eye(6)*10.0
    est = np.zeros((len(meas),6))
    for k in range(len(meas)):
        try:
            filt.predict(F); filt.update(meas[k], H)
            est[k] = get_state(filt)
        except Exception as e:
            est[k] = est[k-1] if k>0 else x0
    return float(np.sqrt(np.mean((track[:,:3]-est[:,:3])**2)))

H = np.zeros((4,6)); H[0,0]=H[1,1]=H[2,2]=H[3,3]=1.0
dt=0.1; F=np.eye(6)
for i in range(3): F[i,i+3]=dt

methods = {"EKF":create_ekf_filter,"UKF":create_ukf_filter,"CKF":create_ckf_filter,
           "AEKF":create_aekf_filter,"RUKF":create_rukf_filter,
           "DeepKF":create_deepkf_filter,"NS-ARKF":create_ns_arkf_filter}

rng = np.random.default_rng(42)
for nt in ["gaussian","impulsive","time_varying"]:
    print(f"\n=== {nt} ===")
    accum = {m:[] for m in methods}
    for _ in range(8):
        track = make_track(150, dt, rng)
        clean = (H@track.T).T
        inj = NoiseInjector(NoiseConfig(seed=int(rng.integers(1<<30))))
        if nt=="gaussian":
            meas = inj.inject_gaussian_mixture(clean, 0.6)
        elif nt=="impulsive":
            meas = inj.inject_salt_pepper(clean, 0.05)
        else:
            meas = inj.inject_time_varying(clean, 0.6, k_period=100)
        for m,fac in methods.items():
            accum[m].append(run(fac, meas, H, F, track))
    for m in methods:
        print(f"  {m:10s} {np.mean(accum[m]):.4f}")
