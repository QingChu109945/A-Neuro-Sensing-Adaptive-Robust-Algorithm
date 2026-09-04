"""Quick diagnostic: why do all filters report identical RMSE?"""
import os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(HERE))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from experiment_system.filtering import (
    create_ekf_filter, create_ukf_filter, create_aekf_filter,
    create_rukf_filter, create_ns_arkf_filter,
)

def make_track(n, dt=0.1, rng=None):
    rng = rng or np.random.default_rng(0)
    x = np.zeros((n, 6))
    x[0] = [20, 3, 0.5, 8, 0.5, 0.0]
    for k in range(1, n):
        acc = rng.normal(0, 0.4, 3)
        x[k, 3:] = x[k-1, 3:] + acc*dt
        x[k, :3] = x[k-1, :3] + x[k, 3:]*dt
    return x

H = np.zeros((4, 6))
H[0,0]=H[1,1]=H[2,2]=H[3,3]=1.0
dt = 0.1
F = np.eye(6)
for i in range(3):
    F[i, i+3] = dt

rng = np.random.default_rng(1)
track = make_track(150, dt, rng)
clean = (H @ track.T).T
meas = clean + rng.normal(0, 0.6, clean.shape)

for name, fac in [("EKF",create_ekf_filter),("UKF",create_ukf_filter),
                  ("AEKF",create_aekf_filter),("RUKF",create_rukf_filter),
                  ("NS-ARKF",create_ns_arkf_filter)]:
    filt = fac(dim_x=6, dim_z=4)
    # inspect what state attribute exists
    has_xhat = hasattr(filt, "x_hat")
    has_uif = hasattr(filt, "uif")
    est = np.zeros((150,6))
    for k in range(150):
        filt.predict(F)
        filt.update(meas[k], H)
        if has_uif:
            est[k] = np.asarray(filt.uif.x_hat).ravel()[:6]
        else:
            est[k] = np.asarray(filt.x_hat).ravel()[:6]
    rmse = float(np.sqrt(np.mean((track[:, :3] - est[:, :3])**2)))
    print(f"{name:10s} has_xhat={has_xhat} has_uif={has_uif} RMSE={rmse:.4f} "
          f"first_est={est[0,:3].round(2)} last_est={est[-1,:3].round(2)} "
          f"last_true={track[-1,:3].round(2)}")
