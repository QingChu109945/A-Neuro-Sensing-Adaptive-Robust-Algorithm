"""Diagnostic 2: bounded emissivity-bearing observables.

Rebuild the forward model but replace the unbounded r_thermal = I_thermal/L_bb
ratio (which explodes when L_bb -> 0) with *bounded* echo-fraction observables
that a dual-mode laser-radar actually reports:

    f_active  = I_reflect_meas / I_echo_meas     in [0,1]  (active/reflected share)
    f_thermal = I_thermal_meas / I_echo_meas     in [0,1]  (passive/emitted share)

and a temperature-normalised thermal radiance
    g_thermal = I_thermal_meas / (sigma T^4)     (bounded because we drop the 1/(1+(D/1e3)^2) geometric attenuation that made L_bb tiny)

We check OLS + per-feature correlation again.
"""
import os, sys
import numpy as np

STEF = 5.670374419e-8
HERE = os.path.dirname(os.path.abspath(__file__))
_PKG_ROOT = os.path.dirname(os.path.dirname(HERE))
if _PKG_ROOT not in sys.path:
    sys.path.insert(0, _PKG_ROOT)
from experiment_system.data.public_dataset_loaders import load_modis_ucsb, load_slum_ir


def build(seed=42):
    rng = np.random.default_rng(seed)
    modis = load_modis_ucsb(); slum = load_slum_ir()
    eps = [float(r["emissivity"]) for r in modis["points"]]
    for sid, vals in slum["by_surface"].items():
        eps.extend(vals[:: max(1, len(vals) // 60)])
    eps = np.clip(np.asarray(eps, float), 1e-2, 0.99)
    n = len(eps)
    D = rng.uniform(100, 5000, n)
    th = np.deg2rad(rng.uniform(0, 75, n))
    T = rng.uniform(300, 900, n)
    rho = np.clip(1 - eps, 1e-2, 0.99)
    geom = np.cos(th)                              # keep angle dependence, drop 1/(1+..) attenuation
    Lbb = STEF * T ** 4                            # blackbody radiance (no attenuation -> never ~0)
    I_thermal = eps * Lbb * geom
    I_reflect = rho * 1.0 * geom * (Lbb * 1e-3)    # active return on comparable scale
    I_echo = I_thermal + I_reflect
    noise = rng.normal(0, 0.03, n) * I_echo.mean()
    I_echo_m = I_echo + noise
    I_reflect_m = np.clip(I_reflect + 0.5 * noise, 0, None)
    I_thermal_m = np.clip(I_echo_m - I_reflect_m, 0, None)
    f_active = np.clip(I_reflect_m / (I_echo_m + 1e-12), 0.0, 1.0)
    f_thermal = np.clip(I_thermal_m / (I_echo_m + 1e-12), 0.0, 1.0)
    g_thermal = np.clip(I_thermal_m / (Lbb * geom + 1e-12), 0.0, 2.0)
    X = np.stack([D, np.rad2deg(th), T, f_active, f_thermal, g_thermal], 1)
    return X, eps


def r2(y, p):
    return 1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2)


X, eps = build()
Xs = (X - X.mean(0)) / (X.std(0) + 1e-10)
rng = np.random.default_rng(0)
idx = rng.permutation(len(eps))
Xs, eps = Xs[idx], eps[idx]
tr = int(0.85 * len(eps))
Xtr, ytr, Xte, yte = Xs[:tr], eps[:tr], Xs[tr:], eps[tr:]
print("feature max:", np.round(Xs.max(0), 2), " min:", np.round(Xs.min(0), 2))
A = np.hstack([Xtr, np.ones((len(Xtr), 1))])
coef = np.linalg.solve(A.T @ A + 1e-2 * np.eye(A.shape[1]), A.T @ ytr)
pred = np.hstack([Xte, np.ones((len(Xte), 1))]) @ coef
print("[Ridge] test R2 = %.4f RMSE=%.4f" % (r2(yte, pred), np.sqrt(np.mean((yte - pred) ** 2))))
names = ["distance", "angle", "temperature", "f_active", "f_thermal", "g_thermal"]
for i, nm in enumerate(names):
    print("  %-12s corr=%.3f" % (nm, np.corrcoef(Xte[:, i], yte)[0, 1]))
