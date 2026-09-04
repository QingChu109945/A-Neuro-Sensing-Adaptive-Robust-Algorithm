"""Diagnostic: is the real-data inversion task well-posed?

We rebuild the dataset exactly as public_dataset_validation._build_real_emissivity_dataset
and then fit a plain ridge/least-squares baseline. If a linear model already
recovers emissivity with high R^2, the task is well-posed and any negative R^2
from the neural factories is a model/training issue, not a data issue.
"""
import os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
_PKG_ROOT = os.path.dirname(os.path.dirname(HERE))
if _PKG_ROOT not in sys.path:
    sys.path.insert(0, _PKG_ROOT)

from experiment_system.data.public_dataset_validation import _build_real_emissivity_dataset

ds = _build_real_emissivity_dataset(seed=42)
Xtr, ytr = ds["X_train"], ds["y_train"][:, 0]
Xte, yte = ds["X_test"], ds["y_test"][:, 0]

print("n_train=%d n_test=%d n_feat=%d" % (Xtr.shape[0], Xte.shape[0], Xtr.shape[1]))
print("feature mean (train):", np.round(Xtr.mean(0), 3))
print("feature std  (train):", np.round(Xtr.std(0), 3))
print("feature max  (train):", np.round(Xtr.max(0), 2))
print("feature min  (train):", np.round(Xtr.min(0), 2))
print("emissivity target range:", round(float(yte.min()), 3), round(float(yte.max()), 3))

# ---- plain OLS with bias ----
def r2(y, p):
    ss = np.sum((y - p) ** 2)
    st = np.sum((y - y.mean()) ** 2)
    return 1.0 - ss / st

A = np.hstack([Xtr, np.ones((len(Xtr), 1))])
coef, *_ = np.linalg.lstsq(A, ytr, rcond=None)
pred = np.hstack([Xte, np.ones((len(Xte), 1))]) @ coef
print("\n[OLS] test R2 = %.4f  RMSE = %.4f" % (r2(yte, pred), np.sqrt(np.mean((yte - pred) ** 2))))

# ridge
for lam in (1e-3, 1e-1, 1.0):
    n = A.shape[1]
    coef = np.linalg.solve(A.T @ A + lam * np.eye(n), A.T @ ytr)
    pred = np.hstack([Xte, np.ones((len(Xte), 1))]) @ coef
    print("[Ridge lam=%.3g] test R2 = %.4f  RMSE = %.4f" % (lam, r2(yte, pred), np.sqrt(np.mean((yte - pred) ** 2))))

# correlation of each feature with target
print("\nper-feature corr with emissivity (test):")
names = ["distance", "angle", "temperature", "r_active", "r_thermal"]
for i, nm in enumerate(names):
    c = np.corrcoef(Xte[:, i], yte)[0, 1]
    print("  %-12s r=%.3f" % (nm, c))
