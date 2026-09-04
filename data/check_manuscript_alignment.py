"""Cross-check that public_validation_results.json matches the manuscript tables.

Reads the JSON produced by public_dataset_validation.py and compares every
value against the numbers hard-coded in the manuscript Tables
``tab:public_filtering`` and ``tab:public_inversion`` (CAL0828.tex).  Prints a
per-cell PASS/FAIL and exits non-zero if any mismatch exceeds the display
tolerance the manuscript rounds to.

Usage
-----
    python -m experiment_system.data.check_manuscript_alignment
    # check against the NumPy/CPU results instead of the default GPU JSON:
    python -m experiment_system.data.check_manuscript_alignment \
        --json public_validation_results.json
"""

import argparse
import json
import os
import sys
from decimal import Decimal, ROUND_HALF_UP

HERE = os.path.dirname(os.path.abspath(__file__))
# The manuscript Table tab:public_inversion now reports the GPU-autograd
# SSM-PINN, so validate against the GPU results by default.
DEFAULT_JSON = "public_validation_results_gpu.json"

# ---- Manuscript Table tab:public_filtering (3D position RMSE, m) ----------- #
# columns: gaussian, impulsive, time_varying, average  (rounded to 3 dp in tex)
MS_FILTERING = {
    "EKF":     (0.778, 11.562, 0.322, 4.221),
    "UKF":     (0.778, 11.562, 0.322, 4.221),
    "CKF":     (0.778, 11.562, 0.322, 4.221),
    "AEKF":    (0.592, 197.651, 0.269, 66.170),
    "PSO-EKF": (0.592, 197.651, 0.269, 66.170),
    "GA-UKF":  (0.778, 11.562, 0.322, 4.221),
    "DeepKF":  (0.778, 11.561, 0.322, 4.220),
    "RUKF":    (0.555, 5.416, 0.320, 2.097),
    "NS-ARKF": (0.459, 4.774, 0.316, 1.850),
}

# ---- Manuscript Table tab:public_inversion -------------------------------- #
# columns: emissivity_rmse (3 dp), r2 (2 dp), kirchhoff_violation_% (1 dp)
MS_INVERSION = {
    "FC-NN":                (0.322, -3.01, 92.8),
    "PINN-FC":              (0.407, -5.40, 13.1),
    "Transformer":          (0.366, -4.18, 0.0),
    "S4-Model":             (0.373, -4.40, 0.0),
    "ResNet":               (0.097, 0.640, 0.0),
    "Mamba":                (0.096, 0.645, 0.0),
    "Hard-Constraint PINN": (0.407, -5.42, 0.0),
    "SSM-PINN":             (0.047, 0.914, 0.0),
}
# SSM-PINN PICP_95 reported as 1.000 (GPU-autograd backend)
MS_SSM_PICP = 1.000


def _round_half_up(x, ndec):
    """Round like a manuscript does (half away from zero), not banker's."""
    q = Decimal(1).scaleb(-ndec)  # 10**-ndec
    return float(Decimal(repr(x)).quantize(q, rounding=ROUND_HALF_UP))


def _cmp(label, got, want, tol):
    # The manuscript prints each cell rounded (half-up) to a fixed number of
    # decimals, so a JSON value may legitimately sit on a rounding half-boundary
    # (e.g. 0.0965 displayed as 0.097).  Accept the cell if the JSON value rounds
    # to the printed value at the printed precision *or* falls within `tol`.
    ndec = len(str(want).split(".")[1]) if "." in str(want) else 0
    ok = _round_half_up(got, ndec) == want or abs(got - want) <= tol
    flag = "PASS" if ok else "FAIL"
    print(f"    [{flag}] {label:32s} got={got:<12.4f} tex={want}")
    return ok


def main():
    ap = argparse.ArgumentParser(description="Check manuscript/JSON alignment.")
    ap.add_argument("--json", default=DEFAULT_JSON,
                    help="results JSON under data/ (default: GPU results)")
    args = ap.parse_args()
    json_path = args.json if os.path.isabs(args.json) else os.path.join(HERE, args.json)
    with open(json_path) as fh:
        data = json.load(fh)

    all_ok = True

    print("== Filtering (Table tab:public_filtering) ==")
    ftab = data["filtering_validation"]["table"]
    for name, (g, i, tv, avg) in MS_FILTERING.items():
        row = ftab.get(name, {})
        all_ok &= _cmp(f"{name}.gaussian", row.get("gaussian", float('nan')), g, 5e-4)
        all_ok &= _cmp(f"{name}.impulsive", row.get("impulsive", float('nan')), i, 5e-3)
        all_ok &= _cmp(f"{name}.time_varying", row.get("time_varying", float('nan')), tv, 5e-4)
        all_ok &= _cmp(f"{name}.average", row.get("average", float('nan')), avg, 5e-3)

    print("== Inversion (Table tab:public_inversion) ==")
    itab = data["inversion_validation"]["table"]
    for name, (rmse, r2, viol_pct) in MS_INVERSION.items():
        row = itab.get(name, {})
        all_ok &= _cmp(f"{name}.emissivity_rmse", row.get("emissivity_rmse", float('nan')), rmse, 5e-4)
        all_ok &= _cmp(f"{name}.r2", row.get("r2", float('nan')), r2, 5e-3)
        got_viol = row.get("kirchhoff_violation_rate", float('nan')) * 100.0
        all_ok &= _cmp(f"{name}.kirchhoff_viol_%", got_viol, viol_pct, 5e-2)
    ssm_picp = itab.get("SSM-PINN", {}).get("picp", float('nan'))
    all_ok &= _cmp("SSM-PINN.PICP_95", ssm_picp, MS_SSM_PICP, 5e-4)

    print()
    if all_ok:
        print("RESULT: all manuscript cells match %s." % os.path.basename(json_path))
        return 0
    print("RESULT: mismatches found (see FAIL rows above).")
    return 1


if __name__ == "__main__":
    sys.exit(main())
