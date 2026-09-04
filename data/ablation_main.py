"""Fine-grained NS-ARKF ablation on the MAIN experiment protocol.

Uses the exact trajectory, measurement model and noise families of
comparison_experiments.FilteringComparison so the ablation numbers are
consistent with Table 7 (filtering) and Table 9 (ablation).  Cumulative
configurations isolate the marginal contribution of each mechanism:

    UIF                     : unknown-input filter only
    + IFHBFNN               : add fuzzy hyper-RBF interference estimator
    + robust gate (IGG-III) : add robust innovation gate (impulse rejection)
    + HBKFO (full)          : add meta-heuristic covariance adaptation

Reports overall-RMSE per noise family and the average, matching the metric
used for Table 7.
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(HERE))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from experiment_system.comparison_experiments import (
    FilteringComparison, ExperimentConfig,
)
from experiment_system.evaluation import compute_filtering_metrics
from experiment_system.filtering import NSARKF
from experiment_system.data_generator import NoiseInjector, NoiseConfig

cfg = ExperimentConfig()
comp = FilteringComparison(cfg)
true_states, clean, F, H = comp._generate_true_states()

noise_specs = ["gaussian", "mixture", "impulsive", "time_varying"]


def make_meas(nt):
    comp.rng = np.random.default_rng(cfg.seed)
    if nt == "time_varying":
        return comp._inject_time_varying_noise(clean)
    spec = comp.NOISE_TYPES[nt]
    nc = NoiseConfig(noise_type=spec["type"], level=spec["level"], seed=cfg.seed)
    return NoiseInjector(nc).inject(clean)


def run_variant(use_if, gate, use_hb, meas):
    filt = NSARKF(cfg.dim_x, cfg.dim_z, use_ifhbfnn=use_if, use_hbkfo=use_hb)
    filt.robust_gate = gate
    states, covs = [], []
    for z in meas:
        filt.predict(F)
        filt.update(z, H)
        s = filt.get_state()
        states.append(s.x_hat)
        covs.append(s.P)
    m = compute_filtering_metrics(true_states, np.array(states), np.array(covs))
    return float(m.overall_rmse)


variants = [
    ("UIF", False, False, False),
    ("+ IFHBFNN", True, False, False),
    ("+ robust gate (IGG-III)", True, True, False),
    ("+ HBKFO (full NS-ARKF)", True, True, True),
]

table = {}
for name, ui, gate, hb in variants:
    row = {}
    for nt in noise_specs:
        row[nt] = run_variant(ui, gate, hb, make_meas(nt))
    row["average"] = float(np.mean([row[nt] for nt in noise_specs]))
    table[name] = row

print("=== NS-ARKF fine-grained ablation (overall RMSE, main protocol) ===")
prev = None
for name, *_ in variants:
    row = table[name]
    delta = "" if prev is None else "  (Delta avg %+.3f)" % (row["average"] - prev)
    print("  %-26s g=%.3f mix=%.3f imp=%.3f tv=%.3f avg=%.3f%s"
          % (name, row["gaussian"], row["mixture"], row["impulsive"],
             row["time_varying"], row["average"], delta))
    prev = row["average"]

with open(os.path.join(HERE, "ablation_main_results.json"), "w") as fh:
    json.dump(table, fh, indent=2)
print("Wrote ablation_main_results.json")
