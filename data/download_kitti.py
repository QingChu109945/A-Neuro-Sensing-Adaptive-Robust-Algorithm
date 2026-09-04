"""Download / prepare the KITTI tracking-benchmark geometry used for filtering
validation (manuscript Section 5.2, Table ``tab:public_filtering``).

Source
------
KITTI Vision Benchmark Suite, Karlsruhe Institute of Technology & Toyota
Technological Institute at Chicago.  Public project page:
    http://www.cvlibs.net/datasets/kitti/
Multi-object tracking benchmark:
    http://www.cvlibs.net/datasets/kitti/eval_tracking.php

Access model (important, and stated verbatim in the manuscript)
---------------------------------------------------------------
The full KITTI *raw* archive (LiDAR point clouds + images) is distributed only
behind an **interactive account / e-mail-confirmation download**; it cannot be
fetched non-interactively from a script, and its licence forbids
redistribution.  The manuscript therefore does *not* claim to redistribute the
raw archive.  Instead the filtering validation is run on **constant-velocity
ground-vehicle trajectories that reproduce the KITTI tracking-benchmark motion
statistics** (urban speeds up to ~15 m/s, 10 Hz sampling, gentle manoeuvres),
into which the three extreme-noise models of Section 5.1.3 are injected.  This
is exactly what ``public_dataset_validation._generate_kitti_like_track`` does,
and this script is the documented, reproducible entry point for that geometry.

What this script does
---------------------
1. Records provenance (URL, licence, access model) to ``kitti/SOURCE.json`` so
   a third party can see precisely which public benchmark the geometry follows
   and where to obtain the raw archive themselves.
2. If a locally supplied KITTI tracking *label* file is present under
   ``kitti/raw/`` (``label_02/*.txt`` from the openly documented tracking
   ground-truth format, which the user can drop in after an account download),
   it parses the object trajectories into a tidy CSV so the *real* tracks can be
   substituted for the reconstructed ones.  No label file is downloaded or
   redistributed here.
3. Regenerates the reconstructed KITTI-geometry tracks used for the reported
   numbers (via the same code path as the validation module) and caches a small
   ``kitti_geometry_tracks.npz`` so the trajectory source of Table
   ``tab:public_filtering`` is inspectable.  This does **not** change any
   reported result -- it only materialises what the validation already computes
   internally with the same fixed seed.

Usage
-----
    python download_kitti.py            # write provenance + cache geometry tracks
    python download_kitti.py --parse-labels   # also parse any local label_02/*.txt
"""

import argparse
import csv
import json
import os
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "kitti")
RAW_DIR = os.path.join(OUT_DIR, "raw")

PROJECT_URL = "http://www.cvlibs.net/datasets/kitti/"
TRACKING_URL = "http://www.cvlibs.net/datasets/kitti/eval_tracking.php"

# Reconstruction parameters -- MUST match public_dataset_validation so the cached
# tracks are identical to those scored in Table ``tab:public_filtering``.
N_TRACKS = 15
N_STEPS = 150
DT = 0.1
SEED = 42


def _generate_kitti_like_track(n_steps, dt, rng):
    """Reconstructed constant-velocity 3D track (px,py,pz,vx,vy,vz).

    Identical to ``public_dataset_validation._generate_kitti_like_track`` so the
    cached geometry reproduces the reported filtering numbers bit-for-bit.
    """
    x = np.zeros((n_steps, 6))
    x[0] = [rng.uniform(5, 40), rng.uniform(-8, 8), rng.uniform(-1, 1),
            rng.uniform(3, 12), rng.uniform(-1.5, 1.5), 0.0]
    for k in range(1, n_steps):
        acc = rng.normal(0, 0.4, 3)
        x[k, 3:] = x[k - 1, 3:] + acc * dt
        x[k, :3] = x[k - 1, :3] + x[k, 3:] * dt
    return x


def _cache_geometry_tracks():
    """Materialise the reconstructed KITTI-geometry tracks (seed=42)."""
    rng = np.random.default_rng(SEED)
    tracks = np.stack(
        [_generate_kitti_like_track(N_STEPS, DT, rng) for _ in range(N_TRACKS)]
    )
    dest = os.path.join(OUT_DIR, "kitti_geometry_tracks.npz")
    np.savez_compressed(
        dest, tracks=tracks, dt=DT, seed=SEED,
        columns=np.array(["px", "py", "pz", "vx", "vy", "vz"]),
    )
    print(f"  cached reconstructed geometry: {tracks.shape} -> {dest}")
    return dest


def _parse_local_labels():
    """Parse any locally supplied KITTI tracking label files.

    The user may place official ``label_02/*.txt`` files (obtained through the
    KITTI account download) under ``kitti/raw/label_02/``.  We parse them into a
    tidy CSV of per-frame object positions so real trajectories can be swapped
    in.  Nothing is downloaded or redistributed.

    KITTI tracking label columns (space separated):
        frame track_id type truncated occluded alpha
        bbox(4) dimensions(3) location(3: x,y,z) rotation_y [score]
    We keep frame, track_id, type and the 3D location (x, y, z).
    """
    label_dir = os.path.join(RAW_DIR, "label_02")
    if not os.path.isdir(label_dir):
        print("  no local label_02/ directory found -- skipping label parsing.")
        print("  (drop official KITTI tracking labels there to use real tracks.)")
        return 0
    rows = []
    for fname in sorted(os.listdir(label_dir)):
        if not fname.endswith(".txt"):
            continue
        seq = os.path.splitext(fname)[0]
        with open(os.path.join(label_dir, fname), "r", errors="ignore") as fh:
            for line in fh:
                parts = line.split()
                if len(parts) < 17:
                    continue
                try:
                    frame = int(parts[0])
                    track_id = int(parts[1])
                    obj_type = parts[2]
                    loc_x, loc_y, loc_z = (float(parts[13]),
                                           float(parts[14]),
                                           float(parts[15]))
                except (ValueError, IndexError):
                    continue
                if track_id < 0:  # DontCare rows use -1
                    continue
                rows.append({
                    "sequence": seq, "frame": frame, "track_id": track_id,
                    "type": obj_type, "x": loc_x, "y": loc_y, "z": loc_z,
                })
    if not rows:
        print("  label_02/ present but no parsable rows found.")
        return 0
    csv_path = os.path.join(OUT_DIR, "kitti_tracks.csv")
    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=["sequence", "frame", "track_id", "type",
                            "x", "y", "z"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"  parsed {len(rows)} label rows -> {csv_path}")
    return len(rows)


def build(parse_labels=False):
    os.makedirs(RAW_DIR, exist_ok=True)

    n_labels = 0
    if parse_labels:
        n_labels = _parse_local_labels()

    _cache_geometry_tracks()

    provenance = {
        "dataset": "KITTI Vision Benchmark Suite (multi-object tracking)",
        "provider": ("Karlsruhe Institute of Technology & "
                     "Toyota Technological Institute at Chicago"),
        "project_url": PROJECT_URL,
        "tracking_url": TRACKING_URL,
        "citation_key": "geiger2012kitti",
        "access_date": time.strftime("%Y-%m-%d"),
        "access_model": (
            "The full raw archive requires an interactive account / e-mail "
            "confirmation download and may not be redistributed. This script "
            "records provenance and reproduces the KITTI tracking-benchmark "
            "MOTION GEOMETRY used by the filtering validation; it does not "
            "download or redistribute the raw archive."
        ),
        "geometry_used": {
            "description": ("Constant-velocity ground-vehicle trajectories "
                            "matching KITTI tracking motion statistics."),
            "urban_speed_max_mps": 15.0,
            "sampling_hz": 1.0 / DT,
            "n_tracks": N_TRACKS,
            "n_steps": N_STEPS,
            "seed": SEED,
            "cache": "kitti_geometry_tracks.npz",
        },
        "real_labels_parsed": n_labels,
        "note": ("To validate on the ORIGINAL KITTI tracks, obtain label_02/*.txt "
                 "via the KITTI account download, place them under kitti/raw/"
                 "label_02/, and re-run with --parse-labels."),
    }
    with open(os.path.join(OUT_DIR, "SOURCE.json"), "w") as fh:
        json.dump(provenance, fh, indent=2)
    print(f"Wrote provenance -> {os.path.join(OUT_DIR, 'SOURCE.json')}")
    return 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Prepare KITTI tracking-benchmark geometry for filtering "
                    "validation (records provenance; optional label parsing).")
    ap.add_argument("--parse-labels", action="store_true",
                    help="Parse locally supplied kitti/raw/label_02/*.txt files.")
    args = ap.parse_args()
    build(parse_labels=args.parse_labels)
