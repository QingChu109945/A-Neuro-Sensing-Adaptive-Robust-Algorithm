"""Loaders for the public datasets used to validate the manuscript.

This module exposes tidy NumPy/dict loaders for the two publicly available
emissivity libraries that were downloaded into this directory:

  * MODIS UCSB Emissivity Library  (``modis_ucsb/modis_ucsb_emissivity.csv``)
  * SLUM urban-materials spectral library (``slum/raw/LUMA_SLUM_IR.csv``)

Both are long-wave-infrared (8-14 um) directional-hemispherical emissivity
libraries.  They are used in Section 5.2 of the manuscript to (a) cross-
validate that the synthetic material-database emissivity ranges (Table 1) fall
inside physically measured ranges and (b) provide a real-data benchmark for the
SSM-PINN emissivity inversion.

Provenance is recorded in each dataset's ``SOURCE.json``.  No file is modified
here; loaders are read-only.
"""

import csv
import os
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))

MODIS_CSV = os.path.join(HERE, "modis_ucsb", "modis_ucsb_emissivity.csv")
SLUM_IR_CSV = os.path.join(HERE, "slum", "raw", "LUMA_SLUM_IR.csv")

LWIR_MIN_UM = 8.0
LWIR_MAX_UM = 14.0


def load_modis_ucsb():
    """Load the MODIS UCSB LWIR emissivity library.

    Returns
    -------
    dict with keys:
        ``points``   : list of dicts, one per (material, wavelength) sample
        ``by_label`` : {material_label: [emissivity, ...]}
        ``by_category`` : {paper_category: [emissivity, ...]}
    """
    points = []
    by_label = defaultdict(list)
    by_category = defaultdict(list)
    if not os.path.exists(MODIS_CSV):
        return {"points": [], "by_label": {}, "by_category": {}}
    with open(MODIS_CSV, newline="") as fh:
        for row in csv.DictReader(fh):
            emis = float(row["emissivity"])
            points.append(row)
            by_label[row["material_label"]].append(emis)
            by_category[row["paper_category"]].append(emis)
    return {
        "points": points,
        "by_label": dict(by_label),
        "by_category": dict(by_category),
    }


def load_slum_ir():
    """Load the SLUM long-wave-infrared emissivity library.

    The CSV stores wavelength in the first column and one emissivity column per
    urban surface (74 samples: A001, A002, ...).  We restrict to the 8-14 um
    window and return per-surface emissivity lists.

    Returns
    -------
    dict with keys:
        ``surfaces``   : list of surface ids
        ``by_surface`` : {surface_id: [emissivity, ...]}  (LWIR window only)
        ``wavelengths``: list of retained wavelengths (um)
    """
    if not os.path.exists(SLUM_IR_CSV):
        return {"surfaces": [], "by_surface": {}, "wavelengths": []}
    by_surface = defaultdict(list)
    wavelengths = []
    with open(SLUM_IR_CSV, newline="") as fh:
        reader = csv.reader(fh)
        header = next(reader)
        surface_ids = header[1:]
        for raw in reader:
            if not raw or not raw[0]:
                continue
            try:
                wl = float(raw[0])
            except ValueError:
                continue
            if not (LWIR_MIN_UM <= wl <= LWIR_MAX_UM):
                continue
            wavelengths.append(wl)
            for sid, val in zip(surface_ids, raw[1:]):
                try:
                    emis = float(val)
                except (ValueError, TypeError):
                    continue
                if 0.0 <= emis <= 1.0:
                    by_surface[sid].append(emis)
    return {
        "surfaces": surface_ids,
        "by_surface": dict(by_surface),
        "wavelengths": wavelengths,
    }


def summarize(values):
    """Return (mean, min, max, n) for a list of floats."""
    if not values:
        return (float("nan"), float("nan"), float("nan"), 0)
    n = len(values)
    return (sum(values) / n, min(values), max(values), n)


if __name__ == "__main__":
    modis = load_modis_ucsb()
    slum = load_slum_ir()
    print("MODIS UCSB: %d materials, %d points"
          % (len(modis["by_label"]), len(modis["points"])))
    for label, vals in modis["by_category"].items():
        m, lo, hi, n = summarize(vals)
        print("  [%-22s] eps=%.3f [%.3f, %.3f] n=%d" % (label, m, lo, hi, n))
    print("SLUM IR: %d urban surfaces, %d LWIR wavelengths"
          % (len(slum["by_surface"]), len(slum["wavelengths"])))
    all_slum = [e for vs in slum["by_surface"].values() for e in vs]
    m, lo, hi, n = summarize(all_slum)
    print("  aggregate eps=%.3f [%.3f, %.3f] n=%d" % (m, lo, hi, n))
