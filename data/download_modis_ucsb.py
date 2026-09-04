"""Download the MODIS UCSB Emissivity Library (public dataset) into data/modis_ucsb.

Source
------
MODIS UCSB Emissivity Library, Z.-M. Wan Group, ICESS, University of
California, Santa Barbara. Public URL:
    https://icess.eri.ucsb.edu/modis/EMIS/html/em.html

The library provides laboratory directional-hemispherical emissivity spectra
measured with a TIR (FTIR) spectrometer for natural and manmade materials.
For the non-cooperative target measurement study we retain the *manmade*
metallic/painted materials whose emissivity ranges overlap the paper's
material database.

Raw file format (``.prn``)
--------------------------
ASCII files with a ``#``-commented header followed by three whitespace
separated columns:

    #X(Micrometer)   #X(cm-1)   #Yvalue
    14.5588          686.869    .88915
    ...

Column 1 is wavelength (um), column 2 is wavenumber (cm^-1), column 3 is the
absolute directional-hemispherical emissivity.  This is emissivity directly
(``#DATATYPE = DIRECTIONAL HEMISPHERICAL EMISSIVITY SPECTRUM``); no Kirchhoff
conversion is required.

This script downloads each spectrum, restricts it to the long-wave-infrared
window (8-14 um) used by the paper's inversion task, and writes a tidy
``modis_ucsb_emissivity.csv`` plus a ``SOURCE.json`` provenance file. Raw
downloads are cached under ``raw/``; re-running is idempotent.

Usage
-----
    python download_modis_ucsb.py            # download + build CSV
    python download_modis_ucsb.py --offline  # rebuild CSV from cached .prn
"""

import argparse
import csv
import json
import os
import sys
import time

BASE = "https://icess.eri.ucsb.edu/modis/EMIS"
IMAGES = BASE + "/images"

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "modis_ucsb")
RAW_DIR = os.path.join(OUT_DIR, "raw")

# Manmade metallic / painted / ceramic spectra whose emissivity overlaps the
# paper's database.  Labels come from each file's #TITLE header; the mapped
# category is used only for cross-validation against Table 1 of the paper.
MANMADE_SPECTRA = [
    # Painted / anodized aluminium and plastic (high-emissivity coatings)
    ("alumdisk.prn", "Black painted anodized aluminum disk", "Anti-Infrared Coating"),
    ("alumgrte.prn", "Painted aluminum grate", "Anti-Infrared Coating"),
    ("disk144.prn", "Computer disk aluminum substrate", "Aluminum Alloy"),
    ("blkplsttp.prn", "Black plastic box top", "Polyurethane Coating"),
    # Aluminium-oxide / gold-coated sandpapers and Krylon paints (metallic + coating)
    ("alumsnd1.prn", "Aluminum-oxide sandpaper (grit 1)", "Aluminum Alloy"),
    ("alumsnd3.prn", "Aluminum-oxide sandpaper (grit 3)", "Aluminum Alloy"),
    ("goldsnd1.prn", "Gold-coated sandpaper", "Anti-Optical Coating"),
    ("krylon1.prn", "Krylon spray paint (type 1)", "Polyurethane Coating"),
    ("krylon2.prn", "Krylon spray paint (type 2)", "Polyurethane Coating"),
    # Ceramic / masonry (ceramic-coating analogues, high emissivity)
    ("clybrkcm.prn", "Clay construction brick", "Ceramic Coating"),
    ("rdconbrk.prn", "Red construction brick", "Ceramic Coating"),
    ("ustile2p.prn", "Ceramic tile", "Ceramic Coating"),
    ("beigetle.prn", "Beige ceramic tile", "Ceramic Coating"),
    ("masonnat.prn", "Natural masonry", "Ceramic Coating"),
    # Pavement / stone (rough dielectric, high emissivity)
    ("asphalt.prn", "Asphalt pavement", "Ceramic Coating"),
    ("flagbuck.prn", "Buckingham flagstone", "Ceramic Coating"),
]

LWIR_MIN_UM = 8.0
LWIR_MAX_UM = 14.0


def _fetch(url, dest, retries=3):
    import requests

    for attempt in range(retries):
        try:
            resp = requests.get(url, timeout=30)
            if resp.status_code == 200 and resp.content:
                with open(dest, "wb") as fh:
                    fh.write(resp.content)
                return True
            return False
        except Exception as exc:  # pragma: no cover - network dependent
            sys.stderr.write(f"  retry {attempt + 1}/{retries} for {url}: {exc}\n")
            time.sleep(1.5)
    return False


def _title(path):
    with open(path, "r", errors="ignore") as fh:
        for line in fh:
            if line.upper().startswith("#TITLE"):
                return line.split("=", 1)[-1].strip()
    return None


def _parse_prn(path):
    """Parse a UCSB .prn spectrum.

    Returns list of (wavelength_um, emissivity).  The three data columns are
    wavelength(um), wavenumber(cm^-1) and absolute emissivity; header lines
    start with ``#`` and are skipped.
    """
    rows = []
    with open(path, "r", errors="ignore") as fh:
        for line in fh:
            if line.lstrip().startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 3:
                continue
            try:
                wl = float(parts[0])
                emis = float(parts[2])
            except ValueError:
                continue
            rows.append((wl, emis))
    return rows


def build(offline=False):
    os.makedirs(RAW_DIR, exist_ok=True)
    records = []
    obtained = []
    for prn, fallback_label, category in MANMADE_SPECTRA:
        dest = os.path.join(RAW_DIR, prn)
        if not os.path.exists(dest) and not offline:
            if not _fetch(f"{IMAGES}/{prn}", dest):
                sys.stderr.write(f"  skip (unavailable): {prn}\n")
                continue
        if not os.path.exists(dest):
            continue
        label = _title(dest) or fallback_label
        spectrum = _parse_prn(dest)
        lwir = [
            (wl, emis)
            for wl, emis in spectrum
            if LWIR_MIN_UM <= wl <= LWIR_MAX_UM and 0.0 <= emis <= 1.0
        ]
        if not lwir:
            continue
        obtained.append(prn)
        for wl, emis in lwir:
            records.append(
                {
                    "material_label": label,
                    "paper_category": category,
                    "wavelength_um": round(wl, 4),
                    "emissivity": round(emis, 5),
                    "reflectivity": round(1.0 - emis, 5),  # Kirchhoff, opaque
                    "source_file": prn,
                }
            )

    if not records:
        sys.stderr.write(
            "No spectra retrieved. Run online once to populate the raw/ cache.\n"
        )
        return 0

    csv_path = os.path.join(OUT_DIR, "modis_ucsb_emissivity.csv")
    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "material_label",
                "paper_category",
                "wavelength_um",
                "emissivity",
                "reflectivity",
                "source_file",
            ],
        )
        writer.writeheader()
        writer.writerows(records)

    summary = {}
    for r in records:
        summary.setdefault(r["material_label"], []).append(r["emissivity"])
    summary_rows = {
        k: {
            "emissivity_mean": round(sum(v) / len(v), 4),
            "emissivity_min": round(min(v), 4),
            "emissivity_max": round(max(v), 4),
            "n_points": len(v),
        }
        for k, v in summary.items()
    }

    provenance = {
        "dataset": "MODIS UCSB Emissivity Library",
        "provider": "Z.-M. Wan Group, ICESS, University of California, Santa Barbara",
        "url": f"{BASE}/html/em.html",
        "access_date": time.strftime("%Y-%m-%d"),
        "license": "Publicly distributed for research use (UCSB ICESS)",
        "spectral_window_um": [LWIR_MIN_UM, LWIR_MAX_UM],
        "datatype": "Directional hemispherical emissivity (FTIR)",
        "n_materials": len(obtained),
        "n_points": len(records),
        "files_obtained": obtained,
        "per_material_summary": summary_rows,
    }
    with open(os.path.join(OUT_DIR, "SOURCE.json"), "w") as fh:
        json.dump(provenance, fh, indent=2)

    print(f"Wrote {len(records)} points for {len(obtained)} materials -> {csv_path}")
    for label, s in summary_rows.items():
        print(f"  {label:42s} eps={s['emissivity_mean']:.3f} "
              f"[{s['emissivity_min']:.3f}, {s['emissivity_max']:.3f}] "
              f"n={s['n_points']}")
    return len(records)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true",
                    help="Rebuild CSV from cached raw/ .prn files only.")
    args = ap.parse_args()
    build(offline=args.offline)
