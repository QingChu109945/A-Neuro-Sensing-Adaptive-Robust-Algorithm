"""Download the SLUM (Spectral Library of Impervious Urban Materials) dataset.

Source
------
Kotthaus, S., Smith, T.E.L., Wooster, M.J., Grimmond, C.S.B. (2014).
"Derivation of an urban materials spectral library through emittance and
reflectance spectroscopy." ISPRS Journal of Photogrammetry and Remote
Sensing 94, 194-212.  Data hosted on Zenodo:
    https://zenodo.org/record/4263841   (DOI: 10.5281/zenodo.4263842)

The archive contains two CSV spectral libraries:
  * LUMA_SLUM_IR.csv  - long-wave infrared emissivity spectra
  * LUMA_SLUM_SW.csv  - short-wave reflectance spectra

For the non-cooperative target inversion task we use the LWIR emissivity file,
which lists 74 impervious urban surfaces (asphalt, brick, concrete, metal,
paint, etc.).  This script downloads the raw CSV(s), caches them under raw/,
and writes a SOURCE.json provenance file.

Usage
-----
    python download_slum.py
"""

import json
import os
import sys
import time

RECORD_API = "https://zenodo.org/api/records/4263842"

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "slum")
RAW_DIR = os.path.join(OUT_DIR, "raw")

WANTED = {"LUMA_SLUM_IR.csv", "LUMA_SLUM_SW.csv"}


def build():
    import requests

    os.makedirs(RAW_DIR, exist_ok=True)
    try:
        meta = requests.get(RECORD_API, timeout=30).json()
    except Exception as exc:  # pragma: no cover - network dependent
        sys.stderr.write(f"Could not query Zenodo: {exc}\n")
        return 0

    obtained = []
    for f in meta.get("files", []):
        key = f.get("key")
        if key not in WANTED:
            continue
        dest = os.path.join(RAW_DIR, key)
        if not os.path.exists(dest):
            url = f.get("links", {}).get("self")
            try:
                resp = requests.get(url, timeout=120)
                if resp.status_code == 200:
                    with open(dest, "wb") as fh:
                        fh.write(resp.content)
            except Exception as exc:  # pragma: no cover
                sys.stderr.write(f"  failed {key}: {exc}\n")
                continue
        if os.path.exists(dest):
            obtained.append(key)
            print(f"  cached {key} ({os.path.getsize(dest)} bytes)")

    provenance = {
        "dataset": "SLUM - Spectral Library of Impervious Urban Materials",
        "authors": "Kotthaus, Smith, Wooster, Grimmond",
        "publication": ("ISPRS Journal of Photogrammetry and Remote Sensing, "
                        "94:194-212, 2014"),
        "doi_paper": "10.1016/j.isprsjprs.2014.05.005",
        "doi_data": "10.5281/zenodo.4263842",
        "url": "https://zenodo.org/record/4263841",
        "access_date": time.strftime("%Y-%m-%d"),
        "license": meta.get("metadata", {}).get("license", {}),
        "files_obtained": obtained,
        "usage": ("LUMA_SLUM_IR.csv provides long-wave infrared emissivity "
                  "spectra for 74 impervious urban surfaces; used to cross-"
                  "validate the paper's synthetic emissivity ranges."),
    }
    with open(os.path.join(OUT_DIR, "SOURCE.json"), "w") as fh:
        json.dump(provenance, fh, indent=2, default=str)
    return len(obtained)


if __name__ == "__main__":
    n = build()
    print(f"Obtained {n} SLUM file(s).")
