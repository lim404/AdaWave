"""Dataset path resolution for the AdaWave evaluation harness.

All dataset locations derive from a single root, resolved in this order:

1. ``$ADAWAVE_DATA_ROOT``
2. ``<repo>/data`` (default)

ScanNet++ often lives on a separate volume, so it additionally honours
``$ADAWAVE_SCANNETPP_ROOT``.

Datasets are *not* redistributed with this repository; see README.md for
the download location and terms of each. Paths are resolved lazily, so
importing this module never requires the data to be present -- call
``require()`` at the point of use to get an actionable error instead of a
bare ``FileNotFoundError`` from deep inside a loader.
"""
import os
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent


def data_root() -> Path:
    return Path(os.environ.get("ADAWAVE_DATA_ROOT", _REPO / "data"))


def _scannetpp_root() -> Path:
    env = os.environ.get("ADAWAVE_SCANNETPP_ROOT")
    return Path(env) if env else data_root() / "ScanNet++" / "data"


# --- dataset locations -------------------------------------------------
# Resolved at call time so that changing the environment variable inside a
# session (or in tests) takes effect.

def vaihingen_dir() -> Path:
    return data_root() / "Vaihingen" / "3DLabeling"


def vaihingen_train_pts() -> Path:
    return vaihingen_dir() / "Vaihingen3D_Traininig.pts"


def vaihingen_eval_pts_gz() -> Path:
    return vaihingen_dir() / "Vaihingen3D_EVAL_WITH_REF.pts.gz"


def dales_root() -> Path:
    return data_root() / "DALESObjects"


def modelnet40_root() -> Path:
    return data_root() / "ModelNet40"


def punet_root() -> Path:
    return data_root() / "PUNet_denoise" / "PUNet" / "pointclouds"


def scannetpp_root() -> Path:
    return _scannetpp_root()


# --- helpers -----------------------------------------------------------

_HINTS = {
    "Vaihingen": "ISPRS Vaihingen 3D — https://www.isprs.org/education/benchmarks/UrbanSemLab/ (registration required)",
    "DALESObjects": "DALES Objects — https://udayton.edu/engineering/research/centers/vision_lab/research/was_data_analysis_and_processing/dale.php",
    "ModelNet40": "ModelNet40 — https://modelnet.cs.princeton.edu/",
    "PUNet_denoise": "PU-Net test set — as distributed with score-denoise (Luo & Hu, ICCV 2021)",
    "ScanNet++": "ScanNet++ — https://kaldir.vc.in.tum.de/scannetpp/ (registration required)",
}


def require(path, what: str = ""):
    """Return ``path`` if it exists, else raise with a setup hint."""
    p = Path(path)
    if p.exists():
        return p
    hint = ""
    for key, text in _HINTS.items():
        if key in str(p):
            hint = f"\n  Obtain it from: {text}"
            break
    raise FileNotFoundError(
        f"Missing dataset file{': ' + what if what else ''}\n"
        f"  Expected at: {p}\n"
        f"  Data root:   {data_root()}  "
        f"({'from $ADAWAVE_DATA_ROOT' if 'ADAWAVE_DATA_ROOT' in os.environ else 'default'})"
        f"{hint}\n"
        f"  Set ADAWAVE_DATA_ROOT to the directory holding the datasets, or "
        f"place them under {data_root()}/ following the layout in README.md."
    )
