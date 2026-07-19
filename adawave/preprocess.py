"""Module 1: Data loading and preprocessing.

Handles point cloud I/O, duplicate removal, and statistical outlier rejection.
"""

import numpy as np
from pathlib import Path


def load_points(source) -> dict:
    """Load point cloud from file or numpy array.

    Parameters
    ----------
    source : str, Path, or np.ndarray
        If str/Path: path to .las/.laz/.xyz/.txt/.npy file.
        If ndarray of shape (N, 3+): columns are X, Y, Z, [intensity, ...].

    Returns
    -------
    dict with keys:
        'xyz'       : (N, 3) float64
        'intensity' : (N,) float64 or None
        'rgb'       : (N, 3) uint8 or None
    """
    if isinstance(source, np.ndarray):
        xyz = source[:, :3].astype(np.float64)
        intensity = source[:, 3] if source.shape[1] > 3 else None
        return {"xyz": xyz, "intensity": intensity, "rgb": None}

    path = Path(source)
    suffix = path.suffix.lower()

    if suffix in (".las", ".laz"):
        return _load_las(path)
    elif suffix == ".pts":
        return _load_pts(path)
    elif suffix in (".xyz", ".txt", ".csv"):
        data = np.loadtxt(path)
        return load_points(data)
    elif suffix == ".npy":
        data = np.load(path)
        return load_points(data)
    else:
        raise ValueError(f"Unsupported file format: {suffix}")


def _load_las(path: Path) -> dict:
    import laspy

    las = laspy.read(str(path))
    xyz = np.column_stack([las.x, las.y, las.z]).astype(np.float64)

    intensity = None
    if hasattr(las, "intensity"):
        intensity = np.asarray(las.intensity, dtype=np.float64)

    rgb = None
    if hasattr(las, "red") and hasattr(las, "green") and hasattr(las, "blue"):
        rgb = np.column_stack([las.red, las.green, las.blue]).astype(np.uint16)
        if rgb.max() > 255:
            rgb = (rgb / 256).astype(np.uint8)
        else:
            rgb = rgb.astype(np.uint8)

    return {"xyz": xyz, "intensity": intensity, "rgb": rgb}


def _load_pts(path: Path) -> dict:
    """Load Vaihingen-style .pts file (X Y Z Intensity return_number number_of_returns [label])."""
    data = np.loadtxt(str(path))
    xyz = data[:, :3].astype(np.float64)
    intensity = data[:, 3].astype(np.float64) if data.shape[1] > 3 else None
    labels = data[:, 6].astype(np.int32) if data.shape[1] >= 7 else None
    return {"xyz": xyz, "intensity": intensity, "rgb": None, "labels": labels}


def remove_duplicates(xyz: np.ndarray, tol: float = 1e-8) -> np.ndarray:
    """Remove duplicate points within tolerance.

    Returns
    -------
    mask : (N,) bool — True for points to keep.
    """
    rounded = np.round(xyz / tol).astype(np.int64)
    _, unique_idx = np.unique(rounded, axis=0, return_index=True)
    mask = np.zeros(len(xyz), dtype=bool)
    mask[unique_idx] = True
    return mask


def statistical_outlier_removal(xyz: np.ndarray, k: int = 20,
                                std_ratio: float = 2.0) -> np.ndarray:
    """Statistical outlier removal based on mean kNN distance.

    Returns
    -------
    mask : (N,) bool — True for inlier points.
    """
    from scipy.spatial import cKDTree

    tree = cKDTree(xyz)
    dists, _ = tree.query(xyz, k=k + 1)  # include self
    mean_dists = dists[:, 1:].mean(axis=1)  # exclude self

    global_mean = mean_dists.mean()
    global_std = mean_dists.std()

    threshold = global_mean + std_ratio * global_std
    mask = mean_dists < threshold
    return mask


def preprocess(source, sor_k: int = 20, sor_std: float = 2.0) -> dict:
    """Full preprocessing pipeline.

    Returns
    -------
    dict with 'xyz', 'intensity', 'rgb', 'mask' (indices into original).
    """
    data = load_points(source)
    xyz = data["xyz"]
    n_original = len(xyz)

    # Step 1: remove duplicates
    dup_mask = remove_duplicates(xyz)
    # Step 2: statistical outlier removal on de-duplicated points
    sor_mask_sub = statistical_outlier_removal(xyz[dup_mask], k=sor_k,
                                              std_ratio=sor_std)

    # Combine masks
    indices_after_dup = np.where(dup_mask)[0]
    keep_indices = indices_after_dup[sor_mask_sub]

    final_mask = np.zeros(n_original, dtype=bool)
    final_mask[keep_indices] = True

    result = {
        "xyz": xyz[final_mask],
        "intensity": data["intensity"][final_mask] if data["intensity"] is not None else None,
        "rgb": data["rgb"][final_mask] if data["rgb"] is not None else None,
        "mask": final_mask,
    }
    print(f"[preprocess] {n_original} -> {final_mask.sum()} points "
          f"(removed {n_original - final_mask.sum()})")
    return result
