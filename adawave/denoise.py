"""Module 7: Structure-aware denoising.

Applies different smoothing strategies for smooth vs. feature points.
Uses normal-consistent neighbor filtering, adaptive step sizes, and
wavefront-inspired structural propagation for continuous protection.
"""

import numpy as np


def denoise_once(xyz: np.ndarray, neighbor_indices: np.ndarray,
                 neighbor_dists: np.ndarray,
                 normals: np.ndarray, tangent1: np.ndarray,
                 singularity_strength: np.ndarray,
                 principal_direction: np.ndarray,
                 is_feature: np.ndarray,
                 neighbor_quality: np.ndarray = None,
                 structure_proximity: np.ndarray = None,
                 smooth_weight: float = 0.5,
                 feature_weight: float = 0.15) -> np.ndarray:
    """One iteration of structure-aware denoising.

    Parameters
    ----------
    xyz : (N, 3) current point positions.
    neighbor_indices : (N, k) neighbor indices.
    neighbor_dists : (N, k) neighbor distances.
    normals : (N, 3) local normals.
    tangent1 : (N, 3) first tangent direction.
    singularity_strength : (N,) per-point singularity strength [0, 1].
    principal_direction : (N,) per-point principal angle in [0, pi).
    is_feature : (N,) bool mask.
    neighbor_quality : (N, k) optional normal-consistency weights [0, 1].
    structure_proximity : (N,) optional continuous structure proximity [0, 1].
        Used to modulate smooth-point step size: points near features get
        reduced smoothing to create a protective buffer zone.
    smooth_weight : float, step size for smooth point update.
    feature_weight : float, step size for feature point update.

    Returns
    -------
    xyz_new : (N, 3) updated point positions.
    """
    n = len(xyz)
    xyz_new = xyz.copy()

    nn1_dists = neighbor_dists[:, 0]
    median_spacing = np.median(nn1_dists)

    for i in range(n):
        nbr_idx = neighbor_indices[i]
        nbr_pts = xyz[nbr_idx]  # (k, 3)
        center = xyz[i]
        normal = normals[i]

        diff = nbr_pts - center  # (k, 3)
        dists = np.linalg.norm(diff, axis=1)

        normal_proj = diff @ normal  # (k,)

        sigma_s = np.median(dists)
        if sigma_s < 1e-12:
            continue

        # Tight bilateral range sigma
        sigma_r = 0.3 * sigma_s

        w_spatial = np.exp(-0.5 * (dists / sigma_s) ** 2)
        w_range = np.exp(-0.5 * (normal_proj / sigma_r) ** 2)

        # Normal-consistency weight from geometry module
        if neighbor_quality is not None:
            w_quality = neighbor_quality[i]
        else:
            w_quality = np.ones(len(nbr_idx))

        if not is_feature[i]:
            # --- Smooth point: bilateral smoothing along normal ---
            w = w_spatial * w_range * w_quality
            w_sum = w.sum()
            if w_sum < 1e-12:
                continue

            shift = np.dot(w, normal_proj) / w_sum
            max_shift = 0.5 * median_spacing
            shift = np.clip(shift, -max_shift, max_shift)

            # Structure proximity modulation: reduce smoothing near features
            effective_weight = smooth_weight
            if structure_proximity is not None:
                sp = structure_proximity[i]
                # Blend: full smooth_weight at sp=0, reduced toward feature_weight at sp=1
                effective_weight = smooth_weight * (1.0 - 0.5 * sp)

            xyz_new[i] += effective_weight * shift * normal

        else:
            # --- Feature point: conservative constrained smoothing ---
            theta = principal_direction[i]
            t1 = tangent1[i]
            t2 = np.cross(normal, t1)
            t2_norm = np.linalg.norm(t2)
            if t2_norm > 1e-12:
                t2 /= t2_norm
            else:
                t2 = np.zeros(3)

            principal_3d = np.cos(theta) * t1 + np.sin(theta) * t2

            # Directional affinity: angular acceptance along principal direction
            diff_norm = diff / (dists[:, None] + 1e-12)
            alignment = np.abs(diff_norm @ principal_3d)
            w_dir = alignment ** 2

            w = w_spatial * w_range * w_quality * w_dir
            w_sum = w.sum()
            if w_sum < 1e-12:
                continue

            shift = np.dot(w, normal_proj) / w_sum

            max_shift = 0.3 * median_spacing
            shift = np.clip(shift, -max_shift, max_shift)
            xyz_new[i] += feature_weight * shift * normal

    return xyz_new


def iterative_denoise(xyz: np.ndarray, neighbor_query, k: int = 30,
                      n_iters: int = 3, n_dirs: int = 8,
                      grid_size: int = 16, smooth_weight: float = 0.5,
                      feature_weight: float = 0.15,
                      patch_type: str = "density",
                      use_learnable_propagation: bool = False,
                      propagation_weights: str = None,
                      verbose: bool = True) -> dict:
    """Full iterative denoising pipeline.

    Parameters
    ----------
    xyz : (N, 3) input point positions.
    neighbor_query : NeighborQuery instance.
    k : int, number of neighbors.
    n_iters : int, number of denoising iterations.
    n_dirs : int, number of discrete directions.
    grid_size : int, patch grid size.
    smooth_weight : float, smooth point step size.
    feature_weight : float, feature point step size.
    patch_type : str, "density" or "height".
    use_learnable_propagation : bool, if True use the trained MLP edge-weight
        kernel from ``propagate_learnable`` instead of the hand-crafted one.
    propagation_weights : str, optional path to MLP weights (default: packaged).
    verbose : bool, print progress.

    Returns
    -------
    dict with:
        'xyz'                   : (N, 3) denoised positions.
        'singularity_strength'  : (N,) final singularity strength.
        'principal_direction'   : (N,) final principal direction.
        'directional_entropy'   : (N,) final directional entropy.
        'is_feature'            : (N,) final feature mask.
    """
    from .geometry import estimate_local_frames, filter_neighbors_by_normal
    from .patch import build_density_patches, build_height_patches
    from .directional import compute_directional_responses
    from .descriptors import compute_descriptors
    from .propagate import propagate_singularity

    patch_builder = build_height_patches if patch_type == "height" else build_density_patches

    learnable_mlp = None
    if use_learnable_propagation:
        from .propagate_learnable import load_mlp, propagate_singularity_learnable
        learnable_mlp = load_mlp(propagation_weights)
        if verbose:
            print("[denoise] using learnable propagation MLP")

    current_xyz = xyz.copy()

    for it in range(n_iters):
        if verbose:
            print(f"[denoise] iteration {it + 1}/{n_iters}")

        # Rebuild KDTree for updated positions
        neighbor_query.rebuild(current_xyz)
        dists, indices = neighbor_query.query_knn(k)

        # Re-estimate local geometry
        frames = estimate_local_frames(current_xyz, indices)

        # Compute normal-consistent neighbor quality weights
        nq_weights = filter_neighbors_by_normal(
            current_xyz, indices, dists, frames["normals"],
            normal_thresh=0.3, residual_sigma_factor=1.5
        )

        if verbose:
            mean_w = nq_weights.mean()
            low_w = (nq_weights < 0.1).mean()
            print(f"  neighbor quality: mean={mean_w:.3f}, "
                  f"suppressed (<0.1)={100*low_w:.1f}%")

        # Build patches with quality-filtered neighbors
        patches = patch_builder(
            current_xyz, indices,
            frames["normals"], frames["tangent1"], frames["tangent2"],
            grid_size=grid_size, neighbor_weights=nq_weights
        )

        # Directional responses
        responses = compute_directional_responses(patches, n_dirs=n_dirs)

        # Descriptors (feature classification stays at 95th percentile)
        desc = compute_descriptors(responses, n_dirs=n_dirs)

        # Wavefront-inspired structural propagation
        # Propagated strength is used as continuous structure_proximity
        # for smooth-point step modulation, NOT for feature reclassification.
        original_ss = desc["singularity_strength"].copy()
        if learnable_mlp is not None:
            propagated_ss = propagate_singularity_learnable(
                current_xyz, indices,
                frames["normals"], frames["tangent1"],
                original_ss,
                desc["principal_direction"],
                learnable_mlp,
                neighbor_quality=nq_weights,
            )
        else:
            propagated_ss = propagate_singularity(
                current_xyz, indices,
                frames["normals"], frames["tangent1"],
                original_ss,
                desc["principal_direction"],
                neighbor_quality=nq_weights,
            )
        # Structure proximity: how much each point was boosted by propagation
        # Normalized to [0, 1] — high values = near structural features
        boost = propagated_ss - original_ss
        boost_max = boost.max()
        if boost_max > 1e-12:
            structure_proximity = boost / boost_max
        else:
            structure_proximity = np.zeros(len(original_ss))

        if verbose:
            n_boosted = (boost > 0.01).sum()
            mean_prox = structure_proximity[structure_proximity > 0.01].mean() \
                if n_boosted > 0 else 0.0
            print(f"  propagation: {n_boosted} points in buffer zone, "
                  f"mean proximity={mean_prox:.3f}")

        # Denoise one step with structure proximity modulation
        current_xyz = denoise_once(
            current_xyz, indices, dists,
            frames["normals"], frames["tangent1"],
            desc["singularity_strength"],
            desc["principal_direction"],
            desc["is_feature"],
            neighbor_quality=nq_weights,
            structure_proximity=structure_proximity,
            smooth_weight=smooth_weight,
            feature_weight=feature_weight,
        )

        if verbose:
            n_feat = desc["is_feature"].sum()
            print(f"  feature points: {n_feat}/{len(current_xyz)} "
                  f"({100 * n_feat / len(current_xyz):.1f}%)")

    return {
        "xyz": current_xyz,
        "singularity_strength": propagated_ss,
        "principal_direction": desc["principal_direction"],
        "directional_entropy": desc["directional_entropy"],
        "is_feature": desc["is_feature"],
    }
