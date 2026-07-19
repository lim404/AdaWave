"""Large-scale deployment test: one full DALES tile (~11M points) with
block-parallel processing (plan §4.3 World C requirements).

Protocol: tile train/5080_54435 (500x500 m), synthetic sigma=5cm Gaussian
noise (seed 42) so a clean reference exists at full scale. Blocks of
100x100 m with a 10 m context margin; each worker denoises block+margin
with the frozen v2 and keeps the interior. Reported: total wall time,
time per million points, peak RSS per worker, overall Z-RMSE improvement,
and seam consistency (improvement within 2 m of block boundaries vs the
interior — equal values = no stitching artefacts).
"""
import os, sys, time, csv
sys.path.insert(0, ".")
import numpy as np
import multiprocessing as mp

BLOCK = 100.0
MARGIN = 10.0
SIGMA = 0.05
SEED = 42
N_WORKERS = 5
TILE = "train/5080_54435_new.ply"


def worker(args):
    import resource
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from adawave.restoration import denoise_wavefront_prior_v2
    (bx, by, noisy_blk, interior_mask) = args
    t0 = time.time()
    out = denoise_wavefront_prior_v2(noisy_blk)
    dt = time.time() - t0
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6  # GB
    return bx, by, out["xyz"][interior_mask], dt, rss, len(noisy_blk)


def main():
    from eval_datasets import _load_dales, z_rmse
    xyz, labels = _load_dales(TILE)
    xyz = xyz[np.isfinite(xyz).all(axis=1)]
    rng = np.random.default_rng(SEED)
    noisy = xyz + rng.normal(0, SIGMA, xyz.shape)
    print(f"tile: {len(xyz):,} points, extent "
          f"{xyz[:,0].max()-xyz[:,0].min():.0f} x {xyz[:,1].max()-xyz[:,1].min():.0f} m",
          flush=True)

    x0, y0 = noisy[:, 0].min(), noisy[:, 1].min()
    nbx = int(np.ceil((noisy[:, 0].max() - x0) / BLOCK))
    nby = int(np.ceil((noisy[:, 1].max() - y0) / BLOCK))
    jobs = []
    idx_of = {}
    for bx in range(nbx):
        for by in range(nby):
            lo = np.array([x0 + bx * BLOCK, y0 + by * BLOCK])
            hi = lo + BLOCK
            m_ctx = ((noisy[:, 0] >= lo[0] - MARGIN) & (noisy[:, 0] < hi[0] + MARGIN)
                     & (noisy[:, 1] >= lo[1] - MARGIN) & (noisy[:, 1] < hi[1] + MARGIN))
            idx_ctx = np.where(m_ctx)[0]
            if len(idx_ctx) < 500:
                continue
            blk = noisy[idx_ctx]
            m_int = ((blk[:, 0] >= lo[0]) & (blk[:, 0] < hi[0])
                     & (blk[:, 1] >= lo[1]) & (blk[:, 1] < hi[1]))
            idx_of[(bx, by)] = idx_ctx[m_int]
            jobs.append((bx, by, blk, m_int))
    print(f"{len(jobs)} blocks (block {BLOCK} m, margin {MARGIN} m), "
          f"median block size {int(np.median([len(j[2]) for j in jobs])):,} pts",
          flush=True)

    t_start = time.time()
    denoised = noisy.copy()
    times, rsss, sizes = [], [], []
    with mp.Pool(N_WORKERS) as pool:
        for bx, by, xyz_int, dt, rss, n in pool.imap_unordered(worker, jobs):
            denoised[idx_of[(bx, by)]] = xyz_int
            times.append(dt); rsss.append(rss); sizes.append(n)
            print(f"block ({bx},{by}) n={n:,} t={dt:.0f}s rss={rss:.2f}GB",
                  flush=True)
    wall = time.time() - t_start
    total_pts = len(xyz)
    print(f"\nWALL {wall/60:.1f} min for {total_pts/1e6:.1f}M points "
          f"({wall/(total_pts/1e6):.0f} s per 1M incl. parallelism, "
          f"{np.sum(times)/(np.sum(sizes)/1e6):.0f} s/1M single-core); "
          f"peak worker RSS {max(rsss):.2f} GB", flush=True)

    zi = (1 - z_rmse(denoised, xyz) / z_rmse(noisy, xyz)) * 100
    # seam consistency: points within 2 m of any internal block boundary
    fx = (noisy[:, 0] - x0) % BLOCK
    fy = (noisy[:, 1] - y0) % BLOCK
    seam = (np.minimum(fx, BLOCK - fx) < 2.0) | (np.minimum(fy, BLOCK - fy) < 2.0)
    zi_seam = (1 - z_rmse(denoised[seam], xyz[seam]) / z_rmse(noisy[seam], xyz[seam])) * 100
    zi_int = (1 - z_rmse(denoised[~seam], xyz[~seam]) / z_rmse(noisy[~seam], xyz[~seam])) * 100
    print(f"Z-RMSE improvement: overall {zi:+.2f}%  seam(2m) {zi_seam:+.2f}%  "
          f"interior {zi_int:+.2f}%  (gap {zi_int-zi_seam:+.2f} pp)", flush=True)
    with open("large_scale_tile_summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["points", "wall_s", "s_per_M_wall", "s_per_M_core",
                    "peak_rss_GB", "Z_overall", "Z_seam", "Z_interior"])
        w.writerow([total_pts, round(wall, 1), round(wall/(total_pts/1e6), 1),
                    round(np.sum(times)/(np.sum(sizes)/1e6), 1),
                    round(max(rsss), 2), round(zi, 2), round(zi_seam, 2),
                    round(zi_int, 2)])


if __name__ == "__main__":
    main()
