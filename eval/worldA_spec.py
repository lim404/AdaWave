"""World A: synthetic-geometry suite — PRE-REGISTERED SPEC (2026-07-18).

Registered BEFORE any method ran on these shapes (FREEZE_v2.md protocol).
The suite maps to the three theory mechanisms and includes negative cases
where the frozen method is EXPECTED to be stressed; results are reported
as-is, one shot.

Shapes (analytic; reference surface = 200k-point dense sample; ell = target
mean spacing ~= 0.025 on a 2x2 footprint, N = 8000):
  positive / mechanism cases
    plane        z = 0                          (2-D diffusion, smooth)
    sphere       R = 1                          (smooth curvature)
    cylinder     R = 0.5, len 2                 (anisotropic curvature)
    ridge90      z = -|x|      (90 deg dihedral; 1-D diffusion along edge)
    ridge120     z = -|x|/sqrt(3)               (milder crease)
    step         z = 0 / 0.3 with vertical riser (boundary + facade)
    corner       three orthogonal quarter-planes (direction ambiguous point)
    cross        z = -min(|x|,|y|)              (two crossing singular lines)
  negative / stress cases (user-mandated)
    ripple       z = 2*ell * sin(2*pi*x/(8*ell))  (high-freq: structure ~ noise scale)
    sheets       two parallel planes, gap 4*ell   (close thin sheets)
  sampling variant
    ridge90-grad density gradient 4:1 across x   (ultra-sparse edge side)

Noise: Gaussian sigma in {0.5, 1.0} * ell for every shape/seed; additional
noise-type variants on plane + ridge90: Laplace(scale = sigma/sqrt(2)) and
Gaussian + 5% uniform outliers in +/-10*ell box, at sigma = 1.0*ell.
Seeds: {7, 19, 101}. Methods: Bilateral, MLS, GLR, Prior-v1, Prior-v2
(frozen defaults for v2; baseline settings identical to dev scripts).
Metrics vs dense reference: point-to-surface RMSE improvement %, split into
  near   (within 3*ell of the singular set; NaN for smooth-only shapes)
  smooth (rest)
  overall
"""
import numpy as np

ELL = 0.025
N_PTS = 8000
N_REF = 200_000
SEEDS = [7, 19, 101]
SIGMAS = [0.5 * ELL, 1.0 * ELL]


def _grid2(n, rng, half=1.0):
    return rng.uniform(-half, half, (n, 2))


def _make(surface, singdist, n, rng, half=1.0, density_grad=False):
    if density_grad:
        # 4:1 density ratio across x: inverse-CDF sampling of p(x) ~ lin
        u = rng.uniform(0, 1, n)
        x = half * (2 * (np.sqrt(u * 15 + 1) - 1) / (np.sqrt(16) - 1) - 1)
        y = rng.uniform(-half, half, n)
        xy = np.column_stack([x, y])
    else:
        xy = _grid2(n, rng, half)
    return surface(xy), singdist


# each generator returns (sample(n, rng) -> (N,3), singular_distance(pts) -> (N,) or None)
def g_plane():
    surf = lambda xy: np.column_stack([xy, np.zeros(len(xy))])
    return lambda n, rng: (surf(_grid2(n, rng)), None)


def g_sphere():
    def sample(n, rng):
        v = rng.normal(size=(n, 3))
        v /= np.linalg.norm(v, axis=1, keepdims=True)
        return v, None
    return sample


def g_cylinder():
    def sample(n, rng):
        th = rng.uniform(0, 2 * np.pi, n)
        z = rng.uniform(-1, 1, n)
        return np.column_stack([0.5 * np.cos(th), 0.5 * np.sin(th), z]), None
    return sample


def _ridge(slope):
    def sample(n, rng):
        xy = _grid2(n, rng)
        pts = np.column_stack([xy[:, 0], xy[:, 1], -slope * np.abs(xy[:, 0])])
        sd = lambda p: np.hypot(p[:, 0], p[:, 2])
        return pts, sd
    return sample


def g_step():
    def sample(n, rng):
        # top plane, bottom plane, vertical riser at x=0 (h=0.3)
        n_r = int(n * 0.3 / (2 + 0.3))
        n_f = (n - n_r) // 2
        xyb = _grid2(n_f, rng); xyb[:, 0] = -np.abs(xyb[:, 0])
        xyt = _grid2(n - n_r - n_f, rng); xyt[:, 0] = np.abs(xyt[:, 0])
        bot = np.column_stack([xyb, np.zeros(len(xyb))])
        top = np.column_stack([xyt, 0.3 * np.ones(len(xyt))])
        y = rng.uniform(-1, 1, n_r); z = rng.uniform(0, 0.3, n_r)
        ris = np.column_stack([np.zeros(n_r), y, z])
        pts = np.vstack([bot, top, ris])
        sd = lambda p: np.minimum(np.hypot(p[:, 0], p[:, 2]),
                                  np.hypot(p[:, 0], p[:, 2] - 0.3))
        return pts, sd
    return sample


def g_corner():
    def sample(n, rng):
        m = n // 3
        a = rng.uniform(0, 1, (m, 2)); b = rng.uniform(0, 1, (m, 2))
        c = rng.uniform(0, 1, (n - 2 * m, 2))
        p1 = np.column_stack([a, np.zeros(m)])              # z=0 plane
        p2 = np.column_stack([b[:, 0], np.zeros(m), b[:, 1]])   # y=0
        p3 = np.column_stack([np.zeros(n - 2 * m), c])          # x=0
        pts = np.vstack([p1, p2, p3])
        def sd(p):
            d1 = np.hypot(p[:, 0], p[:, 1])
            d2 = np.hypot(p[:, 1], p[:, 2])
            d3 = np.hypot(p[:, 0], p[:, 2])
            return np.minimum(np.minimum(d1, d2), d3)
        return pts, sd
    return sample


def g_cross():
    def sample(n, rng):
        xy = _grid2(n, rng)
        z = -np.minimum(np.abs(xy[:, 0]), np.abs(xy[:, 1]))
        pts = np.column_stack([xy, z])
        def sd(p):
            return np.minimum(np.hypot(p[:, 0], p[:, 2] + np.abs(p[:, 1]) * 0),
                              np.hypot(p[:, 1], p[:, 2] + 0)) * 0 + np.minimum(
                np.hypot(p[:, 0], p[:, 2]), np.hypot(p[:, 1], p[:, 2]))
        return pts, sd
    return sample


def g_ripple():
    lam, amp = 8 * ELL, 2 * ELL
    def sample(n, rng):
        xy = _grid2(n, rng)
        return np.column_stack([xy, amp * np.sin(2 * np.pi * xy[:, 0] / lam)]), None
    return sample


def g_sheets():
    gap = 4 * ELL
    def sample(n, rng):
        xy = _grid2(n, rng)
        z = np.where(rng.uniform(0, 1, n) < 0.5, 0.0, gap)
        return np.column_stack([xy, z]), None
    return sample


def g_ridge_grad(slope=1.0):
    def sample(n, rng):
        u = rng.uniform(0, 1, n)
        x = 2 * (np.sqrt(u * 15 + 1) - 1) / 3 - 1     # 4:1 density across x
        y = rng.uniform(-1, 1, n)
        pts = np.column_stack([x, y, -slope * np.abs(x)])
        sd = lambda p: np.hypot(p[:, 0], p[:, 2])
        return pts, sd
    return sample


SHAPES = [
    ("plane",       g_plane()),
    ("sphere",      g_sphere()),
    ("cylinder",    g_cylinder()),
    ("ridge90",     _ridge(1.0)),
    ("ridge120",    _ridge(1.0 / np.sqrt(3.0))),
    ("step",        g_step()),
    ("corner",      g_corner()),
    ("cross",       g_cross()),
    ("ripple",      g_ripple()),
    ("sheets",      g_sheets()),
    ("ridge90-grad", g_ridge_grad()),
]

# noise-type variants (shape, kind) at sigma = 1.0*ELL
NOISE_VARIANTS = [("plane", "laplace"), ("plane", "outlier"),
                  ("ridge90", "laplace"), ("ridge90", "outlier")]
