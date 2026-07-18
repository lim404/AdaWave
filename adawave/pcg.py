"""Matrix-free Jacobi-preconditioned conjugate gradients (facade over the
frozen implementation)."""
from .restoration import _solve_pcg as solve_pcg  # noqa: F401
