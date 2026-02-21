"""
Convergence testing utilities (e.g. golden section search for optimal omega).
Migrated from time_independent.py as-is.
"""
import numpy as np


def golden_section_search(f, a, b, tol=1e-3, iter=0, max_iter=100, f_args=(), c=None, d=None, fc=None, fd=None):
    """Find the value of x in [a, b] that minimizes f(x) using recursive golden section search (migrated as-is)."""
    if iter >= max_iter:
        print("Maximum iterations reached in golden section search")
        return (a + b) / 2
    h = b - a
    if h < tol:
        print(f"Golden section search converged after {iter} iterations")
        return (a + b) / 2
    gr = (1 + np.sqrt(5)) / 2  # golden ratio
    if c is None:
        c = a + h * (1/(gr*gr))
    if d is None:
        d = a + h * (1/gr)
    if fc is None:
        fc = f(c, *f_args)
    if fd is None:
        fd = f(d, *f_args)
    print(f"Iter {iter+1}: a={a:.4f}, b={b:.4f}, c={c:.4f} (f={fc}), d={d:.4f} (f={fd})")
    if fc < fd:
        return golden_section_search(f, a, d, tol, iter+1, max_iter, f_args, d=c, fd=fc)
    else:
        return golden_section_search(f, c, b, tol, iter+1, max_iter, f_args, c=d, fc=fd)