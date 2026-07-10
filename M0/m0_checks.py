"""Tiny self-check helpers for the M0 foundations notebooks.

Import in a notebook with:

    from m0_checks import check, check_tensor, TODO

Each `check(...)` prints a friendly PASS/FAIL so you can verify an exercise
without reading docs. Nothing here is PyTorch-specific magic -- it is just
comparison + pretty printing.
"""
from __future__ import annotations

import torch


class TODO:
    """Sentinel for "you haven't filled this in yet".

    Use as:  answer = TODO   # replace with your code
    The checks give a clear message instead of a confusing type error.
    """
    pass


def _is_todo(x) -> bool:
    return x is TODO or isinstance(x, type) and x is TODO


def check(name: str, got, want, *, atol: float = 1e-6) -> bool:
    """Compare `got` vs `want` and print a PASS/FAIL line.

    Handles tensors, numbers, tuples/lists, and bools. Uses a tolerance for
    floating point so 0.1+0.2 style rounding does not fail you.
    """
    if _is_todo(got):
        print(f"❌ {name}: not implemented yet (still equals TODO). Replace TODO with your code.")
        return False

    ok = False
    detail = ""
    try:
        if isinstance(want, torch.Tensor) or isinstance(got, torch.Tensor):
            g = torch.as_tensor(got)
            w = torch.as_tensor(want)
            if g.shape != w.shape:
                detail = f"shape mismatch: got {tuple(g.shape)}, want {tuple(w.shape)}"
            else:
                if g.dtype.is_floating_point or w.dtype.is_floating_point:
                    ok = torch.allclose(g.float(), w.float(), atol=atol)
                else:
                    ok = torch.equal(g, w)
                if not ok:
                    detail = f"values differ:\n got={g}\n want={w}"
        elif isinstance(want, float) or isinstance(got, float):
            ok = abs(float(got) - float(want)) <= atol
            if not ok:
                detail = f"got {got}, want {want}"
        else:
            ok = got == want
            if not ok:
                detail = f"got {got!r}, want {want!r}"
    except Exception as e:  # noqa: BLE001 - want a friendly message, not a crash
        print(f"❌ {name}: your value raised {type(e).__name__}: {e}")
        return False

    if ok:
        print(f"✅ {name}")
    else:
        print(f"❌ {name}: {detail}")
    return ok


def check_tensor(name: str, got, *, shape=None, dtype=None) -> bool:
    """Check only structural properties (shape and/or dtype) of a tensor."""
    if _is_todo(got):
        print(f"❌ {name}: not implemented yet (still equals TODO).")
        return False
    if not isinstance(got, torch.Tensor):
        print(f"❌ {name}: expected a torch.Tensor, got {type(got).__name__}")
        return False
    problems = []
    if shape is not None and tuple(got.shape) != tuple(shape):
        problems.append(f"shape got {tuple(got.shape)}, want {tuple(shape)}")
    if dtype is not None and got.dtype != dtype:
        problems.append(f"dtype got {got.dtype}, want {dtype}")
    if problems:
        print(f"❌ {name}: " + "; ".join(problems))
        return False
    print(f"✅ {name}")
    return True
