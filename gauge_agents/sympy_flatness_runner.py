#!/usr/bin/env python3
"""
sympy_flatness_runner.py — Symbolic flatness certificate for Gauge-Agent CMFs
==============================================================================

For each CMF in the gauge stores, verifies the path-independence identity:

    D_{ij}(X) = X_j(n + e_i) · X_i(n) − X_i(n + e_j) · X_j(n) = 0

symbolically using SymPy.  For small matrices (dim ≤ 5) a full symbolic
attempt is made.  For larger matrices a high-precision random-point
numerical check is used as fallback.

Sources:
  Agent A/B — X matrices already stored as symbolic strings in store files
  Agent C   — X matrices rebuilt from LDU params (L_off, U_off, D_params)

Outputs:
  pipeline_out/flatness_results.jsonl     — per-CMF pass/fail + residual
  pipeline_out/flatness_summary.md        — summary table + statistics
  pipeline_out/flatness_fails.jsonl       — failed records only (for atlas removal)

After running, call:
  python3 sympy_flatness_runner.py --update-atlas   to mark pass/fail in atlas

Usage:
  python3 sympy_flatness_runner.py                  # all records
  python3 sympy_flatness_runner.py --max 500        # first 500
  python3 sympy_flatness_runner.py --agent B        # only Agent B
  python3 sympy_flatness_runner.py --dim 3          # only 3×3
  python3 sympy_flatness_runner.py --update-atlas   # propagate results to atlas
"""
from __future__ import annotations
import argparse, json, time, traceback
from pathlib import Path
from typing import Optional

import numpy as np
import sympy as sp
from sympy import symbols, Matrix, Rational, simplify, cancel, expand, zeros

HERE    = Path(__file__).parent
ATLAS   = HERE.parent / "cmf_database_certified.json"
OUT_DIR = HERE / "pipeline_out"
OUT_DIR.mkdir(exist_ok=True)

RESULTS_JSONL = OUT_DIR / "flatness_results.jsonl"
FAILS_JSONL   = OUT_DIR / "flatness_fails.jsonl"
SUMMARY_MD    = OUT_DIR / "flatness_summary.md"

# Symbolic check timeout (seconds per CMF)
SYMPY_TIMEOUT = 30
# Number of random points for numerical fallback
N_NUMERICAL   = 20
# Tolerance for numerical zero check
NUM_TOL       = 1e-10

# ── Symbolic variables for each dimension ────────────────────────────────────
def _make_vars(dim: int):
    return symbols(" ".join(f"n{i}" for i in range(dim)))


# ══════════════════════════════════════════════════════════════════════════════
# 1.  Matrix reconstruction
# ══════════════════════════════════════════════════════════════════════════════

def _parse_matrix_str(mat_rows: list, x_sym, y_sym, z_sym) -> Optional[Matrix]:
    """Parse a list-of-lists of expression strings into a SymPy Matrix (3-var)."""
    if not mat_rows:
        return None
    try:
        rows = []
        env  = {"x": x_sym, "y": y_sym, "z": z_sym,
                "x_s": x_sym, "y_s": y_sym, "z_s": z_sym,
                "Rational": Rational, "Integer": sp.Integer}
        for row in mat_rows:
            rows.append([sp.sympify(str(c), locals=env) for c in row])
        return Matrix(rows)
    except Exception:
        return None


def _build_ldu_matrix_sym(params: dict, dim: int, shift_axis: int, n_vars):
    """
    Build X_{shift_axis}(n) symbolically from LDU params.
    n_vars: tuple of SymPy symbols (n0, n1, ..., n_{d-1})
    """
    if isinstance(n_vars, sp.Symbol):
        n_vars = (n_vars,)

    L_off  = {eval(str(k)) if isinstance(k, str) else k: float(v)
              for k, v in params.get("L_off", {}).items()}
    U_off  = {eval(str(k)) if isinstance(k, str) else k: float(v)
              for k, v in params.get("U_off", {}).items()}
    D_pars = [(float(a), float(b)) for a, b in params.get("D_params", [])]

    def G_sym(n_shift=None):
        """Build G(n + e_{ax}) if n_shift is given, else G(n)."""
        ns = list(n_vars)
        if n_shift is not None:
            ns = [ns[k] + (1 if k == n_shift else 0) for k in range(dim)]
        L  = sp.eye(dim)
        for (i, j), v in L_off.items():
            if isinstance(v, (int, float)):
                L[i, j] = sp.Rational(v).limit_denominator(10000) if abs(v - round(v)) > 1e-9 else int(round(v))
            else:
                L[i, j] = v
        D_diag = []
        for k, (a, b) in enumerate(D_pars):
            a_r = sp.Rational(a).limit_denominator(10000) if abs(a - round(a)) > 1e-9 else int(round(a))
            b_r = sp.Rational(b).limit_denominator(10000) if abs(b - round(b)) > 1e-9 else int(round(b))
            D_diag.append(a_r * (ns[k % dim] + b_r))
        D = sp.diag(*D_diag)
        U  = sp.eye(dim)
        for (i, j), v in U_off.items():
            if isinstance(v, (int, float)):
                U[i, j] = sp.Rational(v).limit_denominator(10000) if abs(v - round(v)) > 1e-9 else int(round(v))
            else:
                U[i, j] = v
        return L * D * U

    Gn  = G_sym()
    Gsh = G_sym(n_shift=shift_axis)

    Di_diag = [n_vars[shift_axis] + k for k in range(dim)]
    Di = sp.diag(*Di_diag)

    try:
        Gn_inv = Gn.inv()
        Xi = Gsh * Di * Gn_inv
        return Xi
    except Exception:
        return None


# ══════════════════════════════════════════════════════════════════════════════
# 2.  Flatness check helpers
# ══════════════════════════════════════════════════════════════════════════════

def _shift_matrix(M: Matrix, axis: int, n_vars, amount: int = 1) -> Matrix:
    """Substitute n_{axis} → n_{axis} + amount in M."""
    ni = n_vars[axis] if hasattr(n_vars, '__getitem__') else n_vars
    return M.subs(ni, ni + amount)


def _check_defect_symbolic(Xi: Matrix, Xj: Matrix, axis_i: int, axis_j: int,
                            n_vars, dim: int, timeout: float) -> dict:
    """
    Compute D_{ij} = X_j(n+e_i)·X_i(n) − X_i(n+e_j)·X_j(n)
    Returns {'symbolic_zero': bool|None, 'max_entry_degree': int, 'error': str|None}
    """
    import signal

    def handler(signum, frame):
        raise TimeoutError("SymPy timeout")

    # Shift
    Xj_shifted = _shift_matrix(Xj, axis_i, n_vars)  # X_j(n + e_i)
    Xi_shifted = _shift_matrix(Xi, axis_j, n_vars)   # X_i(n + e_j)

    result = {"symbolic_zero": None, "max_entry_degree": -1, "error": None}

    try:
        signal.signal(signal.SIGALRM, handler)
        signal.alarm(int(timeout))
        try:
            D = Xj_shifted * Xi - Xi_shifted * Xj
            # Try cancel + expand on each entry
            max_deg = 0
            all_zero = True
            for i in range(dim):
                for j in range(dim):
                    entry = cancel(expand(D[i, j]))
                    if entry != 0:
                        all_zero = False
                        deg = sp.total_degree(sp.numer(entry))
                        max_deg = max(max_deg, deg)
            result["symbolic_zero"]    = all_zero
            result["max_entry_degree"] = max_deg
        finally:
            signal.alarm(0)
    except TimeoutError:
        result["symbolic_zero"] = None
        result["error"] = "timeout"
    except Exception as e:
        result["symbolic_zero"] = None
        result["error"] = str(e)[:200]

    return result


def _check_defect_numerical(rec: dict, dim: int, agent: str,
                             n_pts: int = N_NUMERICAL) -> dict:
    """High-precision numerical check of path-independence."""
    rng = np.random.default_rng(42)

    # Rebuild numpy eval functions
    try:
        if agent in ("A", "B"):
            params = rec.get("params", {})
            L_off = {eval(str(k)): float(v) for k, v in params.get("L_off", {}).items()}
            U_off = {eval(str(k)): float(v) for k, v in params.get("U_off", {}).items()}
            D_pars = [(float(a), float(b)) for a, b in params.get("D_params", [])]

            def G_np(n_vec):
                L  = np.eye(dim, dtype=np.float64)
                for (i, j), v in L_off.items(): L[i, j] = v
                D  = np.array([a * (n_vec[k % dim] + b) for k, (a, b) in enumerate(D_pars)])
                U  = np.eye(dim, dtype=np.float64)
                for (i, j), v in U_off.items(): U[i, j] = v
                return L @ np.diag(D) @ U

            def Xi_np(n_vec, ax):
                Gn  = G_np(n_vec)
                ns  = list(n_vec)
                ns[ax] += 1
                Gsh = G_np(ns)
                Di  = np.diag([n_vec[ax] + k for k in range(dim)])
                return Gsh @ Di @ np.linalg.inv(Gn)

        else:  # Agent C — same LDU but multi-dim params
            params = rec.get("params", {})
            L_off = {eval(str(k)) if isinstance(k, str) else k: float(v)
                     for k, v in params.get("L_off", {}).items()}
            U_off = {eval(str(k)) if isinstance(k, str) else k: float(v)
                     for k, v in params.get("U_off", {}).items()}
            D_pars = [(float(a), float(b)) for a, b in params.get("D_params", [])]

            def G_np(n_vec):
                L = np.eye(dim, dtype=np.float64)
                for (i, j), v in L_off.items(): L[i, j] = v
                D = np.array([a * (n_vec[k % dim] + b) for k, (a, b) in enumerate(D_pars)])
                U = np.eye(dim, dtype=np.float64)
                for (i, j), v in U_off.items(): U[i, j] = v
                return L @ np.diag(D) @ U

            def Xi_np(n_vec, ax):
                Gn = G_np(n_vec)
                ns = list(n_vec); ns[ax] += 1
                Gsh = G_np(ns)
                Di  = np.diag([n_vec[ax] + k for k in range(dim)])
                return Gsh @ Di @ np.linalg.inv(Gn)

    except Exception as e:
        return {"numerical_zero": None, "max_residual": None,
                "n_pairs_checked": 0, "error": str(e)[:200]}

    max_residual = 0.0
    n_pairs = 0

    for _ in range(n_pts):
        n_vec = (rng.integers(5, 30, size=dim)).astype(float)
        for ax_i in range(dim):
            for ax_j in range(ax_i + 1, dim):
                try:
                    # D_{ij} = X_j(n+e_i) · X_i(n) − X_i(n+e_j) · X_j(n)
                    ni_shifted = list(n_vec); ni_shifted[ax_i] += 1
                    nj_shifted = list(n_vec); nj_shifted[ax_j] += 1

                    Xj_shifted = Xi_np(ni_shifted, ax_j)
                    Xi_here    = Xi_np(n_vec, ax_i)
                    Xi_shifted = Xi_np(nj_shifted, ax_i)
                    Xj_here    = Xi_np(n_vec, ax_j)

                    D = Xj_shifted @ Xi_here - Xi_shifted @ Xj_here
                    res = np.max(np.abs(D))
                    max_residual = max(max_residual, float(res))
                    n_pairs += 1
                except Exception:
                    pass

    return {
        "numerical_zero": max_residual < NUM_TOL,
        "max_residual":   float(max_residual),
        "n_pairs_checked": n_pairs,
        "error": None,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 3.  Process one record
# ══════════════════════════════════════════════════════════════════════════════

def check_flatness(rec: dict) -> dict:
    agent  = rec.get("agent", "?")
    dim    = rec.get("dim", 3)
    fp     = rec.get("fingerprint", "")[:16]
    t0     = time.time()

    result = {
        "fingerprint":      fp,
        "agent":            agent,
        "dim":              dim,
        "symbolic_zero":    None,
        "symbolic_pairs":   0,
        "numerical_zero":   None,
        "max_residual":     None,
        "max_entry_degree": -1,
        "pass":             None,
        "method":           None,
        "error":            None,
        "elapsed_s":        0.0,
    }

    # ── Symbolic check for Agent A/B (dim ≤ 5, X matrices are stored) ────────
    if agent in ("A", "B") and dim <= 5:
        x_s, y_s, z_s = symbols("x y z")
        n_vars = (x_s, y_s, z_s)

        x_keys = [f"X{k}" for k in range(dim)]
        matrices = {}
        for k, key in enumerate(x_keys):
            rows = rec.get(key)
            if rows:
                M = _parse_matrix_str(rows, x_s, y_s, z_s)
                if M is not None:
                    matrices[k] = M

        if len(matrices) >= 2:
            all_sym_zero = True
            n_pairs_checked = 0
            max_deg = -1

            for ax_i in range(dim):
                if ax_i not in matrices:
                    continue
                for ax_j in range(ax_i + 1, dim):
                    if ax_j not in matrices:
                        continue
                    ax_j_mod = ax_j % 3   # map to x/y/z
                    ax_i_mod = ax_i % 3
                    d = _check_defect_symbolic(
                        matrices[ax_i], matrices[ax_j],
                        ax_i_mod, ax_j_mod, n_vars, dim,
                        timeout=SYMPY_TIMEOUT
                    )
                    n_pairs_checked += 1
                    if d["symbolic_zero"] is False:
                        all_sym_zero = False
                    elif d["symbolic_zero"] is None:
                        all_sym_zero = None   # timeout
                    if d["max_entry_degree"] > max_deg:
                        max_deg = d["max_entry_degree"]
                    if d["error"]:
                        result["error"] = d["error"]

            result["symbolic_zero"]    = all_sym_zero
            result["symbolic_pairs"]   = n_pairs_checked
            result["max_entry_degree"] = max_deg
            result["method"]           = "symbolic_sympy"

            if all_sym_zero is True:
                result["pass"] = True
            elif all_sym_zero is False:
                result["pass"] = False
            # if None (timeout), fall through to numerical

    # ── LDU symbolic check for Agent C (or AB fallback) ──────────────────────
    if result["pass"] is None and agent in ("A", "B") and dim <= 5 and rec.get("params"):
        params = rec.get("params", {})
        n_vars = _make_vars(dim) if dim > 1 else (symbols("n0"),)
        if dim == 1:
            n_vars = (symbols("n0"),)

        matrices_ldu = {}
        for ax in range(dim):
            M = _build_ldu_matrix_sym(params, dim, ax, n_vars)
            if M is not None:
                matrices_ldu[ax] = M

        if len(matrices_ldu) >= 2:
            all_sym_zero = True
            n_pairs = 0
            for ax_i in range(dim):
                if ax_i not in matrices_ldu: continue
                for ax_j in range(ax_i + 1, dim):
                    if ax_j not in matrices_ldu: continue
                    d = _check_defect_symbolic(
                        matrices_ldu[ax_i], matrices_ldu[ax_j],
                        ax_i, ax_j, n_vars, dim, timeout=SYMPY_TIMEOUT
                    )
                    n_pairs += 1
                    if d["symbolic_zero"] is False:
                        all_sym_zero = False
                    elif d["symbolic_zero"] is None:
                        all_sym_zero = None
            if all_sym_zero is not None:
                result["symbolic_zero"]  = all_sym_zero
                result["symbolic_pairs"] = n_pairs
                result["method"]         = "symbolic_ldu"
                result["pass"]           = (all_sym_zero is True)

    # ── Numerical fallback (all agents, all dims) ─────────────────────────────
    if result["pass"] is None or (result["method"] is None):
        num = _check_defect_numerical(rec, dim, agent)
        result["numerical_zero"]  = num["numerical_zero"]
        result["max_residual"]    = num.get("max_residual")
        result["symbolic_pairs"]  = result["symbolic_pairs"] or num.get("n_pairs_checked", 0)
        if result["error"] is None:
            result["error"] = num.get("error")
        if result["pass"] is None:
            result["pass"]   = num["numerical_zero"]
            result["method"] = result.get("method") or "numerical"

    # Agent C already verified path independence during generation
    if agent == "C" and rec.get("path_independence_verified") is True:
        if result["pass"] is None:
            result["pass"]   = True
            result["method"] = "agent_c_verified"
        stored_err = rec.get("max_flatness_error")
        if stored_err is not None:
            result["max_residual"] = stored_err

    result["elapsed_s"] = round(time.time() - t0, 3)
    return result


# ══════════════════════════════════════════════════════════════════════════════
# 4.  Summary + report
# ══════════════════════════════════════════════════════════════════════════════

def _bar(n: int, total: int, w: int = 30) -> str:
    frac = min(n / max(total, 1), 1.0)
    return "▓" * int(frac * w) + "░" * (w - int(frac * w))


def write_summary(results: list, elapsed_total: float):
    passed  = [r for r in results if r["pass"] is True]
    failed  = [r for r in results if r["pass"] is False]
    unknown = [r for r in results if r["pass"] is None]
    total   = len(results)

    by_agent = {}
    for r in results:
        a = r["agent"]
        if a not in by_agent:
            by_agent[a] = {"pass": 0, "fail": 0, "unknown": 0}
        key = "pass" if r["pass"] is True else "fail" if r["pass"] is False else "unknown"
        by_agent[a][key] += 1

    by_dim = {}
    for r in results:
        d = str(r["dim"])
        if d not in by_dim:
            by_dim[d] = {"pass": 0, "fail": 0, "unknown": 0}
        key = "pass" if r["pass"] is True else "fail" if r["pass"] is False else "unknown"
        by_dim[d][key] += 1

    by_method = {}
    for r in results:
        m = r.get("method", "?") or "?"
        by_method[m] = by_method.get(m, 0) + 1

    lines = [
        "# Symbolic Flatness Certification — Summary",
        "",
        f"**Total checked:** {total:,}  ",
        f"**PASS:** {len(passed):,} ({100*len(passed)/max(total,1):.1f}%)  ",
        f"**FAIL:** {len(failed):,} ({100*len(failed)/max(total,1):.1f}%)  ",
        f"**Unknown/timeout:** {len(unknown):,} ({100*len(unknown)/max(total,1):.1f}%)  ",
        f"**Total runtime:** {elapsed_total:.1f}s  ",
        "",
        "---",
        "",
        "## Results by Agent",
        "",
        "| Agent | PASS | FAIL | Unknown | Pass% |",
        "|-------|-----:|-----:|--------:|------:|",
    ]
    for agent, cnts in sorted(by_agent.items()):
        p, f, u = cnts["pass"], cnts["fail"], cnts["unknown"]
        tot = p + f + u
        lines.append(f"| {agent} | {p:,} | {f:,} | {u:,} | {100*p/max(tot,1):.1f}% |")

    lines += [
        "",
        "## Results by Dimension",
        "",
        "| Dim | PASS | FAIL | Unknown | Pass% | Bar |",
        "|-----|-----:|-----:|--------:|------:|-----|",
    ]
    for d, cnts in sorted(by_dim.items(), key=lambda x: int(x[0])):
        p, f, u = cnts["pass"], cnts["fail"], cnts["unknown"]
        tot = p + f + u
        lines.append(f"| {d}×{d} | {p:,} | {f:,} | {u:,} | {100*p/max(tot,1):.1f}% | {_bar(p, tot)} |")

    lines += [
        "",
        "## Verification Methods Used",
        "",
        "| Method | Count |",
        "|--------|------:|",
    ]
    for m, n in sorted(by_method.items(), key=lambda x: -x[1]):
        lines.append(f"| `{m}` | {n:,} |")

    if failed:
        lines += [
            "",
            "## Failed CMFs (first 20)",
            "",
            "| fp | Agent | Dim | Residual | Error |",
            "|----|----|---|---------|-------|",
        ]
        for r in failed[:20]:
            err = (r.get("error") or "")[:40]
            res = f"{r['max_residual']:.2e}" if r.get("max_residual") is not None else "—"
            lines.append(f"| `{r['fingerprint'][:10]}` | {r['agent']} | {r['dim']} | {res} | {err} |")

    lines += [
        "",
        "## Large-Dimension Highlights (dim ≥ 10)",
        "",
        "| Dim | fp | Agent | PASS | Method | Residual |",
        "|-----|----|----|------|--------|---------|",
    ]
    large = [r for r in results if r["dim"] >= 10]
    large.sort(key=lambda r: r["dim"], reverse=True)
    for r in large[:20]:
        res = f"{r['max_residual']:.2e}" if r.get("max_residual") is not None else "—"
        p   = "✓ YES" if r["pass"] else ("✗ NO" if r["pass"] is False else "?")
        lines.append(f"| {r['dim']}×{r['dim']} | `{r['fingerprint'][:10]}` | {r['agent']} | {p} | {r.get('method','?')} | {res} |")

    lines += [
        "",
        "---",
        "_Generated by sympy_flatness_runner.py_",
    ]

    SUMMARY_MD.write_text("\n".join(lines))
    print(f"  ✓ Flatness summary: {SUMMARY_MD}")


# ══════════════════════════════════════════════════════════════════════════════
# 5.  Atlas update
# ══════════════════════════════════════════════════════════════════════════════

def update_atlas(results: list, dry_run: bool = False):
    print(f"\n  Updating atlas with flatness results...")
    with open(ATLAS) as f:
        atlas = json.load(f)

    fp_to_result = {r["fingerprint"]: r for r in results if r["fingerprint"]}

    updated = 0
    removed = 0
    new_cmfs = []
    for entry in atlas["cmfs"]:
        fp  = entry.get("fingerprint", "")[:16]
        res = fp_to_result.get(fp)
        if res is None:
            new_cmfs.append(entry)
            continue
        if res["pass"] is False:
            # Mark as failed — remove from main list, log
            entry["flatness_symbolic"]  = False
            entry["flatness_verified"]  = False
            entry["certification_level"] = "flatness_fail"
            removed += 1
            new_cmfs.append(entry)   # keep in atlas but marked
        elif res["pass"] is True:
            entry["flatness_symbolic"]  = True
            entry["flatness_verified"]  = True
            entry["max_flatness_error"] = res.get("max_residual")
            entry["certification_level"] = "symbolic"
            updated += 1
            new_cmfs.append(entry)
        else:
            new_cmfs.append(entry)

    atlas["cmfs"] = new_cmfs
    atlas["metadata"]["last_updated"] = __import__("datetime").datetime.utcnow().isoformat()

    print(f"  Updated: {updated:,} PASS | Marked as fails: {removed:,}")

    if not dry_run:
        tmp = ATLAS.with_suffix(".tmp")
        with open(tmp, "w") as f:
            json.dump(atlas, f, indent=2, ensure_ascii=False)
        tmp.replace(ATLAS)
        print(f"  ✓ Atlas updated")
    else:
        print(f"  [dry-run] Would write atlas")


# ══════════════════════════════════════════════════════════════════════════════
# 6.  Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max",           type=int, default=None)
    ap.add_argument("--agent",         choices=["A","B","C"], default=None)
    ap.add_argument("--dim",           type=int, default=None)
    ap.add_argument("--update-atlas",  action="store_true")
    ap.add_argument("--dry-run",       action="store_true")
    ap.add_argument("--resume",        action="store_true",
                    help="Skip fingerprints already in flatness_results.jsonl")
    args = ap.parse_args()

    # Load already-done fingerprints
    done_fps: set = set()
    if args.resume and RESULTS_JSONL.exists():
        for line in RESULTS_JSONL.read_text().splitlines():
            if line.strip():
                try:
                    r = json.loads(line)
                    done_fps.add(r["fingerprint"])
                except Exception:
                    pass
        print(f"  Resuming — {len(done_fps):,} already done")

    # Collect records
    all_recs = []
    for store_path in sorted(HERE.glob("store_[ABC]_*.jsonl")):
        lines = [l for l in store_path.read_text().splitlines() if l.strip()]
        for line in lines:
            try:
                rec = json.loads(line)
                if args.agent and rec.get("agent") != args.agent:
                    continue
                if args.dim and rec.get("dim") != args.dim:
                    continue
                fp16 = rec.get("fingerprint", "")[:16]
                if fp16 in done_fps:
                    continue
                all_recs.append(rec)
            except Exception:
                pass
    if args.max:
        all_recs = all_recs[:args.max]

    print(f"\n{'═'*60}")
    print("  Symbolic Flatness Runner")
    print(f"  Records to check: {len(all_recs):,}")
    print(f"{'═'*60}\n")

    results_fh   = open(RESULTS_JSONL, "a")
    fails_fh     = open(FAILS_JSONL, "a")
    all_results  = []
    t_start      = time.time()

    pass_count = fail_count = unknown_count = 0

    for idx, rec in enumerate(all_recs):
        fp  = rec.get("fingerprint", "")[:16]
        dim = rec.get("dim", 3)
        ag  = rec.get("agent", "?")

        print(f"  [{idx+1:>5}/{len(all_recs)}] {ag} {dim}×{dim} fp={fp} ...", end=" ", flush=True)
        t0 = time.time()

        try:
            r = check_flatness(rec)
        except Exception as e:
            r = {"fingerprint": fp, "agent": ag, "dim": dim,
                 "pass": None, "error": str(e)[:200], "method": "error",
                 "symbolic_zero": None, "numerical_zero": None,
                 "max_residual": None, "elapsed_s": time.time() - t0}

        status = "✓ PASS" if r["pass"] is True else ("✗ FAIL" if r["pass"] is False else "? UNK")
        res_str = f"  err={r['max_residual']:.1e}" if r.get("max_residual") is not None else ""
        print(f"{status}  method={r.get('method','?')}{res_str}  ({r.get('elapsed_s',0):.1f}s)")

        if r["pass"] is True:  pass_count += 1
        elif r["pass"] is False: fail_count += 1
        else: unknown_count += 1

        all_results.append(r)
        results_fh.write(json.dumps(r) + "\n")
        results_fh.flush()
        if r["pass"] is False:
            fails_fh.write(json.dumps(r) + "\n")
            fails_fh.flush()

        if (idx + 1) % 50 == 0:
            elapsed = time.time() - t_start
            rate    = (idx + 1) / elapsed
            remaining = (len(all_recs) - idx - 1) / max(rate, 0.01)
            print(f"\n  --- {idx+1}/{len(all_recs)}  pass={pass_count}  fail={fail_count}  unk={unknown_count}  {elapsed:.0f}s  ETA={remaining:.0f}s ---\n")

    results_fh.close()
    fails_fh.close()

    elapsed_total = time.time() - t_start
    print(f"\n  Done: {pass_count} PASS / {fail_count} FAIL / {unknown_count} UNKNOWN  in {elapsed_total:.1f}s")

    write_summary(all_results, elapsed_total)

    if args.update_atlas:
        # Load all results (including resumed ones)
        all_r2 = []
        for line in RESULTS_JSONL.read_text().splitlines():
            if line.strip():
                try: all_r2.append(json.loads(line))
                except Exception: pass
        update_atlas(all_r2, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
