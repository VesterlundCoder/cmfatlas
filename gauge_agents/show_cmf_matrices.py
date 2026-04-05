#!/usr/bin/env python3
"""
show_cmf_matrices.py — Symbolic reconstruction of all Agent C CMF hits
=======================================================================

For each hit in store_C_NxN.jsonl, reconstructs the full symbolic matrices:
  G(n)           — the d×d gauge matrix (product L·D_diag·U)
  D_i(n_i)       — the canonical kernel for each axis i (diagonal)
  X_i(n)         — the i-th CMF matrix (d×d rational functions in n_0…n_{d-1})

Uses the LDU structure for fast symbolic inversion:
  G(n)^{-1} = U^{-1} · D_diag(n)^{-1} · L^{-1}

Output:
  cmf_matrices_report.md   — full readable report
  cmf_matrices_latex.tex   — LaTeX source for all matrices

Usage:
    python3 show_cmf_matrices.py          # all dims
    python3 show_cmf_matrices.py --dim 10 # only 10×10
    python3 show_cmf_matrices.py --dim 6 7 8 --tex  # with LaTeX output
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

import sympy as sp
from sympy import symbols, Matrix, Rational, simplify, factor

HERE = Path(__file__).parent


# ══════════════════════════════════════════════════════════════════════════════
# 1.  Symbolic building blocks
# ══════════════════════════════════════════════════════════════════════════════

def _make_coord_syms(dim: int):
    """Return tuple of sympy symbols (n0, n1, …, n_{d-1})."""
    return symbols(f"n0:{dim}")          # sympy shorthand


def _build_G_sym(params: dict, coords) -> tuple:
    """
    Build G(n) = L · D_diag · U  symbolically.
    Returns (G, L, D_diag_mat, U, G_inv) where G_inv = U^{-1}·D^{-1}·L^{-1}.
    Uses triangular structure for fast inversion — avoids full SymPy det().
    """
    dim = params["dim"]

    # ── L: unit lower triangular ──────────────────────────────────────────────
    def _pk(s):
        return tuple(int(x.strip()) for x in str(s).strip("()").split(","))

    L_mat = sp.eye(dim)
    for k, v in params["L_off"].items():
        i, j = _pk(k)
        L_mat[i, j] = Rational(v).limit_denominator(16)

    # ── D_diag: diagonal with a_k * (n_{k%dim} + b_k) ───────────────────────
    diag_entries = []
    for k, (a, b) in enumerate(params["D_params"]):
        n_k = coords[k % dim]
        entry = Rational(a).limit_denominator(16) * (n_k + Rational(b).limit_denominator(16))
        diag_entries.append(entry)
    D_mat = sp.diag(*diag_entries)

    # ── U: unit upper triangular ──────────────────────────────────────────────
    U_mat = sp.eye(dim)
    for k, v in params["U_off"].items():
        i, j = _pk(k)
        U_mat[i, j] = Rational(v).limit_denominator(16)

    G = L_mat * D_mat * U_mat

    # ── G^{-1} via LDU structure ──────────────────────────────────────────────
    # L^{-1}: unit lower triangular — solve L·X = I column by column
    L_inv = sp.eye(dim)
    for col in range(dim):
        for row in range(col + 1, dim):
            val = -sum(L_mat[row, k] * L_inv[k, col]
                       for k in range(col, row))
            L_inv[row, col] = val

    # D^{-1}: trivial
    D_inv = sp.diag(*[sp.Integer(1) / e for e in diag_entries])

    # U^{-1}: unit upper triangular — solve U·X = I
    U_inv = sp.eye(dim)
    for col in range(dim - 1, -1, -1):
        for row in range(col - 1, -1, -1):
            val = -sum(U_mat[row, k] * U_inv[k, col]
                       for k in range(row + 1, col + 1))
            U_inv[row, col] = val

    G_inv = U_inv * D_inv * L_inv
    return G, L_mat, D_mat, U_mat, G_inv


def _build_Di_sym(dim: int, axis: int, coords) -> Matrix:
    """D_i = diag(n_i, n_i+1, …, n_i+dim-1)."""
    n_i = coords[axis]
    return sp.diag(*[n_i + k for k in range(dim)])


def _build_Xi_sym(G_inv: Matrix, L_mat: Matrix, D_mat: Matrix,
                  U_mat: Matrix, Di: Matrix,
                  params: dict, axis: int, coords, do_simplify: bool) -> Matrix:
    """
    X_i(n) = G(n+e_i) · D_i(n_i) · G(n)^{-1}.
    G(n+e_i) is formed by shifting coords[axis] += 1 in G.
    """
    dim = params["dim"]
    diag_entries_shifted = []
    for k, (a, b) in enumerate(params["D_params"]):
        n_k = coords[k % dim]
        # increment only coordinate `axis`
        n_k_sh = n_k + 1 if (k % dim) == axis else n_k
        entry = Rational(a).limit_denominator(16) * (n_k_sh + Rational(b).limit_denominator(16))
        diag_entries_shifted.append(entry)

    D_sh = sp.diag(*diag_entries_shifted)
    G_sh = L_mat * D_sh * U_mat

    Xi = G_sh * Di * G_inv
    if do_simplify:
        Xi = sp.simplify(Xi)
    return Xi


# ══════════════════════════════════════════════════════════════════════════════
# 2.  Formatting helpers
# ══════════════════════════════════════════════════════════════════════════════

def _mat_to_str_table(M: Matrix, label: str) -> str:
    """Render a d×d matrix as a readable text table."""
    dim = M.shape[0]
    lines = [f"\n{'─'*70}", f"  {label}  ({dim}×{dim})", f"{'─'*70}"]
    # column widths
    cells = [[str(M[r, c]) for c in range(dim)] for r in range(dim)]
    col_w = [max(len(cells[r][c]) for r in range(dim)) for c in range(dim)]
    for row in cells:
        parts = [f"{cell:>{col_w[c]}}" for c, cell in enumerate(row)]
        lines.append("  │  " + "  │  ".join(parts) + "  │")
    lines.append(f"{'─'*70}")
    return "\n".join(lines)


def _mat_to_latex(M: Matrix, label: str) -> str:
    """Render a matrix as LaTeX pmatrix."""
    rows = []
    for r in range(M.shape[0]):
        row = " & ".join(sp.latex(M[r, c]) for c in range(M.shape[1]))
        rows.append("  " + row)
    body = " \\\\\n".join(rows)
    return (f"\\[\n{label} = \\begin{{pmatrix}}\n{body}\n"
            f"\\end{{pmatrix}}\n\\]")


def _diag_to_str(M: Matrix, label: str) -> str:
    """For diagonal matrices, show just the diagonal entries."""
    dim = M.shape[0]
    entries = [str(M[i, i]) for i in range(dim)]
    return f"  {label} = diag( " + ",  ".join(entries) + " )"


# ══════════════════════════════════════════════════════════════════════════════
# 3.  Per-hit report builder
# ══════════════════════════════════════════════════════════════════════════════

def report_hit(rec: dict, do_simplify: bool = False,
               do_Xi: bool = True) -> tuple[str, str]:
    """
    Returns (text_report, latex_report) for one CMF hit.
    do_simplify=True uses sp.simplify() on X_i (expensive for large dim).
    do_Xi=False skips X_i computation for dim > max_xi_dim (just show G,D,L,U).
    """
    dim  = rec["dim"]
    fp   = rec.get("fingerprint", "?")
    dlts = rec.get("deltas", [])
    n_pi = rec.get("n_pairs_checked", dim*(dim-1)//2)
    pi_e = rec.get("max_flatness_error", 0.0)
    bst  = rec.get("best_delta", 0.0)
    n_cv = rec.get("n_converging_axes", sum(1 for d in dlts if d > 1.0))

    coords = _make_coord_syms(dim)
    params = rec["params"]
    params["dim"] = dim   # ensure dim is set

    G, L_mat, D_mat, U_mat, G_inv = _build_G_sym(params, coords)

    # ── Text header ───────────────────────────────────────────────────────────
    hdr = (f"\n{'═'*70}\n"
           f"  {dim}×{dim} CMF  ({dim} MATRICES of {dim}×{dim})\n"
           f"  fingerprint  : {fp}\n"
           f"  best_delta   : {bst:.3f}  converging axes: {n_cv}/{dim}\n"
           f"  C({dim},2)={n_pi} path-independence pairs verified\n"
           f"  max flatness error: {pi_e:.2e}\n"
           f"  deltas per axis  : {[round(d,2) for d in dlts]}\n"
           f"{'═'*70}\n"
           f"  Coordinates: {', '.join(str(c) for c in coords)}\n"
           f"  G(n) = L · D_diag(n) · U\n")

    txt = hdr
    tex = (f"\\section*{{{dim}\\times {dim}\\text{{ CMF }}  "
           f"\\texttt{{{fp[:8]}...}}}}\n"
           f"\\textbf{{best\\_delta}}={bst:.3f},\\quad "
           f"\\textbf{{converging axes}}={n_cv}/{dim}\n\n")

    # ── G, L, D, U ────────────────────────────────────────────────────────────
    txt += _mat_to_str_table(G, f"G(n)  —  gauge matrix  [L·D·U]")
    tex += _mat_to_latex(G, "G(\\mathbf{n})")

    txt += "\n"
    txt += _mat_to_str_table(L_mat, "L  (unit lower triangular, constant)")
    tex += _mat_to_latex(L_mat, "L")

    txt += "\n"
    txt += "\n" + _diag_to_str(D_mat,
              "D_diag(n)  [diagonal entries of G before L,U mix]")
    tex += "\n" + _mat_to_latex(D_mat, "D_{\\rm diag}(\\mathbf{n})")

    txt += "\n"
    txt += _mat_to_str_table(U_mat, "U  (unit upper triangular, constant)")
    tex += _mat_to_latex(U_mat, "U")

    # ── D_i canonical kernels ─────────────────────────────────────────────────
    txt += "\n\n  ── Canonical kernel matrices D_i (same structure for all hits) ──"
    for i in range(dim):
        Di = _build_Di_sym(dim, i, coords)
        txt += "\n" + _diag_to_str(Di, f"D_{i}(n{i})")

    # ── X_i matrices ──────────────────────────────────────────────────────────
    if do_Xi:
        txt += "\n\n  ── CMF matrices  X_i(n) = G(n+e_i) · D_i(n_i) · G(n)^{{-1}} ──\n"
        tex += "\n\\subsection*{CMF matrices $X_i$}\n"
        for i in range(dim):
            Di = _build_Di_sym(dim, i, coords)
            Xi = _build_Xi_sym(G_inv, L_mat, D_mat, U_mat, Di,
                               params, i, coords, do_simplify)
            lbl = f"X_{i}(n)  —  matrix along axis {i}  (shift n{i} → n{i}+1)"
            txt += _mat_to_str_table(Xi, lbl)
            txt += "\n"
            tex += _mat_to_latex(Xi, f"X_{{{i}}}(\\mathbf{{n}})")
    else:
        txt += (f"\n  [X_i reconstruction skipped for dim={dim} — "
                f"run with --simplify or --force-xi to enable]\n")
        txt += ("  Formula: X_i(n) = G(n+e_i) · diag(n_i, n_i+1, …, n_i+{dim-1})"
                " · G(n)^{{-1}}\n")

    return txt, tex


# ══════════════════════════════════════════════════════════════════════════════
# 4.  Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dim", type=int, nargs="*",
                    help="Dimension(s) to show (default: all)")
    ap.add_argument("--simplify", action="store_true",
                    help="Run sp.simplify() on X_i entries (slow for large dim)")
    ap.add_argument("--max-xi-dim", type=int, default=10,
                    help="Max dim for full X_i reconstruction (default 10)")
    ap.add_argument("--force-xi", action="store_true",
                    help="Force X_i reconstruction even for large dim")
    args = ap.parse_args()

    out_md  = HERE / "cmf_matrices_report.md"
    out_tex = HERE / "cmf_matrices_latex.tex"

    # Find store files
    stores = sorted(HERE.glob("store_C_*x*.jsonl"))
    if not stores:
        print("No store_C_*.jsonl files found. Run agent_c_large.py first.")
        sys.exit(0)

    all_text = ["# Agent C CMF Matrices — Symbolic Reconstruction\n\n"]
    all_tex  = [
        "\\documentclass{article}\n"
        "\\usepackage{amsmath,amssymb,geometry}\n"
        "\\geometry{margin=1.5cm,landscape}\n"
        "\\begin{document}\n"
        "\\title{Agent C CMF Matrices}\n\\maketitle\n"
    ]

    total_hits = 0

    for store_path in stores:
        # Extract dim from filename
        name = store_path.stem   # e.g. store_C_10x10
        try:
            d = int(name.split("_")[2].split("x")[0])
        except Exception:
            continue

        if args.dim is not None and d not in args.dim:
            continue

        lines = [l for l in store_path.read_text().splitlines() if l.strip()]
        if not lines:
            print(f"  store_C_{d}x{d}.jsonl — empty, skipping")
            continue

        print(f"\n{'─'*60}")
        print(f"  Processing store_C_{d}x{d}.jsonl  ({len(lines)} hit(s))")

        for idx, line in enumerate(lines):
            rec = json.loads(line)
            rec_dim = rec.get("dim", d)

            do_Xi = (args.force_xi or rec_dim <= args.max_xi_dim)

            print(f"  [{idx+1}/{len(lines)}] dim={rec_dim}  "
                  f"fp={rec.get('fingerprint','?')}  "
                  f"reconstructing G + {'X_i' if do_Xi else 'G only'} …",
                  end="", flush=True)

            txt, tex = report_hit(rec, do_simplify=args.simplify, do_Xi=do_Xi)
            all_text.append(txt)
            all_tex.append(tex)
            total_hits += 1
            print(" done")

    all_tex.append("\\end{document}\n")

    # Write outputs
    with open(out_md, "w") as f:
        f.write("\n".join(all_text))

    with open(out_tex, "w") as f:
        f.write("\n".join(all_tex))

    print(f"\n{'═'*60}")
    print(f"  {total_hits} hits processed")
    print(f"  Text report : {out_md}")
    print(f"  LaTeX source: {out_tex}")
    print(f"{'═'*60}")

    # Print text report to stdout so user can see it directly
    print("\n" + "\n".join(all_text))


if __name__ == "__main__":
    main()
