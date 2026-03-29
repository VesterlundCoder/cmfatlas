#!/usr/bin/env sage
# -*- coding: utf-8 -*-
"""
CMF Atlas — Symbolic Flatness Verifier
=======================================
Checks the flatness condition

    K₁(k,m) · K₂(k+1, m)  =  K₂(k,m) · K₁(k, m+1)

for every CMF in atlas_2d.db.  Works in two modes:

  MODE 1 — Polynomial CMFs (2D, have f_poly / fbar_poly)
    Uses QQ[k,m] — exact polynomial arithmetic, very fast.

  MODE 2 — Explicit matrix CMFs (RamanujanTools etc.)
    Uses SR (Symbolic Ring) + simplify_rational().
    Checks ALL axis pairs: K_i · K_j(n+e_i) = K_j · K_i(n+e_j).

Usage
-----
    sage verify_cmfs.sage                    # all CMFs
    sage verify_cmfs.sage --limit 20         # first 20
    sage verify_cmfs.sage --id 5590          # single entry
    sage verify_cmfs.sage --source RamanujanTools
    sage verify_cmfs.sage --source "CMF Hunter" --limit 50
    sage verify_cmfs.sage --mode poly        # polynomial CMFs only
    sage verify_cmfs.sage --mode explicit    # explicit matrix CMFs only
"""

import sqlite3
import json
import sys
import os
import re
import time
import signal
from datetime import datetime

# ── Paths ────────────────────────────────────────────────────────────────────
_DIR = os.path.dirname(os.path.abspath(__file__))
DB   = os.path.join(_DIR, '..', 'data', 'atlas_2d.db')
REPORTS_DIR = os.path.join(_DIR, 'reports')
os.makedirs(REPORTS_DIR, exist_ok=True)

TIMESTAMP = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
REPORT_JSON = os.path.join(REPORTS_DIR, f'verification_{TIMESTAMP}.json')
REPORT_TXT  = os.path.join(REPORTS_DIR, f'verification_{TIMESTAMP}.txt')

# ── CLI argument parsing ─────────────────────────────────────────────────────
_args = sys.argv[1:]

def _get_arg(flag, default=None):
    try:
        idx = _args.index(flag)
        return _args[idx + 1]
    except (ValueError, IndexError):
        return default

LIMIT      = int(_get_arg('--limit', 0)) or None
FILTER_ID  = int(_get_arg('--id',    0)) or None
FILTER_SRC = _get_arg('--source')
FILTER_MODE= _get_arg('--mode')        # 'poly', 'explicit', or None (both)
TIMEOUT_S  = int(_get_arg('--timeout', 30))

# ── Polynomial ring for 2D CMFs ──────────────────────────────────────────────
R2 = PolynomialRing(QQ, names='k,m')
k, m = R2.gens()

# ── Timeout helper ───────────────────────────────────────────────────────────
class TimeoutError(Exception):
    pass

def _timeout_handler(signum, frame):
    raise TimeoutError("timeout")

def with_timeout(fn, seconds=TIMEOUT_S):
    signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(seconds)
    try:
        result = fn()
        signal.alarm(0)
        return result
    except TimeoutError:
        signal.alarm(0)
        raise

# ── MODE 1: Polynomial CMF helpers ──────────────────────────────────────────
def _parse_poly2d(expr_str):
    """Parse a Python polynomial string in x,y into QQ[k,m]."""
    s = expr_str.strip().replace('x', 'k').replace('y', 'm')
    return R2(eval(s, {'k': k, 'm': m}))

def verify_poly_cmf(f_str, fbar_str):
    """
    Build K₁, K₂ from f_poly/fbar_poly and verify flatness in QQ[k,m].

    K₁(k,m) = [[0,        1      ],
                [b(k+1),  a(k,m) ]]

    K₂(k,m) = [[ḡ(k,m),  1      ],
                [b(k),    g(k,m) ]]

    where  g = f_poly,  ḡ = fbar_poly,
           b(k) = g(k,0)·ḡ(k,0),
           a(k,m) = g(k,m) − ḡ(k+1,m).

    Returns (is_flat: bool, detail: str, diff_matrix)
    """
    g    = _parse_poly2d(f_str)
    gbar = _parse_poly2d(fbar_str)

    b    = R2(g.subs({m: R2(0)}) * gbar.subs({m: R2(0)}))   # b(k)
    b1   = R2(b.subs({k: k + 1}))                            # b(k+1)
    a    = R2(g - gbar.subs({k: k + 1}))                     # a(k,m)

    K1 = matrix(R2, [[R2(0), R2(1)], [b1, a]])
    K2 = matrix(R2, [[gbar,  R2(1)], [b,  g]])

    # K₂(k+1, m)
    K2_k1 = matrix(R2, [
        [R2(gbar.subs({k: k+1})), R2(1)],
        [b1,                      R2(g.subs({k: k+1}))],
    ])
    # K₁(k, m+1)  — b1 has no m so b1.subs(m=m+1) = b1
    K1_m1 = matrix(R2, [
        [R2(0), R2(1)],
        [b1,    R2(a.subs({m: m+1}))],
    ])

    diff = K1 * K2_k1 - K2 * K1_m1
    flat = all(e == R2(0) for e in diff.list())

    if flat:
        return True, "K₁·K₂(k+1,m) − K₂·K₁(k,m+1) = 0  [exact in QQ[k,m]]", diff
    else:
        nz = [(i, j, str(diff[i,j]))
              for i in range(diff.nrows())
              for j in range(diff.ncols())
              if diff[i,j] != R2(0)]
        detail = "Non-zero residuals at " + ", ".join(
            f"[{r},{c}]={v[:60]}" for r,c,v in nz[:4])
        return False, detail, diff

# ── MODE 2: Explicit matrix helpers ─────────────────────────────────────────
def _collect_symbols(mats_dict):
    """Return a dict {name: SR variable} covering all axis names + expression tokens."""
    syms = set()
    skip = {'True','False','None','and','or','not','for','if','else',
            'return','import','from','as','range','in','is','lambda',
            'None','pass','break','continue','class','def'}
    for key, val in mats_dict.items():
        syms.add(str(key))
        for tok in re.findall(r'\b([a-zA-Z_][a-zA-Z0-9_]*)\b', str(val)):
            syms.add(tok)
    var_map = {}
    for s in syms - skip:
        try:
            var_map[s] = SR.var(s)
        except Exception:
            pass
    return var_map

def _parse_explicit_matrix(mat_str, var_map):
    """Parse a stored Python list-of-lists string into a SageMath SR matrix."""
    rows = eval(mat_str, {'__builtins__': {}}, var_map)
    n_rows = len(rows)
    n_cols = len(rows[0])
    entries = []
    for row in rows:
        for e in row:
            entries.append(SR(e))
    return matrix(SR, n_rows, n_cols, entries)

def verify_explicit_cmf(mats_dict):
    """
    For each pair of axes (i, j), check:
        K_i · K_j.subs(axis_i → axis_i+1)  =  K_j · K_i.subs(axis_j → axis_j+1)

    Returns (is_flat: bool, detail: str)
    """
    var_map = _collect_symbols(mats_dict)
    axes = list(mats_dict.keys())

    # Parse all matrices
    parsed = {}
    for ax in axes:
        parsed[ax] = _parse_explicit_matrix(mats_dict[ax], dict(var_map))

    pair_results = []
    failed = []

    for ii in range(len(axes)):
        for jj in range(ii + 1, len(axes)):
            ax_i, ax_j = axes[ii], axes[jj]
            v_i = var_map.get(ax_i, SR.var(ax_i))
            v_j = var_map.get(ax_j, SR.var(ax_j))

            Mi  = parsed[ax_i]
            Mj  = parsed[ax_j]

            # K_j shifted along axis i
            Mj_si = Mj.subs({v_i: v_i + 1})
            # K_i shifted along axis j
            Mi_sj = Mi.subs({v_j: v_j + 1})

            diff = (Mi * Mj_si - Mj * Mi_sj).apply_map(
                lambda e: e.simplify_rational()
            )

            flat = all(e == SR(0) for e in diff.list())
            pair_results.append(f"({ax_i},{ax_j})={'✓' if flat else '✗'}")
            if not flat:
                failed.append(f"({ax_i},{ax_j})")

    detail = "; ".join(pair_results)
    return (len(failed) == 0), detail

# ── Load CMFs ────────────────────────────────────────────────────────────────
con = sqlite3.connect(DB)
cur = con.cursor()

rows = cur.execute("""
    SELECT c.id, c.dimension, c.cmf_payload, r.canonical_payload, s.name
    FROM cmf c
    JOIN representation r ON r.id = c.representation_id
    JOIN series s ON s.id = r.series_id
    ORDER BY c.id
""").fetchall()
con.close()

# Apply filters
if FILTER_ID is not None:
    rows = [r for r in rows if r[0] == FILTER_ID]
if FILTER_SRC is not None:
    rows = [r for r in rows if
            FILTER_SRC.lower() in (json.loads(r[2]) or {}).get('source_category','').lower()]
if FILTER_MODE == 'poly':
    rows = [r for r in rows if (json.loads(r[2]) or {}).get('f_poly','')]
if FILTER_MODE == 'explicit':
    rows = [r for r in rows if (json.loads(r[3]) or {}).get('matrices')]
if LIMIT:
    rows = rows[:LIMIT]

print(f"\nCMF Atlas — Symbolic Flatness Verifier")
print(f"SageMath {version()}")
print(f"Database: {os.path.abspath(DB)}")
print(f"Checking {len(rows)} CMFs  (timeout={TIMEOUT_S}s/entry)")
print("=" * 72)

# ── Main verification loop ───────────────────────────────────────────────────
results   = []
n_pass = n_fail = n_skip = n_error = n_timeout = 0

for row in rows:
    cid, dim, cpay, rpay, sname = row
    payload = json.loads(cpay) if cpay else {}
    canon   = json.loads(rpay) if rpay else {}

    f_poly    = payload.get('f_poly', '').strip()
    fbar_poly = payload.get('fbar_poly', '').strip()
    src_cat   = payload.get('source_category', '?')
    cert      = payload.get('certification_level', '?')
    explicit  = canon.get('matrices', {})

    rec = {
        "id": cid, "dimension": dim,
        "source_category": src_cat, "cert_level": cert,
        "f_poly": f_poly, "fbar_poly": fbar_poly,
        "result": None, "detail": "", "time_s": 0.0,
        "mode": None,
    }

    t0 = time.time()

    # ── Decide mode ─────────────────────────────────────────────────────
    if f_poly and fbar_poly:
        rec["mode"] = "poly"
        try:
            def _run():
                return verify_poly_cmf(f_poly, fbar_poly)
            flat, detail, _ = with_timeout(_run, TIMEOUT_S)
            rec["result"] = "PASS" if flat else "FAIL"
            rec["detail"] = detail
            if flat: n_pass += 1
            else:     n_fail += 1
        except TimeoutError:
            rec["result"] = "TIMEOUT"
            rec["detail"] = f"Exceeded {TIMEOUT_S}s"
            n_timeout += 1
        except Exception as ex:
            rec["result"] = "ERROR"
            rec["detail"] = str(ex)[:300]
            n_error += 1

    elif explicit and isinstance(explicit, dict) and len(explicit) >= 2:
        rec["mode"] = "explicit"
        try:
            def _run():
                return verify_explicit_cmf(explicit)
            flat, detail = with_timeout(_run, TIMEOUT_S)
            rec["result"] = "PASS" if flat else "FAIL"
            rec["detail"] = detail
            if flat: n_pass += 1
            else:     n_fail += 1
        except TimeoutError:
            rec["result"] = "TIMEOUT"
            rec["detail"] = f"Exceeded {TIMEOUT_S}s"
            n_timeout += 1
        except Exception as ex:
            rec["result"] = "ERROR"
            rec["detail"] = str(ex)[:300]
            n_error += 1

    else:
        rec["mode"] = "skip"
        rec["result"] = "SKIP"
        rec["detail"] = "No f_poly or explicit matrices available"
        n_skip += 1

    rec["time_s"] = round(time.time() - t0, 3)
    results.append(rec)

    icon = {"PASS": "✓", "FAIL": "✗", "SKIP": "–",
            "ERROR": "!", "TIMEOUT": "⏱", None: "?"}.get(rec["result"], "?")
    poly_lbl = f_poly[:38] if f_poly else (f"{dim}D explicit [{','.join(list(explicit.keys())[:3])}]" if explicit else "—")
    print(f"  [{icon}] #{cid:<6d} {src_cat:<20s} {str(rec['result']):<8s} "
          f"{rec['time_s']:5.1f}s  {poly_lbl}")

# ── Summary ──────────────────────────────────────────────────────────────────
total = len(results)
print()
print("=" * 72)
print(f"  TOTAL   : {total}")
print(f"  ✓ PASS  : {n_pass}  ({100*n_pass//max(total,1)}%)")
print(f"  ✗ FAIL  : {n_fail}")
print(f"  ⏱ TIMEOUT: {n_timeout}")
print(f"  ! ERROR : {n_error}")
print(f"  – SKIP  : {n_skip}")
print()

# Failures / errors detail
fails = [r for r in results if r["result"] in ("FAIL", "ERROR", "TIMEOUT")]
if fails:
    print("── Failures / Errors / Timeouts ──────────────────────────────────────")
    for r in fails:
        print(f"  #{r['id']}  [{r['result']}]  {r['detail'][:120]}")
    print()

# ── Write JSON report ────────────────────────────────────────────────────────
report = {
    "generated_at": datetime.utcnow().isoformat() + "Z",
    "sage_version": str(version()),
    "db": os.path.abspath(DB),
    "filters": {
        "id": FILTER_ID, "source": FILTER_SRC,
        "mode": FILTER_MODE, "limit": LIMIT,
    },
    "total": total,
    "pass": n_pass, "fail": n_fail,
    "skip": n_skip, "error": n_error, "timeout": n_timeout,
    "results": results,
}
class _SageEncoder(json.JSONEncoder):
    def default(self, obj):
        try:
            return int(obj)
        except (TypeError, ValueError):
            pass
        try:
            return float(obj)
        except (TypeError, ValueError):
            pass
        return super().default(obj)

with open(REPORT_JSON, "w") as fh:
    json.dump(report, fh, indent=2, cls=_SageEncoder)

# ── Write human-readable summary ─────────────────────────────────────────────
with open(REPORT_TXT, "w") as fh:
    fh.write(f"CMF Atlas Symbolic Verification Report\n")
    fh.write(f"Generated : {report['generated_at']}\n")
    fh.write(f"SageMath  : {report['sage_version']}\n")
    fh.write(f"Total     : {total}  |  PASS={n_pass}  FAIL={n_fail}  "
             f"TIMEOUT={n_timeout}  ERROR={n_error}  SKIP={n_skip}\n")
    fh.write("=" * 72 + "\n\n")
    for r in results:
        icon = {"PASS": "PASS", "FAIL": "FAIL", "SKIP": "SKIP",
                "ERROR": "ERR ", "TIMEOUT": "TIME"}.get(r["result"], "????")
        fh.write(f"[{icon}] #{r['id']:<6d} dim={r['dimension']}  "
                 f"src={r['source_category']:<20s}  {r['time_s']:.2f}s\n")
        if r["f_poly"]:
            fh.write(f"       f = {r['f_poly']}\n")
        fh.write(f"       {r['detail'][:120]}\n\n")

print(f"JSON report : {REPORT_JSON}")
print(f"Text report : {REPORT_TXT}")
