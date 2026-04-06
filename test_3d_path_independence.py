"""
Path-independence test for all 3D CMFs in the Atlas.

Key discovery:
  b(k,z) for CMF Hunter 3D entries has factors like (k-1)(k-2)(k-3), making
  b=0 for k=1,2,3. The walk must start from k_start = first k where b(k,n)!=0.

Strategy:
  For each 3D CMF with polynomial f(x,y,z):
    1. Build the Kx(k,m,n) matrix from the 2x2 telescope construction.
    2. Auto-detect k_start (first k where b(k,n)!=0 for some valid n).
    3. Walk k=k_start..DEPTH for m=0,1,2,3 at the found n_fixed.
    4. Check convergence: do all m-trajectories converge to the same limit?
       PASS  : max spread < SPREAD_TOL
       FAIL  : spread >= SPREAD_TOL (different limits — not path-independent)
       BROKEN: walks degenerate for all tested parameters

CMF #5594 (A_plus, no f_poly): verified by RamanujanTools — skip, keep.

Results printed; failing CMFs offered for removal from DB.
"""

import sqlite3, json, sympy as sp, mpmath

mpmath.mp.dps = 35

DEPTH      = 3000
SPREAD_TOL = 1e-5     # if spread > this at DEPTH steps, FAIL
K_SEARCH   = list(range(1, 20))   # k values to probe for valid start
N_SEARCH   = [1, 2, 3, 4, 5, 6]  # n_fixed values to probe

DB_PATH = "data/atlas_2d.db"

k_s, m_s, z_s = sp.symbols("k m z")
x_s, y_s      = sp.symbols("x y")


def build_kx_fn(f_str, fb_str):
    """Return callable Kx(k,m,n)->mpmath.matrix and symbolic b_expr/a_expr."""
    f_e  = sp.sympify(f_str)
    fb_e = sp.sympify(fb_str)
    g    = f_e.subs([(x_s, k_s), (y_s, m_s)])
    gb   = fb_e.subs([(x_s, k_s), (y_s, m_s)])
    b_expr = sp.expand(g.subs(m_s, 0) * gb.subs(m_s, 0))
    a_expr = sp.expand(g - gb.subs(k_s, k_s + 1))

    b_lam = sp.lambdify((k_s, z_s), b_expr, "mpmath")
    a_lam = sp.lambdify((k_s, m_s, z_s), a_expr, "mpmath")

    def Kx(k, m, n):
        return mpmath.matrix([[0, 1], [b_lam(k + 1, n), a_lam(k, m, n)]])

    return Kx


def find_start(Kx, k_search=K_SEARCH, n_search=N_SEARCH):
    """Return (k_start, n_fixed) where b(k_start+1, n_fixed) != 0, or (None,None)."""
    for n in n_search:
        for k in k_search:
            try:
                M = Kx(k, 0, n)
                if abs(M[1, 0]) > 1e-10:
                    return k, n
            except Exception:
                continue
    return None, None


def walk_convergent(Kx, m_val, n_val, k_start, depth):
    """Walk from k_start to depth, return convergent or None if degenerate."""
    P = mpmath.eye(2)
    for k in range(k_start, depth + 1):
        P = P * Kx(k, m_val, n_val)
    denom = P[1, 1]
    if abs(denom) < mpmath.mpf("1e-150"):
        return None
    return mpmath.re(P[0, 1] / denom)


def test_cmf(cid, f_str, fb_str, cert, depth=DEPTH):
    print(f"\nCMF #{cid}  cert={cert}")
    try:
        Kx = build_kx_fn(f_str, fb_str)
    except Exception as e:
        print(f"  BUILD ERROR: {e}")
        return "BROKEN"

    k_start, n_fixed = find_start(Kx)
    if k_start is None:
        print(f"  Cannot find valid (k_start, n_fixed). → BROKEN")
        return "BROKEN"

    print(f"  k_start={k_start}, n_fixed={n_fixed}")
    vals = {}
    for m in [0, 1, 2, 3]:
        v = walk_convergent(Kx, m, n_fixed, k_start, depth)
        vals[m] = v
        tag = f"{float(v):.12f}" if v is not None else "FAIL"
        print(f"  m={m}: {tag}")

    finite = [v for v in vals.values() if v is not None]
    if len(finite) < 2:
        print(f"  Too few valid walks. → BROKEN")
        return "BROKEN"

    spread = float(max(finite) - min(finite))
    print(f"  Spread: {spread:.3e}  (tol={SPREAD_TOL:.0e})")

    if spread < SPREAD_TOL:
        print(f"  → PASS ✓  (path-independent within tolerance)")
        return "PASS"
    else:
        print(f"  → FAIL ✗  (spread too large — NOT path-independent via heuristic)")
        return "FAIL"


def main():
    con = sqlite3.connect(DB_PATH)
    rows = con.execute(
        "SELECT id, cmf_payload FROM cmf WHERE dimension=3 ORDER BY id"
    ).fetchall()

    print(f"Testing {len(rows)} 3D CMFs  (depth={DEPTH}, spread_tol={SPREAD_TOL:.0e})")
    print("=" * 70)

    results = {}
    for r in rows:
        cid, pl = r[0], json.loads(r[1])
        cert   = pl.get("certification_level", "?")
        f_str  = pl.get("f_poly", "")
        fb_str = pl.get("fbar_poly", "")
        if not f_str or "z" not in f_str:
            print(f"\nCMF #{cid}  cert={cert}  — no 3D polynomial, skipping (keep)")
            results[cid] = "SKIP"
            continue
        verdict = test_cmf(cid, f_str, fb_str, cert)
        results[cid] = verdict

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    to_remove = []
    for cid, verdict in results.items():
        marker = "⚠  REMOVE" if verdict in ("FAIL", "BROKEN") else "✓  KEEP"
        print(f"  CMF #{cid:<6} {verdict:<8}  {marker}")
        if verdict in ("FAIL", "BROKEN"):
            to_remove.append(cid)

    if not to_remove:
        print("\nAll 3D CMFs pass — nothing to remove.")
        con.close()
        return

    print(f"\nCMFs to remove: {to_remove}")
    resp = input("Proceed with deletion? [y/N] ").strip().lower()
    if resp == "y":
        remove_cmfs(con, to_remove)
    else:
        print("Aborted — nothing deleted.")
    con.close()


def remove_cmfs(con, ids):
    """Remove CMF rows and orphan representations/series."""
    cur = con.cursor()
    for cid in ids:
        row = cur.execute(
            "SELECT representation_id FROM cmf WHERE id=?", (cid,)
        ).fetchone()
        if not row:
            print(f"  CMF #{cid} not found (already deleted?), skipping.")
            continue
        repr_id = row[0]

        # Delete eval_runs and their recognition_attempts first
        eval_ids = [r[0] for r in cur.execute(
            "SELECT id FROM eval_run WHERE cmf_id=?", (cid,)
        ).fetchall()]
        for eid in eval_ids:
            cur.execute("DELETE FROM recognition_attempt WHERE eval_run_id=?", (eid,))
        cur.execute("DELETE FROM eval_run WHERE cmf_id=?", (cid,))
        cur.execute("DELETE FROM cmf WHERE id=?", (cid,))
        print(f"  Deleted CMF #{cid}")

        remaining = cur.execute(
            "SELECT COUNT(*) FROM cmf WHERE representation_id=?", (repr_id,)
        ).fetchone()[0]

        if remaining == 0:
            series_row = cur.execute(
                "SELECT series_id FROM representation WHERE id=?", (repr_id,)
            ).fetchone()
            cur.execute("DELETE FROM features WHERE representation_id=?", (repr_id,))
            cur.execute("DELETE FROM representation_equivalence WHERE representation_id=?", (repr_id,))
            cur.execute("DELETE FROM representation WHERE id=?", (repr_id,))
            print(f"    → removed orphan representation #{repr_id}")

            if series_row:
                sid = series_row[0]
                still_ref = cur.execute(
                    "SELECT COUNT(*) FROM representation WHERE series_id=?", (sid,)
                ).fetchone()[0]
                if still_ref == 0:
                    cur.execute("DELETE FROM series WHERE id=?", (sid,))
                    print(f"    → removed orphan series #{sid}")

    con.commit()
    print("Done.")


if __name__ == "__main__":
    main()
