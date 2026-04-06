#!/usr/bin/env python3
"""
cmf_push_identifications.py
============================
Merge walk_identify_results.jsonl (v1) and walk_identify_results_v2.jsonl (v2),
pick the best identification per CMF, and write it into atlas.db cmf_payload.

Fields written:
  cmf_payload.identified_constant       — human-readable expression
  cmf_payload.identification_method     — float / pslq / pconst / isc
  cmf_payload.identification_digits     — matching decimal digits
  cmf_payload.identification_updated_at — ISO timestamp

Run:
    python3 cmf_push_identifications.py             # dry-run (print only)
    python3 cmf_push_identifications.py --write     # actually update atlas.db
"""
import argparse, json, math, sqlite3, time
from pathlib import Path

BASE   = Path(__file__).parent
DB_PATH = BASE / "data" / "atlas_2d.db"
if not DB_PATH.exists():
    DB_PATH = BASE / "data" / "atlas.db"

V1_JSONL = BASE / "walk_identify_results.jsonl"
V2_JSONL = BASE / "walk_identify_results_v2.jsonl"

import re
_RAT_EXP_PAT = re.compile(r'\*\*\s*\(\s*\d+\s*/\s*\d+\s*\)')   # matches **(p/q)

# ISC noise: expressions with rational exponents on primes
def _is_noisy_isc(expr: str) -> bool:
    if not expr:
        return True
    return bool(_RAT_EXP_PAT.search(expr))

def _digits_from_hit(h: dict) -> int:
    """Compute confidence digits from 'digits' key or 'rel_err'.
    ISC always reports rel_err=0.0 even for noise, so never use rel_err for ISC."""
    d = h.get("digits", 0)
    if not d:
        method = h.get("method", "")
        if method in ("isc",):   # ISC rel_err=0.0 is meaningless
            return 0
        rel = h.get("rel_err", 1.0)
        if rel is not None and rel > 0 and rel < 1e-6:
            d = max(1, int(-math.log10(rel)))
    return d


def _score(rec: dict) -> int:
    """Higher = more trustworthy identification."""
    # pconst match: gold standard
    pm = rec.get("pconst_match") or {}
    if pm.get("matches"):
        return 20

    hits = rec.get("identified") or []
    if not hits:
        return 0
    h = hits[0]
    d = _digits_from_hit(h)
    expr = h.get("expr", "")

    if _is_noisy_isc(expr):
        return 0
    # ISC without ** is OK but low confidence
    if h.get("method") == "isc" and "**" not in expr:
        return max(d, 8) if d else 6
    return d


def _best_identification(rec: dict):
    """Return {'expr', 'method', 'digits'} or None."""
    pm = rec.get("pconst_match") or {}
    if pm.get("matches"):
        pc = rec.get("primary_constant", "?")
        return {"expr": str(pc), "method": "pconst", "digits": 20}

    hits = rec.get("identified") or []
    for h in hits:
        expr = h.get("expr", "")
        if _is_noisy_isc(expr):
            continue
        method = h.get("method", "?")
        d = _digits_from_hit(h)
        # Clean ISC with no ** (simple fraction/rational) → assign d=8 as minimum
        if method == "isc" and not d:
            d = 8
        if d >= 6:
            return {"expr": expr, "method": method, "digits": d}
    return None


def load_jsonl(path: Path) -> dict:
    result = {}
    if not path.exists():
        return result
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
            result[r["cmf_id"]] = r
        except Exception:
            pass
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true", help="Actually write to atlas.db")
    args = ap.parse_args()

    v1 = load_jsonl(V1_JSONL)
    v2 = load_jsonl(V2_JSONL)
    print(f"v1: {len(v1)} entries  |  v2: {len(v2)} entries")

    all_ids = sorted(set(list(v1.keys()) + list(v2.keys())))

    # Merge: pick best scored result per CMF
    merged = {}
    for cid in all_ids:
        r1 = v1.get(cid)
        r2 = v2.get(cid)
        if r1 and r2:
            merged[cid] = r1 if _score(r1) >= _score(r2) else r2
        else:
            merged[cid] = r1 or r2

    hits = [(cid, r, _best_identification(r))
            for cid, r in merged.items()
            if _best_identification(r) is not None]
    hits.sort(key=lambda x: -x[2]["digits"])

    print(f"\nConfirmed identifications: {len(hits)}\n")
    print(f"{'CMF':>6}  {'Digits':>6}  {'Method':>8}  {'f_poly':30}  Expression")
    print("-" * 100)
    for cid, rec, ident in hits:
        print(f"#{cid:5d}  {ident['digits']:>6d}  {ident['method']:>8}  "
              f"{rec.get('f_poly','')[:30]:30}  {ident['expr'][:60]}")

    if not args.write:
        print(f"\n[dry-run] Pass --write to update atlas.db at {DB_PATH}")
        return

    if not DB_PATH.exists():
        print(f"ERROR: DB not found at {DB_PATH}")
        return

    con = sqlite3.connect(DB_PATH)
    ts = time.strftime("%Y-%m-%dT%H:%M:%SZ")
    updated = 0

    for cid, rec, ident in hits:
        row = con.execute("SELECT cmf_payload FROM cmf WHERE id=?", (cid,)).fetchone()
        if row is None:
            print(f"  #{cid}: not found in DB — skip")
            continue
        payload = json.loads(row[0]) if isinstance(row[0], str) else row[0]
        payload["identified_constant"]       = ident["expr"]
        payload["identification_method"]     = ident["method"]
        payload["identification_digits"]     = ident["digits"]
        payload["identification_updated_at"] = ts
        con.execute("UPDATE cmf SET cmf_payload=? WHERE id=?",
                    (json.dumps(payload), cid))
        updated += 1

    con.commit()
    con.close()
    print(f"\n✓ Updated {updated} entries in {DB_PATH}")

    # Write a tidy summary JSONL
    out = BASE / "identification_summary.jsonl"
    with open(out, "w") as f:
        for cid, rec, ident in hits:
            f.write(json.dumps({
                "cmf_id": cid,
                "f_poly": rec.get("f_poly"),
                "fbar_poly": rec.get("fbar_poly"),
                "cert": rec.get("cert"),
                "identified_constant": ident["expr"],
                "method": ident["method"],
                "digits": ident["digits"],
                "estimate": rec.get("estimate", ""),
                "deep_delta": rec.get("deep_delta"),
            }) + "\n")
    print(f"✓ Summary → {out}")


if __name__ == "__main__":
    main()
