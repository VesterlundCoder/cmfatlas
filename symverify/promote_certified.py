#!/usr/bin/env python3
"""
Promote CMFs that passed symbolic verification to A_certified.

Usage:
    python3 promote_certified.py reports/verification_YYYYMMDD_HHMMSS.json
    python3 promote_certified.py reports/verification_YYYYMMDD_HHMMSS.json --dry-run
"""
import sqlite3
import json
import sys
import os

DRY_RUN = '--dry-run' in sys.argv
args = [a for a in sys.argv[1:] if not a.startswith('--')]

if not args:
    print("Usage: python3 promote_certified.py <report.json> [--dry-run]")
    sys.exit(1)

REPORT = args[0]
DB = os.path.join(os.path.dirname(__file__), '..', 'data', 'atlas_2d.db')

with open(REPORT) as f:
    report = json.load(f)

passed = [r for r in report["results"] if r["result"] == "PASS"]
print(f"Report   : {REPORT}")
print(f"Passed   : {len(passed)} / {report['total']}")
print(f"Dry run  : {DRY_RUN}")
print()

# Only promote CMF Hunter entries — RamanujanTools are already A_plus
# and Gauge Transformed are already A_certified for known families
PROMOTABLE_SOURCES = {"CMF Hunter"}
PROMOTE_FROM = {"B_verified_numeric", "C_scouting"}
PROMOTE_TO   = "A_certified"

to_promote = [
    r for r in passed
    if r.get("source_category") in PROMOTABLE_SOURCES
    and r.get("cert_level") in PROMOTE_FROM
]

print(f"Eligible for promotion ({' / '.join(PROMOTABLE_SOURCES)}, "
      f"from {PROMOTE_FROM}): {len(to_promote)}")

if not to_promote:
    print("Nothing to promote.")
    sys.exit(0)

con = sqlite3.connect(DB)
promoted = 0

for r in to_promote:
    cid  = r["id"]
    row  = con.execute("SELECT cmf_payload FROM cmf WHERE id=?", (cid,)).fetchone()
    if not row:
        print(f"  WARN: CMF #{cid} not found in DB")
        continue
    p = json.loads(row[0])
    old_cert = p.get("certification_level", "?")
    p["certification_level"] = PROMOTE_TO
    p["symbolic_verification"] = {
        "verified_at": report["generated_at"],
        "sage_version": report.get("sage_version", ""),
        "method": "exact polynomial arithmetic in QQ[k,m]" if r.get("mode") == "poly"
                  else "simplify_rational() in SR",
        "detail": r["detail"],
    }
    new_payload = json.dumps(p)
    print(f"  #{cid}  {r['source_category']:<20s}  {old_cert} → {PROMOTE_TO}  "
          f"  f={r.get('f_poly','')[:40]}")
    if not DRY_RUN:
        con.execute("UPDATE cmf SET cmf_payload=? WHERE id=?", (new_payload, cid))
        promoted += 1

if not DRY_RUN:
    con.commit()
    print(f"\nPromoted {promoted} CMFs to {PROMOTE_TO}.")
else:
    print(f"\nDry run — no changes written. Remove --dry-run to apply.")

con.close()
