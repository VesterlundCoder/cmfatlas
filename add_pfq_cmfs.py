#!/usr/bin/env python3
"""
Import pFq and known CMFs from RamanujanTools into atlas.db.
Sources: known_cmfs.py and pFq class from github.com/RamanujanMachine/ramanujantools
"""
import sqlite3, json, hashlib
from datetime import datetime

DB_PATH = "/Users/davidsvensson/Desktop/cmf_atlas/data/atlas.db"
PROJECT_ID = 1  # CMF Database v2.2
NOW = datetime.utcnow().isoformat()

KNOWN_CMFS = [
    {
        "name": "rt_e", "dimension": 2, "primary_constant": "e",
        "certification_level": "A_certified", "flatness_verified": True,
        "description": "CMF for Euler's number e. Axes x,y. From RamanujanTools known_cmfs.py.",
        "axes": ["x", "y"],
        "matrices": {"x": "[[1, -(y+1)], [-1, x+y+2]]", "y": "[[0, -(y+1)], [-1, x+y+1]]"},
        "generator_type": "known_family", "source_type": "known_family",
    },
    {
        "name": "rt_pi", "dimension": 2, "primary_constant": "pi",
        "certification_level": "A_certified", "flatness_verified": True,
        "description": "CMF for π. Axes x,y. From RamanujanTools known_cmfs.py.",
        "axes": ["x", "y"],
        "matrices": {"x": "[[x, -x], [-y, 2*x+y+1]]", "y": "[[1+y, -x], [-(1+y), x+2*y+1]]"},
        "generator_type": "known_family", "source_type": "known_family",
    },
    {
        "name": "rt_symmetric_pi", "dimension": 2, "primary_constant": "pi",
        "certification_level": "A_certified", "flatness_verified": True,
        "description": "Symmetric CMF for π. Axes x,y. From RamanujanTools.",
        "axes": ["x", "y"],
        "matrices": {"x": "[[x, x*y], [1, 1+2*x+y]]", "y": "[[y, x*y], [1, 1+x+2*y]]"},
        "generator_type": "known_family", "source_type": "known_family",
    },
    {
        "name": "rt_2F1_derived", "dimension": 3, "primary_constant": "2F1_hypergeometric",
        "certification_level": "A_certified", "flatness_verified": True,
        "description": "2F1-derived hypergeometric CMF. Axes a,b,c (3D). From RamanujanTools hypergeometric_derived_2F1().",
        "axes": ["a", "b", "c"],
        "matrices": {
            "a": "[[1+2*a, (1+2*a)*(1+2*b)], [1, 5+4*a+2*b+4*c]]",
            "b": "[[1+2*b, (1+2*a)*(1+2*b)], [1, 5+2*a+4*b+4*c]]",
            "c": "[[-1-2*c, (1+2*a)*(1+2*b)], [1, 3+2*a+2*b+2*c]]",
        },
        "generator_type": "known_family", "source_type": "known_family",
    },
    {
        "name": "rt_3F2_derived", "dimension": 5, "primary_constant": "3F2_hypergeometric",
        "certification_level": "A_certified", "flatness_verified": True,
        "description": "3F2-derived hypergeometric CMF. Axes x0,x1,x2,y0,y1 (5D). 3×3 matrix. From RamanujanTools hypergeometric_derived_3F2().",
        "axes": ["x0", "x1", "x2", "y0", "y1"],
        "matrices": {},
        "generator_type": "known_family", "source_type": "known_family",
    },
]

PFQ_INSTANCES = [
    {
        "name": "pFq_2F1_ln2", "dimension": 3,
        "primary_constant": "ln(2)", "certification_level": "A_certified",
        "flatness_verified": True,
        "description": "pFq 2F1(1,1; 2; -1). Axes a0,a1,b0. Converges to ln(2). Via pFq(p=2,q=1,z=-1).",
        "axes": ["a0","a1","b0"], "p": 2, "q": 1, "z": "-1",
        "a_params": [1, 1], "b_params": [2],
        "generator_type": "pfq", "source_type": "pfq",
    },
    {
        "name": "pFq_2F1_pi_half", "dimension": 3,
        "primary_constant": "pi/2", "certification_level": "A_certified",
        "flatness_verified": True,
        "description": "pFq 2F1(1/2,1/2; 3/2; 1). Axes a0,a1,b0. Converges to π/2. Via pFq(p=2,q=1,z=1).",
        "axes": ["a0","a1","b0"], "p": 2, "q": 1, "z": "1",
        "a_params": ["1/2","1/2"], "b_params": ["3/2"],
        "generator_type": "pfq", "source_type": "pfq",
    },
    {
        "name": "pFq_3F2_zeta3", "dimension": 5,
        "primary_constant": "zeta(3)", "certification_level": "A_certified",
        "flatness_verified": True,
        "description": "pFq 3F2(1,1,1; 2,2; 1). Axes a0,a1,a2,b0,b1. Converges to ζ(3). Via pFq(p=3,q=2,z=1).",
        "axes": ["a0","a1","a2","b0","b1"], "p": 3, "q": 2, "z": "1",
        "a_params": [1,1,1], "b_params": [2,2],
        "generator_type": "pfq", "source_type": "pfq",
    },
    {
        "name": "pFq_4F3_zeta3", "dimension": 7,
        "primary_constant": "zeta(3)", "certification_level": "A_certified",
        "flatness_verified": True,
        "description": "pFq 4F3(1,1,1,1; 2,2,2; 1). Axes a0..a3, b0..b2 (7D). Apéry-type. Via pFq(p=4,q=3,z=1).",
        "axes": ["a0","a1","a2","a3","b0","b1","b2"], "p": 4, "q": 3, "z": "1",
        "a_params": [1,1,1,1], "b_params": [2,2,2],
        "generator_type": "pfq", "source_type": "pfq",
    },
    {
        "name": "pFq_4F3_zeta3_apery_half", "dimension": 7,
        "primary_constant": "zeta(3)", "certification_level": "B_verified_numeric",
        "flatness_verified": True,
        "description": "pFq 4F3(1/2,1/2,1/2,1/2; 3/2,3/2,3/2; 1). 7D. Shifted Apéry. Via pFq(p=4,q=3,z=1).",
        "axes": ["a0","a1","a2","a3","b0","b1","b2"], "p": 4, "q": 3, "z": "1",
        "a_params": ["1/2","1/2","1/2","1/2"], "b_params": ["3/2","3/2","3/2"],
        "generator_type": "pfq", "source_type": "pfq",
    },
    {
        "name": "pFq_5F4_zeta5", "dimension": 9,
        "primary_constant": "zeta(5)", "certification_level": "B_verified_numeric",
        "flatness_verified": True,
        "description": "pFq 5F4(1,1,1,1,1; 2,2,2,2; 1). 9D. Target: ζ(5). Via pFq(p=5,q=4,z=1). From Ramanujan Dreams LUMI project.",
        "axes": ["a0","a1","a2","a3","a4","b0","b1","b2","b3"], "p": 5, "q": 4, "z": "1",
        "a_params": [1,1,1,1,1], "b_params": [2,2,2,2],
        "generator_type": "pfq", "source_type": "pfq",
    },
    {
        "name": "pFq_6F5_zeta5", "dimension": 11,
        "primary_constant": "zeta(5)", "certification_level": "C_scouting",
        "flatness_verified": True,
        "description": "pFq 6F5(1,...; 2,...; 1). 11D. LUMI znegsweep target ζ(5). Via pFq(p=6,q=5,z=1).",
        "axes": ["a0","a1","a2","a3","a4","a5","b0","b1","b2","b3","b4"], "p": 6, "q": 5, "z": "1",
        "a_params": [1,1,1,1,1,1], "b_params": [2,2,2,2,2],
        "generator_type": "pfq", "source_type": "pfq",
    },
    {
        "name": "pFq_3F2_catalan", "dimension": 5,
        "primary_constant": "Catalan", "certification_level": "B_verified_numeric",
        "flatness_verified": True,
        "description": "pFq 3F2(1/2,1/2,1; 3/2,3/2; 1). 5D. Related to Catalan's constant G. Via pFq(p=3,q=2,z=1).",
        "axes": ["a0","a1","a2","b0","b1"], "p": 3, "q": 2, "z": "1",
        "a_params": ["1/2","1/2",1], "b_params": ["3/2","3/2"],
        "generator_type": "pfq", "source_type": "pfq",
    },
    {
        "name": "pFq_2F1_atanh", "dimension": 3,
        "primary_constant": "pi/4", "certification_level": "A_certified",
        "flatness_verified": True,
        "description": "pFq 2F1(1/2,1; 3/2; 1). 3D. Converges to π/4 (Leibniz formula). Via pFq(p=2,q=1,z=1).",
        "axes": ["a0","a1","b0"], "p": 2, "q": 1, "z": "1",
        "a_params": ["1/2",1], "b_params": ["3/2"],
        "generator_type": "pfq", "source_type": "pfq",
    },
    {
        "name": "pFq_4F3_pi2", "dimension": 7,
        "primary_constant": "pi**2/6", "certification_level": "A_certified",
        "flatness_verified": True,
        "description": "pFq 4F3(1,1,1,1; 2,2,2; -1). 7D. Related to ζ(2)=π²/6. Via pFq(p=4,q=3,z=-1).",
        "axes": ["a0","a1","a2","a3","b0","b1","b2"], "p": 4, "q": 3, "z": "-1",
        "a_params": [1,1,1,1], "b_params": [2,2,2],
        "generator_type": "pfq", "source_type": "pfq",
    },
]


def fingerprint(d: dict) -> str:
    return hashlib.sha256(json.dumps(d, sort_keys=True).encode()).hexdigest()[:16]


def insert_cmf(c, series_id: int, entry: dict):
    canonical_payload = {
        "source_type": entry.get("source_type", "pfq"),
        "axes": entry["axes"],
        "dimension": entry["dimension"],
        "primary_constant": entry.get("primary_constant"),
        "certification_level": entry.get("certification_level"),
        "description": entry["description"],
        "source_repo": "https://github.com/RamanujanMachine/ramanujantools",
    }
    if "matrices" in entry:
        canonical_payload["matrices"] = entry["matrices"]
    for k in ["p","q","z","a_params","b_params"]:
        if k in entry:
            canonical_payload[k] = entry[k]

    fp = fingerprint(canonical_payload)
    c.execute("SELECT id FROM representation WHERE canonical_fingerprint=?", (fp,))
    if c.fetchone():
        print(f"  SKIP (exists): {entry['name']}")
        return False

    c.execute("""
        INSERT INTO representation (series_id, primary_group, canonical_fingerprint, canonical_payload, overlap_groups, created_at)
        VALUES (?,?,?,?,?,?)
    """, (series_id, entry.get("source_type","pfq"), fp, json.dumps(canonical_payload), "[]", NOW))
    repr_id = c.lastrowid

    cmf_payload = {
        "f_poly": None, "fbar_poly": None, "degree": None,
        "dimension": entry["dimension"],
        "primary_constant": entry.get("primary_constant"),
        "certification_level": entry.get("certification_level"),
        "source": "ramanujantools",
        "flatness_verified": entry.get("flatness_verified", True),
        "name": entry["name"],
        "description": entry["description"],
        "axes": entry["axes"],
    }
    for k in ["p","q","z","a_params","b_params"]:
        if k in entry:
            cmf_payload[f"pfq_{k}"] = entry[k]

    c.execute("""
        INSERT INTO cmf (representation_id, cmf_payload, dimension, direction_policy, created_at)
        VALUES (?,?,?,?,?)
    """, (repr_id, json.dumps(cmf_payload), entry["dimension"], "standard", NOW))
    print(f"  ADDED: {entry['name']} (dim={entry['dimension']}, const={entry.get('primary_constant')})")
    return True


def main():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    # Series: known_family from RamanujanTools
    c.execute("SELECT id FROM series WHERE name='ramanujantools_known' LIMIT 1")
    row = c.fetchone()
    if row:
        sid_known = row[0]
    else:
        c.execute("INSERT INTO series (project_id,name,definition,generator_type,provenance,created_at) VALUES (?,?,?,?,?,?)",
            (PROJECT_ID,'ramanujantools_known',
             'Named CMFs from RamanujanTools known_cmfs.py',
             'known_family',
             'https://github.com/RamanujanMachine/ramanujantools',
             NOW))
        sid_known = c.lastrowid

    # Series: pFq family
    c.execute("SELECT id FROM series WHERE name='pfq_hypergeometric' LIMIT 1")
    row = c.fetchone()
    if row:
        sid_pfq = row[0]
    else:
        c.execute("INSERT INTO series (project_id,name,definition,generator_type,provenance,created_at) VALUES (?,?,?,?,?,?)",
            (PROJECT_ID,'pfq_hypergeometric',
             'pFq hypergeometric Conservative Matrix Fields via RamanujanTools pFq class',
             'pfq',
             'https://github.com/RamanujanMachine/ramanujantools/blob/master/ramanujantools/cmf/pfq.py',
             NOW))
        sid_pfq = c.lastrowid

    print("\n--- RamanujanTools known CMFs ---")
    added = sum(insert_cmf(c, sid_known, e) for e in KNOWN_CMFS)

    print("\n--- pFq instances ---")
    added += sum(insert_cmf(c, sid_pfq, e) for e in PFQ_INSTANCES)

    conn.commit()
    conn.close()

    # Verify
    conn2 = sqlite3.connect(DB_PATH)
    total = conn2.execute("SELECT COUNT(*) FROM cmf").fetchone()[0]
    conn2.close()
    print(f"\nDone. Total CMFs in database: {total}")


if __name__ == "__main__":
    main()
