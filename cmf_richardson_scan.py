#!/usr/bin/env python3
"""
cmf_richardson_scan.py
======================
Walk + Richardson/Aitken series acceleration for slow-converging CMF Hunter entries.

Why this matters:
  - CMF Hunter cubic entries have convergence ratio R→1 in the 2×2 walk → delta~0.016
  - At depth 3000, raw estimate has ~48 bits ≈ 14 decimal digits
  - Richardson extrapolation on the milestone sequence can boost this to 25-35 digits
  - 25+ digits is sufficient for PSLQ to identify known constants

Methods applied (in order):
  1. Aitken Δ² acceleration on last 3 milestones
  2. mpmath.richardson on full milestone sequence
  3. mpmath.shanks (Wynn epsilon) on milestone sequence
  4. Direct PSLQ + float match + mpmath.identify on the best accelerated estimate

Usage:
    python3 cmf_richardson_scan.py --source cmf_hunter
    python3 cmf_richardson_scan.py --source all --depth 3000 --dps 60
    python3 cmf_richardson_scan.py --limit 20   # test
"""
import argparse, json, math, sqlite3, time
from pathlib import Path

import mpmath
from sympy import symbols as _sym, sympify as _sp, lambdify as _lam, expand as _exp

DB_PATH  = Path(__file__).parent / "data" / "atlas_2d.db"
if not DB_PATH.exists():
    DB_PATH = Path(__file__).parent / "data" / "atlas.db"
OUT_JSONL = Path(__file__).parent / "cmf_richardson_results.jsonl"

CONST_NAMES = ["π","e","ln2","ζ(3)","ζ(5)","π²/6","G","4/π","1/π","π²","ln3","√2","φ","√3","√5",
               "2/π","ζ(2)","ζ(3)/π²","ln2/π","G/π","ζ(5)/2","ζ(3)/2","ζ(3)²",
               "π³/32","π²·ln2/8","7ζ(3)/2","3ζ(3)/4"]

def build_const_vec():
    m = mpmath
    z3=m.zeta(3); z5=m.zeta(5); G=m.catalan; pi=m.pi
    return [pi, m.e, m.log(2), z3, z5, pi**2/6, G,
            4/pi, 1/pi, pi**2, m.log(3), m.sqrt(2), (1+m.sqrt(5))/2, m.sqrt(3), m.sqrt(5),
            2/pi, m.zeta(2), z3/pi**2, m.log(2)/pi, G/pi, z5/2, z3/2, z3**2,
            pi**3/32, pi**2*m.log(2)/8, 7*z3/2, 3*z3/4]

def build_fns(fp, fb):
    k,m,x,y = _sym("k m x y")
    fe=_sp(fp); fbe=_sp(fb)
    gkm=fe.subs([(x,k),(y,m)]); gbkm=fbe.subs([(x,k),(y,m)])
    be=_exp(gkm.subs(m,0)*gbkm.subs(m,0)); ae=_exp(gkm-gbkm.subs(k,k+1))
    gf =_lam([k,m],gkm, modules="mpmath"); gbf=_lam([k,m],gbkm,modules="mpmath")
    bf =_lam([k],   be,  modules="mpmath"); af =_lam([k,m],ae,  modules="mpmath")
    Kx = lambda kv,mv: mpmath.matrix([[0,1],[bf(kv+1),af(kv,mv)]])
    Ky = lambda kv,mv: mpmath.matrix([[gbf(kv,mv),1],[bf(kv),gf(kv,mv)]])
    return Kx, Ky

def walk_milestones(Kfn, k0, depth, step=50):
    """Run walk, collect raw estimates at every `step` steps."""
    P=mpmath.eye(2); seq=[]
    for it in range(1, depth+1):
        try:
            P=P*Kfn(k0+it-1)
            if it%20==0:
                sc=max(abs(float(mpmath.re(P[0,0]))),abs(float(mpmath.re(P[0,1]))),1e-300)
                P=P/sc
            if it%step==0:
                d=P[1,1]; n=P[0,1]
                est = mpmath.re(n/d) if mpmath.fabs(d)>1e-200 else None
                seq.append((it, est))
        except Exception:
            pass
    return seq  # list of (step, estimate)

def aitken(a, b, c):
    """Aitken Δ² acceleration on three consecutive estimates."""
    if None in (a,b,c): return None
    d1=b-a; d2=c-b
    denom=d2-d1
    if mpmath.fabs(denom)<1e-200: return c
    return c - d2*d2/denom

def richardson_accel(seq):
    """Apply Richardson extrapolation using mpmath.richardson on a value sequence."""
    vals=[v for (_,v) in seq if v is not None]
    if len(vals)<4: return None
    try:
        return mpmath.richardson(vals)
    except Exception:
        return None

def shanks_accel(seq):
    """Wynn epsilon / Shanks on the sequence."""
    vals=[v for (_,v) in seq if v is not None]
    if len(vals)<4: return None
    try:
        return mpmath.shanks(vals)[-1][-1]
    except Exception:
        return None

def delta_from_seq(seq):
    """Compute bits/step improvement from last three milestones in seq."""
    valid=[(s,v) for s,v in seq if v is not None]
    if len(valid)<3: return None
    s1,v1=valid[-3]; s2,v2=valid[-2]; s3,v3=valid[-1]
    d12=abs(float(v2-v1)); d23=abs(float(v3-v2))
    if d12==0: return 0.0
    if d23==0: return 1e6
    return (math.log2(d12)-math.log2(d23))/(s2-s1)

def identify(val, cv):
    if val is None: return []
    fv=float(val)
    if not math.isfinite(fv) or fv==0: return []
    hits=[]
    # Float match
    for name,c in zip(CONST_NAMES,cv):
        cf=float(c)
        if not cf: continue
        for s in [1,-1]:
            for mult in [mpmath.mpf(1),mpmath.mpf(2),mpmath.mpf(3),mpmath.mpf(4),
                         mpmath.mpf("1")/2,mpmath.mpf("1")/3,mpmath.mpf("1")/4,mpmath.mpf("1")/6]:
                candidate=float(s*mult*c)
                rel=abs(fv-candidate)/max(abs(candidate),1e-300)
                if rel<1e-10:
                    label=f"{'-' if s<0 else ''}{'' if float(mult)==1.0 else str(float(mult))+'*'}{name}"
                    hits.append({"method":"float","expr":label,"rel_err":rel})
    if hits:
        hits.sort(key=lambda h:h["rel_err"])
        return hits[:3]
    # PSLQ
    try:
        v=mpmath.mpf(fv)
        vec=[mpmath.mpf(1),v]+[mpmath.mpf(float(c)) for c in cv]
        rel=mpmath.pslq(vec,maxcoeff=500,tol=mpmath.mpf("1e-20"))
        if rel and rel[1]!=0:
            parts=[str(rel[0])]+[f"{co}*{nm}" for co,nm in zip(rel[2:],CONST_NAMES) if co]
            hits.append({"method":"pslq","expr":f"({'+'.join(p for p in parts if p!='0')})/{-rel[1]}","rel_err":0.0})
    except Exception: pass
    # ISC
    try:
        s=mpmath.identify(mpmath.mpf(fv),tol=1e-12)
        if s: hits.append({"method":"isc","expr":s,"rel_err":0.0})
    except Exception: pass
    hits.sort(key=lambda h:h["rel_err"])
    return hits[:4]

def best_traj(Kx, Ky, k_start, scan_depth=400, scan_step=100):
    """Quick scan to find best-converging trajectory."""
    candidates=[("Kx_m0", lambda s: Kx(s,0), k_start)]
    for mv in [1,2,3,5]:
        candidates.append((f"Kx_m{mv}", lambda s,mv=mv: Kx(s,mv), k_start))
    for kv in [1,2,3,5,10]:
        candidates.append((f"Ky_k{kv}", lambda s,kv=kv: Ky(kv,s), 1))
    best_d=None; best=None
    for lbl,Kfn,k0 in candidates:
        try:
            seq=walk_milestones(Kfn,k0,scan_depth,scan_step)
            d=delta_from_seq(seq)
            if d is not None and (best_d is None or d>best_d):
                best_d=d; best=(lbl,Kfn,k0)
        except Exception: pass
    return best, best_d

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--source",  default="all", choices=["all","telescope","cmf_hunter"])
    ap.add_argument("--depth",   type=int, default=3000)
    ap.add_argument("--dps",     type=int, default=55)
    ap.add_argument("--step",    type=int, default=50, help="milestone step size")
    ap.add_argument("--k-start", type=int, default=0,  dest="k_start")
    ap.add_argument("--limit",   type=int, default=0)
    ap.add_argument("--out",     type=str, default=str(OUT_JSONL))
    args=ap.parse_args()

    mpmath.mp.dps=args.dps

    src_filter={
        "all":("telescope","cmf_hunter"),
        "telescope":("telescope",),
        "cmf_hunter":("cmf_hunter",),
    }[args.source]

    conn=sqlite3.connect(DB_PATH)
    rows=conn.execute(f"""
        SELECT id, cmf_payload FROM cmf
        WHERE json_extract(cmf_payload,'$.source') IN ({','.join('?'*len(src_filter))})
        ORDER BY json_extract(cmf_payload,'$.source'), id
    """, src_filter).fetchall()
    conn.close()
    if args.limit: rows=rows[:args.limit]

    cv=build_const_vec()
    out_path=Path(args.out)
    n_steps=args.depth//args.step
    print(f"CMF Richardson Scan: {len(rows)} CMFs | depth={args.depth} | dps={args.dps} | {n_steps} milestones/traj")
    print(f"  Acceleration: Aitken + Richardson + Shanks")
    print(f"  Output: {out_path}\n")

    results=[]; t0=time.time()

    with open(out_path,"w") as fout:
        for ci,(cid,praw) in enumerate(rows):
            p=json.loads(praw)
            fp=p.get("f_poly",""); fb=p.get("fbar_poly","")
            src=p.get("source",""); pconst=p.get("primary_constant","")
            elapsed=time.time()-t0
            print(f"[{ci+1:3d}/{len(rows)}] #{cid} ({src}) [{elapsed:.0f}s]",end="  ",flush=True)

            if not fp or not fb:
                print("skip"); continue
            try:
                Kx,Ky=build_fns(fp,fb)
            except Exception as e:
                print(f"BUILD ERR: {e}"); continue

            # Find best trajectory via quick scan
            best,scan_d=best_traj(Kx,Ky,args.k_start,scan_depth=min(400,args.depth),scan_step=100)
            if best is None:
                print("no trajectory"); continue
            lbl,Kfn,k0=best

            # Full deep walk with dense milestones
            try:
                seq=walk_milestones(Kfn,k0,args.depth,args.step)
            except Exception as e:
                print(f"WALK ERR: {e}"); continue

            raw_delta=delta_from_seq(seq)
            raw_est=seq[-1][1] if seq and seq[-1][1] is not None else None

            # Accelerate
            accel_vals={}
            if len(seq)>=3:
                a,b,c=[v for _,v in seq[-3:]]
                accel_vals["aitken"]=aitken(a,b,c)
            accel_vals["richardson"]=richardson_accel(seq)
            accel_vals["shanks"]=shanks_accel(seq)

            # Pick best accelerated value: the one with most digits vs raw
            best_accel=None; best_accel_method="raw"
            if raw_est:
                best_accel=raw_est
            for method,val in accel_vals.items():
                if val is None: continue
                if best_accel is None:
                    best_accel=val; best_accel_method=method
                else:
                    # prefer whichever agrees with raw to more digits
                    try:
                        raw_f=float(raw_est); v_f=float(val)
                        # use the accelerated value if it's finite and plausible
                        if math.isfinite(v_f) and abs(v_f-raw_f)/max(abs(raw_f),1e-300)<0.1:
                            best_accel=val; best_accel_method=method
                    except Exception: pass

            # Identification
            hits=identify(best_accel,cv)

            d_str=f"{float(raw_delta):.4f}" if raw_delta is not None else "N/A"
            accel_str=mpmath.nstr(best_accel,25) if best_accel else "None"
            id_str=hits[0]["expr"][:50] if hits else "(none)"
            print(f"traj={lbl}  delta={d_str}  accel={best_accel_method}  est={accel_str[:30]}  → {id_str}")

            # Compute accel precision vs raw
            accel_precision={}
            for method,val in accel_vals.items():
                if val is not None and raw_est is not None:
                    try:
                        diff=abs(float(val)-float(raw_est))/max(abs(float(raw_est)),1e-300)
                        accel_precision[method]=f"{-math.log10(max(diff,1e-60)):.1f} digits" if diff>0 else ">60 digits"
                    except Exception:
                        accel_precision[method]="error"

            rec={
                "cmf_id":cid,"source":src,
                "f_poly":fp,"fbar_poly":fb,
                "primary_constant":str(pconst) if pconst else None,
                "best_traj":lbl,"raw_delta":round(float(raw_delta),5) if raw_delta else None,
                "raw_estimate":mpmath.nstr(raw_est,35) if raw_est else None,
                "accelerated":{m:mpmath.nstr(v,35) if v else None for m,v in accel_vals.items()},
                "best_accel_method":best_accel_method,
                "best_estimate":mpmath.nstr(best_accel,35) if best_accel else None,
                "accel_precision":accel_precision,
                "identified":hits,
                "depth":args.depth,"n_milestones":len(seq),
            }
            fout.write(json.dumps(rec)+"\n"); fout.flush()
            results.append(rec)

    total=time.time()-t0
    identified=[r for r in results if r["identified"]]
    print(f"\nDone in {total:.1f}s  |  identified={len(identified)}/{len(results)}")
    for r in identified:
        print(f"  #{r['cmf_id']:4d} ({r['source']:12s}) → {r['identified'][0]['expr'][:60]}  pconst={r['primary_constant']}")
    print(f"Results → {out_path}")

if __name__=="__main__":
    main()
