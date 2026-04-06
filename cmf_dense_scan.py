#!/usr/bin/env python3
"""
cmf_dense_scan.py
=================
Dense trajectory scan for CMF Hunter (and telescope) entries.

Key improvement over walk_scan.py:
  - Much wider (k_fixed, m_fixed) grid: k ∈ 1..15, m ∈ 0..15
  - Non-zero k_start (default 30): skips the degenerate low-k zone where b(k)≈0
  - Diagonal trajectories: k=m+c for c ∈ {0,1,2,3}
  - Higher depth (default 600) + dps (default 50)
  - Saves ALL trajectory results (not just best), for post-analysis

Usage:
    python3 cmf_dense_scan.py                           # all sources
    python3 cmf_dense_scan.py --source cmf_hunter
    python3 cmf_dense_scan.py --depth 800 --dps 60 --k-start 50
    python3 cmf_dense_scan.py --limit 10               # test
"""
import argparse, json, math, sqlite3, time
from pathlib import Path

import mpmath
from sympy import symbols as _sym, sympify as _sp, lambdify as _lam, expand as _exp

DB_PATH   = Path(__file__).parent / "data" / "atlas_2d.db"
if not DB_PATH.exists():
    DB_PATH = Path(__file__).parent / "data" / "atlas.db"
OUT_JSONL = Path(__file__).parent / "cmf_dense_scan.jsonl"

CONST_NAMES = ["π","e","ln2","ζ(3)","ζ(5)","π²/6","G","4/π","1/π","π²","ln3","√2","φ","√3","√5","2/π","ζ(2)","ζ(3)/π²"]

def build_const_vec():
    m = mpmath
    z3=m.zeta(3); z5=m.zeta(5); z2=m.zeta(2); G=m.catalan; pi=m.pi
    return [pi,m.e,m.log(2),z3,z5,pi**2/6,G,4/pi,1/pi,pi**2,m.log(3),m.sqrt(2),(1+m.sqrt(5))/2,m.sqrt(3),m.sqrt(5),2/pi,z2,z3/pi**2]

def build_fns(fp, fb):
    k,m,x,y = _sym("k m x y")
    fe=_sp(fp); fbe=_sp(fb)
    gkm=fe.subs([(x,k),(y,m)]); gbkm=fbe.subs([(x,k),(y,m)])
    be=_exp(gkm.subs(m,0)*gbkm.subs(m,0)); ae=_exp(gkm-gbkm.subs(k,k+1))
    gf=_lam([k,m],gkm,modules="mpmath"); gbf=_lam([k,m],gbkm,modules="mpmath")
    bf=_lam([k],be,modules="mpmath");    af=_lam([k,m],ae,modules="mpmath")
    Kx = lambda kv,mv: mpmath.matrix([[0,1],[bf(kv+1),af(kv,mv)]])
    Ky = lambda kv,mv: mpmath.matrix([[gbf(kv,mv),1],[bf(kv),gf(kv,mv)]])
    return Kx, Ky

def walk_delta_est(Kfn, k0, depth):
    m1,m2,m3 = depth//3, 2*depth//3, depth
    P=mpmath.eye(2); ests={}
    for it in range(1,depth+1):
        try:
            P=P*Kfn(k0+it-1)
            if it%20==0:
                sc=max(abs(float(mpmath.re(P[0,0]))),abs(float(mpmath.re(P[0,1]))),1e-300)
                P=P/sc
            if it in (m1,m2,m3):
                d=P[1,1]; n=P[0,1]
                ests[it]= mpmath.re(n/d) if mpmath.fabs(d)>1e-200 else None
        except Exception: ests[it]=None
    e1,e2,e3 = ests.get(m1),ests.get(m2),ests.get(m3)
    if None in (e1,e2,e3): return None, e3
    d12=abs(float(e2-e1)); d23=abs(float(e3-e2))
    if d12==0: return 0.0, e3
    if d23==0: return 1e6, e3
    return (math.log2(d12)-math.log2(d23))/(m2-m1), e3

def pslq_check(val, cv):
    if val is None: return []
    fv=float(val)
    if not math.isfinite(fv) or fv==0: return []
    hits=[]
    for name,c in zip(CONST_NAMES,cv):
        cf=float(c)
        if not cf: continue
        for s in [1,-1]:
            for mult in [mpmath.mpf(1),mpmath.mpf(2),mpmath.mpf("1")/2,mpmath.mpf(3),mpmath.mpf("1")/3,mpmath.mpf(4),mpmath.mpf("1")/4]:
                if abs(fv - float(s*mult*c))/max(abs(float(s*mult*c)),1e-300) < 1e-9:
                    hits.append({"method":"float","expr":f"{'-' if s<0 else ''}{'' if mult==1 else str(mult)+'*'}{name}"})
    if not hits:
        try:
            v=mpmath.mpf(fv)
            vec=[mpmath.mpf(1),v]+[mpmath.mpf(float(c)) for c in cv]
            rel=mpmath.pslq(vec,maxcoeff=300,tol=mpmath.mpf("1e-30"))
            if rel and rel[1]!=0:
                parts=[str(rel[0])]+[f"{co}*{nm}" for co,nm in zip(rel[2:],CONST_NAMES) if co]
                hits.append({"method":"pslq","expr":f"({'+'.join(p for p in parts if p!='0')})/{-rel[1]}"})
        except Exception: pass
        try:
            s=mpmath.identify(mpmath.mpf(fv),tol=1e-12)
            if s: hits.append({"method":"isc","expr":s})
        except Exception: pass
    return hits[:3]

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--source",   default="all", choices=["all","telescope","cmf_hunter"])
    ap.add_argument("--depth",    type=int, default=600)
    ap.add_argument("--dps",      type=int, default=50)
    ap.add_argument("--k-start",  type=int, default=30, dest="k_start")
    ap.add_argument("--k-max",    type=int, default=15, dest="k_max")
    ap.add_argument("--m-max",    type=int, default=15, dest="m_max")
    ap.add_argument("--diag",     action="store_true", help="also run diagonal trajectories")
    ap.add_argument("--limit",    type=int, default=0)
    ap.add_argument("--out",      type=str, default=str(OUT_JSONL))
    ap.add_argument("--delta-threshold", type=float, default=-0.5, dest="threshold")
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

    # Build trajectory list
    trajs=[]
    for mv in range(args.m_max+1):
        trajs.append(("Kx", mv, "m"))
    for kv in range(1, args.k_max+1):
        trajs.append(("Ky", kv, "k"))
    if args.diag:
        for c in [0,1,2,3,5]:
            trajs.append(("diag", c, "c"))

    cv=build_const_vec()
    k_start=args.k_start
    out_path=Path(args.out)
    t0=time.time()

    print(f"CMF Dense Scan: {len(rows)} CMFs | {len(trajs)} trajectories each | depth={args.depth} | dps={args.dps} | k_start={k_start}")
    print(f"  Grid: Kx m∈[0,{args.m_max}]  Ky k∈[1,{args.k_max}]  {'+ diag' if args.diag else ''}")
    print(f"  Threshold: {args.threshold}  Output: {out_path}\n")

    n_interesting=0; n_err=0

    with open(out_path,"w") as fout:
        for ci,(cid,praw) in enumerate(rows):
            p=json.loads(praw)
            fp=p.get("f_poly",""); fb=p.get("fbar_poly","")
            src=p.get("source",""); pconst=p.get("primary_constant","")
            elapsed=time.time()-t0
            print(f"[{ci+1:3d}/{len(rows)}] #{cid} ({src}) [{elapsed:.0f}s]", end="  ", flush=True)

            if not fp or not fb:
                print("skip"); continue

            try:
                Kx,Ky=build_fns(fp,fb)
            except Exception as e:
                print(f"BUILD ERR: {e}"); n_err+=1; continue

            best_d=None; best_traj=None; best_est=None
            all_traj_results=[]

            for dir_type, param, param_name in trajs:
                try:
                    if dir_type=="Kx":
                        Kfn=lambda s,mv=param: Kx(s,mv)
                        lbl=f"Kx_m{param}_ks{k_start}"
                    elif dir_type=="Ky":
                        Kfn=lambda s,kv=param: Ky(kv,s)
                        lbl=f"Ky_k{param}_ks{k_start}"
                    else:
                        c=param
                        Kfn=lambda s,c=c: Kx(s,s+c) if s+c>=0 else Kx(s,0)
                        lbl=f"diag_c{c}_ks{k_start}"

                    d,est=walk_delta_est(Kfn, k_start, args.depth)
                    if d is None: continue

                    all_traj_results.append({
                        "traj":lbl,"delta":round(float(d),5),
                        "estimate":mpmath.nstr(est,25) if est else None
                    })
                    if best_d is None or d>best_d:
                        best_d=d; best_traj=lbl; best_est=est
                except Exception:
                    continue

            if not all_traj_results:
                print("no valid trajectories"); continue

            best_d_f=float(best_d) if best_d is not None else -999
            is_interesting = best_d_f >= args.threshold

            # PSLQ only on interesting ones with best estimate
            pslq_hits=[]
            if is_interesting and best_est is not None:
                pslq_hits=pslq_check(best_est,cv)

            d_str=f"{best_d_f:.3f}" if math.isfinite(best_d_f) else "N/A"
            flag="★" if is_interesting else " "
            pslq_str=pslq_hits[0]["expr"][:40] if pslq_hits else ""
            print(f"best={best_traj}  delta={d_str}  {flag}  {pslq_str}")

            if is_interesting: n_interesting+=1

            rec={
                "cmf_id":cid,"source":src,
                "f_poly":fp,"fbar_poly":fb,
                "primary_constant":str(pconst) if pconst else None,
                "k_start":k_start,
                "best_traj":best_traj,
                "best_delta":round(best_d_f,5),
                "best_estimate":mpmath.nstr(best_est,30) if best_est else None,
                "pslq":pslq_hits,
                "trajectories":all_traj_results,
            }
            fout.write(json.dumps(rec)+"\n"); fout.flush()

    total=time.time()-t0
    print(f"\nDone in {total:.1f}s | interesting={n_interesting}/{len(rows)} | errors={n_err}")
    print(f"Results → {out_path}")

if __name__=="__main__":
    main()
