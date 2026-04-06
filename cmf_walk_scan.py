#!/usr/bin/env python3
"""Batch matrix walk + PSLQ for all CMF Hunter and Telescope CMFs."""
import argparse, json, math, sqlite3, time
from pathlib import Path
import mpmath
from sympy import symbols as _sym, sympify as _sp, lambdify as _lam, expand as _exp

DB_PATH  = Path(__file__).parent / "data" / "atlas_2d.db"
if not DB_PATH.exists():
    DB_PATH = Path(__file__).parent / "data" / "atlas.db"
OUT_PATH = Path(__file__).parent / "walk_scan_results.jsonl"

CONST_NAMES = ["π","e","ln2","ζ3","ζ5","π²/6","G","4/π","1/π","π²","ln3","√2","φ","√3"]

def const_vec():
    m = mpmath
    return [m.pi, m.e, m.log(2), m.zeta(3), m.zeta(5), m.pi**2/6, m.catalan,
            4/m.pi, 1/m.pi, m.pi**2, m.log(3), m.sqrt(2), (1+m.sqrt(5))/2, m.sqrt(3)]

def build_fns(fp, fb):
    k,m,x,y = _sym("k m x y")
    fe = _sp(fp); fb_e = _sp(fb)
    is3d = 'z' in str(fe.free_symbols)
    if is3d:
        _fr = fe.free_symbols
        xs = next(s for s in _fr if s.name=='x')
        ys = next(s for s in _fr if s.name=='y')
        zs = next(s for s in _fr if s.name=='z')
        gkm  = fe.subs([(xs,k),(ys,m)])
        gbkm = fb_e.subs([(xs,k),(ys,m)])
        be   = _exp(gkm.subs(m,0)*gbkm.subs(m,0))
        ae   = _exp(gkm - gbkm.subs(k,k+1))
        ev   = lambda e,kv,mv,nv: complex(e.subs([(k,kv),(m,mv),(zs,nv)]))
        evb  = lambda e,kv,nv:   complex(e.subs([(k,kv),(zs,nv)]))
        Kx = lambda kv,mv,nv: mpmath.matrix([[0,1],[evb(be,kv+1,nv),ev(ae,kv,mv,nv)]])
        Ky = lambda kv,mv,nv: mpmath.matrix([[ev(gbkm,kv,mv,nv),1],[evb(be,kv,nv),ev(gkm,kv,mv,nv)]])
        Kz = Ky
        return Kx,Ky,Kz,True
    gkm  = fe.subs([(x,k),(y,m)])
    gbkm = fb_e.subs([(x,k),(y,m)])
    be   = _exp(gkm.subs(m,0)*gbkm.subs(m,0))
    ae   = _exp(gkm - gbkm.subs(k,k+1))
    gf   = _lam([k,m],gkm,modules="mpmath")
    gbf  = _lam([k,m],gbkm,modules="mpmath")
    bf   = _lam([k],be,modules="mpmath")
    af   = _lam([k,m],ae,modules="mpmath")
    Kx = lambda kv,mv,nv=0: mpmath.matrix([[0,1],[bf(kv+1),af(kv,mv)]])
    Ky = lambda kv,mv,nv=0: mpmath.matrix([[gbf(kv,mv),1],[bf(kv),gf(kv,mv)]])
    return Kx,Ky,None,False

def walk(Kfn, k0, depth, miles):
    P = mpmath.eye(2); out = {}
    for it in range(1, depth+1):
        try:
            P = P * Kfn(k0+it-1)
            if it%20==0:
                sc = max(abs(float(mpmath.re(P[0,0]))),abs(float(mpmath.re(P[0,1]))),1e-300)
                P  = P/sc
            if it in miles:
                d = P[1,1]; n = P[0,1]
                out[it] = mpmath.re(n/d) if mpmath.fabs(d)>1e-200 else None
        except:
            if it in miles: out[it]=None
    return out

def delta(mv, m1, m2, m3):
    e1,e2,e3 = mv.get(m1),mv.get(m2),mv.get(m3)
    if None in (e1,e2,e3): return None
    d12=abs(float(e2-e1)); d23=abs(float(e3-e2))
    if d12==0: return 0.0
    if d23==0: return 1e6
    return (math.log2(d12)-math.log2(d23))/(m2-m1)

def kz_start(Kz,k,m):
    for s in range(1,30):
        try:
            if abs(float(mpmath.re(Kz(k,m,s)[1,0])))>1e-10: return s
        except: pass
    return 1

def trajectories(is3d, Kx, Ky, Kz):
    t=[]; mvs=[0,1,2,3,5,10]; kvs=[1,2,3,5]; nvs=[1,2,5] if is3d else [0]
    for mv in mvs:
        for nv in nvs:
            t.append((f"Kx_m{mv}_n{nv}", lambda s,mv=mv,nv=nv: Kx(s,mv,nv), 0))
    for kv in kvs:
        for nv in nvs:
            t.append((f"Ky_k{kv}_n{nv}", lambda s,kv=kv,nv=nv: Ky(kv,s,nv), 1))
    if is3d and Kz:
        for kv in [1,2,3]:
            for mv in [0,1,2]:
                t.append((f"Kz_k{kv}_m{mv}", lambda s,kv=kv,mv=mv: Kz(kv,mv,s), kz_start(Kz,kv,mv)))
    return t

def pslq_check(val, cvec):
    if val is None: return []
    fv = float(val)
    if not math.isfinite(fv) or fv==0: return []
    hits=[]
    for name,c in zip(CONST_NAMES,cvec):
        cf=float(c)
        if cf and abs(fv-cf)/max(abs(cf),1e-300)<1e-6:
            hits.append({"method":"float","name":name,"rel_err":abs(fv-cf)/abs(cf)})
    try:
        mpmath.mp.dps=50
        v=mpmath.mpf(fv)
        vec=[mpmath.mpf(1),v]+[mpmath.mpf(float(c)) for c in cvec]
        rel=mpmath.pslq(vec,maxcoeff=200,tol=mpmath.mpf("1e-25"))
        if rel and rel[1]!=0:
            parts=[f"{rel[0]}"]
            for co,nm in zip(rel[2:],CONST_NAMES):
                if co: parts.append(f"{co}·{nm}")
            hits.append({"method":"pslq","name":f"({'+'.join(p for p in parts if p!='0')})/{-rel[1]}","coeffs":list(rel),"rel_err":0.0})
    except: pass
    return sorted(hits,key=lambda x:x["rel_err"])[:5]

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--depth",    type=int,   default=300)
    ap.add_argument("--dps",      type=int,   default=40)
    ap.add_argument("--threshold",type=float, default=-0.5)
    ap.add_argument("--out",      type=str,   default=str(OUT_PATH))
    ap.add_argument("--limit",    type=int,   default=0)
    args=ap.parse_args()

    mpmath.mp.dps=args.dps
    depth=args.depth; thr=args.threshold
    m1,m2,m3=depth//3, 2*depth//3, depth
    miles=set([m1,m2,m3])
    cv=const_vec()

    conn=sqlite3.connect(DB_PATH)
    rows=conn.execute("""
        SELECT id, cmf_payload FROM cmf
        WHERE json_extract(cmf_payload,'$.source') IN ('cmf_hunter','telescope')
        ORDER BY id
    """).fetchall()
    conn.close()
    if args.limit>0: rows=rows[:args.limit]

    print(f"CMF Walk Scan: {len(rows)} CMFs | depth={depth} | milestones=[{m1},{m2},{m3}] | dps={args.dps}")
    print(f"  threshold={thr}  output={args.out}\n")

    n_int=0; n_skip=0; n_err=0; t0=time.time()

    with open(args.out,"w") as fout:
        for ci,(cid,praw) in enumerate(rows):
            p=json.loads(praw) if isinstance(praw,str) else praw
            fp=p.get("f_poly",""); fb=p.get("fbar_poly","")
            src=p.get("source",""); cert=p.get("certification_level","")
            pconst=p.get("primary_constant",None)
            print(f"  [{ci+1}/{len(rows)}] #{cid} ({src}) [{time.time()-t0:.0f}s]",end="",flush=True)
            if not fp or not fb:
                print(" — skip (no poly)"); n_skip+=1; continue
            try:
                Kx,Ky,Kz,is3d=build_fns(fp,fb)
            except Exception as e:
                print(f" — BUILD ERR: {e}"); n_err+=1; continue

            trajs=trajectories(is3d,Kx,Ky,Kz)
            best_d=None; best_lbl=None; best_est=None; trecs=[]

            for lbl,Kfn,k0 in trajs:
                try:
                    mv=walk(Kfn,k0,depth,miles)
                    d=delta(mv,m1,m2,m3)
                    est=mv.get(m3)
                    trecs.append({"label":lbl,"delta":round(float(d),4) if d is not None else None,
                                  "estimate":mpmath.nstr(est,20) if est is not None else None})
                    if d is not None and (best_d is None or d>best_d):
                        best_d=d; best_lbl=lbl; best_est=est
                except Exception as e:
                    trecs.append({"label":lbl,"delta":None,"estimate":None,"error":str(e)[:100]})

            is_int=(best_d is not None and best_d>thr)
            bd_str=f"{best_d:.3f}" if best_d is not None else "N/A"
            print(f"  best_delta={bd_str} {'★ INTERESTING' if is_int else ''}",end="")

            pslq=[]
            if is_int and best_est is not None:
                pslq=pslq_check(best_est,cv)
                if pslq: print(f"  → {pslq[0]['name']}",end="")
            print()

            if is_int:
                n_int+=1
                rec={"cmf_id":cid,"source":src,"cert":cert,"f_poly":fp,"fbar_poly":fb,
                     "is_3d":is3d,"primary_constant":str(pconst) if pconst else None,
                     "best_delta":round(float(best_d),4),"best_traj":best_lbl,
                     "best_estimate":mpmath.nstr(best_est,25) if best_est is not None else None,
                     "pslq":pslq,"trajectories":trecs}
                fout.write(json.dumps(rec)+"\n"); fout.flush()

    total=time.time()-t0
    print(f"\nDone in {total:.1f}s  |  interesting={n_int}/{len(rows)}  |  errors={n_err}  |  skipped={n_skip}")
    print(f"Results → {args.out}")

if __name__=="__main__":
    main()
