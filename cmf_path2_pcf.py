#!/usr/bin/env python3
"""
cmf_path2_pcf.py — Shanks/Levin/Richardson acceleration + extended PSLQ
Re-evaluates Euler sums with faster-converging methods and a richer constant basis
including zeta(odd), digamma at rationals, and L-function values.

Usage:
    python3 cmf_path2_pcf.py --reuse cmf_euler_full.jsonl --dps 80
    python3 cmf_path2_pcf.py --source cmf_hunter --dps 80 --limit 20
"""
import argparse, json, math, sqlite3, time
from pathlib import Path
import mpmath
import sympy as sp
from sympy import Symbol, sympify, lambdify, Poly

DB_PATH   = Path(__file__).parent / "data" / "atlas_2d.db"
if not DB_PATH.exists():
    DB_PATH = Path(__file__).parent / "data" / "atlas.db"
OUT_JSONL = Path(__file__).parent / "cmf_path2_results.jsonl"

x_sym = Symbol('x'); y_sym = Symbol('y'); _xp = Symbol('_xp')

# ── Extended constant basis ────────────────────────────────────────
CNAMES = ["ζ(3)","ζ(5)","ζ(7)","ζ(9)","ζ(3)/2","ζ(5)/2","ζ(7)/2",
          "ζ(3)/4","ζ(3)²","7ζ(3)/2","3ζ(3)/4","5ζ(3)/4",
          "π","π²","π³","π²/6","π²/12","G","ln2","ln3","ln2/π","G/π",
          "ψ(½)","ψ(⅓)","ψ(¼)","ψ(⅔)","ψ(¾)",
          "ψ'(½)","ψ'(⅓)","ψ'(¼)","ζ(2)","ζ(4)","ζ(3)/π²","ζ(5)/π⁴"]

def build_cv():
    m = mpmath; z3=m.zeta(3); z5=m.zeta(5); z7=m.zeta(7); z9=m.zeta(9)
    G=m.catalan; pi=m.pi
    return [z3,z5,z7,z9,z3/2,z5/2,z7/2,z3/4,z3**2,7*z3/2,3*z3/4,5*z3/4,
            pi,pi**2,pi**3,pi**2/6,pi**2/12,G,m.log(2),m.log(3),m.log(2)/pi,G/pi,
            m.digamma(m.mpf('1')/2),m.digamma(m.mpf('1')/3),
            m.digamma(m.mpf('1')/4),m.digamma(m.mpf('2')/3),m.digamma(m.mpf('3')/4),
            m.zeta(2,m.mpf('1')/2),m.zeta(2,m.mpf('1')/3),m.zeta(2,m.mpf('1')/4),
            m.zeta(2),m.zeta(4),z3/pi**2,z5/pi**4]

def wynn_epsilon(seq):
    n = len(seq)
    if n < 4: return None
    e = [[mpmath.mpf(0)]*(n+1) for _ in range(n+2)]
    for i in range(n): e[i][1] = seq[i]
    best = seq[-1]
    for k in range(2, n+1):
        for i in range(n-k+1):
            try:
                d = e[i+1][k-1] - e[i][k-1]
                if mpmath.fabs(d) < mpmath.mpf(10)**(-(mpmath.mp.dps-5)): return e[i][k-1]
                e[i][k] = e[i+1][k-2] + 1/d
                if k%2==0 and mpmath.isfinite(e[i][k]): best = e[i][k]
            except Exception: pass
    return best

def euler_sum_fast(f_str, m_val, dps):
    fe = sympify(f_str); f_m = fe.subs(y_sym, m_val)
    try: deg = Poly(f_m.subs(x_sym, _xp).expand(), _xp).degree()
    except: deg = 0
    if deg < 2: return {}
    f_fn = lambdify(x_sym, f_m, modules="mpmath")
    try:
        samp = [complex(f_fn(k)) for k in range(1, 51)]
        if any(abs(v) < 1e-10 for v in samp): return {}
        sign = -1 if all(v.real < 0 for v in samp) else 1
    except: return {}
    old = mpmath.mp.dps; mpmath.mp.dps = dps+20; res = {}
    try:
        fn = lambda k: mpmath.mpf(sign)/f_fn(k)
        # A: euler-maclaurin baseline
        try:
            res['em'] = mpmath.nsum(fn, [1, mpmath.inf], method='euler-maclaurin')
        except: pass
        # B: Shanks via Wynn epsilon on partial sums
        try:
            N = max(300, dps*5); acc=mpmath.mpf(0); pts=[]
            for k in range(1, N+1):
                acc += fn(k)
                if k >= N//4: pts.append(acc)
            v = wynn_epsilon(pts)
            if v is not None: res['shanks'] = v
        except: pass
        # C: Levin
        try: res['levin'] = mpmath.nsum(fn, [1, mpmath.inf], method='levin')
        except: pass
    finally: mpmath.mp.dps = old
    return res

def identify_val(val, cv, dps):
    if val is None: return []
    try: fv = float(val)
    except: return []
    if not math.isfinite(fv) or fv == 0: return []
    hits = []
    for name,c in zip(CNAMES, cv):
        cf = float(c)
        if not cf: continue
        for s in [1,-1]:
            for mult in [1,2,3,4,mpmath.mpf('1')/2,mpmath.mpf('1')/3,mpmath.mpf('1')/4]:
                cand = float(s*mult*float(cf))
                rel = abs(fv-cand)/max(abs(cand),1e-300)
                if rel < 1e-15:
                    hits.append({'method':'float','expr':f"{'-'if s<0 else ''}{'' if float(mult)==1 else str(float(mult))+'*'}{name}",'rel_err':rel})
    if hits: hits.sort(key=lambda h:h['rel_err']); return hits[:3]
    try:
        vec=[mpmath.mpf(1),mpmath.mpf(fv)]+[mpmath.mpf(float(c)) for c in cv]
        r=mpmath.pslq(vec,maxcoeff=1000,tol=mpmath.mpf(10)**(-(dps-15)))
        if r and r[1]!=0:
            pts=[str(r[0])]+[f"{co}*{nm}" for co,nm in zip(r[2:],CNAMES) if co]
            hits.append({'method':'pslq','expr':f"({'+'.join(p for p in pts if p!='0')})/{-r[1]}",'rel_err':0.0})
    except: pass
    try:
        s=mpmath.identify(mpmath.mpf(fv),tol=mpmath.mpf(10)**(-(dps//2)))
        if s: hits.append({'method':'isc','expr':s,'rel_err':0.0})
    except: pass
    return hits[:4]

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--source',default='cmf_hunter',choices=['all','telescope','cmf_hunter'])
    ap.add_argument('--reuse',default='cmf_euler_full.jsonl')
    ap.add_argument('--m-max',type=int,default=8,dest='m_max')
    ap.add_argument('--dps',type=int,default=80)
    ap.add_argument('--limit',type=int,default=0)
    ap.add_argument('--out',default=str(OUT_JSONL))
    args=ap.parse_args()
    mpmath.mp.dps=args.dps

    # Load previous Euler scan if available (skip re-computing)
    euler_cache={}
    rp=Path(__file__).parent/args.reuse
    if rp.exists():
        for line in open(rp):
            r=json.loads(line); euler_cache[r['cmf_id']]=r
        print(f"Loaded {len(euler_cache)} cached Euler entries from {rp.name}")

    src_filter={'all':('telescope','cmf_hunter'),'telescope':('telescope',),'cmf_hunter':('cmf_hunter',)}[args.source]
    conn=sqlite3.connect(DB_PATH)
    rows=conn.execute(f"SELECT id,cmf_payload FROM cmf WHERE json_extract(cmf_payload,'$.source') IN ({','.join('?'*len(src_filter))}) ORDER BY id",src_filter).fetchall()
    conn.close()
    if args.limit: rows=rows[:args.limit]

    cv=build_cv(); out_path=Path(args.out); t0=time.time()
    print(f"Path-2 PCF scan: {len(rows)} CMFs | dps={args.dps} | basis={len(CNAMES)} constants\n")

    n_id=0
    with open(out_path,'w') as fout:
        for ci,(cid,praw) in enumerate(rows):
            p=json.loads(praw); fp=p.get('f_poly',''); fb=p.get('fbar_poly','')
            src=p.get('source',''); pconst=p.get('primary_constant','')
            print(f"[{ci+1:3d}/{len(rows)}] #{cid} [{time.time()-t0:.0f}s]",end='  ',flush=True)
            if not fp: print('skip'); continue

            polys={'f':fp}
            if fb and fb!=fp: polys['fbar']=fb
            all_lines=[]; best_hit=None; best_m=None; best_method=None

            for m_val in range(args.m_max+1):
                for poly_name,poly_str in polys.items():
                    # Use cached value if available
                    cached_val=None
                    if cid in euler_cache:
                        for ml in euler_cache[cid].get('m_lines',[]):
                            if ml['m_val']==m_val and ml['poly']==poly_name:
                                cached_val=ml.get('value'); break

                    if cached_val:
                        method_vals={'cached':mpmath.mpf(cached_val)}
                    else:
                        method_vals=euler_sum_fast(poly_str,m_val,args.dps)
                    if not method_vals: continue

                    # Pick best value (prefer shanks > levin > em > cached)
                    for meth in ['shanks','levin','em','cached']:
                        if meth in method_vals:
                            best_v=method_vals[meth]; best_meth=meth; break
                    else: continue

                    hits=identify_val(best_v,cv,args.dps)
                    line={'m_val':m_val,'poly':poly_name,'method':best_meth,
                          'value':mpmath.nstr(best_v,dps-5 if (dps:=args.dps) else 30),
                          'identified':hits}
                    all_lines.append(line)
                    if hits and best_hit is None:
                        best_hit=hits; best_m=m_val; best_method=best_meth

            if best_hit: n_id+=1; flag='★'; id_str=best_hit[0]['expr'][:40]
            else: flag=' '; id_str='(none)'
            print(f"lines={len(all_lines)}  {flag}  best@m={best_m}  [{best_method}]  {id_str}")

            rec={'cmf_id':cid,'source':src,'f_poly':fp,'fbar_poly':fb,
                 'primary_constant':str(pconst) if pconst else None,
                 'm_lines':all_lines,'best_m':best_m,'best_hits':best_hit}
            fout.write(json.dumps(rec)+'\n'); fout.flush()

    print(f"\nDone {time.time()-t0:.1f}s | identified={n_id}/{len(rows)} | → {out_path}")

if __name__=='__main__': main()
