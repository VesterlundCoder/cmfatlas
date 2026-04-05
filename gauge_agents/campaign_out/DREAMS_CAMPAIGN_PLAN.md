# CMF Atlas — Ramanujan Dreams Campaign Plan
**50-seed, 10×-scaled exploration. 3,360 total trajectories. DPS=80.**

---

## A. Seed Portfolio (50 seeds)

### Tier S — 8 seeds × 100 trajectories × depth 2000/500
Highest-confidence multi-signal seeds. All 5 rational-limit CMFs included.

| ID | dim | Bucket | Score | Style | Limit | Mode mix emphasis |
|----|-----|--------|-------|-------|-------|-------------------|
| gauge_A_c02c792c | 3×3 | B3 | 0.853 | top_score_rational_limit | **−2** | exploit/pslq_chase/padic/congruence |
| gauge_A_259f0d0b | 3×3 | B3 | 0.822 | top_score_multi_signal | — | exploit/padic/congruence |
| gauge_A_4341aad2 | 3×3 | B2 | 0.820 | rational_limit_high_dio | **5/4** | exploit/pslq_chase/dio/padic |
| gauge_A_6525df80 | 3×3 | B2 | 0.812 | top_score_dio_dominant | — | exploit/dio/padic |
| gauge_A_6fd0cbca | 3×3 | B2 | 0.810 | top_score_padic_multi | — | exploit/padic/congruence |
| gauge_A_af50bd58 | 3×3 | B2 | 0.805 | rational_limit_max_delta | **1/7** | exploit/pslq_chase/padic/dio |
| gauge_A_6ce2fcaa | 3×3 | B2 | 0.773 | rational_limit_negative | **−5/2** | exploit/pslq_chase/dio |
| gauge_A_35e44a13 | 3×3 | B2 | 0.772 | rational_limit_third | **1/3** | exploit/pslq_chase/dio |

### Tier A — 12 seeds × 80 trajectories × depth 1500/400
Strong frontier; dimensional diversity (dim=3,4,5); specialist specialists.

| ID | dim | Style | Reason |
|----|-----|-------|--------|
| gauge_A_a5d075c6 | 3×3 | frontier_padic_multi | Rank 6, B3, depth=5 |
| gauge_A_0ec1c56f | 3×3 | padic_specialist_b4 | P=0.533 (top p-adic), B4 |
| gauge_A_549cf52a | 3×3 | high_delta_b2 | delta=45, symbolic flat |
| gauge_A_b92ba7c4 | 3×3 | max_delta_b2 | delta=60 (max), B2 |
| gauge_A_75118d01 | 3×3 | top_dio_b3 | DIO=0.768, delta=60, B3 |
| gauge_A_dd1bffd4 | 3×3 | balanced_frontier_b2 | DIO=0.728, balanced |
| gauge_A_336a9969 | 3×3 | b4_high_score | Best balanced B4 |
| gauge_A_bcea5a3f | 4×4 | top_dim4_b4 | Top 4×4, DIO=0.698 |
| gauge_A_0837e3c5 | 4×4 | dim4_max_delta | delta=60, dim=4 |
| gauge_A_da4374ab | 3×3 | strong_padic_b3 | P=0.527, B3 |
| gauge_A_42b1d2e2 | 5×5 | top_dim5_b4 | Top 5×5, B4 |
| gauge_A_d9d19033 | 5×5 | dim5_dio_delta | DIO=0.740, delta=53.5 |

### Tier B — 10 seeds × 60 trajectories × depth 1200/300
Regime specialists: p-adic, DIO, Ore, dimensional.

| ID | dim | Style | Key metric |
|----|-----|-------|-----------|
| gauge_A_7476fc55 | 3×3 | max_padic_specialist | P=0.599 (dataset max) |
| gauge_A_a9e44de6 | 3×3 | padic_b3_strong | P=0.528, B3 |
| gauge_A_dc17ce38 | 3×3 | padic_b3_control | P=0.530, B3 control pair |
| gauge_A_ece045d2 | 5×5 | max_dio_dim5 | DIO=0.773 (dataset max) |
| gauge_A_da79e292 | 4×4 | dio_specialist_dim4_b3 | DIO=0.767, B3 |
| gauge_A_0036aeb0 | 4×4 | dio_specialist_dim4_b2 | DIO=0.768, B2 |
| gauge_A_bd55d9f4 | 5×5 | ore_specialist_dim5 | ore=2.5 (max), dim=5 |
| gauge_A_a0561d69 | 4×4 | ore_specialist_dim4 | ore=2.0, B3 |
| gauge_A_5680c8cc | 5×5 | dim5_max_delta | delta=60 (max), dim=5 |
| gauge_A_c2fb96a9 | 5×5 | dim5_b4_balanced | DIO=0.696, B4, dim=5 |

### Tier C — 20 seeds × 50 trajectories × depth 1000/200
10 lower-dim control/speculative + 10 high-dim Agent C (dims 7–27).

**Lower-dim (control/speculative):**
gauge_A_22848068 (3, B4), gauge_A_2e08a9d5 (3, B4), gauge_A_43c0d9eb (3, B4),
gauge_A_6db222f8 (4, B2), gauge_A_65c2ae8d (5, B2), gauge_A_c67f64aa (5, B2),
gauge_A_ec3e00f3 (3, B2, low-score control), gauge_A_589a4d47 (3, B2),
gauge_A_11d7b808 (3, B2, high-DIO control), gauge_A_71eb5518 (4, B2)

**High-dimensional Agent C (stratified dims 7–27):**

| ID | dim | delta | bidir | Key property |
|----|-----|-------|-------|-------------|
| gauge_C_b2b4dc50 | 7×7 | 8.9 | 0.250 | Only B3 bidir in 7–9 range |
| gauge_C_7931c7a2 | 9×9 | 10.1 | 0.000 | Covers 9×9 gap |
| gauge_C_587cb95b | 11×11 | 10.5 | 0.000 | Only analyzed high-dim >10 |
| gauge_C_cdba20c4 | 14×14 | 11.1 | 0.000 | Best delta in 13–16 |
| gauge_C_6c07de8a | 17×17 | 12.3 | 0.176 | Only bidir in 14–20 range |
| gauge_C_71f28daa | 19×19 | 13.3 | 0.000 | Covers 19×19 |
| gauge_C_5b5d3dcc | 21×21 | 14.3 | 0.064 | Covers 21×21 |
| gauge_C_1123900d | 23×23 | 14.5 | 0.000 | Rising delta trend |
| gauge_C_a6747bf3 | 25×25 | 15.9 | 0.000 | Strong flatness at frontier |
| gauge_C_b7f2dc96 | 27×27 | 16.8 | 0.000 | Ultimate high-dim frontier |

---

## B. Experimental Matrix

```
Tier S:  8 seeds × 100 traj × (2/3 at depth 2000 + 1/3 at depth  500) =  800 traj
Tier A: 12 seeds ×  80 traj × (2/3 at depth 1500 + 1/3 at depth  400) =  960 traj
Tier B: 10 seeds ×  60 traj × (2/3 at depth 1200 + 1/3 at depth  300) =  600 traj
Tier C: 20 seeds ×  50 traj × (2/3 at depth 1000 + 1/3 at depth  200) = 1000 traj
                                                               TOTAL:    3360 traj
```

### Trajectory mode distribution per tier

| Tier | exploit | pslq_chase | padic | congruence | dio | ore | novelty | crossdim | anti_score | long_random |
|------|---------|------------|-------|------------|-----|-----|---------|----------|------------|-------------|
| S    | 30% | 20% | 20% | 10% | 10% | — | 10% | — | — | — |
| A    | 25% | 10% | 15% | 10% | 10% | — | 15% | 15% | — | — |
| B    | 15% | 10% | varies | varies | varies | varies | 15% | 10% | — | — |
| C (low) | 10% | 5% | 10% | — | 10% | 5% | 25% | 15% | 15% | 15% |
| C (high) | 20% | 10% | — | — | — | — | 25% | 20% | 15% | 10% |

### Checkpoint schedule (5 per trajectory)
- CP0: step 0 (start state)
- CP1: step = 25% of depth
- CP2: step = 50% of depth
- CP3: step = 75% of depth
- CP4: step = 100% of depth (final)

PSLQ attempted at each checkpoint. Self-delta computed on all axis linear combinations (eᵢ, eᵢ+eⱼ) at depth/4.

---

## C. Metrics Schema

Every trajectory record (`campaign_results.jsonl`) contains:

```jsonc
{
  // Identity
  "seed_id": "gauge_A_...",   "seed_fp": "...",
  "agent":   "A|B|C",         "dim": 3,
  "bucket":  2,                "tier": "S",
  "style":   "...",

  // Trajectory parameters
  "traj_idx": 0,    "mode": "exploit",
  "start":    [...], "ray": [...],
  "depth":    2000,  "n_ax": 3,

  // Seed analysis scores (from CSV)
  "seed_total_score", "seed_A_score", "seed_B_score", "seed_C_score",
  "seed_P_score", "seed_DIO_score", "seed_congruence_depth",
  "seed_ore", "seed_delta", "seed_bonus_flags", "seed_symbolic_flat",
  "seed_identified_limit",

  // 5 checkpoints with PSLQ
  "checkpoints": [
    {
      "fraction": 0.0|0.25|0.5|0.75|1.0,
      "step":     int,
      "limit":    float|null,
      "converged": bool,
      "pslq": {
        "formula": "...",
        "residual": float,
        "confidence": "numerically_plausible",
        "precision_dps": 80
      } | null
    },
    ...  // × 5
  ],

  // Self-delta on linear combinations
  "self_delta_main":    float|null,     // |L(start) - L(start+1)| / scale
  "self_delta_lincombs": {              // per-ray self-delta
    "e0": float|null, "e1": float|null, "e2": float|null,
    "e0+e1": float|null, "e0+e2": float|null, "e1+e2": float|null
  },

  // Summary
  "final_limit":   float|null,
  "stability":     float|null,   // |L(cp4) - L(cp3)| / |L(cp4)|
  "novelty_dist":  float|null,   // distance from known limit
  "top_formula":   "...",        // best PSLQ hit at any checkpoint

  // Promotion signals
  "promote_pslq":         bool,
  "promote_stability":    bool,  // stability < 1e-6
  "promote_self_delta":   bool,  // self_delta_main < 1e-4
  "promote_formula_match": bool,

  "elapsed": float
}
```

---

## D. Promotion and Triage Rules

### Trajectory-level signals (any of these → flag trajectory)
- `promote_pslq = True` — PSLQ matched at any checkpoint
- `promote_stability = True` — |L(cp4)−L(cp3)|/|L(cp4)| < 1e-6
- `promote_self_delta = True` — self_delta_main < 1e-4
- Formula appears at ≥2 checkpoints — increasing confidence

### Seed-level promotion (applied after all trajectories)

| Level | Condition | Action |
|-------|-----------|--------|
| **PAPER** | ≥10 PSLQ hits OR ≥5 trajectories with same formula OR >50% convergent runs | Full symbolic follow-up + paper-level writeup |
| **DEEP** | ≥3 PSLQ hits OR ≥5 stable runs OR ≥2 formula matches | Extend with 2× more trajectories + DPS=120 verification |
| **WATCH** | ≥1 PSLQ hit OR ≥2 stable runs OR ≥5 self-delta passes | Log for next campaign round |
| **NONE** | None of the above | No escalation |

### Symbolic follow-up triggers
1. **Same formula repeated ≥3×** → attempt mpmath verification at DPS=150
2. **PSLQ at CP1 (25%)** → walk is already in basin; schedule long_random run from same start
3. **High-dim (dim≥7) convergence** → attempt dimensional collapse analysis
4. **Limit oscillates across checkpoints** → flag as unstable; check for bifurcation

### PSLQ confidence levels
- `"numerically_plausible"` — PSLQ relation found, residual < 1e-10
- `"verified"` — independently confirmed via mpmath at DPS≥120
- Never claim `"proven"` from PSLQ alone

---

## E. Scientific Interpretation

### What would be especially significant

**1. Regime transitions**
A trajectory that starts in one convergence basin and crosses to another at some checkpoint.
Indicator: |L(cp2) − L(cp1)| >> |L(cp4) − L(cp3)| (large early drift, stable late).

**2. High-dim → low-dim collapse**
An Agent C (dim≥7) seed whose limit matches a known dim=3 constant.
Indicator: PSLQ identifies π, ζ(3), G, log(2) etc. from a 21×21 walk.
This would suggest a non-trivial dimensional reduction of the CMF family.

**3. Persistent p-adic structure**
PSLQ hitting the same formula from p-adic-mode starts (prime-valued starts) but not exploit-mode starts.
Indicator: mode=padic promote_pslq rate >> mode=exploit promote_pslq rate for same seed.

**4. New candidate constants from PSLQ**
Formula not in the standard basis (π, ζ(3), G, log(2), ζ(5)...).
Requires: residual < 1e-12, confirmed at DPS=100+, reproduced from 3+ independent trajectories.

**5. Reproducible novel families**
Multiple seeds from different tiers converging to the same formula from independent starts.
Indicator: top_formula collisions across seed_ids in promotions.json.

**6. Congruence-depth amplification**
Seeds with depth=5 (capped) showing extreme PSLQ stability — suggests arithmetic structure deeper than current analyzer can measure.

**7. Self-delta divergence at linear combination e0+e1 vs e0**
Means the CMF is directionally asymmetric — mathematically interesting for gauge-inequivalence analysis.

---

## Execution

```bash
# Launch (background, resumable)
cd "cmf_harvester"
nohup python3 gauge_agents/dreams_campaign.py \
  --focused-basis --dps 80 --resume \
  > gauge_agents/campaign_out/campaign_stdout.log 2>&1 &

# Monitor
tail -f gauge_agents/campaign_out/campaign.log

# Check promotions so far
python3 -c "
import json
p = json.loads(open('gauge_agents/campaign_out/promotions.json').read())
for x in sorted(p, key=lambda x: -x['n_pslq_hits'])[:10]:
    print(x['seed_id'][:30], x['promote_level'], x['n_pslq_hits'], x.get('top_formula',''))
"

# Resume after interruption
python3 gauge_agents/dreams_campaign.py --focused-basis --dps 80 --resume
```
