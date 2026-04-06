#!/usr/bin/env python3
"""
run_flatness_parallel.py
Run sympy_flatness_runner.py for all agents in parallel, dim=3 only.
Results merged into gauge_agents/pipeline_out/flatness_results.jsonl
Usage: nohup python3 run_flatness_parallel.py > /tmp/flatness_parallel.log 2>&1 &
"""
from __future__ import annotations
import json, subprocess, sys, time
from multiprocessing import Pool
from pathlib import Path

HERE = Path(__file__).parent
PYTHON = sys.executable
RUNNER = HERE / "gauge_agents" / "sympy_flatness_runner.py"
AGENTS = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]

def run_agent(agent: str):
    log = Path(f"/tmp/flatness_{agent}_3x3.log")
    print(f"[{agent}] Starting dim=3 symbolic check …", flush=True)
    t0 = time.time()
    with open(log, "w") as fh:
        ret = subprocess.run(
            [PYTHON, str(RUNNER), "--agent", agent, "--dim", "3"],
            cwd=str(HERE), stdout=fh, stderr=subprocess.STDOUT
        )
    elapsed = time.time() - t0
    # Count results from log
    lines = log.read_text(errors="replace")
    passed = lines.count("✓ PASS")
    failed = lines.count("✗ FAIL")
    unknown = lines.count("? UNKNOWN") + lines.count("TIMEOUT")
    print(f"[{agent}] Done in {elapsed/60:.1f}min  PASS={passed}  FAIL={failed}  UNKNOWN={unknown}", flush=True)
    return agent, passed, failed, unknown

if __name__ == "__main__":
    print(f"Launching {len(AGENTS)} parallel agents for dim=3 symbolic flatness …", flush=True)
    t_start = time.time()

    with Pool(processes=len(AGENTS)) as pool:
        results = pool.map(run_agent, AGENTS)

    total_pass = sum(r[1] for r in results)
    total_fail = sum(r[2] for r in results)
    total_unk  = sum(r[3] for r in results)
    elapsed = (time.time() - t_start) / 60

    print(f"\n{'='*60}")
    print(f"  ALL AGENTS DONE in {elapsed:.1f} min")
    print(f"  PASS={total_pass}  FAIL={total_fail}  UNKNOWN={total_unk}")
    for agent, p, f, u in results:
        print(f"    Agent {agent}: PASS={p} FAIL={f} UNKNOWN={u}")
    print(f"{'='*60}")
