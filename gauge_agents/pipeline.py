#!/usr/bin/env python3
"""
pipeline.py — Master controller: run all 3 agents toward 50k unique CMFs
=========================================================================

Launches Agents A, B, C as background subprocesses.
Every 60s: counts total unique fingerprints across ALL store files.
Every 600s: re-runs coupling_classifier + dreams_runner on new B4 hits.
Stops when total unique ≥ TARGET or STOP_AGENTS sentinel is created.

Usage:
    python3 pipeline.py                    # run to 50k (default)
    python3 pipeline.py --target 10000     # custom target
    python3 pipeline.py --no-dreams        # skip dreams runner
    python3 pipeline.py --stop             # create sentinel to stop all agents
"""
from __future__ import annotations
import argparse, json, os, subprocess, sys, time
from pathlib import Path

HERE   = Path(__file__).parent
PYTHON = "/Users/davidsvensson/Desktop/rd-lumi-z3/venv/bin/python3"
LOGS   = HERE / "logs"
LOGS.mkdir(exist_ok=True)

TARGET_DEFAULT  = 50_000
POLL_INTERVAL   = 60        # seconds between count polls
CLASSIFY_EVERY  = 600       # seconds between classify + dreams runs
SENTINEL        = HERE / "STOP_AGENTS"

AGENTS = [
    ("agent_explorer.py",  LOGS / "agent_A.log"),
    ("agent_holonomic.py", LOGS / "agent_B.log"),
    ("agent_c_large.py",   LOGS / "agent_C.log"),
]


# ══════════════════════════════════════════════════════════════════════════════
# Counting helpers
# ══════════════════════════════════════════════════════════════════════════════

def count_unique_total() -> dict:
    """Count unique fingerprints across all store files. Returns {agent: count, total: N}."""
    counts = {"A": 0, "B": 0, "C": 0, "total": 0}
    seen_fps: set = set()

    for store_path in sorted(HERE.glob("store_[ABC]_*.jsonl")):
        agent = ("A" if "_A_" in store_path.name else
                 "B" if "_B_" in store_path.name else "C")
        for line in store_path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                fp  = rec.get("fingerprint", "")
                if fp and fp not in seen_fps:
                    seen_fps.add(fp)
                    counts[agent] = counts.get(agent, 0) + 1
                    counts["total"] += 1
            except Exception:
                pass
    return counts


def count_b4() -> int:
    """Count B4 records in classified_bucket4.jsonl."""
    p = HERE / "classified_bucket4.jsonl"
    if not p.exists():
        return 0
    return sum(1 for l in p.read_text().splitlines() if l.strip())


# ══════════════════════════════════════════════════════════════════════════════
# Launch helpers
# ══════════════════════════════════════════════════════════════════════════════

def launch_agent(script: str, log_path: Path) -> subprocess.Popen:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a") as flog:
        p = subprocess.Popen(
            [PYTHON, str(HERE / script)],
            cwd=str(HERE),
            stdout=flog,
            stderr=flog,
            stdin=subprocess.DEVNULL,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
            start_new_session=True,
        )
    return p


def is_alive(p: subprocess.Popen) -> bool:
    return p is not None and p.poll() is None


def run_classifier():
    """Run coupling_classifier.py in-process (fast, no subprocess overhead)."""
    try:
        result = subprocess.run(
            [PYTHON, str(HERE / "coupling_classifier.py")],
            cwd=str(HERE),
            capture_output=True, text=True, timeout=300,
        )
        # Print bucket summary line from output
        for line in result.stdout.splitlines():
            if "TOTAL" in line or "B4" in line or "B3" in line:
                print(f"  [classify] {line.strip()}")
    except Exception as e:
        print(f"  [classify] error: {e}")


def run_dreams(max_recs: int = 100):
    """Run dreams_runner.py on new B4 CMFs."""
    try:
        result = subprocess.run(
            [PYTHON, str(HERE / "dreams_runner.py"), "--max", str(max_recs), "--b4-only"],
            cwd=str(HERE),
            capture_output=True, text=True, timeout=600,
        )
        for line in result.stdout.splitlines():
            if "✓" in line or "hit" in line.lower() or "formula" in line.lower():
                print(f"  [dreams] {line.strip()}")
    except Exception as e:
        print(f"  [dreams] error: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# Progress bar
# ══════════════════════════════════════════════════════════════════════════════

def _bar(n: int, total: int, width: int = 40) -> str:
    frac = min(n / max(total, 1), 1.0)
    filled = int(frac * width)
    return "█" * filled + "░" * (width - filled)


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target",     type=int, default=TARGET_DEFAULT)
    ap.add_argument("--no-dreams",  action="store_true")
    ap.add_argument("--stop",       action="store_true",
                    help="Create STOP_AGENTS sentinel and exit")
    args = ap.parse_args()

    if args.stop:
        SENTINEL.touch()
        print("  Created STOP_AGENTS sentinel. All agents will exit cleanly.")
        return

    if SENTINEL.exists():
        SENTINEL.unlink()
        print("  Removed stale STOP_AGENTS sentinel.")

    target = args.target
    print(f"\n{'═'*70}")
    print(f"  CMF Pipeline  |  target={target:,} unique CMFs")
    print(f"  Agents: A (explorer) + B (holonomic) + C (dim-climbing)")
    print(f"  Stop:   touch {SENTINEL.name}  or  python3 pipeline.py --stop")
    print(f"{'═'*70}\n")

    # Check existing count
    initial = count_unique_total()
    print(f"  Existing: A={initial['A']:,}  B={initial['B']:,}  "
          f"C={initial['C']:,}  total={initial['total']:,}")
    if initial["total"] >= target:
        print(f"  Already at target ({initial['total']:,} ≥ {target:,}). Done.")
        return

    # Launch all 3 agents
    procs: dict[str, subprocess.Popen] = {}
    for script, log_path in AGENTS:
        agent_id = script.split("_")[1][0].upper()   # A, B, C
        p = launch_agent(script, log_path)
        procs[agent_id] = p
        print(f"  ▶ Launched {script:<25}  PID={p.pid}  log={log_path.name}")

    print()
    t_start       = time.time()
    t_last_classify = t_start

    try:
        while True:
            time.sleep(POLL_INTERVAL)

            if SENTINEL.exists():
                print("\n  STOP_AGENTS sentinel detected — stopping pipeline.")
                break

            # Restart any crashed agent
            for script, log_path in AGENTS:
                aid = script.split("_")[1][0].upper()
                if not is_alive(procs.get(aid)):
                    print(f"  ⚠ Agent {aid} crashed — restarting...")
                    procs[aid] = launch_agent(script, log_path)

            counts  = count_unique_total()
            total   = counts["total"]
            elapsed = time.time() - t_start
            rate    = max(total - initial["total"], 0) / max(elapsed, 1) * 3600

            bar = _bar(total, target)
            pct = 100 * total / target
            print(f"  [{time.strftime('%H:%M:%S')}]  "
                  f"{bar}  {total:>7,}/{target:,} ({pct:.1f}%)  "
                  f"A={counts['A']:,} B={counts['B']:,} C={counts['C']:,}  "
                  f"{rate:.0f}/hr", flush=True)

            # Periodic classify + dreams
            if time.time() - t_last_classify >= CLASSIFY_EVERY:
                t_last_classify = time.time()
                print("  [periodic] Running classifier...")
                run_classifier()
                if not args.no_dreams:
                    print("  [periodic] Running Dreams...")
                    run_dreams(max_recs=50)

            if total >= target:
                print(f"\n  ✅ TARGET REACHED: {total:,} unique CMFs collected!")
                SENTINEL.touch()
                break

    except KeyboardInterrupt:
        print("\n  Interrupted. Creating STOP_AGENTS sentinel...")
        SENTINEL.touch()

    # Final summary
    final = count_unique_total()
    elapsed = time.time() - t_start
    print(f"\n{'═'*70}")
    print(f"  PIPELINE DONE  |  {elapsed/3600:.1f}h  |  {final['total']:,} unique CMFs")
    print(f"  A={final['A']:,}  B={final['B']:,}  C={final['C']:,}")
    print(f"{'═'*70}")

    # Final classify + dreams
    print("\n  Running final classification...")
    run_classifier()
    if not args.no_dreams:
        print("  Running final Dreams pass...")
        run_dreams(max_recs=500)


if __name__ == "__main__":
    main()
