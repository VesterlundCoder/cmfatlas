#!/usr/bin/env python3
"""Launch both CMF gauge agents as detached daemons."""
import os, sys, subprocess
from pathlib import Path

HERE   = Path(__file__).parent
PYTHON = "/Users/davidsvensson/Desktop/rd-lumi-z3/venv/bin/python3"

agents = [
    ("agent_explorer.py",  "logs/agent_A.log"),
    ("agent_holonomic.py", "logs/agent_B.log"),
]

for script, logfile in agents:
    log_path = HERE / logfile
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a") as flog:
        p = subprocess.Popen(
            [PYTHON, str(HERE / script)],
            cwd=str(HERE),
            stdout=flog,
            stderr=flog,
            stdin=subprocess.DEVNULL,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
            start_new_session=True,   # detach from parent session (macOS-safe)
        )
        print(f"  Launched {script}  PID={p.pid}  log={logfile}")

print("Done. Check logs/ for output.")
