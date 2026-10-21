"""Run every planned Paper 03 experiment without results.json (same claim mechanism as Paper 02's runner).
Usage: python run_batch.py <Paper_03 dir> [forward|reverse]"""
import os
import subprocess
import sys

import pandas as pd

root = sys.argv[1]
order = sys.argv[2] if len(sys.argv) > 2 else "forward"
job = os.environ.get("SLURM_JOB_ID", "local-%d" % os.getpid())
reg = pd.read_csv(os.path.join(root, "experiments", "experiment_registry.csv"))
rows = list(reg.to_dict("records"))
for r in (rows[::-1] if order == "reverse" else rows):
    out = os.path.join(root, "experiments", r["exp_id"])
    if os.path.exists(os.path.join(out, "results.json")):
        continue
    os.makedirs(out, exist_ok=True)
    try:
        fd = os.open(os.path.join(out, ".claim"), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, job.encode()); os.close(fd)
    except FileExistsError:
        owner = open(os.path.join(out, ".claim")).read().strip()
        if subprocess.run(["squeue", "-h", "-j", owner], capture_output=True, text=True).stdout.strip():
            continue
        open(os.path.join(out, ".claim"), "w").write(job)
    cmd = [sys.executable, os.path.join(root, "src", "train.py"), "--proc", os.path.join(root, "data", "processed"), "--out", out,
           "--model", r["model"], "--seed", str(r["seed"])] + (["--random-init-eval"] if r["random_init"] else [])
    print(r["exp_id"], " ".join(cmd[2:]), flush=True)
    print(r["exp_id"], "exit", subprocess.call(cmd), flush=True)
