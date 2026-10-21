"""Run every planned experiment that has no results.json yet, one after another, inside one SLURM job.

Usage: python run_batch.py <Paper_01 dir> [budget_hours] [forward|reverse]
Several of these jobs may run at once: each experiment is claimed with an exclusive .claim file holding the
SLURM job id, and a claim whose job is no longer in the queue is treated as stale and taken over.
Safe to restart: finished experiments are skipped, an interrupted one is re-run from scratch in its
own directory (its partial log is kept as training.log.interrupted-N).
"""
import os
import subprocess
import sys
import time

import pandas as pd

root = sys.argv[1]  # Paper_01 directory
budget_h = float(sys.argv[2]) if len(sys.argv) > 2 else 70.0
order = sys.argv[3] if len(sys.argv) > 3 else "forward"
job = os.environ.get("SLURM_JOB_ID", "local-%d" % os.getpid())
reg = os.environ.get("REGISTRY") or os.path.join(root, "experiments", "experiment_registry.csv")  # REGISTRY: e.g. lr_registry.csv
t0 = time.time()


def job_active(j):
    if not j or j == job:
        return False
    try:
        return bool(subprocess.run(["squeue", "-h", "-j", j], capture_output=True, text=True, timeout=60).stdout.strip())
    except Exception:
        return True  # if the queue cannot be asked, do not steal


def claim(out):
    os.makedirs(out, exist_ok=True)
    c = os.path.join(out, ".claim")
    try:
        fd = os.open(c, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, job.encode())
        os.close(fd)
        return True
    except FileExistsError:
        owner = open(c).read().strip()
        if job_active(owner):
            return False
        open(c, "w").write(job)  # stale claim from a finished or killed job
        return True


fails, streak = {}, 0
while True:
    df = pd.read_csv(reg)  # re-read each time: phases may be appended while the job runs
    todo = [r for _, r in df.iterrows() if not os.path.exists(os.path.join(root, "experiments", r["exp_id"], "results.json"))
            and not os.path.exists(os.path.join(root, "experiments", r["exp_id"], ".failed"))]
    if order == "reverse":
        todo = todo[::-1]
    started = False
    for r in todo:
        out = os.path.join(root, "experiments", r["exp_id"])
        if os.path.exists(os.path.join(out, "results.json")) or not claim(out):
            continue
        if (time.time() - t0) / 3600 > budget_h:
            print("time budget reached; stopping cleanly")
            os.remove(os.path.join(out, ".claim"))
            sys.exit(0)
        if os.path.exists(os.path.join(out, "training.log")):
            n = len([f for f in os.listdir(out) if f.startswith("training.log.interrupted")])
            os.rename(os.path.join(out, "training.log"), os.path.join(out, "training.log.interrupted-%d" % (n + 1)))
        cmd = [sys.executable, os.path.join(root, "src", "train.py"), "--proc", os.path.join(root, "data", "processed"), "--out", out,
               "--model", r["model"], "--protocol", r["protocol"], "--fold", str(r["fold"]), "--seed", str(r["seed"]),
               "--lambda-sal", str(r["lambda_sal"]), "--split-seed", str(int(r.get("split_seed", 0) or 0))]
        for col, flag, cast in [("dilate", "--dilate", int), ("train_frac", "--train-frac", float), ("lr", "--lr", float)]:  # optional ablation columns
            if col in r and pd.notna(r[col]):
                cmd += [flag, str(cast(r[col]))]
        print(time.strftime("%F %T"), r["exp_id"], " ".join(cmd[2:]), flush=True)
        rc = subprocess.call(cmd)
        # FAIL_STOP: three failures of one experiment mark it .failed; three failed experiments in a row stop the job
        if rc != 0:
            fails[r["exp_id"]] = fails.get(r["exp_id"], 0) + 1
            if fails[r["exp_id"]] >= 3:
                open(os.path.join(out, ".failed"), "w").write("exit %d three times\n" % rc)
                streak += 1
                if streak >= 3:
                    print(time.strftime("%F %T"), "three experiments failed in a row; stopping", flush=True)
                    sys.exit(1)
        else:
            streak = 0
        print(time.strftime("%F %T"), r["exp_id"], "exit", rc, flush=True)
        started = True
        break  # re-read the registry and the claims after every run
    if not started:
        break
done = sum(os.path.exists(os.path.join(root, "experiments", e, "results.json")) for e in pd.read_csv(reg)["exp_id"])
print("completed %d of %d planned runs" % (done, len(pd.read_csv(reg))))
