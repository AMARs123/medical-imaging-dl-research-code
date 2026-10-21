"""Run every planned experiment that has no results.json yet, one after another, inside one SLURM job.

Usage: python run_batch.py <Paper_02 dir> [budget_hours] [forward|reverse]
Several of these jobs may run at once: each experiment is claimed with an exclusive .claim file holding the
SLURM job id; a claim whose job is no longer in the queue is stale and is taken over.
Safe to restart: finished experiments are skipped; an interrupted one is re-run from scratch in its own
directory (its partial log is kept as training.log.interrupted-N).
"""
import os
import subprocess
import sys
import time

import pandas as pd

root = sys.argv[1]
budget_h = float(sys.argv[2]) if len(sys.argv) > 2 else 70.0
order = sys.argv[3] if len(sys.argv) > 3 else "forward"
job = os.environ.get("SLURM_JOB_ID", "local-%d" % os.getpid())
reg = os.path.join(root, "experiments", "experiment_registry.csv")
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
        open(c, "w").write(job)
        return True


# 17 Oct 2026: ONLY_MODELS / SKIP_MODELS (comma-separated timm names) let a job take only the runs that fit its GPU share
# (MaxViT-T needs about 6.3 GB, more than a 3-shard job may use); FAIL_STOP as in the other papers' runners
ONLY = [m for m in os.environ.get("ONLY_MODELS", "").split(",") if m]
SKIP = [m for m in os.environ.get("SKIP_MODELS", "").split(",") if m]
fails, streak = {}, 0
while True:
    df = pd.read_csv(reg)
    todo = [r for _, r in df.iterrows() if not os.path.exists(os.path.join(root, "experiments", r["exp_id"], "results.json"))
            and not os.path.exists(os.path.join(root, "experiments", r["exp_id"], ".failed"))
            and (not ONLY or r["model"] in ONLY) and r["model"] not in SKIP]
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
               "--loss", r["loss"], "--split-seed", str(int(r["split_seed"])),
               "--workers", str(max(1, int(os.environ.get("SLURM_CPUS_PER_TASK", "7")) - 1))]  # one data worker per spare CPU
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
        break
    if not started:
        break
done = sum(os.path.exists(os.path.join(root, "experiments", e, "results.json")) for e in pd.read_csv(reg)["exp_id"])
print("completed %d of %d planned runs" % (done, len(pd.read_csv(reg))))
