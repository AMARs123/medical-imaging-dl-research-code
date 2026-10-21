"""Train one ECG model on PTB-XL diagnostic superclasses (multi-label) with the recommended folds.

Folds 1-8 train, fold 9 validation, fold 10 test (patient-wise folds from the dataset; folds 9-10 were human-validated).
Example: python train.py --proc data/processed --out experiments/EXP001 --model xresnet1d50 --seed 0
Outputs: config.yaml, command.txt, environment.txt, training.log, metrics.csv, predictions_val.csv, predictions.csv,
best_model.pt, last_model.pt (fp16), norm.npz (per-lead mean/std fitted on the training folds), results.json.
--random-init-eval: save an untrained (randomly initialised) model and its test predictions only; used as the
model-randomisation sanity check of the attribution analysis.
"""
import argparse
import json
import os
import platform
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models import ECGNet  # noqa: E402

SUPER = ["NORM", "MI", "STTC", "CD", "HYP"]
p = argparse.ArgumentParser()
p.add_argument("--proc", required=True)
p.add_argument("--out", required=True)
p.add_argument("--model", default="xresnet1d50", choices=["xresnet1d50", "inception1d", "convtran1d"])
p.add_argument("--seed", type=int, default=0)
p.add_argument("--epochs", type=int, default=40)
p.add_argument("--batch-size", type=int, default=64)
p.add_argument("--lr", type=float, default=1e-3)
p.add_argument("--weight-decay", type=float, default=1e-2)
p.add_argument("--random-init-eval", action="store_true")
p.add_argument("--max-steps", type=int, default=0)
p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
args = p.parse_args()

os.makedirs(args.out, exist_ok=True)
if os.path.exists(os.path.join(args.out, "results.json")):
    print("results.json exists in", args.out, "- refusing to overwrite")
    sys.exit(0)
log_f = open(os.path.join(args.out, "training.log"), "a")


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    log_f.write(s + "\n")
    log_f.flush()


yaml.safe_dump(vars(args), open(os.path.join(args.out, "config.yaml"), "w"))
open(os.path.join(args.out, "command.txt"), "w").write(" ".join([sys.executable] + sys.argv) + "\n")
env = dict(python=sys.version.split()[0], torch=torch.__version__, cuda=torch.version.cuda, platform=platform.platform(),
           gpu=torch.cuda.get_device_name(0) if args.device == "cuda" else "none", slurm_job=os.environ.get("SLURM_JOB_ID", "none"))
open(os.path.join(args.out, "environment.txt"), "w").write("\n".join("%s: %s" % kv for kv in env.items()) + "\n")

torch.manual_seed(args.seed); np.random.seed(args.seed)
torch.backends.cudnn.benchmark = False
X = np.load(os.path.join(args.proc, "signals.npy"), mmap_mode="r")
meta = pd.read_csv(os.path.join(args.proc, "meta.csv"))
Y = meta[SUPER].values.astype(np.float32)
tr = np.where(meta.strat_fold <= 8)[0]
va = np.where(meta.strat_fold == 9)[0]
te = np.where(meta.strat_fold == 10)[0]
assert not set(meta.patient_id.values[tr]) & set(meta.patient_id.values[te]), "patient overlap train/test"
mu = X[tr].mean((0, 2), keepdims=True)[0].astype(np.float32)
sd = X[tr].std((0, 2), keepdims=True)[0].astype(np.float32)
np.savez(os.path.join(args.out, "norm.npz"), mean=mu, std=sd)
Xn = torch.from_numpy(((np.asarray(X) - mu) / sd).astype(np.float32))
Yt = torch.from_numpy(Y)
split = dict(n_train=len(tr), n_val=len(va), n_test=len(te), patients_shared_train_test=0,
             train_prevalence=Y[tr].mean(0).round(4).tolist(), test_prevalence=Y[te].mean(0).round(4).tolist())
log("split", json.dumps(split))

model = ECGNet(args.model, 12, len(SUPER)).to(args.device)
n_params = sum(q.numel() for q in model.parameters())


def batches(idx, shuffle):
    idx = np.random.permutation(idx) if shuffle else idx
    for s in range(0, len(idx), args.batch_size):
        b = idx[s:s + args.batch_size]
        if shuffle and len(b) < 2:
            continue
        yield Xn[b], Yt[b], b


@torch.no_grad()
def predict(idx):
    model.eval()
    out = []
    for x, _, _ in batches(idx, False):
        out.append(model(x.to(args.device)).float().cpu().numpy())
    return np.concatenate(out)


def metrics(y, logit):
    from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
    pr = 1 / (1 + np.exp(-logit))
    m = dict(macro_auc=float(roc_auc_score(y, pr, average="macro")),
             auc=[float(roc_auc_score(y[:, k], pr[:, k])) for k in range(len(SUPER))],
             macro_ap=float(average_precision_score(y, pr, average="macro")),
             macro_f1_at_05=float(f1_score(y, pr > 0.5, average="macro", zero_division=0)))
    return m


def save_preds(idx, logit, fn):
    df = pd.DataFrame(dict(ecg_id=meta.ecg_id.values[idx], patient_id=meta.patient_id.values[idx], array_index=idx))
    for k, c in enumerate(SUPER):
        df["y_" + c] = Y[idx, k]
        df["logit_" + c] = logit[:, k]
    df.to_csv(os.path.join(args.out, fn), index=False)


t0 = time.time()
if args.random_init_eval:  # sanity-check model: no training at all
    torch.save(model.state_dict(), os.path.join(args.out, "best_model.pt"))
    lo = predict(te)
    save_preds(te, lo, "predictions.csv")
    json.dump(dict(config=vars(args), split=split, n_params=n_params, test=metrics(Y[te], lo), environment=env, random_init=True),
              open(os.path.join(args.out, "results.json"), "w"), indent=1)
    log("DONE random init")
    sys.exit(0)

opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
steps = int(np.ceil(len(tr) / args.batch_size))
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.epochs * steps, pct_start=0.1)
rows, best, best_ep = [], -1, 0
for ep in range(args.epochs):
    model.train()
    tl = n = 0.0
    for st, (x, y, _) in enumerate(batches(tr, True)):
        if args.max_steps and st >= args.max_steps:
            break
        x, y = x.to(args.device), y.to(args.device)
        # augmentation: random circular time shift (up to 1 s) and amplitude scaling 0.9-1.1, label-preserving
        x = torch.roll(x, int(np.random.randint(-100, 101)), -1) * float(np.random.uniform(0.9, 1.1))
        loss = F.binary_cross_entropy_with_logits(model(x), y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if sched.last_epoch < args.epochs * steps - 1:
            sched.step()
        tl += loss.item() * len(y); n += len(y)
    lv = predict(va)
    vm = metrics(Y[va], lv)
    rows.append(dict(epoch=ep + 1, train_loss=tl / max(n, 1), val_macro_auc=vm["macro_auc"], val_macro_ap=vm["macro_ap"], seconds=time.time() - t0))
    pd.DataFrame(rows).to_csv(os.path.join(args.out, "metrics.csv"), index=False)
    log("epoch %d loss %.4f | val auc %.4f ap %.4f | %.0fs" % (ep + 1, rows[-1]["train_loss"], vm["macro_auc"], vm["macro_ap"], rows[-1]["seconds"]))
    if vm["macro_auc"] > best:
        best, best_ep = vm["macro_auc"], ep + 1
        torch.save(model.state_dict(), os.path.join(args.out, "best_model.pt"))
torch.save({k: v.half() if v.is_floating_point() else v for k, v in model.state_dict().items()}, os.path.join(args.out, "last_model.pt"))
model.load_state_dict(torch.load(os.path.join(args.out, "best_model.pt"), map_location=args.device))
lv, lt = predict(va), predict(te)
save_preds(va, lv, "predictions_val.csv")
save_preds(te, lt, "predictions.csv")
res = dict(config=vars(args), split=split, n_params=n_params, best_epoch=best_ep, best_val_macro_auc=best, train_seconds=time.time() - t0,
           val=metrics(Y[va], lv), test=metrics(Y[te], lt), environment=env,
           peak_gpu_mb=torch.cuda.max_memory_allocated() / 2**20 if args.device == "cuda" else 0)
json.dump(res, open(os.path.join(args.out, "results.json"), "w"), indent=1)
log("TEST", json.dumps({k: v for k, v in res["test"].items()}))
log("DONE")
