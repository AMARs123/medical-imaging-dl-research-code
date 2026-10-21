"""Train one pneumonia classifier for chest-xray-pneumonia-shift and evaluate it on every group.

Example: python train.py --proc data/processed --out experiments/EXP001 --model densenet121.tv_in1k --train-groups G --fold 0
Training groups are split into 5 patient-grouped folds (fold k test, fold k+1 validation, 3 folds training).
Evaluation: the test fold of every training group, and *all* images of every group that was not used for training.
Outputs: config.yaml, command.txt, environment.txt, training.log, metrics.csv, predictions.csv (one row per evaluated
image: group, in_training_group, fold, label, logit), best_model.pt, last_model.pt (fp16), results.json.
The decision threshold is chosen on the validation fold (Youden index) and stored; no test image influences any choice.
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
from common import CXRSet, binary_metrics, build_model, folds_for, set_seed  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--proc", required=True)
p.add_argument("--out", required=True)
p.add_argument("--model", default="densenet121.tv_in1k")
p.add_argument("--train-groups", default="G")
p.add_argument("--fold", type=int, default=0)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--aug", default="base", choices=["none", "base", "strong"])
p.add_argument("--prep", default="raw", choices=["raw", "clahe"])
p.add_argument("--epochs", type=int, default=20)
p.add_argument("--batch-size", type=int, default=32)
p.add_argument("--lr", type=float, default=1e-4)
p.add_argument("--weight-decay", type=float, default=0.05)
p.add_argument("--workers", type=int, default=3)
p.add_argument("--max-steps", type=int, default=0)
p.add_argument("--limit", type=int, default=0, help="smoke test only: keep this many images per set")
p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
args = p.parse_args()
os.makedirs(args.out, exist_ok=True)
# optional WEIGHTS_DIR (17 Oct 2026, disk quota): weights kept outside /home; their location is recorded in the run folder
WDIR = os.path.join(os.environ['WEIGHTS_DIR'], os.path.basename(os.path.normpath(args.out))) if os.environ.get('WEIGHTS_DIR') else args.out
os.makedirs(WDIR, exist_ok=True)
if WDIR != args.out:
    open(os.path.join(args.out, 'WEIGHTS_LOCATION.txt'), 'w').write('best_model.pt and last_model.pt of this run are stored in %s (outside the /home quota, 17 Oct 2026); copy them here after the quota raise.\n' % WDIR)
if os.path.exists(os.path.join(args.out, "results.json")):
    print("results.json exists; refusing to overwrite"); sys.exit(0)
log_f = open(os.path.join(args.out, "training.log"), "a")


def log(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); log_f.write(s + "\n"); log_f.flush()


yaml.safe_dump(vars(args), open(os.path.join(args.out, "config.yaml"), "w"))
open(os.path.join(args.out, "command.txt"), "w").write(" ".join([sys.executable] + sys.argv) + "\n")
env = dict(python=sys.version.split()[0], torch=torch.__version__, cuda=torch.version.cuda, platform=platform.platform(),
           gpu=torch.cuda.get_device_name(0) if args.device == "cuda" else "none", slurm_job=os.environ.get("SLURM_JOB_ID", "none"))
try:
    import timm; env["timm"] = timm.__version__
except Exception:
    pass
open(os.path.join(args.out, "environment.txt"), "w").write("\n".join("%s: %s" % kv for kv in env.items()) + "\n")

set_seed(args.seed)
meta = pd.read_csv(os.path.join(args.proc, "meta.csv"), keep_default_na=False, na_values=[""])
images = np.load(os.path.join(args.proc, "images.npy" if args.prep == "raw" else "images_clahe.npy"), mmap_mode="r")
y = meta.label.values
tg = args.train_groups.split(",")
fold = folds_for(meta, tg, seed=0)
tr = np.where((fold >= 0) & (fold != args.fold) & (fold != (args.fold + 1) % 5))[0]
va = np.where(fold == (args.fold + 1) % 5)[0]
te_in = np.where(fold == args.fold)[0]
te_out = np.where(~meta.group.isin(tg).values & meta.ok.values)[0]
assert not set(meta.patient.values[tr]) & set(meta.patient.values[te_in]), "patient overlap"
if args.limit:
    rng_l = np.random.default_rng(0)
    tr, va, te_in, te_out = [np.sort(rng_l.choice(a, min(len(a), args.limit), replace=False)) for a in (tr, va, te_in, te_out)]
split = dict(n_train=len(tr), n_val=len(va), n_test_in=len(te_in), n_test_out=len(te_out), train_pos=int(y[tr].sum()),
             patients_shared_train_test=0, train_by_group={k: int(v) for k, v in meta.iloc[tr].group.value_counts().items()})
log("split", json.dumps(split))
np.savez(os.path.join(args.out, "split_indices.npz"), train=tr, val=va, test_in=te_in, test_out=te_out)


def loader(idx, train):
    ds = CXRSet(images, y, idx, args.aug if train else "none")
    return torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=train, num_workers=args.workers,
                                       drop_last=train and len(idx) > args.batch_size, pin_memory=args.device == "cuda")


model = build_model(args.model).to(args.device)
n_params = sum(q.numel() for q in model.parameters())
opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
trl = loader(tr, True)
steps = args.max_steps or len(trl)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.epochs * steps, pct_start=0.1)
amp = args.device == "cuda"
scaler = torch.amp.GradScaler(enabled=amp)
pos_w = torch.tensor((len(tr) - y[tr].sum()) / max(1, y[tr].sum()), device=args.device)  # balances the two classes in the loss


@torch.no_grad()
def predict(idx):
    model.eval()
    out = []
    for x, _, _ in loader(idx, False):
        with torch.autocast(args.device, enabled=amp):
            out.append(model(x.to(args.device)).float()[:, 0].cpu().numpy())
    return np.concatenate(out) if out else np.zeros(0)


rows, best, best_ep, t0 = [], -1, 0, time.time()
for ep in range(args.epochs):
    model.train(); tl = n = 0.0
    for st, (x, t, _) in enumerate(trl):
        if args.max_steps and st >= args.max_steps:
            break
        x, t = x.to(args.device), t.float().to(args.device)
        with torch.autocast(args.device, enabled=amp):
            lo = model(x)[:, 0]
        loss = F.binary_cross_entropy_with_logits(lo.float(), t, pos_weight=pos_w)
        opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
        tl += loss.item() * len(t); n += len(t)
    pv = predict(va)
    vm = binary_metrics(y[va], 1 / (1 + np.exp(-pv)))
    rows.append(dict(epoch=ep + 1, train_loss=tl / max(n, 1), val_auc=vm["auc"], val_ap=vm["ap"], seconds=time.time() - t0))
    pd.DataFrame(rows).to_csv(os.path.join(args.out, "metrics.csv"), index=False)
    log("epoch %d loss %.4f | val auc %.4f ap %.4f | %.0fs" % (ep + 1, rows[-1]["train_loss"], vm["auc"], vm["ap"], rows[-1]["seconds"]))
    if vm["auc"] > best:
        best, best_ep = vm["auc"], ep + 1
        torch.save(model.state_dict(), os.path.join(WDIR, "best_model.pt"))
torch.save({k: v.half() if v.is_floating_point() else v for k, v in model.state_dict().items()}, os.path.join(WDIR, "last_model.pt"))
model.load_state_dict(torch.load(os.path.join(WDIR, "best_model.pt"), map_location=args.device))

pv = predict(va)
prob_v = 1 / (1 + np.exp(-pv))
from sklearn.metrics import roc_curve  # noqa: E402
fpr, tpr, thr = roc_curve(y[va], prob_v)
threshold = float(thr[np.argmax(tpr - fpr)])  # Youden index on validation
lt_in, lt_out = predict(te_in), predict(te_out)
pred = pd.concat([
    pd.DataFrame(dict(array_index=va, part="val", logit=pv)),
    pd.DataFrame(dict(array_index=te_in, part="test_in", logit=lt_in)),
    pd.DataFrame(dict(array_index=te_out, part="test_out", logit=lt_out))])
pred["image"] = meta.image.values[pred.array_index]
pred["group"] = meta.group.values[pred.array_index]
pred["patient"] = meta.patient.values[pred.array_index]
pred["label"] = y[pred.array_index]
pred.to_csv(os.path.join(args.out, "predictions.csv"), index=False)
res = dict(config=vars(args), split=split, n_params=n_params, best_epoch=best_ep, best_val_auc=best, threshold=threshold,
           train_seconds=time.time() - t0, environment=env,
           peak_gpu_mb=torch.cuda.max_memory_allocated() / 2**20 if amp else 0, test={})
for g in ["G", "NA", "NP"]:
    q = pred[(pred.group == g) & (pred.part != "val")]
    if len(q):
        res["test"][g] = binary_metrics(q.label.values, 1 / (1 + np.exp(-q.logit.values)), threshold)
        res["test"][g]["in_training_group"] = g in tg
json.dump(res, open(os.path.join(args.out, "results.json"), "w"), indent=1)
log("TEST", json.dumps({g: {k: round(v, 4) for k, v in m.items() if isinstance(v, float)} for g, m in res["test"].items()}))
log("DONE")
