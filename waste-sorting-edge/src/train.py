"""waste-sorting-edge: train one material classifier (5 classes) and evaluate it on studio photos (TrashNet) and litter in the wild (TACO).

Example: python train.py --root . --out experiments/EXP001 --model mobilenetv3_large_100.ra_in1k --train-on trashnet --seed 0
Fixed splits per source (seed 0): 70% train / 15% validation / 15% test, stratified by class and grouped by photograph
(TACO crops of one photo stay together). The two test splits are the same for every run.
--train-on trashnet | taco | both. Checkpoint chosen on the validation balanced accuracy of the training source(s).
Outputs: config.yaml, command.txt, environment.txt, training.log, metrics.csv, predictions.csv (test sets of both sources,
logits), best_model.pt, last_model.pt (fp16), results.json.
"""
import argparse
import json
import os
import platform
import random
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import CLASSES, ImgSet, splits  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--root", required=True)
p.add_argument("--out", required=True)
p.add_argument("--model", default="resnet18.a1_in1k")
p.add_argument("--train-on", default="both", choices=["trashnet", "taco", "both"])
p.add_argument("--seed", type=int, default=0)
p.add_argument("--epochs", type=int, default=25)
p.add_argument("--batch-size", type=int, default=48)
p.add_argument("--lr", type=float, default=3e-4)
p.add_argument("--workers", type=int, default=3)
p.add_argument("--max-steps", type=int, default=0)
p.add_argument("--limit", type=int, default=0)
p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
args = p.parse_args()
os.makedirs(args.out, exist_ok=True)
if os.path.exists(os.path.join(args.out, "results.json")):
    print("results.json exists; refusing to overwrite"); sys.exit(0)
log_f = open(os.path.join(args.out, "training.log"), "a")


def log(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); log_f.write(s + "\n"); log_f.flush()


yaml.safe_dump(vars(args), open(os.path.join(args.out, "config.yaml"), "w"))
open(os.path.join(args.out, "command.txt"), "w").write(" ".join([sys.executable] + sys.argv) + "\n")
env = dict(python=sys.version.split()[0], torch=torch.__version__, cuda=torch.version.cuda, platform=platform.platform(),
           gpu=torch.cuda.get_device_name(0) if args.device == "cuda" else "none", slurm_job=os.environ.get("SLURM_JOB_ID", "none"))
open(os.path.join(args.out, "environment.txt"), "w").write("\n".join("%s: %s" % kv for kv in env.items()) + "\n")
random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
meta = pd.read_csv(os.path.join(args.root, "data", "processed", "meta.csv"))
A = np.load(os.path.join(args.root, "data", "processed", "images.npy"), mmap_mode="r")
part = splits(meta)
src = {"trashnet": ["TrashNet"], "taco": ["TACO"], "both": ["TrashNet", "TACO"]}[args.train_on]
tr = np.where(meta.source.isin(src) & (part == "train"))[0]
va = np.where(meta.source.isin(src) & (part == "val"))[0]
tests = {"test_trashnet": np.where((meta.source == "TrashNet") & (part == "test"))[0], "test_taco": np.where((meta.source == "TACO") & (part == "test"))[0]}
if args.limit:
    tr, va = tr[:args.limit], va[:args.limit]; tests = {k: v[:args.limit] for k, v in tests.items()}
y = meta.label.values
log("split", json.dumps(dict(train=len(tr), val=len(va), **{k: len(v) for k, v in tests.items()})))


def loader(idx, train):
    return torch.utils.data.DataLoader(ImgSet(A, y, idx, train), batch_size=args.batch_size, shuffle=train, num_workers=args.workers,
                                       drop_last=train and len(idx) > args.batch_size, pin_memory=args.device == "cuda")


import timm  # noqa: E402
model = timm.create_model(args.model, pretrained=True, num_classes=len(CLASSES)).to(args.device)
opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
trl = loader(tr, True)
steps = args.max_steps or len(trl)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.epochs * steps, pct_start=0.1)
amp = args.device == "cuda"
scaler = torch.amp.GradScaler(enabled=amp)
cnt = np.bincount(y[tr], minlength=len(CLASSES)).astype(float)
cw = torch.tensor(cnt.sum() / (len(CLASSES) * np.maximum(cnt, 1)), dtype=torch.float32, device=args.device)  # class-balanced loss


@torch.no_grad()
def predict(idx):
    model.eval(); out = []
    for x, _ in loader(idx, False):
        with torch.autocast(args.device, enabled=amp):
            out.append(model(x.to(args.device)).float().cpu().numpy())
    return np.concatenate(out)


def metr(yy, lo):
    from sklearn.metrics import balanced_accuracy_score, f1_score, recall_score
    pr = lo.argmax(1)
    return dict(n=int(len(yy)), accuracy=float((pr == yy).mean()), balanced_accuracy=float(balanced_accuracy_score(yy, pr)),
                macro_f1=float(f1_score(yy, pr, average="macro")),
                recall=[float(v) for v in recall_score(yy, pr, average=None, labels=list(range(len(CLASSES))), zero_division=0)])


rows, best, best_ep, t0 = [], -1, 0, time.time()
for ep in range(args.epochs):
    model.train(); tl = n = 0.0
    for st, (x, t) in enumerate(trl):
        if args.max_steps and st >= args.max_steps:
            break
        x, t = x.to(args.device), t.to(args.device)
        with torch.autocast(args.device, enabled=amp):
            loss = F.cross_entropy(model(x).float(), t, weight=cw, label_smoothing=0.1)
        opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
        tl += loss.item() * len(t); n += len(t)
    vm = metr(y[va], predict(va))
    rows.append(dict(epoch=ep + 1, train_loss=tl / max(n, 1), val_bacc=vm["balanced_accuracy"], seconds=time.time() - t0))
    pd.DataFrame(rows).to_csv(os.path.join(args.out, "metrics.csv"), index=False)
    log("epoch %d loss %.4f | val bacc %.4f | %.0fs" % (ep + 1, rows[-1]["train_loss"], vm["balanced_accuracy"], rows[-1]["seconds"]))
    if vm["balanced_accuracy"] > best:
        best, best_ep = vm["balanced_accuracy"], ep + 1
        torch.save(model.state_dict(), os.path.join(args.out, "best_model.pt"))
torch.save({k: v.half() if v.is_floating_point() else v for k, v in model.state_dict().items()}, os.path.join(args.out, "last_model.pt"))
model.load_state_dict(torch.load(os.path.join(args.out, "best_model.pt"), map_location=args.device))
res = dict(config=vars(args), best_epoch=best_ep, best_val_bacc=best, environment=env, train_seconds=time.time() - t0,
           n_params=sum(q.numel() for q in model.parameters()))
pred = []
for k, idx in tests.items():
    lo = predict(idx); res[k] = metr(y[idx], lo)
    d = pd.DataFrame(dict(array_index=idx, set=k, label=y[idx], pred=lo.argmax(1)))
    for c in range(lo.shape[1]):
        d["logit_%d" % c] = lo[:, c]
    pred.append(d)
pd.concat(pred).to_csv(os.path.join(args.out, "predictions.csv"), index=False)
res["peak_gpu_mb"] = torch.cuda.max_memory_allocated() / 2**20 if amp else 0
json.dump(res, open(os.path.join(args.out, "results.json"), "w"), indent=1)
log("RESULT", json.dumps({k: {kk: round(v, 4) for kk, v in res[k].items() if isinstance(v, float)} for k in tests}))
log("DONE")
