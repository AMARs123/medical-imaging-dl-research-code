"""Train and evaluate one model on one fold. Everything needed to reproduce the run is saved.

Example:
  python train.py --proc data/processed --out experiments/EXP001 --model resnet50 \
      --protocol root --fold 0 --seed 0 --lambda-sal 0.0

Outputs in --out: config.yaml, command.txt, environment.txt, training.log, metrics.csv (per epoch),
predictions.csv (test set), evidence_test.npy (class-evidence maps of the test set), best_model.pt,
last_model.pt, results.json.
The test fold is never used for model selection: the checkpoint is chosen on validation macro-F1.
The tumour masks are used only by the training loss (when --lambda-sal > 0) and by the evaluation
of explanations; they are never an input to the network.
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
from common import (EvidenceNet, TumourSet, classification_metrics, load_arrays, localisation_metrics,  # noqa: E402
                    make_split, patient_overlap, saliency_loss, set_seed)

p = argparse.ArgumentParser()
p.add_argument("--proc", required=True)
p.add_argument("--out", required=True)
p.add_argument("--model", default="resnet50")
p.add_argument("--protocol", default="root", choices=["root", "pid", "slice", "authors"])
p.add_argument("--fold", type=int, default=0)
p.add_argument("--split-seed", type=int, default=0)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--epochs", type=int, default=30)
p.add_argument("--batch-size", type=int, default=32)
p.add_argument("--lr", type=float, default=1e-4)
p.add_argument("--weight-decay", type=float, default=0.05)
p.add_argument("--lambda-sal", type=float, default=0.0)
p.add_argument("--dilate", type=int, default=1)
p.add_argument("--img-size", type=int, default=224)
p.add_argument("--workers", type=int, default=4)
p.add_argument("--no-pretrained", action="store_true")
p.add_argument("--max-steps", type=int, default=0, help="smoke test only: stop each epoch after this many steps")
p.add_argument("--limit", type=int, default=0, help="smoke test only: use this many images per split")
p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
p.add_argument("--mem-fraction", type=float, default=0.0)
p.add_argument("--train-frac", type=float, default=1.0,
               help="keep this share of training ID roots (per class); the subset depends only on fold and fraction")
args = p.parse_args()

os.makedirs(args.out, exist_ok=True)
# P01_WEIGHTS_SCRATCH (set only for the learning-rate check): keep weights outside /home and remove them after the run
WDIR = os.path.join(os.environ["P01_WEIGHTS_SCRATCH"], os.path.basename(os.path.normpath(args.out))) if os.environ.get("P01_WEIGHTS_SCRATCH") else args.out
os.makedirs(WDIR, exist_ok=True)
if os.path.exists(os.path.join(args.out, "results.json")):
    print("results.json already exists in", args.out, "- refusing to overwrite an experiment")
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
           device=args.device, gpu=torch.cuda.get_device_name(0) if args.device == "cuda" else "none",
           slurm_job=os.environ.get("SLURM_JOB_ID", "none"))
try:
    import timm
    env["timm"] = timm.__version__
except Exception:
    pass
open(os.path.join(args.out, "environment.txt"), "w").write("\n".join("%s: %s" % kv for kv in env.items()) + "\n")

if args.device == "cuda" and args.mem_fraction > 0:
    torch.cuda.set_per_process_memory_fraction(args.mem_fraction)
set_seed(args.seed)
images, masks, meta = load_arrays(args.proc)
tr, va, te = make_split(meta, args.protocol, args.fold, args.split_seed)
if args.train_frac < 1.0:  # data-efficiency ablation: drop whole ID roots from the training split only
    sub_rng = np.random.default_rng(1000 + 10 * args.fold + int(round(100 * args.train_frac)))
    keep = []
    for lab in range(3):
        roots = np.unique(meta["root"].values[tr][meta["label"].values[tr] == lab])
        keep.extend(sub_rng.choice(roots, max(1, int(round(args.train_frac * len(roots)))), replace=False))
    tr = tr[np.isin(meta["root"].values[tr], keep)]
if args.limit:
    tr, va, te = tr[:args.limit], va[:args.limit], te[:args.limit]
split_info = dict(n_train=len(tr), n_val=len(va), n_test=len(te),
                  pids_shared_train_test=patient_overlap(meta, tr, te),
                  pids_shared_train_val=patient_overlap(meta, tr, va),
                  pids_shared_val_test=patient_overlap(meta, va, te),
                  roots_shared_train_test=patient_overlap(meta, tr, te, "root"),
                  roots_shared_train_val=patient_overlap(meta, tr, va, "root"),
                  roots_shared_val_test=patient_overlap(meta, va, te, "root"),
                  train_class_counts=np.bincount(meta["label"].values[tr], minlength=3).tolist(),
                  test_class_counts=np.bincount(meta["label"].values[te], minlength=3).tolist())
log("split", json.dumps(split_info))
np.savez(os.path.join(args.out, "split_indices.npz"), train=tr, val=va, test=te)


def loader(idx, train):
    ds = TumourSet(images, masks, meta, idx, args.img_size, train)
    return torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=train, num_workers=args.workers,
                                       drop_last=train and len(idx) > args.batch_size, pin_memory=args.device == "cuda")


model = EvidenceNet(args.model, 3, pretrained=not args.no_pretrained, img_size=args.img_size).to(args.device)
n_params = sum(q.numel() for q in model.parameters())
counts = np.bincount(meta["label"].values[tr], minlength=3).astype(np.float32)
class_w = torch.tensor(counts.sum() / (3 * np.maximum(counts, 1)), device=args.device)
opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
steps_per_epoch = max(1, len(loader(tr, True)))
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.epochs * steps_per_epoch, pct_start=0.1)
use_amp = args.device == "cuda"
scaler = torch.amp.GradScaler(enabled=use_amp)


@torch.no_grad()
def evaluate(idx, keep_evidence=False):
    model.eval()
    probs, ys, ids, loc, evid = [], [], [], [], []
    for x, y, m, j in loader(idx, False):
        x = x.to(args.device)
        with torch.autocast(args.device, enabled=use_amp):
            logits, e = model(x, return_evidence=True)
        pr = logits.float().softmax(1).cpu().numpy()
        e = e.float()
        for b in range(len(y)):
            # explanation for the predicted class, as a user of the model would see it
            c = int(pr[b].argmax())
            heat = F.interpolate(F.relu(e[b:b + 1, c:c + 1]), size=m.shape[-2:], mode="bilinear", align_corners=False)[0, 0].cpu().numpy()
            loc.append(localisation_metrics(heat, m[b, 0].numpy()))
        if keep_evidence:
            evid.append(e.cpu().numpy().astype(np.float16))
        probs.append(pr); ys.append(y.numpy()); ids.append(j.numpy())
    prob, y, ids = np.concatenate(probs), np.concatenate(ys), np.concatenate(ids)
    met = classification_metrics(y, prob)
    locdf = pd.DataFrame(loc)
    for k in locdf:
        met["loc_" + k] = float(np.nanmean(locdf[k]))
    return met, prob, y, ids, locdf, (np.concatenate(evid) if keep_evidence else None)


rows, best, t0 = [], -1.0, time.time()
for ep in range(args.epochs):
    model.train()
    tl = ts = n = 0.0
    for step, (x, y, m, _) in enumerate(loader(tr, True)):
        if args.max_steps and step >= args.max_steps:
            break
        x, y, m = x.to(args.device), y.to(args.device), m.to(args.device)
        with torch.autocast(args.device, enabled=use_amp):
            logits, e = model(x, return_evidence=True)
            ce = F.cross_entropy(logits.float(), y, weight=class_w)
            sal = saliency_loss(e.float(), y, m, args.dilate)
            loss = ce + args.lambda_sal * sal
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt); scaler.update(); sched.step()
        tl += ce.item() * len(y); ts += sal.item() * len(y); n += len(y)
    vm, *_ = evaluate(va)
    row = dict(epoch=ep + 1, train_ce=tl / max(n, 1), train_outside=ts / max(n, 1), lr=sched.get_last_lr()[0],
               seconds=time.time() - t0, **{"val_" + k: v for k, v in vm.items() if not isinstance(v, list)})
    rows.append(row)
    pd.DataFrame(rows).to_csv(os.path.join(args.out, "metrics.csv"), index=False)
    log("epoch %d ce %.4f outside %.3f | val acc %.4f f1 %.4f point %.3f mass %.3f" % (
        ep + 1, row["train_ce"], row["train_outside"], vm["accuracy"], vm["macro_f1"], vm["loc_pointing"], vm["loc_mass_inside"]))
    torch.save(model.state_dict(), os.path.join(WDIR, "last_model.pt"))
    if vm["macro_f1"] > best:
        best, best_epoch = vm["macro_f1"], ep + 1
        torch.save(model.state_dict(), os.path.join(WDIR, "best_model.pt"))

# last-epoch weights kept in fp16 to save disk space (earlier runs were converted the same way on 2026-10-15)
_lp = os.path.join(WDIR, "last_model.pt")
torch.save({k: v.half() if v.is_floating_point() else v for k, v in torch.load(_lp, map_location="cpu").items()}, _lp)
if WDIR == args.out:
    open(os.path.join(args.out, "last_model_NOTE.txt"), "w").write("last_model.pt stored in fp16 to save disk space; best_model.pt is fp32.\n")
model.load_state_dict(torch.load(os.path.join(WDIR, "best_model.pt"), map_location=args.device))
tm, prob, y, ids, locdf, evid = evaluate(te, keep_evidence=True)
pred = pd.DataFrame(dict(array_index=ids, idx=meta["idx"].values[ids], pid=meta["pid"].values[ids], root=meta["root"].values[ids], label=y, pred=prob.argmax(1),
                         p0=prob[:, 0], p1=prob[:, 1], p2=prob[:, 2]))
pd.concat([pred, locdf.reset_index(drop=True)], axis=1).to_csv(os.path.join(args.out, "predictions.csv"), index=False)
np.save(os.path.join(args.out, "evidence_test.npy"), evid)
res = dict(config=vars(args), split=split_info, n_params=n_params, best_epoch=best_epoch, best_val_macro_f1=best,
           train_seconds=time.time() - t0, test=tm, environment=env,
           peak_gpu_mb=torch.cuda.max_memory_allocated() / 2**20 if args.device == "cuda" else 0,
           smoke_test=bool(args.max_steps or args.limit))
json.dump(res, open(os.path.join(args.out, "results.json"), "w"), indent=1)
log("TEST", json.dumps({k: v for k, v in tm.items() if not isinstance(v, list)}))
if WDIR != args.out:  # weights were kept only in node-local scratch (learning-rate check, disk quota); remove the scratch copies
    for _f in ("best_model.pt", "last_model.pt"):
        if os.path.exists(os.path.join(WDIR, _f)):
            os.remove(os.path.join(WDIR, _f))
    open(os.path.join(args.out, "WEIGHTS_NOTE.txt"), "w").write("Weights of this learning-rate-check run were held in node-local scratch during training and not kept (disk quota, 17 Oct 2026); logs, metrics, predictions and evidence maps are kept.\n")
log("DONE")
