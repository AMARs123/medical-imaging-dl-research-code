"""Train and evaluate one model on one fold of HAM10000; also evaluate on the official ISIC 2018 Task 3 test set.

Example:
  python train.py --proc data/processed --out experiments/EXP001 --model resnet50.a1_in1k \
      --protocol lesion --fold 0 --seed 0 --loss ce

Outputs in --out: config.yaml, command.txt, environment.txt, training.log, metrics.csv (per epoch),
predictions_val.csv, predictions.csv (HAM10000 test fold), predictions_isic.csv (ISIC 2018 test set),
best_model.pt (fp32), last_model.pt (fp16, to save disk), split_indices.npz, results.json.
Predictions files hold the raw logits, so post-hoc corrections can be computed later without re-running.
The checkpoint is chosen on validation balanced accuracy; neither test set is used for any choice.
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
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import CLASSES, K, SkinSet, build_model, classification_metrics, lesion_overlap, load_arrays, loss_fn, make_split, set_seed  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--proc", required=True)
p.add_argument("--out", required=True)
p.add_argument("--model", default="resnet50.a1_in1k")
p.add_argument("--protocol", default="lesion", choices=["lesion", "image"])
p.add_argument("--fold", type=int, default=0)
p.add_argument("--split-seed", type=int, default=0)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--loss", default="ce", choices=["ce", "wce", "focal", "la", "sampler"])
p.add_argument("--epochs", type=int, default=25)
p.add_argument("--batch-size", type=int, default=32)
p.add_argument("--lr", type=float, default=1e-4)
p.add_argument("--weight-decay", type=float, default=0.05)
p.add_argument("--img-size", type=int, default=224)
p.add_argument("--workers", type=int, default=6)
p.add_argument("--max-steps", type=int, default=0, help="smoke test only: stop each epoch after this many steps")
p.add_argument("--limit", type=int, default=0, help="smoke test only: use this many images per split")
p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
args = p.parse_args()

os.makedirs(args.out, exist_ok=True)
# optional WEIGHTS_DIR (17 Oct 2026, disk quota): weights kept outside /home; their location is recorded in the run folder
WDIR = os.path.join(os.environ['WEIGHTS_DIR'], os.path.basename(os.path.normpath(args.out))) if os.environ.get('WEIGHTS_DIR') else args.out
os.makedirs(WDIR, exist_ok=True)
if WDIR != args.out:
    open(os.path.join(args.out, 'WEIGHTS_LOCATION.txt'), 'w').write('best_model.pt and last_model.pt of this run are stored in %s (outside the /home quota, 17 Oct 2026); copy them here after the quota raise.\n' % WDIR)
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

set_seed(args.seed)
images, meta = load_arrays(args.proc, "ham")
ext_images, ext_meta = load_arrays(args.proc, "test")
y_all, y_ext = meta["label"].values, ext_meta["label"].values
tr, va, te = make_split(meta, args.protocol, args.fold, args.split_seed)
ext = np.arange(len(ext_meta))
if args.limit:
    tr, va, te, ext = tr[:args.limit], va[:args.limit], te[:args.limit], ext[:args.limit]
split_info = dict(n_train=len(tr), n_val=len(va), n_test=len(te), n_isic=len(ext),
                  lesions_shared_train_test=lesion_overlap(meta, tr, te), lesions_shared_train_val=lesion_overlap(meta, tr, va),
                  lesions_shared_val_test=lesion_overlap(meta, va, te),
                  train_class_counts=np.bincount(y_all[tr], minlength=K).tolist(),
                  test_class_counts=np.bincount(y_all[te], minlength=K).tolist())
log("split", json.dumps(split_info))
np.savez(os.path.join(args.out, "split_indices.npz"), train=tr, val=va, test=te)

counts = np.bincount(y_all[tr], minlength=K).astype(np.float64)
prior = torch.tensor(counts / counts.sum(), dtype=torch.float32, device=args.device)


def loader(imgs, labels, idx, train):
    ds = SkinSet(imgs, labels, idx, args.img_size, train)
    kw = dict(batch_size=args.batch_size, num_workers=args.workers, pin_memory=args.device == "cuda",
              persistent_workers=False, drop_last=train and len(idx) > args.batch_size)
    if train and args.loss == "sampler":  # class-balanced sampling: every class drawn with equal probability
        w = 1.0 / counts[labels[idx]]
        sampler = torch.utils.data.WeightedRandomSampler(torch.tensor(w, dtype=torch.double), len(idx), replacement=True)
        return torch.utils.data.DataLoader(ds, sampler=sampler, **kw)
    return torch.utils.data.DataLoader(ds, shuffle=train, **kw)


model = build_model(args.model).to(args.device)
n_params = sum(q.numel() for q in model.parameters())
opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
train_loader = loader(images, y_all, tr, True)
steps = args.max_steps if args.max_steps else len(train_loader)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.epochs * steps, pct_start=0.1,
                                            anneal_strategy="cos")
use_amp = args.device == "cuda"
scaler = torch.amp.GradScaler(enabled=use_amp)


@torch.no_grad()
def predict(imgs, labels, idx):
    model.eval()
    logits, ys, ids = [], [], []
    for x, y, j in loader(imgs, labels, idx, False):
        with torch.autocast(args.device, enabled=use_amp):
            lo = model(x.to(args.device, non_blocking=True))
        logits.append(lo.float().cpu().numpy()); ys.append(y.numpy()); ids.append(j.numpy())
    lo = np.concatenate(logits)
    pr = torch.from_numpy(lo).softmax(1).numpy()
    return lo, pr, np.concatenate(ys), np.concatenate(ids)


rows, best, best_epoch, t0 = [], -1.0, 0, time.time()
for ep in range(args.epochs):
    model.train()
    tl = n = 0.0
    for step, (x, y, _) in enumerate(train_loader):
        if args.max_steps and step >= args.max_steps:
            break
        x, y = x.to(args.device, non_blocking=True), y.to(args.device, non_blocking=True)
        with torch.autocast(args.device, enabled=use_amp):
            logits = model(x)
        loss = loss_fn(args.loss, logits.float(), y, prior)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt); scaler.update(); sched.step()
        tl += loss.item() * len(y); n += len(y)
    _, pv, yv, _ = predict(images, y_all, va)
    vm = classification_metrics(yv, pv)
    row = dict(epoch=ep + 1, train_loss=tl / max(n, 1), lr=sched.get_last_lr()[0], seconds=time.time() - t0,
               **{"val_" + k: v for k, v in vm.items() if not isinstance(v, list)})
    rows.append(row)
    pd.DataFrame(rows).to_csv(os.path.join(args.out, "metrics.csv"), index=False)
    log("epoch %d loss %.4f | val acc %.4f bacc %.4f f1 %.4f auc %.4f | %.0fs" % (
        ep + 1, row["train_loss"], vm["accuracy"], vm["balanced_accuracy"], vm["macro_f1"], vm["macro_auc"], row["seconds"]))
    if vm["balanced_accuracy"] > best:
        best, best_epoch = vm["balanced_accuracy"], ep + 1
        torch.save(model.state_dict(), os.path.join(WDIR, "best_model.pt"))
torch.save({k: v.half() if v.is_floating_point() else v for k, v in model.state_dict().items()}, os.path.join(WDIR, "last_model.pt"))

model.load_state_dict(torch.load(os.path.join(WDIR, "best_model.pt"), map_location=args.device))
results = dict(config=vars(args), split=split_info, n_params=n_params, best_epoch=best_epoch, best_val_balanced_accuracy=best,
               train_seconds=time.time() - t0, environment=env, smoke_test=bool(args.max_steps or args.limit))
train_lesions = set(meta["lesion_id"].values[tr])
for name, imgs, labels, idx, m in [("val", images, y_all, va, meta), ("test", images, y_all, te, meta), ("isic", ext_images, y_ext, ext, ext_meta)]:
    lo, pr, y, ids = predict(imgs, labels, idx)
    results[name] = classification_metrics(y, pr)
    df = pd.DataFrame(dict(array_index=ids, image_id=m["image_id"].values[ids], lesion_id=m["lesion_id"].values[ids], label=y,
                           pred=pr.argmax(1), source=m["dataset"].values[ids]))
    if name == "test":
        df["lesion_in_train"] = m["lesion_id"].iloc[ids].isin(train_lesions).values
    for k, c in enumerate(CLASSES):
        df["logit_" + c] = lo[:, k]
    fn = {"val": "predictions_val.csv", "test": "predictions.csv", "isic": "predictions_isic.csv"}[name]
    df.to_csv(os.path.join(args.out, fn), index=False)
    log(name.upper(), json.dumps({k: round(v, 4) for k, v in results[name].items() if not isinstance(v, list)}))
results["peak_gpu_mb"] = torch.cuda.max_memory_allocated() / 2**20 if args.device == "cuda" else 0
json.dump(results, open(os.path.join(args.out, "results.json"), "w"), indent=1)
log("DONE")
