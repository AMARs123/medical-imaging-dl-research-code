"""Paper 07: train one glaucoma classifier on Chaksu disc crops and test it on held-out cameras and external datasets.

Example: python train.py --root . --out experiments/EXP001 --model resnet50.a1_in1k --config all --target hard --seed 0
--config all    : train on the official Chaksu training split (all three cameras); in-domain test = official test split.
--config all_n271 : size-matched control added on 17 Oct 2026 after the main results: a label-stratified random subset
                  (seed 0) of the official training split with as many images as the leave-out-Remidio pool (two cameras,
                  271 images); in-domain test = official test split. It separates the effect of the missing camera from that
                  of the smaller training set.
--config loco_X : train on all Chaksu images (both splits) of the two cameras other than X; in-domain test = none;
                  held-out-camera test = all images of camera X.
Validation: 15% of the training images (stratified, seed of the run's split = 0), used to choose the checkpoint (AUC).
--target hard   : label = majority of the five experts (>= 3 of 5 say glaucoma suspect)
--target soft   : target = fraction of experts saying suspect (n_suspect / 5); evaluation always uses the hard label.
Evaluated sets: chaksu_test (config all), held_out_camera (config loco), PAPILA (glaucoma vs healthy; suspects excluded,
scored separately as papila_suspect), RIMONE. Outputs as in the other papers; predictions.csv holds every scored image.
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

p = argparse.ArgumentParser()
p.add_argument("--root", required=True)
p.add_argument("--out", required=True)
p.add_argument("--model", default="resnet50.a1_in1k")
p.add_argument("--config", default="all", choices=["all", "all_n271", "loco_Remidio", "loco_Bosch", "loco_Forus"])
p.add_argument("--target", default="hard", choices=["hard", "soft"])
p.add_argument("--seed", type=int, default=0)
p.add_argument("--epochs", type=int, default=30)
p.add_argument("--batch-size", type=int, default=24)
p.add_argument("--lr", type=float, default=1e-4)
p.add_argument("--size", type=int, default=320)
p.add_argument("--workers", type=int, default=3)
p.add_argument("--max-steps", type=int, default=0)
p.add_argument("--limit", type=int, default=0)
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
open(os.path.join(args.out, "environment.txt"), "w").write("\n".join("%s: %s" % kv for kv in env.items()) + "\n")
random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)

meta = pd.read_csv(os.path.join(args.root, "data", "processed", "meta.csv"))
A = np.load(os.path.join(args.root, "data", "processed", "images.npy"), mmap_mode="r")
ch = meta.dataset == "Chaksu"
if args.config in ("all", "all_n271"):
    pool = np.where(ch & (meta.split == "train"))[0]
    if args.config == "all_n271":
        from sklearn.model_selection import train_test_split as _tts
        n_match = int((ch & (meta.camera != "Remidio")).sum())
        pool, _ = _tts(pool, train_size=n_match, stratify=meta.label.values[pool], random_state=0)
        pool = np.sort(pool)
    tests = {"chaksu_test": np.where(ch & (meta.split == "test"))[0]}
else:
    cam = args.config.split("_")[1]
    pool = np.where(ch & (meta.camera != cam))[0]
    tests = {"held_out_camera": np.where(ch & (meta.camera == cam))[0]}
tests["papila"] = np.where((meta.dataset == "PAPILA") & (meta.label >= 0))[0]
tests["papila_suspect"] = np.where((meta.dataset == "PAPILA") & (meta.label == -1))[0]
tests["rimone"] = np.where(meta.dataset == "RIMONE")[0]
from sklearn.model_selection import train_test_split  # noqa: E402
tr, va = train_test_split(pool, test_size=0.15, stratify=meta.label.values[pool], random_state=0)
if args.limit:
    tr, va = tr[:args.limit], va[:args.limit]; tests = {k: v[:args.limit] for k, v in tests.items()}
y_hard = meta.label.values.astype(float)
y_soft = (meta.n_suspect / meta.n_experts).values
target = y_soft if args.target == "soft" else y_hard
log("split", json.dumps(dict(train=len(tr), val=len(va), train_pos=int(y_hard[tr].sum()), **{k: len(v) for k, v in tests.items()})))
MEAN = torch.tensor([0.485, 0.456, 0.406])[:, None, None]; STD = torch.tensor([0.229, 0.224, 0.225])[:, None, None]


class DS(torch.utils.data.Dataset):
    def __init__(self, idx, train):
        self.idx, self.train = idx, train

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        j = int(self.idx[i])
        x = torch.from_numpy(np.ascontiguousarray(A[j])).permute(2, 0, 1).float()[None] / 255.0
        if self.train:
            ang = np.deg2rad(random.uniform(-30, 30)); sc = random.uniform(0.75, 1.25)  # scale range covers RIM-ONE's tighter crops
            tx, ty = random.uniform(-0.08, 0.08), random.uniform(-0.08, 0.08)
            th = torch.tensor([[[np.cos(ang) / sc, -np.sin(ang) / sc, tx], [np.sin(ang) / sc, np.cos(ang) / sc, ty]]], dtype=torch.float32)
            x = F.grid_sample(x, F.affine_grid(th, (1, 3, args.size, args.size), align_corners=False), align_corners=False)
            if random.random() < 0.5:
                x = x.flip(-1)
            x = x * random.uniform(0.8, 1.2); m = x.mean(); x = (x - m) * random.uniform(0.8, 1.2) + m
            g = x.mean(1, keepdim=True); x = ((x - g) * random.uniform(0.8, 1.2) + g).clamp(0, 1)
        else:
            x = F.interpolate(x, size=args.size, mode="bilinear", antialias=True, align_corners=False)
        return (x[0] - MEAN) / STD, float(target[j]) if self.train else float(y_hard[j]), j


def loader(idx, train):
    return torch.utils.data.DataLoader(DS(idx, train), batch_size=args.batch_size, shuffle=train, num_workers=args.workers,
                                       drop_last=train and len(idx) > args.batch_size, pin_memory=args.device == "cuda")


import timm  # noqa: E402
model = timm.create_model(args.model, pretrained=True, num_classes=1).to(args.device)
opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
trl = loader(tr, True)
steps = args.max_steps or len(trl)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.epochs * steps, pct_start=0.1)
amp = args.device == "cuda"
scaler = torch.amp.GradScaler(enabled=amp)
pos_w = torch.tensor((1 - y_hard[tr].mean()) / max(y_hard[tr].mean(), 1e-3), device=args.device)


@torch.no_grad()
def predict(idx):
    model.eval(); out = []
    for x, _, _ in loader(idx, False):
        with torch.autocast(args.device, enabled=amp):
            out.append(model(x.to(args.device)).float()[:, 0].cpu().numpy())
    return np.concatenate(out) if out else np.zeros(0)


def metr(y, lo, thr=0.5):
    from sklearn.metrics import average_precision_score, roc_auc_score
    pr = 1 / (1 + np.exp(-lo)); y = np.asarray(y)
    d = dict(n=int(len(y)), n_pos=int(y.sum()))
    if 0 < y.sum() < len(y):
        d.update(auc=float(roc_auc_score(y, pr)), ap=float(average_precision_score(y, pr)))
    d.update(sens=float((pr[y == 1] >= thr).mean()) if (y == 1).any() else float("nan"),
             spec=float((pr[y == 0] < thr).mean()) if (y == 0).any() else float("nan"), mean_prob=float(pr.mean()))
    return d


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
    vm = metr(y_hard[va], predict(va))
    rows.append(dict(epoch=ep + 1, train_loss=tl / max(n, 1), val_auc=vm.get("auc", np.nan), seconds=time.time() - t0))
    pd.DataFrame(rows).to_csv(os.path.join(args.out, "metrics.csv"), index=False)
    log("epoch %d loss %.4f | val auc %.4f | %.0fs" % (ep + 1, rows[-1]["train_loss"], rows[-1]["val_auc"], rows[-1]["seconds"]))
    if vm.get("auc", -1) > best:
        best, best_ep = vm["auc"], ep + 1
        torch.save(model.state_dict(), os.path.join(WDIR, "best_model.pt"))
if best_ep == 0:
    torch.save(model.state_dict(), os.path.join(WDIR, "best_model.pt"))
torch.save({k: v.half() if v.is_floating_point() else v for k, v in model.state_dict().items()}, os.path.join(WDIR, "last_model.pt"))
model.load_state_dict(torch.load(os.path.join(WDIR, "best_model.pt"), map_location=args.device))
from sklearn.metrics import roc_curve  # noqa: E402
lv = predict(va); fpr, tpr, th = roc_curve(y_hard[va], 1 / (1 + np.exp(-lv)))
thr = float(th[np.argmax(tpr - fpr)])
res = dict(config=vars(args), best_epoch=best_ep, best_val_auc=best, threshold=thr, environment=env, train_seconds=time.time() - t0,
           n_params=sum(q.numel() for q in model.parameters()))
pred = [pd.DataFrame(dict(array_index=va, set="val", logit=lv))]
for k, idx in tests.items():
    lo = predict(idx)
    res[k] = metr(np.clip(y_hard[idx], 0, 1), lo, thr)
    pred.append(pd.DataFrame(dict(array_index=idx, set=k, logit=lo)))
pred = pd.concat(pred)
pred["label"] = y_hard[pred.array_index]; pred["n_suspect"] = meta.n_suspect.values[pred.array_index]
pred["camera"] = meta.camera.values[pred.array_index]
pred.to_csv(os.path.join(args.out, "predictions.csv"), index=False)
res["peak_gpu_mb"] = torch.cuda.max_memory_allocated() / 2**20 if amp else 0
json.dump(res, open(os.path.join(args.out, "results.json"), "w"), indent=1)
log("RESULT", json.dumps({k: {kk: round(vv, 4) for kk, vv in res[k].items() if isinstance(vv, float)} for k in tests}))
log("DONE")
