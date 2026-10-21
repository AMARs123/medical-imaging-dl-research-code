"""Train one segmentation model (Paper 09) and compute per-image uncertainty scores on every evaluation set.

Example: python train.py --proc data/processed --out experiments/EXP001 --data kvasir --encoder resnet34.a1_in1k --seed 0
Evaluation sets: val, test (in-domain), shift (HAM only: vidir_molemax images), and corrupted versions of test
(blur: Gaussian sigma 2 px; noise: Gaussian sigma 0.08; contrast: factor 0.4).
Saved per run:
  maps_<set>.npy   uint8 [N, 256, 256] deterministic foreground probability x 255 (for ensembles across seeds)
  scores_<set>.csv per image: Dice of deterministic / MC-dropout-mean / TTA-mean prediction, predicted area, and
                   uncertainty scores (see UNC below)
  pix_<set>.npz    for 512 random pixels per image: error indicator of the deterministic prediction and pixel
                   uncertainty of each method (for pixel-level error detection)
UNC: ent_det (mean binary entropy of the deterministic map), ent_det_fg (mean entropy within the predicted
foreground dilated by 5 px), ent_mc, mi_mc (mutual information = predictive entropy - mean sample entropy),
pdice_mc (mean pairwise Dice between 10 MC-dropout samples), ent_tta, pdice_tta (8 dihedral views).
Higher scores mean more uncertain for ent/mi; for pairwise Dice, lower agreement means more uncertain.
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
from common import MEAN, STD, SegSet, UNet, dice, enable_mc_dropout, set_seed  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--proc", required=True)
p.add_argument("--out", required=True)
p.add_argument("--data", default="kvasir", choices=["kvasir", "ham"])
p.add_argument("--encoder", default="resnet34.a1_in1k")
p.add_argument("--seed", type=int, default=0)
p.add_argument("--epochs", type=int, default=0, help="0 = default (60 kvasir, 25 ham)")
p.add_argument("--batch-size", type=int, default=16)
p.add_argument("--lr", type=float, default=3e-4)
p.add_argument("--workers", type=int, default=3)
p.add_argument("--mc", type=int, default=10)
p.add_argument("--max-steps", type=int, default=0)
p.add_argument("--limit", type=int, default=0)
p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
args = p.parse_args()
args.epochs = args.epochs or (60 if args.data == "kvasir" else 25)
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
set_seed(args.seed)
X = np.load(os.path.join(args.proc, "%s_images.npy" % args.data), mmap_mode="r")
M = np.load(os.path.join(args.proc, "%s_masks.npy" % args.data), mmap_mode="r")
meta = pd.read_csv(os.path.join(args.proc, "%s_meta.csv" % args.data))
ok = meta.has_mask.values if "has_mask" in meta else np.ones(len(meta), bool)
sets = {s: np.where((meta.split == s).values & ok)[0] for s in ["train", "val", "test", "shift"]}
sets = {k: v for k, v in sets.items() if len(v)}
if args.limit:
    sets = {k: v[:args.limit] for k, v in sets.items()}
log("split", json.dumps({k: len(v) for k, v in sets.items()}))

model = UNet(args.encoder).to(args.device)
opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
trl = torch.utils.data.DataLoader(SegSet(X, M, sets["train"], True), batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
                                  drop_last=True, pin_memory=args.device == "cuda")
steps = args.max_steps or len(trl)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.epochs * steps, pct_start=0.1)
amp = args.device == "cuda"
scaler = torch.amp.GradScaler(enabled=amp)


def seg_loss(logit, m):
    bce = F.binary_cross_entropy_with_logits(logit, m)
    pr = torch.sigmoid(logit)
    inter = (pr * m).sum((1, 2)); den = pr.sum((1, 2)) + m.sum((1, 2))
    return bce + (1 - (2 * inter + 1) / (den + 1)).mean()


def corrupt(x, kind):
    """x: [B,3,H,W] in [0,1]."""
    if kind == "blur":
        k = torch.arange(-6, 7, dtype=torch.float32, device=x.device); g = torch.exp(-k ** 2 / 8.0); g = g / g.sum()
        x = F.conv2d(F.pad(x, (6, 6, 0, 0), mode="reflect"), g.view(1, 1, 1, 13).repeat(3, 1, 1, 1), groups=3)
        return F.conv2d(F.pad(x, (0, 0, 6, 6), mode="reflect"), g.view(1, 1, 13, 1).repeat(3, 1, 1, 1), groups=3)
    if kind == "noise":
        gen = torch.Generator(device=x.device).manual_seed(0)
        return (x + 0.08 * torch.randn(x.shape, generator=gen, device=x.device)).clamp(0, 1)
    if kind == "contrast":
        mu = x.mean((1, 2, 3), keepdim=True); return (mu + 0.4 * (x - mu)).clamp(0, 1)
    return x


def norm(x):
    return (x - torch.tensor(MEAN, device=x.device)[None, :, None, None]) / torch.tensor(STD, device=x.device)[None, :, None, None]


def ent(p):
    p = p.clamp(1e-6, 1 - 1e-6); return -(p * p.log() + (1 - p) * (1 - p).log())


def pairwise_dice(samples):  # samples [T, H, W] bool
    T = len(samples); d = []
    for a in range(T):
        for b in range(a + 1, T):
            d.append(dice(samples[a], samples[b]))
    return float(np.mean(d))


@torch.no_grad()
def val_dice(idx):
    model.eval(); ds = []
    for b in range(0, len(idx), 32):
        j = idx[b:b + 32]
        x = norm(torch.from_numpy(np.asarray(X[j])).permute(0, 3, 1, 2).float().div(255).to(args.device))
        with torch.autocast(args.device, enabled=amp):
            pr = torch.sigmoid(model(x).float()).cpu().numpy() > 0.5
        ds += [dice(pr[k], np.asarray(M[jj])) for k, jj in enumerate(j)]
    return float(np.mean(ds))


rows, best, best_ep, t0 = [], -1, 0, time.time()
for ep in range(args.epochs):
    model.train(); tl = n = 0.0
    for st, (x, m, _) in enumerate(trl):
        if args.max_steps and st >= args.max_steps:
            break
        x, m = x.to(args.device), m.to(args.device)
        with torch.autocast(args.device, enabled=amp):
            lo = model(x)
        loss = seg_loss(lo.float(), m)
        opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
        tl += loss.item() * len(m); n += len(m)
    vd = val_dice(sets["val"])
    rows.append(dict(epoch=ep + 1, train_loss=tl / max(n, 1), val_dice=vd, seconds=time.time() - t0))
    pd.DataFrame(rows).to_csv(os.path.join(args.out, "metrics.csv"), index=False)
    log("epoch %d loss %.4f | val dice %.4f | %.0fs" % (ep + 1, rows[-1]["train_loss"], vd, rows[-1]["seconds"]))
    if vd > best:
        best, best_ep = vd, ep + 1
        torch.save(model.state_dict(), os.path.join(args.out, "best_model.pt"))
torch.save({k: v.half() if v.is_floating_point() else v for k, v in model.state_dict().items()}, os.path.join(args.out, "last_model.pt"))
model.load_state_dict(torch.load(os.path.join(args.out, "best_model.pt"), map_location=args.device))

rng = np.random.default_rng(0)
eval_sets = [(s, s, None) for s in ["val", "test", "shift"] if s in sets] + [("test_" + k, "test", k) for k in ["blur", "noise", "contrast"]]
res = dict(config=vars(args), split={k: len(v) for k, v in sets.items()}, best_epoch=best_ep, best_val_dice=best,
           n_params=sum(q.numel() for q in model.parameters()), environment=env)
dil = torch.nn.MaxPool2d(11, 1, 5)
for name, src, kind in eval_sets:
    idx = sets[src]
    maps = np.zeros((len(idx), X.shape[1], X.shape[2]), np.uint8)
    rec, pix = [], {k: [] for k in ["err", "ent_det", "ent_mc", "mi_mc", "ent_tta"]}
    for b in range(0, len(idx), 16):
        j = idx[b:b + 16]
        x = torch.from_numpy(np.asarray(X[j])).permute(0, 3, 1, 2).float().div(255).to(args.device)
        x = norm(corrupt(x, kind))
        with torch.no_grad():
            model.eval()
            with torch.autocast(args.device, enabled=amp):
                pdet = torch.sigmoid(model(x).float())
            enable_mc_dropout(model)
            with torch.autocast(args.device, enabled=amp):
                pmc = torch.stack([torch.sigmoid(model(x).float()) for _ in range(args.mc)])
            model.eval()
            views = []
            for fl in range(8):  # dihedral group: 4 rotations x optional flip; predictions mapped back
                xv = torch.rot90(x.flip(-1) if fl >= 4 else x, fl % 4, (2, 3))
                with torch.autocast(args.device, enabled=amp):
                    pv = torch.sigmoid(model(xv).float())
                pv = torch.rot90(pv, -(fl % 4), (1, 2))
                views.append(pv.flip(-1) if fl >= 4 else pv)
            ptta = torch.stack(views)
        e_det = ent(pdet); m_mc = pmc.mean(0); e_mc = ent(m_mc); mi = e_mc - ent(pmc).mean(0); m_tta = ptta.mean(0); e_tta = ent(m_tta)
        fg = dil((pdet > 0.5).float()[:, None])[:, 0] > 0
        for k, jj in enumerate(j):
            gt = np.asarray(M[jj]).astype(bool)
            d_det = pdet[k].cpu().numpy() > 0.5
            maps[b + k] = (pdet[k].cpu().numpy() * 255).round().astype(np.uint8)
            fgk = fg[k]
            rec.append(dict(array_index=int(jj), dice_det=dice(d_det, gt), dice_mc=dice(m_mc[k].cpu().numpy() > 0.5, gt),
                            dice_tta=dice(m_tta[k].cpu().numpy() > 0.5, gt), area_pred=float(d_det.mean()), area_gt=float(gt.mean()),
                            ent_det=float(e_det[k].mean()), ent_det_fg=float(e_det[k][fgk].mean()) if fgk.any() else 0.0,
                            ent_mc=float(e_mc[k].mean()), mi_mc=float(mi[k].mean()),
                            pdice_mc=pairwise_dice((pmc[:, k] > 0.5).cpu().numpy()), ent_tta=float(e_tta[k].mean()),
                            pdice_tta=pairwise_dice((ptta[:, k] > 0.5).cpu().numpy())))
            sel = rng.choice(gt.size, 512, replace=False)
            pix["err"].append((d_det.ravel() != gt.ravel())[sel])
            for key, t in [("ent_det", e_det), ("ent_mc", e_mc), ("mi_mc", mi), ("ent_tta", e_tta)]:
                pix[key].append(t[k].cpu().numpy().ravel()[sel].astype(np.float16))
    np.save(os.path.join(args.out, "maps_%s.npy" % name), maps)
    df = pd.DataFrame(rec); df.to_csv(os.path.join(args.out, "scores_%s.csv" % name), index=False)
    np.savez_compressed(os.path.join(args.out, "pix_%s.npz" % name), **{k: np.concatenate(v) for k, v in pix.items()})
    res[name] = dict(n=len(df), dice_det=float(df.dice_det.mean()), dice_mc=float(df.dice_mc.mean()), dice_tta=float(df.dice_tta.mean()))
    log(name.upper(), json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in res[name].items()}))
res["train_seconds"] = time.time() - t0
res["peak_gpu_mb"] = torch.cuda.max_memory_allocated() / 2**20 if amp else 0
json.dump(res, open(os.path.join(args.out, "results.json"), "w"), indent=1)
log("DONE")
