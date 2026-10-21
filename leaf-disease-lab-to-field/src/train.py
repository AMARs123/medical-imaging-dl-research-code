"""leaf-disease-lab-to-field: train a plant-disease classifier on PlantVillage (27 shared classes) and test it in the lab and in the field.

Example: python train.py --root . --out experiments/EXP001 --model convnext_tiny.fb_in1k --split leaf --seed 0
Split of PlantVillage (fixed by --split-seed 0): 70% train / 10% validation / 20% test, stratified by class, either by
leaf group ('leaf'; images of one known leaf stay together) or by image ('random').
Conditions:
  --source color | segmented | mixed   training images: original photos, background-removed photos (PlantVillage's
                                       segmented version), or each image drawn as either with probability 0.5
  --aug base | strong                  strong adds colour jitter, blur, random erasing and random greyish background
                                       tinting, to reduce reliance on the uniform lab background
  --fewshot k                          add k PlantDoc *training* images per class (drawn with the run seed) to training
Evaluation (checkpoint chosen on PlantVillage validation accuracy, as a practitioner without field data would):
  pv_test (PlantVillage test split, colour photos), pd_all (all PlantDoc images not used for training), pd_test (official
  PlantDoc test split, excluding nothing because few-shot images come only from the PlantDoc training split).
Outputs: config.yaml, command.txt, environment.txt, training.log, metrics.csv, predictions_<set>.csv (logits),
best_model.pt, last_model.pt (fp16), results.json.
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
from PIL import Image, ImageFilter

p = argparse.ArgumentParser()
p.add_argument("--root", required=True)
p.add_argument("--out", required=True)
p.add_argument("--model", default="convnext_tiny.fb_in1k")
p.add_argument("--split", default="leaf", choices=["leaf", "random"])
p.add_argument("--source", default="color", choices=["color", "segmented", "mixed"])
p.add_argument("--aug", default="base", choices=["base", "strong"])
p.add_argument("--fewshot", type=int, default=0)
p.add_argument("--freeze", type=int, default=0, help="1 = linear probe: backbone frozen in eval mode, only the classifier is trained")
p.add_argument("--seed", type=int, default=0)
p.add_argument("--split-seed", type=int, default=0)
p.add_argument("--epochs", type=int, default=8)
p.add_argument("--batch-size", type=int, default=48)
p.add_argument("--lr", type=float, default=1e-4)
p.add_argument("--workers", type=int, default=3)
p.add_argument("--size", type=int, default=224)
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
try:
    import timm; env["timm"] = timm.__version__
except Exception:
    pass
open(os.path.join(args.out, "environment.txt"), "w").write("\n".join("%s: %s" % kv for kv in env.items()) + "\n")
random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)

idx = pd.read_csv(os.path.join(args.root, "data", "index.csv"), keep_default_na=False)
K = idx.label.max() + 1
pv = idx[idx.source == "PV"].copy()
rs = np.random.default_rng(args.split_seed)
part = {}
for c, g in pv.groupby("label"):
    units = g.group.unique() if args.split == "leaf" else g.index.values
    units = units[rs.permutation(len(units))]
    n = len(units)
    for k, u in enumerate(units):
        part[(c, u)] = "train" if k < 0.7 * n else ("val" if k < 0.8 * n else "test")
pv["part"] = [part[(r.label, r.group if args.split == "leaf" else i)] for i, r in pv.iterrows()]
pd_ = idx[idx.source == "PD"].copy()
rf = np.random.default_rng(1000 + args.seed)
few = []
if args.fewshot:
    for c, g in pd_[pd_.split == "train"].groupby("label"):
        few += list(rf.choice(g.index.values, min(args.fewshot, len(g)), replace=False))
pd_["part"] = np.where(pd_.index.isin(few), "fewshot", "eval")
tr = pv[pv.part == "train"]
if args.source == "segmented":
    tr = tr[tr.seg_path != ""]
train = pd.concat([tr.assign(use_seg=args.source == "segmented"), pd_.loc[few].assign(use_seg=False)])
sets = {"val": pv[pv.part == "val"], "pv_test": pv[pv.part == "test"], "pd_all": pd_[pd_.part == "eval"],
        "pd_test": pd_[(pd_.part == "eval") & (pd_.split == "test")]}
if args.limit:
    train = train.sample(min(len(train), args.limit), random_state=0)
    sets = {k: v.head(args.limit) for k, v in sets.items()}
leak = len(set(tr.group) & set(pv[pv.part == "test"].group))
split_info = dict(n_train=len(train), n_fewshot=len(few), **{"n_" + k: len(v) for k, v in sets.items()}, groups_shared_train_test=leak)
log("split", json.dumps(split_info))

MEAN = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
STD = torch.tensor([0.229, 0.224, 0.225])[:, None, None]


class DS(torch.utils.data.Dataset):
    def __init__(self, df, train):
        self.df, self.train = df.reset_index(), train

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        seg = r.get("use_seg", False) or (self.train and args.source == "mixed" and r.seg_path and random.random() < 0.5)
        im = Image.open(os.path.join(args.root, r.seg_path if seg and r.seg_path else r.path)).convert("RGB")
        if self.train:
            w, h = im.size; s = random.uniform(0.6, 1.0) ** 0.5
            cw, ch = int(w * s), int(h * s); x0, y0 = random.randint(0, w - cw), random.randint(0, h - ch)
            im = im.crop((x0, y0, x0 + cw, y0 + ch)).resize((args.size, args.size), Image.BILINEAR)
            if random.random() < 0.5:
                im = im.transpose(Image.FLIP_LEFT_RIGHT)
            if random.random() < 0.5:
                im = im.transpose(Image.FLIP_TOP_BOTTOM)
            if args.aug == "strong" and random.random() < 0.5:
                im = im.filter(ImageFilter.GaussianBlur(random.uniform(0.1, 2.0)))
        else:
            im = im.resize((args.size, args.size), Image.BILINEAR)
        x = torch.from_numpy(np.asarray(im, np.float32) / 255.0).permute(2, 0, 1)
        if self.train:
            if args.aug == "strong":
                x = x * random.uniform(0.6, 1.4)
                m = x.mean(); x = (x - m) * random.uniform(0.6, 1.4) + m
                g = x.mean(0, keepdim=True); x = (x - g) * random.uniform(0.5, 1.5) + g
                x = x + torch.tensor([random.uniform(-0.08, 0.08) for _ in range(3)])[:, None, None]  # colour cast
                if random.random() < 0.5:  # random erasing
                    eh, ew = random.randint(20, 90), random.randint(20, 90)
                    ey, ex = random.randint(0, args.size - eh), random.randint(0, args.size - ew)
                    x[:, ey:ey + eh, ex:ex + ew] = torch.rand(3, 1, 1)
                if seg and random.random() < 0.7:  # replace the black background of segmented images by a random tint
                    bg = (x.max(0, keepdim=True).values < 0.04).float()
                    x = x * (1 - bg) + bg * (torch.rand(3, 1, 1) * 0.6 + 0.2 + 0.08 * torch.randn(1, args.size, args.size))
            else:
                x = x * random.uniform(0.9, 1.1)
            x = x.clamp(0, 1)
        return (x - MEAN) / STD, int(r.label)


def loader(df, train):
    return torch.utils.data.DataLoader(DS(df, train), batch_size=args.batch_size, shuffle=train, num_workers=args.workers,
                                       drop_last=train, pin_memory=args.device == "cuda", persistent_workers=False)


import timm  # noqa: E402
kw = dict(img_size=args.size) if "dinov2" in args.model else {}  # DINOv2 weights were trained at 518 px; positions are interpolated
model = timm.create_model(args.model, pretrained=True, num_classes=int(K), **kw).to(args.device)
head = model.get_classifier()
if args.freeze:  # linear probe (added 16 Oct 2026, before any leaf-disease-lab-to-field run): only the new classifier layer is trained
    for q in model.parameters():
        q.requires_grad_(False)
    for q in head.parameters():
        q.requires_grad_(True)
opt = torch.optim.AdamW([q for q in model.parameters() if q.requires_grad], lr=args.lr, weight_decay=0.05)
trl = loader(train, True)
steps = args.max_steps or len(trl)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.epochs * steps, pct_start=0.1)
amp = args.device == "cuda"
scaler = torch.amp.GradScaler(enabled=amp)


@torch.no_grad()
def predict(df):
    model.eval(); out = []
    for x, _ in loader(df, False):
        with torch.autocast(args.device, enabled=amp):
            out.append(model(x.to(args.device)).float().cpu().numpy())
    return np.concatenate(out)


def metrics(y, lo):
    from sklearn.metrics import balanced_accuracy_score, f1_score
    pr = lo.argmax(1)
    return dict(n=int(len(y)), accuracy=float((pr == y).mean()), balanced_accuracy=float(balanced_accuracy_score(y, pr)),
                macro_f1=float(f1_score(y, pr, average="macro")))


rows, best, best_ep, t0 = [], -1, 0, time.time()
for ep in range(args.epochs):
    model.train(); tl = n = 0.0
    if args.freeze:  # frozen backbone stays in eval mode (no dropout, drop-path or normalisation updates)
        model.eval(); head.train()
    for st, (x, y) in enumerate(trl):
        if args.max_steps and st >= args.max_steps:
            break
        x, y = x.to(args.device), y.to(args.device)
        with torch.autocast(args.device, enabled=amp):
            loss = F.cross_entropy(model(x).float(), y, label_smoothing=0.1)
        opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
        tl += loss.item() * len(y); n += len(y)
    vm = metrics(sets["val"].label.values, predict(sets["val"]))
    rows.append(dict(epoch=ep + 1, train_loss=tl / max(n, 1), val_acc=vm["accuracy"], seconds=time.time() - t0))
    pd.DataFrame(rows).to_csv(os.path.join(args.out, "metrics.csv"), index=False)
    log("epoch %d loss %.4f | val acc %.4f | %.0fs" % (ep + 1, rows[-1]["train_loss"], vm["accuracy"], rows[-1]["seconds"]))
    if vm["accuracy"] > best:
        best, best_ep = vm["accuracy"], ep + 1
        torch.save(model.state_dict(), os.path.join(WDIR, "best_model.pt"))
torch.save({k: v.half() if v.is_floating_point() else v for k, v in model.state_dict().items()}, os.path.join(WDIR, "last_model.pt"))
model.load_state_dict(torch.load(os.path.join(WDIR, "best_model.pt"), map_location=args.device))
res = dict(config=vars(args), split=split_info, best_epoch=best_ep, best_val_acc=best, environment=env,
           n_params=sum(q.numel() for q in model.parameters()))
for name in ["pv_test", "pd_all", "pd_test"]:
    df = sets[name]; lo = predict(df)
    res[name] = metrics(df.label.values, lo)
    out = pd.DataFrame(dict(path=df.path.values, label=df.label.values, pred=lo.argmax(1)))
    for k in range(lo.shape[1]):
        out["logit_%d" % k] = lo[:, k]
    out.to_csv(os.path.join(args.out, "predictions_%s.csv" % name), index=False)
res["train_seconds"] = time.time() - t0
res["peak_gpu_mb"] = torch.cuda.max_memory_allocated() / 2**20 if amp else 0
json.dump(res, open(os.path.join(args.out, "results.json"), "w"), indent=1)
log("RESULT", json.dumps({k: res[k] for k in ["pv_test", "pd_all", "pd_test"]}))
log("DONE")
