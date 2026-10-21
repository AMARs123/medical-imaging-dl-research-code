"""histopathology-hospital-shift: one Camelyon17-WILDS run (tumour vs normal patch), evaluated on ID validation, OOD validation and test hospitals.

Example: python train.py --root . --out experiments/EXP001 --backbone r50_imagenet --aug none --mode finetune --seed 0
Data: uint8 arrays made by prepare_hf.py from the official WILDS splits (directory P10_DATA, default data/camelyon17_np;
the official download was unreachable on 17 Oct 2026, see prepare_hf.py): train and ID validation from hospitals 0, 3, 4 (WILDS
split 0 / split 1), OOD validation = hospital 1, test = hospital 2. Earlier wording of this docstring: (split 0 of hospitals 0,3,4 = train,
split 1 of hospitals 0,3,4 = ID validation, hospital 1 = OOD validation, hospital 2 = test).
--backbone r50_imagenet | r50_lunit | vits_imagenet | vits_lunit   (same architecture, ImageNet vs pathology SSL weights)
--aug none | hed  (HED stain augmentation, Tellez et al. 2019: each of the H, E, D channels scaled by U(1-0.05,1+0.05)
                   and shifted by U(-0.05,0.05) in optical-density space, plus flips and 90-degree rotations for both settings)
--mode finetune | linear  (linear: frozen backbone, only the classification layer is trained)
Training: AdamW, a fixed budget of 1,000 steps of 128 patches (128,000 patches, about 42% of one epoch, drawn by shuffling),
patches upsampled from 96 to 224 px. (Planned: 2 full epochs; reduced on 17 Oct 2026 before any run because 100 runs of
2 epochs would not fit the available GPU time; backup train_before_budget.py.)
The accuracy after every quarter of the budget (on fixed random subsets of 10,000 patches per set; final evaluation on all
patches) is stored for all three evaluation sets, so selection rules can be compared later.
Outputs: config.yaml, command.txt, environment.txt, training.log, metrics.csv, predictions_<set>.csv (final model),
best_model.pt (= final), last_model.pt (fp16), results.json.
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
from PIL import Image

BACK = {"r50_imagenet": "resnet50.a1_in1k", "r50_lunit": "hf-hub:1aurent/resnet50.lunit_swav",
        "vits_imagenet": "vit_small_patch16_224.augreg_in1k", "vits_lunit": "hf-hub:1aurent/vit_small_patch16_224.lunit_dino"}
p = argparse.ArgumentParser()
p.add_argument("--root", required=True)
p.add_argument("--out", required=True)
p.add_argument("--backbone", default="r50_imagenet", choices=list(BACK))
p.add_argument("--aug", default="none", choices=["none", "hed"])
p.add_argument("--mode", default="finetune", choices=["finetune", "linear"])
p.add_argument("--seed", type=int, default=0)
p.add_argument("--epochs", type=int, default=1)  # 17 Oct 2026, before any run: fixed budget of 1,000 steps (see --max-steps)
p.add_argument("--batch-size", type=int, default=128)
p.add_argument("--micro-batch", type=int, default=64, help="patches per forward/backward pass; gradients accumulated over the batch")
p.add_argument("--lr", type=float, default=1e-4)
p.add_argument("--eval-every", type=float, default=0.25, help="evaluate every this fraction of the budget (17 Oct 2026: 4 points)")
p.add_argument("--eval-subsample", type=int, default=10000, help="patches per evaluation set during training (final eval uses all)")
p.add_argument("--workers", type=int, default=5)
p.add_argument("--max-steps", type=int, default=1000, help="steps per epoch (0 = full epoch); 1,000 x 128 = 128,000 patches")
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

DATA = os.environ.get("P10_DATA", os.path.join(args.root, "data", "camelyon17_np"))
PARTS = ["train", "id_val", "ood_val", "test"]
XS = {k: np.load(os.path.join(DATA, k + "_x.npy"), mmap_mode="r") for k in PARTS}
YS = {k: np.load(os.path.join(DATA, k + "_y.npy")) for k in PARTS}
OFF = np.cumsum([0] + [len(YS[k]) for k in PARTS])
sets = {k: np.arange(OFF[i], OFF[i + 1]) for i, k in enumerate(PARTS)}


def get_img(j):
    i = int(np.searchsorted(OFF, j, side="right") - 1)
    return XS[PARTS[i]][j - OFF[i]]


if args.limit:
    sets = {k: v[:args.limit] for k, v in sets.items()}
y = np.concatenate([YS[k] for k in PARTS]).astype(int)
log("split", json.dumps({k: len(v) for k, v in sets.items()}), "train tumour share %.3f" % y[sets["train"]].mean())
# stain vectors of Ruifrok & Johnston (2001) as used by scikit-image: rows = optical density of H, E, D in R, G, B
HED2RGB = torch.tensor([[0.65, 0.70, 0.29], [0.07, 0.99, 0.11], [0.27, 0.57, 0.78]])
HED2RGB[2] = torch.linalg.cross(HED2RGB[0], HED2RGB[1])  # skimage replaces the third vector by the cross product
HED2RGB = HED2RGB / HED2RGB.norm(dim=1, keepdim=True)
RGB2HED = torch.linalg.inv(HED2RGB)  # stains = od (row vector) @ RGB2HED
MEAN = torch.tensor([0.485, 0.456, 0.406])[:, None, None]; STD = torch.tensor([0.229, 0.224, 0.225])[:, None, None]


class DS(torch.utils.data.Dataset):
    def __init__(self, idx, train):
        self.idx, self.train = idx, train

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        j = int(self.idx[i])
        x = torch.from_numpy(np.asarray(get_img(j), np.float32) / 255.0).permute(2, 0, 1)
        if self.train:
            if random.random() < 0.5:
                x = x.flip(-1)
            x = torch.rot90(x, random.randint(0, 3), (1, 2))
            if args.aug == "hed":
                od = -torch.log(x.clamp(1e-6, 1))  # optical density
                hed = torch.einsum("ij,jhw->ihw", RGB2HED.T, od)
                a = torch.tensor([random.uniform(0.95, 1.05) for _ in range(3)])[:, None, None]
                b = torch.tensor([random.uniform(-0.05, 0.05) for _ in range(3)])[:, None, None]
                od = torch.einsum("ij,jhw->ihw", HED2RGB.T, hed * a + b)
                x = torch.exp(-od).clamp(0, 1)
        x = F.interpolate(x[None], size=224, mode="bilinear", align_corners=False)[0]
        return (x - MEAN) / STD, int(y[j])


def loader(idx, train, bs=None):
    return torch.utils.data.DataLoader(DS(idx, train), batch_size=bs or args.batch_size, shuffle=train, num_workers=args.workers,
                                       drop_last=train, pin_memory=args.device == "cuda", persistent_workers=False)


import timm  # noqa: E402
model = timm.create_model(BACK[args.backbone], pretrained=True, num_classes=2).to(args.device)
# 17 Oct 2026, before any run: normalise with the statistics published with each set of weights (timm / hub pretrained_cfg):
# ImageNet mean/std for both ResNet-50s, 0.5/0.5 for both ViT-S (the AugReg ViT-S expects 0.5; ImageNet values were used before)
MEAN = torch.tensor(model.pretrained_cfg["mean"], dtype=torch.float32)[:, None, None]
STD = torch.tensor(model.pretrained_cfg["std"], dtype=torch.float32)[:, None, None]
log("normalisation mean %s std %s" % (list(model.pretrained_cfg["mean"]), list(model.pretrained_cfg["std"])))
if args.mode == "linear":
    for n_, q in model.named_parameters():
        q.requires_grad = any(k in n_ for k in ["fc.", "head."]) and not n_.startswith("head.norm")
params = [q for q in model.parameters() if q.requires_grad]
opt = torch.optim.AdamW(params, lr=args.lr if args.mode == "finetune" else 1e-3, weight_decay=0.05 if args.mode == "finetune" else 0.0)
trl = loader(sets["train"], True)
steps_total = args.epochs * (args.max_steps or len(trl))
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=opt.param_groups[0]["lr"], total_steps=steps_total, pct_start=0.05)
amp = args.device == "cuda"
scaler = torch.amp.GradScaler(enabled=amp)
rs = np.random.default_rng(0)
sub = {k: (np.sort(rs.choice(v, min(len(v), args.eval_subsample), replace=False)) if k != "train" else None) for k, v in sets.items()}


@torch.no_grad()
def predict(idx):
    model.eval(); out = []
    for x, _ in loader(idx, False, 128):
        with torch.autocast(args.device, enabled=amp):
            out.append(torch.softmax(model(x.to(args.device)).float(), 1)[:, 1].cpu().numpy())
    model.train()
    return np.concatenate(out)


rows, t0, step = [], time.time(), 0
every = max(1, int((args.max_steps or len(trl)) * args.eval_every))
for ep in range(args.epochs):
    model.train()
    if args.mode == "linear":
        model.eval()  # frozen BatchNorm / dropout
    for st, (x, t) in enumerate(trl):
        if args.max_steps and st >= args.max_steps:
            break
        x, t = x.to(args.device), t.to(args.device)
        # 17 Oct 2026, before any run finished: the mini-batch of 128 is processed as micro-batches of --micro-batch patches with
        # gradient accumulation (same loss and update; BatchNorm statistics per micro-batch), so that GPU memory stays within the
        # job's share (the server stops jobs that exceed it)
        opt.zero_grad(set_to_none=True); loss = 0.0
        for xm, tm in zip(x.split(args.micro_batch), t.split(args.micro_batch)):
            with torch.autocast(args.device, enabled=amp):
                lm = F.cross_entropy(model(xm).float(), tm) * (len(tm) / len(t))
            scaler.scale(lm).backward(); loss = loss + lm.detach()
        scaler.step(opt); scaler.update(); sched.step(); step += 1
        if step % every == 0 or step == steps_total:
            r = dict(step=step, epoch=step / (args.max_steps or len(trl)), loss=float(loss.item()), seconds=time.time() - t0)
            for k in ["id_val", "ood_val", "test"]:
                pr = predict(sub[k]); r[k + "_acc"] = float(((pr > 0.5) == y[sub[k]]).mean())
            rows.append(r); pd.DataFrame(rows).to_csv(os.path.join(args.out, "metrics.csv"), index=False)
            log("step %d ep %.2f loss %.4f | id %.4f ood %.4f test %.4f | %.0fs" % (step, r["epoch"], r["loss"], r["id_val_acc"], r["ood_val_acc"], r["test_acc"], r["seconds"]))
torch.save(model.state_dict(), os.path.join(WDIR, "best_model.pt"))
torch.save({k: v.half() if v.is_floating_point() else v for k, v in model.state_dict().items()}, os.path.join(WDIR, "last_model.pt"))
open(os.path.join(args.out, "CHECKPOINT_NOTE.txt"), "w").write("No checkpoint selection inside a run: best_model.pt is the final model; selection rules are compared across runs.\n")
res = dict(config=vars(args), environment=env, train_seconds=time.time() - t0, n_params=sum(q.numel() for q in model.parameters()),
           n_trainable=sum(q.numel() for q in params))
for k in ["id_val", "ood_val", "test"]:
    pr = predict(sets[k])
    res[k] = dict(n=int(len(pr)), acc=float(((pr > 0.5) == y[sets[k]]).mean()))
    pd.DataFrame(dict(index=sets[k], label=y[sets[k]], prob=pr.astype(np.float32))).to_csv(os.path.join(args.out, "predictions_%s.csv" % k), index=False)
res["peak_gpu_mb"] = torch.cuda.max_memory_allocated() / 2**20 if amp else 0
json.dump(res, open(os.path.join(args.out, "results.json"), "w"), indent=1)
log("RESULT", json.dumps({k: res[k] for k in ["id_val", "ood_val", "test"]}))
log("DONE")
