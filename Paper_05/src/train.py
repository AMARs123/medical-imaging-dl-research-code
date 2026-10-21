"""Train one DR grading model (Paper 05) on the DDR training split; evaluate on DDR validation, DDR test and DeepDRiD.

Example: python train.py --proc data/processed --out experiments/EXP001 --model resnet50.a1_in1k --form corn --seed 0
Outputs: config.yaml, command.txt, environment.txt, training.log, metrics.csv, outputs_<set>.npy (raw outputs, shape
[N, 4 views, n_out]; views = identity, horizontal flip, vertical flip, both), labels_<set>.npy, best_model.pt,
last_model.pt (fp16), results.json (metrics of the identity view; reg thresholds tuned on DDR validation).
The checkpoint is chosen on DDR-validation QWK; DDR test and DeepDRiD are never used for any choice.
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
from common import FundusSet, N_OUT, build_model, decode, grading_metrics, loss_fn, set_seed, tune_thresholds  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--proc", required=True)
p.add_argument("--out", required=True)
p.add_argument("--model", default="resnet50.a1_in1k")
p.add_argument("--form", default="ce", choices=list(N_OUT))
p.add_argument("--seed", type=int, default=0)
p.add_argument("--epochs", type=int, default=20)
p.add_argument("--batch-size", type=int, default=12)  # 12 (not 24): keeps GPU memory near 2.5 GB, inside the shard allowance (2026-10-16)
p.add_argument("--lr", type=float, default=1e-4)
p.add_argument("--weight-decay", type=float, default=0.05)
p.add_argument("--size", type=int, default=384)
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
try:
    import timm; env["timm"] = timm.__version__
except Exception:
    pass
open(os.path.join(args.out, "environment.txt"), "w").write("\n".join("%s: %s" % kv for kv in env.items()) + "\n")

set_seed(args.seed)
meta = pd.read_csv(os.path.join(args.proc, "meta.csv"))
images = np.load(os.path.join(args.proc, "images.npy"), mmap_mode="r")
y = meta.grade.values.astype(int)
ok = meta.ok.values.astype(bool)
sets = {"train": np.where((meta.dataset == "DDR") & (meta.split == "train") & ok)[0],
        "val": np.where((meta.dataset == "DDR") & (meta.split == "valid") & ok)[0],
        "test": np.where((meta.dataset == "DDR") & (meta.split == "test") & ok)[0],
        "ext": np.where((meta.dataset == "DeepDRiD") & ok)[0]}
if args.limit:
    sets = {k: v[:args.limit] for k, v in sets.items()}
log("split", json.dumps({k: len(v) for k, v in sets.items()}), "train grades", np.bincount(y[sets["train"]], minlength=5).tolist())


def loader(idx, train):
    return torch.utils.data.DataLoader(FundusSet(images, y, idx, train, args.size), batch_size=args.batch_size, shuffle=train,
                                       num_workers=args.workers, drop_last=train, pin_memory=args.device == "cuda")


model = build_model(args.model, args.form).to(args.device)
n_params = sum(q.numel() for q in model.parameters())
opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
trl = loader(sets["train"], True)
steps = args.max_steps or len(trl)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.epochs * steps, pct_start=0.1)
amp = args.device == "cuda"
scaler = torch.amp.GradScaler(enabled=amp)


@torch.no_grad()
def raw_outputs(idx, views=1):
    model.eval()
    outs = []
    for x, _, _ in loader(idx, False):
        x = x.to(args.device)
        vs = [x, x.flip(-1), x.flip(-2), x.flip(-1).flip(-2)][:views]
        with torch.autocast(args.device, enabled=amp):
            outs.append(torch.stack([model(v).float() for v in vs], 1).cpu().numpy())
    return np.concatenate(outs)


rows, best, best_ep, t0 = [], -1, 0, time.time()
for ep in range(args.epochs):
    model.train(); tl = n = 0.0
    for st, (x, t, _) in enumerate(trl):
        if args.max_steps and st >= args.max_steps:
            break
        x, t = x.to(args.device), t.to(args.device)
        with torch.autocast(args.device, enabled=amp):
            out = model(x)
        loss = loss_fn(args.form, out.float(), t)
        opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
        tl += loss.item() * len(t); n += len(t)
    ov = raw_outputs(sets["val"])[:, 0]
    _, gv, sv = decode(args.form, ov)
    if args.form == "reg":
        gv = np.searchsorted(tune_thresholds(sv, y[sets["val"]]), sv)
    vm = grading_metrics(y[sets["val"]], gv)
    rows.append(dict(epoch=ep + 1, train_loss=tl / max(n, 1), val_qwk=vm["qwk"], val_acc=vm["accuracy"], val_severe=vm["severe_error_rate"],
                     seconds=time.time() - t0))
    pd.DataFrame(rows).to_csv(os.path.join(args.out, "metrics.csv"), index=False)
    log("epoch %d loss %.4f | val qwk %.4f acc %.4f severe %.4f | %.0fs" % (ep + 1, rows[-1]["train_loss"], vm["qwk"], vm["accuracy"],
                                                                         vm["severe_error_rate"], rows[-1]["seconds"]))
    if vm["qwk"] > best:
        best, best_ep = vm["qwk"], ep + 1
        torch.save(model.state_dict(), os.path.join(args.out, "best_model.pt"))
torch.save({k: v.half() if v.is_floating_point() else v for k, v in model.state_dict().items()}, os.path.join(args.out, "last_model.pt"))
model.load_state_dict(torch.load(os.path.join(args.out, "best_model.pt"), map_location=args.device))

res = dict(config=vars(args), split={k: len(v) for k, v in sets.items()}, n_params=n_params, best_epoch=best_ep, best_val_qwk=best,
           train_seconds=time.time() - t0, environment=env, peak_gpu_mb=torch.cuda.max_memory_allocated() / 2**20 if amp else 0)
thr = None
for name in ["val", "test", "ext"]:
    o = raw_outputs(sets[name], views=4)
    np.save(os.path.join(args.out, "outputs_%s.npy" % name), o.astype(np.float32))
    np.save(os.path.join(args.out, "labels_%s.npy" % name), y[sets[name]])
    np.save(os.path.join(args.out, "index_%s.npy" % name), sets[name])
    _, g, s = decode(args.form, o[:, 0])
    if args.form == "reg":
        if name == "val":
            thr = tune_thresholds(s, y[sets["val"]])
            res["reg_thresholds"] = thr.tolist()
        g = np.searchsorted(thr, s)
    res[name] = grading_metrics(y[sets[name]], g)
json.dump(res, open(os.path.join(args.out, "results.json"), "w"), indent=1)
log("TEST", json.dumps({k: round(v, 4) for k, v in res["test"].items() if isinstance(v, float)}))
log("EXT", json.dumps({k: round(v, 4) for k, v in res["ext"].items() if isinstance(v, float)}))
log("DONE")
