"""diabetic-retinopathy-ordinal-grading: ordinal formulations for 5-grade DR (0 none, 1 mild, 2 moderate, 3 severe NPDR, 4 PDR), data set, metrics.

Every formulation maps the network output to (a) a class distribution p over the 5 grades and (b) a predicted grade:
  ce    : 5 logits, softmax; grade = argmax p.
  sord  : 5 logits trained against soft ordinal targets q_k ~ exp(-(k-y)^2) (Diaz & Marathe 2019); grade = argmax p.
  reg   : 1 output s trained with MSE to y; grade = cut s at thresholds tuned on validation (default 0.5,1.5,2.5,3.5);
          p = the one-hot of the grade (a single regression model gives no distribution).
  coral : K-1 logits z_k = w.f + b_k (shared weights, separate biases; Cao et al. 2020), P(y>k) = sigmoid(z_k);
          grade = sum_k [P(y>k) > 0.5]; p from consecutive differences of the (monotonised) cumulative probabilities.
  corn  : K-1 logits trained conditionally, task k only on images with y >= k (Shi et al. 2023),
          P(y>k) = prod_{j<=k} sigmoid(z_j); grade and p as for coral.
"""
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

K = 5
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
N_OUT = {"ce": K, "sord": K, "reg": 1, "coral": K - 1, "corn": K - 1}


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


class CoralHead(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.w = nn.Linear(d, 1, bias=False)
        self.b = nn.Parameter(torch.arange(K - 1, 0, -1).float() / (K - 1))  # decreasing initial biases

    def forward(self, f):
        return self.w(f) + self.b


def build_model(name, form, pretrained=True):
    import timm
    m = timm.create_model(name, pretrained=pretrained, num_classes=0)
    d = m.num_features
    head = CoralHead(d) if form == "coral" else nn.Linear(d, N_OUT[form])
    return nn.Sequential(m, head)


def loss_fn(form, out, y):
    if form == "ce":
        return F.cross_entropy(out, y)
    if form == "sord":
        k = torch.arange(K, device=y.device).float()
        q = torch.softmax(-(k[None] - y[:, None].float()) ** 2, 1)
        return -(q * F.log_softmax(out, 1)).sum(1).mean()
    if form == "reg":
        return F.mse_loss(out[:, 0], y.float())
    if form == "coral":
        t = (y[:, None] > torch.arange(K - 1, device=y.device)[None]).float()
        return F.binary_cross_entropy_with_logits(out, t, reduction="none").sum(1).mean()
    if form == "corn":
        loss, n = 0.0, 0
        for k in range(K - 1):
            s = y >= k
            if s.any():
                loss = loss + F.binary_cross_entropy_with_logits(out[s, k], (y[s] > k).float(), reduction="sum")
                n += int(s.sum())
        return loss / max(n, 1)
    raise ValueError(form)


def decode(form, out, thresholds=(0.5, 1.5, 2.5, 3.5)):
    """out: numpy [N, n_out] raw outputs -> (p [N, K], grade [N], score [N] continuous severity)."""
    out = np.asarray(out, np.float64)
    if form in ("ce", "sord"):
        e = np.exp(out - out.max(1, keepdims=True)); p = e / e.sum(1, keepdims=True)
        return p, p.argmax(1), p @ np.arange(K)
    if form == "reg":
        s = out[:, 0]
        g = np.searchsorted(np.asarray(thresholds), s)
        return np.eye(K)[g], g, s
    sig = 1 / (1 + np.exp(-out))
    cum = sig if form == "coral" else np.cumprod(sig, 1)  # P(y > k), k = 0..K-2
    cum = np.minimum.accumulate(cum, axis=1)  # enforce monotone non-increasing
    ext = np.concatenate([np.ones((len(cum), 1)), cum, np.zeros((len(cum), 1))], 1)
    p = np.clip(ext[:, :-1] - ext[:, 1:], 0, 1)
    return p, (cum > 0.5).sum(1), cum.sum(1)


def tune_thresholds(score, y):
    """Coordinate search for reg thresholds maximising QWK on validation (common practice for DR regression)."""
    from sklearn.metrics import cohen_kappa_score
    t = np.array([0.5, 1.5, 2.5, 3.5])
    for _ in range(3):
        for i in range(4):
            best, bv = t[i], -1
            for v in np.linspace(t[i] - 0.5, t[i] + 0.5, 41):
                tt = t.copy(); tt[i] = v
                if np.all(np.diff(tt) > 0):
                    q = cohen_kappa_score(y, np.searchsorted(tt, score), weights="quadratic")
                    if q > bv:
                        best, bv = v, q
            t[i] = best
    return t


def grading_metrics(y, g):
    from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix
    y, g = np.asarray(y), np.asarray(g)
    ref_y, ref_g = y >= 2, g >= 2
    vt_y, vt_g = y >= 3, g >= 3
    return dict(n=int(len(y)), qwk=float(cohen_kappa_score(y, g, weights="quadratic")), accuracy=float(accuracy_score(y, g)),
                mae=float(np.abs(y - g).mean()), severe_error_rate=float((np.abs(y - g) >= 2).mean()),
                referable_sens=float(ref_g[ref_y].mean()) if ref_y.any() else float("nan"),
                referable_spec=float((~ref_g)[~ref_y].mean()) if (~ref_y).any() else float("nan"),
                vt_sens=float(vt_g[vt_y].mean()) if vt_y.any() else float("nan"),
                confusion=confusion_matrix(y, g, labels=list(range(K))).tolist())


class FundusSet(torch.utils.data.Dataset):
    """uint8 [N, S, S, 3] arrays of disc-cropped fundus images. Training: flips, rotation, mild colour jitter, random crop."""

    def __init__(self, images, labels, index, train=False, size=384):
        self.images, self.labels, self.index, self.train, self.size = images, labels, index, train, size

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        j = int(self.index[i])
        x = torch.from_numpy(np.ascontiguousarray(self.images[j])).permute(2, 0, 1).float()[None] / 255.0
        if self.train:
            ang = np.deg2rad(random.uniform(-180, 180)); sc = random.uniform(0.9, 1.1)
            th = torch.tensor([[[np.cos(ang) / sc, -np.sin(ang) / sc, 0], [np.sin(ang) / sc, np.cos(ang) / sc, 0]]], dtype=torch.float32)
            x = F.grid_sample(x, F.affine_grid(th, (1, 3, self.size, self.size), align_corners=False), align_corners=False)
            if random.random() < 0.5:
                x = x.flip(-1)
            x = x * random.uniform(0.85, 1.15)
            m = x.mean(); x = ((x - m) * random.uniform(0.85, 1.15) + m).clamp(0, 1)
        elif x.shape[-1] != self.size:
            x = F.interpolate(x, size=self.size, mode="bilinear", antialias=True, align_corners=False)
        x = (x[0] - torch.from_numpy(MEAN)[:, None, None]) / torch.from_numpy(STD)[:, None, None]
        return x, int(self.labels[j]), j
