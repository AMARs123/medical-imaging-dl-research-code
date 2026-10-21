"""1-D ECG models for ecg-attribution-testing. Every model ends with  feature map [B, C, T'] -> global average pool -> linear,
so a class-activation map (CAM) exists for all three architectures and can be compared on equal terms.

xresnet1d50   : 1-D XResNet-50 (bottleneck ResNet with the "bag of tricks" stem), as in Strodthoff et al. (2021).
inception1d   : InceptionTime-style network (Fawaz et al. 2020), 1-D convolutions with kernel sizes 39/19/9.
convtran1d    : convolutional stem (downsampling x4) + 4 transformer encoder layers with learned positions.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------------------------------- XResNet1d
def conv_bn(ci, co, k=3, s=1, act=True, zero_bn=False):
    layers = [nn.Conv1d(ci, co, k, s, padding=k // 2, bias=False), nn.BatchNorm1d(co)]
    if zero_bn:
        nn.init.zeros_(layers[1].weight)
    if act:
        layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, ci, mid, stride, k=5):
        super().__init__()
        co = mid * self.expansion
        self.convs = nn.Sequential(conv_bn(ci, mid, 1), conv_bn(mid, mid, k, stride), conv_bn(mid, co, 1, act=False, zero_bn=True))
        idp = []
        if stride != 1:
            idp.append(nn.AvgPool1d(stride, ceil_mode=True))
        if ci != co:
            idp.append(conv_bn(ci, co, 1, act=False))
        self.idpath = nn.Sequential(*idp)

    def forward(self, x):
        return F.relu(self.convs(x) + self.idpath(x))


class XResNet1d(nn.Module):
    def __init__(self, n_in=12, n_out=5, layers=(3, 4, 6, 3), k=5):
        super().__init__()
        self.stem = nn.Sequential(conv_bn(n_in, 32, k, 2), conv_bn(32, 32, k), conv_bn(32, 64, k), nn.MaxPool1d(3, 2, 1))
        blocks, ci = [], 64
        for i, (n, mid) in enumerate(zip(layers, (64, 128, 256, 512))):
            for j in range(n):
                blocks.append(Bottleneck(ci, mid, 2 if (j == 0 and i > 0) else 1, k))
                ci = mid * Bottleneck.expansion
        self.body = nn.Sequential(*blocks)
        self.head = nn.Linear(ci, n_out)
        self.n_features = ci

    def features(self, x):
        return self.body(self.stem(x))


# ----------------------------------------------------------------------------- InceptionTime 1d
class InceptionBlock(nn.Module):
    def __init__(self, ci, nf=32, ks=(39, 19, 9), bottleneck=32):
        super().__init__()
        self.bottleneck = nn.Conv1d(ci, bottleneck, 1, bias=False) if ci > 1 else nn.Identity()
        b = bottleneck if ci > 1 else ci
        self.convs = nn.ModuleList([nn.Conv1d(b, nf, k, padding=k // 2, bias=False) for k in ks])
        self.pool = nn.Sequential(nn.MaxPool1d(3, 1, 1), nn.Conv1d(ci, nf, 1, bias=False))
        self.bn = nn.BatchNorm1d(nf * 4)

    def forward(self, x):
        b = self.bottleneck(x)
        return F.relu(self.bn(torch.cat([c(b) for c in self.convs] + [self.pool(x)], 1)))


class Inception1d(nn.Module):
    def __init__(self, n_in=12, n_out=5, depth=6, nf=32):
        super().__init__()
        self.blocks, self.shortcuts = nn.ModuleList(), nn.ModuleList()
        ci, res = n_in, n_in
        for d in range(depth):
            self.blocks.append(InceptionBlock(ci, nf))
            ci = nf * 4
            if d % 3 == 2:
                self.shortcuts.append(nn.Sequential(nn.Conv1d(res, ci, 1, bias=False), nn.BatchNorm1d(ci)))
                res = ci
        self.head = nn.Linear(ci, n_out)
        self.n_features = ci

    def features(self, x):
        res, s = x, 0
        for d, blk in enumerate(self.blocks):
            x = blk(x)
            if d % 3 == 2:
                x = F.relu(x + self.shortcuts[s](res))
                res, s = x, s + 1
        return x


# ----------------------------------------------------------------------------- conv + transformer
class ConvTran1d(nn.Module):
    def __init__(self, n_in=12, n_out=5, d=128, layers=4, heads=4, length=1000):
        super().__init__()
        self.stem = nn.Sequential(conv_bn(n_in, 64, 7, 2), conv_bn(64, d, 7, 2))  # 1000 -> 250 tokens of 40 ms
        self.pos = nn.Parameter(torch.zeros(1, d, (length + 3) // 4))
        nn.init.trunc_normal_(self.pos, std=0.02)
        enc = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout=0.1, batch_first=True, norm_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(enc, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, n_out)
        self.n_features = d

    def features(self, x):
        f = self.stem(x)
        f = f + self.pos[:, :, :f.shape[-1]]
        f = self.norm(self.encoder(f.transpose(1, 2)))
        return f.transpose(1, 2)


class ECGNet(nn.Module):
    """Wrapper: forward returns logits; cam() returns the class-activation map [B, K, T'] whose time mean + bias = logit."""

    def __init__(self, name, n_in=12, n_out=5):
        super().__init__()
        self.name = name
        self.net = {"xresnet1d50": XResNet1d, "inception1d": Inception1d, "convtran1d": ConvTran1d}[name](n_in, n_out)

    def forward(self, x):
        return self.net.head(self.net.features(x).mean(-1))

    def cam(self, x):
        f = self.net.features(x)
        return torch.einsum("kc,bct->bkt", self.net.head.weight, f)


if __name__ == "__main__":
    for n in ["xresnet1d50", "inception1d", "convtran1d"]:
        m = ECGNet(n).eval()  # eval: no dropout, BatchNorm uses running statistics
        x = torch.randn(2, 12, 1000)
        lo, cam = m(x), m.cam(x)
        assert torch.allclose(cam.mean(-1) + m.net.head.bias, lo, atol=1e-4)
        print(n, sum(p.numel() for p in m.parameters()), tuple(lo.shape), tuple(cam.shape))
