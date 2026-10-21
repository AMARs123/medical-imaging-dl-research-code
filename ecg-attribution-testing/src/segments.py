"""Waveform segment masks at 100 Hz from the PTB-XL+ ECGDeli global fiducial points.

For each beat:  P   = [p-wave onset, p-wave offset]
                QRS = [QRS onset, QRS offset]
                STT = (QRS offset, t-wave offset]     (ST segment + T wave)
                other = every remaining time step (mostly the T-P baseline)
An onset is paired with the next matching offset only if it follows within a physiological limit
(P <= 250 ms, QRS <= 250 ms, QRS offset to T offset <= 700 ms) and before the next QRS onset; unpaired points are ignored.
segment_masks(ecg_id) returns a bool array [4, 1000] (P, QRS, STT, other) or None if no complete QRS was found.
"""
import os

import numpy as np

_FID = None
LIMIT = {"P": 25, "QRS": 25, "STT": 70}  # in 10 ms steps


def _load():
    global _FID
    if _FID is None:
        root = os.environ.get("P03_ROOT", os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
        _FID = np.load(os.path.join(root, "data", "processed", "fiducials_raw.npz"), allow_pickle=False)
    return _FID


def masks_from_points(sample, aux, length=1000):
    """sample: positions already at 100 Hz; aux: label strings."""
    order = np.argsort(sample, kind="stable")
    sample, aux = np.asarray(sample)[order], np.asarray(aux)[order]
    m = np.zeros((4, length), bool)
    qrs_on = sample[aux == "QRS onset"]

    def pair(on_label, off_label, lim, row, start_after_on=False):
        ons = sample[aux == on_label]
        offs = sample[aux == off_label]
        for o in ons:
            nxt_qrs = qrs_on[qrs_on > o]
            stop = nxt_qrs[0] if (len(nxt_qrs) and on_label != "p-wave onset") else length + 10 ** 6
            c = offs[(offs > o) & (offs - o <= lim) & (offs <= stop)]
            if len(c):
                a = o + 1 if start_after_on else o
                m[row, max(0, a):min(length, c[0] + 1)] = True

    pair("p-wave onset", "p-wave offset", LIMIT["P"], 0)
    pair("QRS onset", "QRS offset", LIMIT["QRS"], 1)
    pair("QRS offset", "t-wave offset", LIMIT["STT"], 2, start_after_on=True)
    m[0] &= ~m[1]  # QRS has priority where a P offset overlaps a QRS onset after rounding
    m[2] &= ~m[1]
    m[3] = ~(m[0] | m[1] | m[2])
    return m


def segment_masks(ecg_id):
    f = _load()
    k = "%d_sample" % ecg_id
    if k not in f.files:
        return None
    m = masks_from_points(f[k], f["%d_aux" % ecg_id])
    return m if m[1].any() else None


if __name__ == "__main__":
    # check on the example of record 1 (positions at 500 Hz from the PTB-XL+ file, converted here)
    s = np.array([11, 83, 86, 147, 292, 476, 544, 549, 606, 763])
    a = np.array(["p-wave onset", "p-wave offset", "QRS onset", "QRS offset", "t-wave offset"] * 2)
    mk = masks_from_points(np.round(s / 5).astype(int), a)
    print({n: int(mk[i].sum()) for i, n in enumerate(["P", "QRS", "STT", "other"])})
