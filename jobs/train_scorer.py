#!/usr/bin/env python3
"""
Train the state -> grasp-success scorer (the "train milestone" from the HANDOFF).

Pure tabular ML, NO Isaac: reads the labeled grasp-attempt dataset produced by
jobs/gen_data.py, featurizes each (part, candidate-grasp) via graspsort.features,
and fits a tiny numpy MLP (graspsort.scorer) that predicts P(the grasp holds).

Deliverables written to $GS_MODEL_OUT (default ./data/model):
  model.npz                 the trained scorer (loads in eval_sort / the Job image)
  training_history.json     per-epoch loss/acc/AUC + the dataset-size learning curve
  plots/*.png               loss curve, AUC/accuracy curve, learning curve, breakdown
  training_progress.mp4     the metrics curves animated epoch-by-epoch (the "cycles")

Config (env vars):
  GS_DATASET     dataset file or dir (default data/dataset ; uses <dir>/records.jsonl)
  GS_MODEL_OUT   output dir                          (default data/model)
  GS_EPOCHS      training epochs                     (default 300)
  GS_HIDDEN      hidden units                        (default 24)
  GS_LR          Adam learning rate                  (default 0.01)
  GS_BATCH       minibatch size                      (default 32)
  GS_VAL_FRAC    validation fraction                 (default 0.25)
  GS_SEED        RNG seed                            (default 0)
  GS_TRAIN_VIDEO 1 render the progress mp4 (default), 0 skip
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from graspsort import features as F
from graspsort import scorer as S
from graspsort.videoio import to_h264

DATASET = os.environ.get("GS_DATASET", os.path.join("data", "dataset"))
OUT = os.environ.get("GS_MODEL_OUT", os.path.join("data", "model"))
EPOCHS = int(os.environ.get("GS_EPOCHS", "300"))
HIDDEN = int(os.environ.get("GS_HIDDEN", "24"))
LR = float(os.environ.get("GS_LR", "0.01"))
BATCH = int(os.environ.get("GS_BATCH", "32"))
VAL_FRAC = float(os.environ.get("GS_VAL_FRAC", "0.25"))
SEED = int(os.environ.get("GS_SEED", "0"))
TRAIN_VIDEO = os.environ.get("GS_TRAIN_VIDEO", "1") != "0"


# ── metrics ────────────────────────────────────────────────────────────────
def roc_auc(y, p) -> float:
    """Mann-Whitney AUC with average ranks (handles ties). No sklearn."""
    y = np.asarray(y).astype(int)
    p = np.asarray(p, float)
    n1 = int(y.sum())
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    order = np.argsort(p, kind="mergesort")
    ps = p[order]
    ranks = np.empty(len(p), float)
    i, r = 0, 1
    while i < len(p):
        j = i
        while j + 1 < len(p) and ps[j + 1] == ps[i]:
            j += 1
        ranks[order[i:j + 1]] = (r + (r + (j - i))) / 2.0
        r += (j - i + 1)
        i = j + 1
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def bce(y, p) -> float:
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


# ── data ────────────────────────────────────────────────────────────────────
def load_dataset(path):
    jl = path if path.endswith(".jsonl") else os.path.join(path, "records.jsonl")
    X, y, meta = [], [], []
    with open(jl) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            oc = r.get("outcome", {})
            if oc.get("success") is None:
                continue
            X.append(F.featurize_record(r))
            y.append(1.0 if oc.get("success") else 0.0)
            pt = r.get("obs", {}).get("part", {})
            meta.append((pt.get("kind"), pt.get("pose"), pt.get("size")))
    return np.asarray(X, np.float32), np.asarray(y, np.float32), meta, jl


def stratified_split(y, val_frac, seed):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(y))
    val = []
    for cls in (0.0, 1.0):
        c = idx[y == cls]
        rng.shuffle(c)
        k = max(1, int(round(len(c) * val_frac))) if len(c) > 1 else 0
        val.extend(c[:k].tolist())
    val = np.asarray(sorted(val))
    train = np.asarray(sorted(set(idx.tolist()) - set(val.tolist())))
    return train, val


# ── model (numpy MLP + Adam, class-balanced BCE) ────────────────────────────
def init_params(n_in, hidden, seed):
    rng = np.random.default_rng(seed)
    return {
        "W1": (rng.standard_normal((n_in, hidden)) * np.sqrt(2.0 / n_in)).astype(np.float32),
        "b1": np.zeros(hidden, np.float32),
        "W2": (rng.standard_normal((hidden, 1)) * np.sqrt(1.0 / hidden)).astype(np.float32),
        "b2": np.zeros(1, np.float32),
    }


def fit(Xtr, ytr, Xva, yva, feat_mean, feat_std, epochs, hidden, lr, batch, seed,
        record=True):
    """Train the MLP; return (params, history)."""
    p = init_params(Xtr.shape[1], hidden, seed)
    for k in p:
        p[k] = p[k]  # keep float32
    p["feat_mean"], p["feat_std"] = feat_mean, feat_std
    # class-balanced sample weights (positives ~80% -> upweight the rare failures)
    n_pos = max(1.0, float(ytr.sum()))
    n_neg = max(1.0, float(len(ytr) - ytr.sum()))
    w_pos = len(ytr) / (2.0 * n_pos)
    w_neg = len(ytr) / (2.0 * n_neg)
    sw = np.where(ytr > 0.5, w_pos, w_neg).astype(np.float32)

    adam = {k: (np.zeros_like(p[k]), np.zeros_like(p[k])) for k in ("W1", "b1", "W2", "b2")}
    b1a, b2a, eps = 0.9, 0.999, 1e-8
    rng = np.random.default_rng(seed + 1)
    hist = {"epoch": [], "train_loss": [], "val_loss": [],
            "train_acc": [], "val_acc": [], "val_auc": []}

    def standardize(X):
        return (X - feat_mean) / feat_std

    Xtr_s = standardize(Xtr)
    Xva_s = standardize(Xva) if len(Xva) else Xva
    n = len(Xtr_s)
    t = 0
    for ep in range(epochs):
        perm = rng.permutation(n)
        for s in range(0, n, batch):
            bi = perm[s:s + batch]
            xb, yb, wb = Xtr_s[bi], ytr[bi], sw[bi]
            # forward
            z1 = xb @ p["W1"] + p["b1"]
            a1 = np.maximum(0.0, z1)
            z2 = a1 @ p["W2"] + p["b2"]
            pr = S.sigmoid(z2).reshape(-1)
            # weighted BCE grad wrt z2
            g2 = (wb * (pr - yb)).reshape(-1, 1) / len(bi)      # (m,1)
            gW2 = a1.T @ g2
            gb2 = g2.sum(axis=0)
            da1 = g2 @ p["W2"].T
            dz1 = da1 * (z1 > 0)
            gW1 = xb.T @ dz1
            gb1 = dz1.sum(axis=0)
            grads = {"W1": gW1, "b1": gb1, "W2": gW2, "b2": gb2}
            t += 1
            for k in grads:
                m, v = adam[k]
                m = b1a * m + (1 - b1a) * grads[k]
                v = b2a * v + (1 - b2a) * (grads[k] ** 2)
                adam[k] = (m, v)
                mh = m / (1 - b1a ** t)
                vh = v / (1 - b2a ** t)
                p[k] = p[k] - lr * mh / (np.sqrt(vh) + eps)
        if record:
            ptr = S.forward(p, Xtr)
            pva = S.forward(p, Xva) if len(Xva) else np.array([])
            hist["epoch"].append(ep + 1)
            hist["train_loss"].append(bce(ytr, ptr))
            hist["train_acc"].append(float(np.mean((ptr > 0.5) == (ytr > 0.5))))
            if len(Xva):
                hist["val_loss"].append(bce(yva, pva))
                hist["val_acc"].append(float(np.mean((pva > 0.5) == (yva > 0.5))))
                hist["val_auc"].append(roc_auc(yva, pva))
            else:
                hist["val_loss"].append(float("nan"))
                hist["val_acc"].append(float("nan"))
                hist["val_auc"].append(float("nan"))
    return p, hist


# ── plots + progress video ──────────────────────────────────────────────────
def make_plots(hist, lc, breakdown, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    pd = os.path.join(out_dir, "plots")
    os.makedirs(pd, exist_ok=True)
    ep = hist["epoch"]

    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(ep, hist["train_loss"], label="train")
    ax.plot(ep, hist["val_loss"], label="val")
    ax.set_xlabel("epoch"); ax.set_ylabel("BCE loss"); ax.set_title("Scorer training loss")
    ax.legend(); ax.grid(alpha=0.3); fig.tight_layout()
    fig.savefig(os.path.join(pd, "loss.png"), dpi=110); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(ep, hist["val_auc"], label="val ROC-AUC", color="C2")
    ax.plot(ep, hist["val_acc"], label="val accuracy", color="C3", alpha=0.7)
    ax.set_xlabel("epoch"); ax.set_ylabel("metric"); ax.set_ylim(0, 1)
    ax.set_title("Scorer validation quality"); ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(pd, "auc.png"), dpi=110); plt.close(fig)

    if lc:
        fig, ax = plt.subplots(figsize=(7, 4.2))
        ax.plot(lc["n"], lc["val_auc"], marker="o")
        ax.set_xlabel("# training grasp attempts"); ax.set_ylabel("val ROC-AUC")
        ax.set_ylim(0.4, 1.0); ax.set_title("Learning curve — more grasp cycles -> better scorer")
        ax.grid(alpha=0.3); fig.tight_layout()
        fig.savefig(os.path.join(pd, "learning_curve.png"), dpi=110); plt.close(fig)

    if breakdown:
        labels = [f"{k}\n{po}" for (k, po) in breakdown["keys"]]
        fig, ax = plt.subplots(figsize=(8, 4.2))
        ax.bar(range(len(labels)), breakdown["rate"], color="C0")
        ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylabel("grasp success rate"); ax.set_ylim(0, 1)
        ax.set_title("Dataset success rate by part kind x pose (n above bars)")
        for i, (r, c) in enumerate(zip(breakdown["rate"], breakdown["count"])):
            ax.text(i, r + 0.02, str(c), ha="center", fontsize=8)
        fig.tight_layout(); fig.savefig(os.path.join(pd, "success_breakdown.png"), dpi=110)
        plt.close(fig)
    return pd


def make_progress_video(hist, out_path, fps=20, max_frames=150):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import cv2
    ep = hist["epoch"]
    n = len(ep)
    if n == 0:
        return None
    step = max(1, n // max_frames)
    frames = []
    for e in range(step, n + 1, step):
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4))
        a1.plot(ep[:e], hist["train_loss"][:e], label="train")
        a1.plot(ep[:e], hist["val_loss"][:e], label="val")
        a1.set_xlim(1, n); a1.set_ylim(0, max(hist["train_loss"] + hist["val_loss"]) * 1.05 + 1e-6)
        a1.set_xlabel("epoch"); a1.set_ylabel("BCE loss"); a1.set_title("training loss")
        a1.legend(loc="upper right"); a1.grid(alpha=0.3)
        a2.plot(ep[:e], hist["val_auc"][:e], color="C2", label="val ROC-AUC")
        a2.plot(ep[:e], hist["val_acc"][:e], color="C3", alpha=0.7, label="val acc")
        a2.set_xlim(1, n); a2.set_ylim(0, 1)
        a2.set_xlabel("epoch"); a2.set_title("validation quality"); a2.legend(loc="lower right")
        a2.grid(alpha=0.3)
        cur_auc = hist["val_auc"][e - 1]
        fig.suptitle(f"grasp-success scorer — epoch {ep[e-1]}/{n}   "
                     f"val AUC={cur_auc:.3f}", fontsize=12)
        fig.tight_layout()
        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
        frames.append(cv2.cvtColor(buf, cv2.COLOR_RGB2BGR).copy())
        plt.close(fig)
    if not frames:
        return None
    h, w = frames[0].shape[:2]
    vw = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for fr in frames:                          # hold last frame a beat
        vw.write(fr)
    for _ in range(fps):
        vw.write(frames[-1])
    vw.release()
    return to_h264(out_path)


def main():
    os.makedirs(OUT, exist_ok=True)
    X, y, meta, jl = load_dataset(DATASET)
    print(f"[train] loaded {len(y)} records from {jl}", flush=True)
    print(f"[train] class balance: success={int(y.sum())} fail={int(len(y)-y.sum())} "
          f"({100*y.mean():.1f}% positive)", flush=True)

    # dataset success breakdown by (kind,pose) — the physics realism the sim teaches
    bd_keys, bd_rate, bd_cnt = [], [], []
    for k in sorted(set((m[0], m[1]) for m in meta)):
        mask = [i for i, m in enumerate(meta) if (m[0], m[1]) == k]
        if mask:
            bd_keys.append(k); bd_cnt.append(len(mask))
            bd_rate.append(float(np.mean([y[i] for i in mask])))
    breakdown = {"keys": bd_keys, "rate": bd_rate, "count": bd_cnt}
    for k, r, c in zip(bd_keys, bd_rate, bd_cnt):
        print(f"[train]   {k[0]:4} {str(k[1]):9}: {100*r:5.1f}% success (n={c})", flush=True)

    tr, va = stratified_split(y, VAL_FRAC, SEED)
    feat_mean = X[tr].mean(axis=0)
    feat_std = X[tr].std(axis=0)
    feat_std = np.where(feat_std > 1e-6, feat_std, 1.0).astype(np.float32)
    feat_mean = feat_mean.astype(np.float32)

    params, hist = fit(X[tr], y[tr], X[va], y[va], feat_mean, feat_std,
                       EPOCHS, HIDDEN, LR, BATCH, SEED)
    final_auc = hist["val_auc"][-1]
    final_acc = hist["val_acc"][-1]
    print(f"[train] DONE fit: val AUC={final_auc:.3f} val acc={final_acc:.3f} "
          f"(train n={len(tr)}, val n={len(va)})", flush=True)

    # dataset-size learning curve (retrain on growing subsets of the TRAIN split)
    lc = {"n": [], "val_auc": []}
    rng = np.random.default_rng(SEED + 7)
    tr_sh = tr.copy(); rng.shuffle(tr_sh)
    for frac in (0.15, 0.3, 0.5, 0.7, 1.0):
        m = max(BATCH * 2, int(round(len(tr_sh) * frac)))
        m = min(m, len(tr_sh))
        sub = tr_sh[:m]
        _p, _h = fit(X[sub], y[sub], X[va], y[va], feat_mean, feat_std,
                     max(120, EPOCHS // 2), HIDDEN, LR, BATCH, SEED, record=True)
        lc["n"].append(int(m)); lc["val_auc"].append(_h["val_auc"][-1])
        print(f"[train]   learning-curve n={m}: val AUC={_h['val_auc'][-1]:.3f}", flush=True)

    # save the model (trained on the full train split) + history + plots
    meta_blob = {
        "hidden": HIDDEN, "epochs": EPOCHS, "lr": LR, "batch": BATCH,
        "seed": SEED, "n_train": int(len(tr)), "n_val": int(len(va)),
        "val_auc": final_auc, "val_acc": final_acc,
        "positive_rate": float(y.mean()), "n_total": int(len(y)),
        "feature_names": list(F.FEATURE_NAMES),
        "note": "P(grasp holds); higher is a better candidate grasp",
    }
    model_path = os.path.join(OUT, "model.npz")
    S.save_model(model_path, params["W1"], params["b1"], params["W2"], params["b2"],
                 feat_mean, feat_std, meta_blob)
    with open(os.path.join(OUT, "training_history.json"), "w") as f:
        json.dump({"history": hist, "learning_curve": lc, "meta": meta_blob,
                   "breakdown": {"keys": [list(k) for k in bd_keys],
                                 "rate": bd_rate, "count": bd_cnt}}, f, indent=2)
    pd = make_plots(hist, lc, breakdown, OUT)
    print(f"[train] wrote {model_path} + plots in {pd}", flush=True)

    if TRAIN_VIDEO:
        vp = make_progress_video(hist, os.path.join(OUT, "training_progress.mp4"))
        if vp:
            print(f"[train] wrote progress video {vp}", flush=True)

    # sanity: the scorer should rank the heuristic (dyaw=0, offset 0) ABOVE a wild grasp
    sc = S.GraspScorer(params, meta_blob)
    good = sc.score({"kind": "nut", "size": "m12", "pose": "flat"},
                    {"grasp_yaw": 0.0, "heuristic_yaw": 0.0, "xy_offset": [0, 0],
                     "grasp_dz": 0.0, "approach_dh": 0.12})
    bad = sc.score({"kind": "nut", "size": "m12", "pose": "flat"},
                   {"grasp_yaw": 1.2, "heuristic_yaw": 0.0, "xy_offset": [0.02, 0.02],
                    "grasp_dz": 0.004, "approach_dh": 0.12})
    print(f"[train] sanity: P(heuristic-like)={good:.3f}  P(wild-offset)={bad:.3f} "
          f"-> {'OK (heuristic higher)' if good > bad else 'WARN (wild higher)'}", flush=True)


if __name__ == "__main__":
    main()
