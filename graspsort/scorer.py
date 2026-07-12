"""
Grasp-success scorer — a tiny pure-numpy MLP, saved as a single .npz.

Deliberately NOT onnx/sklearn: Isaac's bundled python has neither, and the model
is a 17-input / one-hidden-layer net — trivial to evaluate with two matmuls. Keeping
inference dependency-free means the SAME artifact loads in the Nebius Job image, in
Isaac's python for the sort eval, and in a plain numpy REPL.

Artifact (model.npz) contents:
    W1,b1,W2,b2     layer weights (float32)
    feat_mean,feat_std   input standardization (fit on the training set)
    feature_names   the featurizer column order this net was fit on (parity check)
    meta_json       json blob: hidden size, train/val metrics, class balance, notes

Forward:  p(success) = sigmoid( relu((x-mean)/std @ W1 + b1) @ W2 + b2 )
"""
from __future__ import annotations

import json
from typing import Dict, List, Sequence

import numpy as np

from . import features as F


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30.0, 30.0)))


def forward(params: Dict, X: np.ndarray) -> np.ndarray:
    """X (N, n_features) -> p(success) (N,). Standardizes then runs the MLP."""
    X = np.asarray(X, dtype=np.float32)
    Xs = (X - params["feat_mean"]) / params["feat_std"]
    a1 = np.maximum(0.0, Xs @ params["W1"] + params["b1"])       # ReLU
    z2 = a1 @ params["W2"] + params["b2"]
    return sigmoid(z2).reshape(-1)


class GraspScorer:
    """Loads model.npz and scores (part, candidate-grasp) pairs."""

    def __init__(self, params: Dict, meta: Dict):
        self.params = params
        self.meta = meta
        names = list(params.get("feature_names", []))
        if names and names != list(F.FEATURE_NAMES):
            # train/serve featurizer drift — loudly, don't silently mis-score
            raise ValueError("scorer feature order != graspsort.features.FEATURE_NAMES; "
                             "retrain or align features.py")

    @classmethod
    def load(cls, path: str) -> "GraspScorer":
        d = np.load(path, allow_pickle=False)
        params = {k: d[k] for k in ("W1", "b1", "W2", "b2", "feat_mean", "feat_std")}
        params["feature_names"] = [str(s) for s in d["feature_names"]]
        meta = json.loads(str(d["meta_json"])) if "meta_json" in d else {}
        return cls(params, meta)

    def score(self, part: Dict, action: Dict, scene: Dict | None = None) -> float:
        x = F.featurize(part, action, scene)[None, :]
        return float(forward(self.params, x)[0])

    def score_batch(self, part: Dict, actions: Sequence[Dict],
                    scene: Dict | None = None) -> np.ndarray:
        """Score many candidate grasps for the SAME part (+scene) -> (K,) probs."""
        X = np.stack([F.featurize(part, a, scene) for a in actions], axis=0)
        return forward(self.params, X)


def save_model(path: str, W1, b1, W2, b2, feat_mean, feat_std, meta: Dict):
    np.savez(
        path,
        W1=np.asarray(W1, np.float32), b1=np.asarray(b1, np.float32),
        W2=np.asarray(W2, np.float32), b2=np.asarray(b2, np.float32),
        feat_mean=np.asarray(feat_mean, np.float32),
        feat_std=np.asarray(feat_std, np.float32),
        feature_names=np.asarray(list(F.FEATURE_NAMES)),
        meta_json=json.dumps(meta),
    )
