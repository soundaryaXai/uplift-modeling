#!/usr/bin/env python3
"""
Uplift modeling pipeline  (PRD build steps 1-5)

    synthetic (known truth)  ->  Hillstrom (real, randomized)  ->  Criteo (real, randomized)

For every dataset it:
  1. produces OUT-OF-FOLD uplift scores from three scikit-uplift estimators
     (S-Learner, Two-Model / T-Learner, Class Transformation) plus a naive
     "will they convert?" model that ignores treatment;
  2. scores each ranking with sklift's Qini AUC, AUUC and Uplift@k, with
     bootstrap 95% intervals;
  3. builds the budget simulation: extra events gained when you contact the
     top-k% ranked by each model vs. random contact;
  4. runs dataset-specific trust checks (randomization balance, known-truth
     recovery on synthetic data, named-feature drivers on Hillstrom).

Usage
-----
  python uplift_pipeline.py --datasets synthetic hillstrom --hillstrom-path data/hillstrom.csv
  python uplift_pipeline.py --datasets criteo --criteo-path criteo-research-uplift-v2.1.csv.gz --sample-frac 0.1

Results are merged into results.json, so datasets can be run separately.
Then:  python build_dashboard.py
"""
import argparse
import hashlib
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

import sklift
from sklift.metrics import qini_auc_score, uplift_auc_score, uplift_at_k
from sklift.models import ClassTransformation, SoloModel, TwoModels

warnings.filterwarnings("ignore")

SEED = 42
GRID = np.arange(0, 101)                       # % of audience contacted
UPLIFT_MODELS = ["S-Learner", "Two-Model", "Class Transformation"]
NAIVE = "Probability ranking"                  # "will they convert?" ignoring treatment
ALL_RANKINGS = UPLIFT_MODELS + [NAIVE]
KS = (10, 20, 30)


# --------------------------------------------------------------------------- #
# Datasets
# --------------------------------------------------------------------------- #
class Dataset:
    def __init__(self, key, label, X, y, t, outcome, unit, description, raw=None, truth=None,
                 named_features=False, source="", checks=None):
        self.key, self.label, self.X, self.y, self.t = key, label, X, y, t
        self.outcome, self.unit, self.description = outcome, unit, description
        self.raw, self.truth, self.named_features, self.source = raw, truth, named_features, source
        self.checks = checks


def load_synthetic(n=50_000, seed=SEED):
    """Hidden segments with a KNOWN true uplift. Treatment is randomized 50/50.

    Persuadable   20%  base 10%  uplift +15 pts   <- the only group worth contacting
    Sleeping Dog  10%  base 25%  uplift  -8 pts   <- contact makes them convert LESS
    Sure Thing    ~22% base 60%  uplift   0
    Lost Cause    ~48% base  3%  uplift   0
    The model only sees noisy proxies of the two variables that define the segments.
    """
    rng = np.random.default_rng(seed)
    z = rng.normal(size=(n, 8))
    persuadable = z[:, 0] > 0.84
    dog = z[:, 0] < -1.28
    rest = ~(persuadable | dog)
    sure = rest & (z[:, 1] > 0.5)
    lost = rest & ~sure
    conds = [persuadable, dog, sure, lost]
    seg = np.select(conds, ["Persuadable", "Sleeping Dog", "Sure Thing", "Lost Cause"], default="Lost Cause")
    base = np.select(conds, [0.10, 0.25, 0.60, 0.03])
    tau = np.select(conds, [0.15, -0.08, 0.0, 0.0])
    t = rng.integers(0, 2, n)
    y = (rng.random(n) < base + tau * t).astype(int)
    obs = z.copy()
    obs[:, :2] += rng.normal(scale=0.35, size=(n, 2))          # measurement noise
    X = pd.DataFrame(obs, columns=[f"x{i}" for i in range(1, 9)])
    truth = pd.DataFrame({"segment": seg, "tau": tau, "base": base})
    return Dataset("synthetic", "Synthetic", X, pd.Series(y), pd.Series(t), "conversion", "conversions",
                   "Simulated customers with a hidden, known uplift", truth=truth,
                   source="Generated in uplift_pipeline.py (seed 42)")


def load_hillstrom(path):
    df = pd.read_csv(path)
    assert len(df) == 64_000, "Unexpected Hillstrom row count"
    df["zip_code"] = df["zip_code"].replace({"Surburban": "Suburban"})     # typo in the source file
    t = (df["segment"] != "No E-Mail").astype(int)              # any e-mail vs none (binary, per PRD)
    y = df["visit"].astype(int)
    X = pd.get_dummies(df[["recency", "history", "mens", "womens", "newbie", "zip_code", "channel"]],
                       columns=["zip_code", "channel"], dtype=int)
    sha = hashlib.sha256(Path(path).read_bytes()).hexdigest().upper()
    arms = df["segment"].value_counts()
    ctrl = df[df["segment"] == "No E-Mail"]
    men, wom = df[df["segment"] == "Mens E-Mail"], df[df["segment"] == "Womens E-Mail"]
    checks = [
        {"label": "File fingerprint (SHA-256)", "published": "0E589332...F291AECE",
         "ours": sha[:8] + "..." + sha[-8:], "ok": sha.startswith("0E589332") and sha.endswith("F291AECE")},
        {"label": "Customers per arm (none / men's / women's)", "published": "21,306 / 21,307 / 21,387",
         "ours": f"{arms['No E-Mail']:,} / {arms['Mens E-Mail']:,} / {arms['Womens E-Mail']:,}",
         "ok": (arms["No E-Mail"], arms["Mens E-Mail"], arms["Womens E-Mail"]) == (21306, 21307, 21387)},
        {"label": "Extra spend per customer, men's e-mail", "published": "$0.77",
         "ours": f"${men['spend'].mean() - ctrl['spend'].mean():.2f}",
         "ok": round(men["spend"].mean() - ctrl["spend"].mean(), 2) == 0.77},
        {"label": "Extra spend per customer, women's e-mail", "published": "$0.42",
         "ours": f"${wom['spend'].mean() - ctrl['spend'].mean():.2f}",
         "ok": round(wom["spend"].mean() - ctrl["spend"].mean(), 2) == 0.42},
        {"label": "Purchase-rate lift, men's e-mail", "published": "+0.68 pts",
         "ours": f"+{(men['conversion'].mean() - ctrl['conversion'].mean()) * 100:.2f} pts",
         "ok": round((men["conversion"].mean() - ctrl["conversion"].mean()) * 100, 2) == 0.68},
    ]
    return Dataset("hillstrom", "Hillstrom", X, y, t, "visit", "site visits",
                   "E-mail campaign test, 64,000 past customers", raw=df, named_features=True,
                   source="MineThatData E-Mail Analytics Challenge (2008)", checks=checks)


CHUNK = 250_000          # small chunks keep peak memory low on laptops


def load_criteo(path=None, fetch=False, target="visit", max_rows=None, sample_frac=None, seed=SEED):
    """Load the official Criteo file (criteo-research-uplift-v2.1.csv.gz, ~14M rows).

    The file is streamed in small chunks and randomly sampled as it is read (`sample_frac`, e.g. 0.1 ->
    ~1.4M rows), so memory stays low and the sample is not biased by row order. The sample is cached
    next to the input file, so a re-run skips the slow read.
    """
    if fetch:
        raise SystemExit(
            "--criteo-fetch no longer works: scikit-uplift's Criteo mirror now returns 403 Forbidden.\n"
            "Download criteo-research-uplift-v2.1.csv.gz (311 MB) from https://ailab.criteo.com/criteo-uplift-prediction-dataset/\n"
            "or https://huggingface.co/datasets/criteo/criteo-uplift, then run with --criteo-path <file> --sample-frac 0.1")
    if not path:
        raise SystemExit("criteo needs --criteo-path FILE")
    feats = [f"f{i}" for i in range(12)]
    # NOTE: `exposure` is measured AFTER treatment, so it must never be used as a feature (not loaded).
    cols = feats + ["treatment", "visit", "conversion"]
    cache = Path(f"{path}.sample{sample_frac}_seed{seed}.pkl") if (sample_frac and not max_rows) else None

    if cache is not None and cache.exists():
        print(f"    using cached sample {cache.name}", flush=True)
        df = pd.read_pickle(cache)
    else:
        dtypes = {f: "float32" for f in feats}
        dtypes.update({"treatment": "int8", "visit": "int8", "conversion": "int8"})
        rng = np.random.default_rng(seed)
        parts, read, kept = [], 0, 0
        print("    reading file in chunks (a few minutes)...", flush=True)
        for i, chunk in enumerate(pd.read_csv(path, usecols=cols, chunksize=CHUNK, dtype=dtypes)):
            read += len(chunk)
            if sample_frac:
                chunk = chunk[rng.random(len(chunk)) < sample_frac]
            parts.append(chunk)
            kept += len(chunk)
            if i % 8 == 0:
                print(f"    read {read:>10,} rows | kept {kept:>9,}", flush=True)
            if max_rows and kept >= max_rows:
                break
        df = pd.concat(parts, ignore_index=True)
        del parts
        if max_rows:
            df = df.iloc[:max_rows]
        if cache is not None:
            df.to_pickle(cache)
            print(f"    saved sample to {cache.name}", flush=True)

    X, y, t = df[feats], df[target].astype("int64"), df["treatment"].astype("int64")
    src = "Criteo Uplift Prediction Dataset v2.1 (Diemert et al.)"
    if sample_frac:
        src += f", {sample_frac:.0%} random sample"
    return Dataset("criteo", "Criteo", X, y, t, target, "visits" if target == "visit" else "conversions",
                   "Ad incrementality test", named_features=False, source=src)


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #
def clf(seed=SEED):
    # Deliberately shallow + regularised: uplift signal is weak, deep trees just memorise noise.
    return HistGradientBoostingClassifier(max_depth=3, learning_rate=0.06, max_iter=150,
                                          min_samples_leaf=200, l2_regularization=1.0, random_state=seed)


def uplift_estimators(seed=SEED):
    return {
        "S-Learner": SoloModel(clf(seed), method="dummy"),
        "Two-Model": TwoModels(clf(seed), clf(seed), method="vanilla"),
        "Class Transformation": ClassTransformation(clf(seed)),
    }


def oof_scores(ds, folds, seed=SEED):
    """5-fold cross-fitting: every row is scored by models that never saw it."""
    X, y, t = ds.X, ds.y.values, ds.t.values
    n = len(y)
    strat = t * 2 + y
    scores = {m: np.zeros(n) for m in ALL_RANKINGS}
    for f, (tr, va) in enumerate(StratifiedKFold(folds, shuffle=True, random_state=seed).split(X, strat)):
        Xtr, Xva = X.iloc[tr], X.iloc[va]
        for name, est in uplift_estimators(seed).items():
            est.fit(Xtr, y[tr], t[tr])
            scores[name][va] = np.asarray(est.predict(Xva)).ravel()
        scores[NAIVE][va] = clf(seed).fit(Xtr, y[tr]).predict_proba(Xva)[:, 1]
        print(f"    fold {f + 1}/{folds} done", flush=True)
    return scores


# --------------------------------------------------------------------------- #
# Curves and metrics
# --------------------------------------------------------------------------- #
def rank_order(score):
    # same tie-breaking as sklift (stable ascending sort, reversed)
    return np.argsort(score, kind="mergesort")[::-1]


def gain_curve(order, y, t, w=None):
    """Extra events if the top-k% (by `order`) were contacted, k = 0..100.
    = (treated rate - control rate) x audience size  ==  sklift's uplift curve."""
    n = len(order)
    yo, to = y[order].astype(float), t[order].astype(float)
    wo = np.ones(n) if w is None else w[order].astype(float)
    nn = np.cumsum(wo)
    nt, nc = np.cumsum(wo * to), np.cumsum(wo * (1 - to))
    yt, yc = np.cumsum(wo * to * yo), np.cumsum(wo * (1 - to) * yo)
    pos = np.clip(np.searchsorted(nn, GRID / 100 * nn[-1] - 1e-9, side="left"), 0, n - 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        rate = np.where(nt[pos] > 0, yt[pos] / nt[pos], 0.0) - np.where(nc[pos] > 0, yc[pos] / nc[pos], 0.0)
    out = rate * nn[pos]
    out[0] = 0.0
    return out


def metric_bundle(y, s, t):
    out = {"qini_auc": qini_auc_score(y, s, t), "auuc": uplift_auc_score(y, s, t)}
    for k in KS:
        out[f"uplift_at_{k}"] = uplift_at_k(y, s, t, "overall", k / 100) * 100      # percentage points
    return out


def r(a, d=4):
    return np.round(np.asarray(a, dtype=float), d).tolist()


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #
def evaluate(ds, scores, boot, seed=SEED):
    y, t, n = ds.y.values, ds.t.values, len(ds.y)
    rng = np.random.default_rng(seed)
    orders = {m: rank_order(scores[m]) for m in ALL_RANKINGS}

    point = {m: metric_bundle(y, scores[m], t) for m in ALL_RANKINGS}
    curves = {m: gain_curve(orders[m], y, t) for m in ALL_RANKINGS}

    # Cross-check my curve against sklift's own uplift_curve (they must agree)
    from sklift.metrics import uplift_curve
    jit = scores["Two-Model"] + np.random.default_rng(seed).normal(scale=1e-12, size=n)   # tie-free copy
    xs, ys = uplift_curve(y, jit, t)
    ks = np.ceil(GRID / 100 * n - 1e-9).astype(int)
    curve_check = float(np.max(np.abs(np.interp(ks[1:], xs, ys) - gain_curve(rank_order(jit), y, t)[1:])))

    ate = y[t == 1].mean() - y[t == 0].mean()
    random_line = ate * n * GRID / 100

    bm = {m: {k: [] for k in point[m]} for m in ALL_RANKINGS}
    bc = {m: [] for m in ALL_RANKINGS}
    print(f"    bootstrap x{boot}", flush=True)
    for b in range(boot):
        idx = rng.integers(0, n, n)
        w = np.bincount(idx, minlength=n)
        yb, tb = y[idx], t[idx]
        for m in ALL_RANKINGS:
            for k, v in metric_bundle(yb, scores[m][idx], tb).items():
                bm[m][k].append(v)
            bc[m].append(gain_curve(orders[m], y, t, w))

    metrics = {}
    for m in ALL_RANKINGS:
        metrics[m] = {}
        for k, v in point[m].items():
            lo, hi = np.percentile(bm[m][k], [2.5, 97.5])
            metrics[m][k] = {"v": float(v), "lo": float(lo), "hi": float(hi)}

    bands = {m: {"lo": r(np.percentile(bc[m], 2.5, axis=0), 2), "hi": r(np.percentile(bc[m], 97.5, axis=0), 2)}
             for m in ALL_RANKINGS}
    delta = {}
    for m in UPLIFT_MODELS:
        d = np.array(bc[m]) - np.array(bc[NAIVE])
        delta[m] = {"lo": r(np.percentile(d, 2.5, axis=0), 2), "hi": r(np.percentile(d, 97.5, axis=0), 2)}

    best = max(UPLIFT_MODELS, key=lambda m: point[m]["qini_auc"])

    # Estimator disagreement (PRD section 9): overlap of the top-20% audiences
    k20 = int(0.2 * n)
    masks = {}
    for m in ALL_RANKINGS:
        mk = np.zeros(n, bool)
        mk[orders[m][:k20]] = True
        masks[m] = mk
    overlap = [[float((masks[a] & masks[b]).sum() / k20) for b in ALL_RANKINGS] for a in ALL_RANKINGS]
    sub = rng.choice(n, min(n, 200_000), replace=False)
    rho = [[float(spearmanr(scores[a][sub], scores[b][sub])[0]) for b in ALL_RANKINGS] for a in ALL_RANKINGS]

    return {
        "orders": orders, "curves": curves, "random": random_line, "ate": ate, "best": best,
        "metrics": metrics, "bands": bands, "delta": delta, "curve_check": curve_check,
        "agreement": {"names": ALL_RANKINGS, "top20_overlap": overlap, "spearman": rho},
    }


def balance_check(ds, seed=SEED):
    """Did randomization actually give comparable groups? (supports the causal assumption)"""
    X, t = ds.X, ds.t.values
    if len(X) > 300_000:
        sub = np.random.default_rng(seed).choice(len(X), 300_000, replace=False)
        X, t = X.iloc[sub], t[sub]
    sd = X.std().replace(0, 1)
    smd = ((X[t == 1].mean() - X[t == 0].mean()) / sd).abs()
    p = cross_val_predict(clf(seed), X, t, cv=3, method="predict_proba")[:, 1]
    return {"max_abs_smd": float(smd.max()), "worst_feature": str(smd.idxmax()),
            "treatment_auc": float(roc_auc_score(t, p)),
            "treated_share": float(np.mean(ds.t.values))}


# --------------------------------------------------------------------------- #
# Dataset-specific analysis
# --------------------------------------------------------------------------- #
def diff_row(feature, level, y, t):
    y1, y0 = y[t == 1], y[t == 0]
    n1, n0 = len(y1), len(y0)
    if n1 < 30 or n0 < 30:
        return None
    p1, p0 = y1.mean(), y0.mean()
    se = np.sqrt(p1 * (1 - p1) / n1 + p0 * (1 - p0) / n0)
    d = (p1 - p0) * 100
    return {"feature": feature, "level": level, "n": int(n1 + n0), "control": float(p0 * 100),
            "treated": float(p1 * 100), "uplift": float(d), "lo": float(d - 196 * se), "hi": float(d + 196 * se)}


def hillstrom_drivers(ds, scores):
    df, y, t = ds.raw, ds.y.values, ds.t.values
    groups = {
        "Months since last purchase": pd.cut(df["recency"], [0, 2, 5, 8, 12],
                                             labels=["1-2", "3-5", "6-8", "9-12"]).astype(str),
        "Past-year spend": pd.qcut(df["history"], 4, labels=False, duplicates="drop").map(
            lambda q: f"Quartile {int(q) + 1}").astype(str),
        "New customer": df["newbie"].map({1: "Yes", 0: "No"}),
        "Bought men's items": df["mens"].map({1: "Yes", 0: "No"}),
        "Bought women's items": df["womens"].map({1: "Yes", 0: "No"}),
        "Area type": df["zip_code"],
        "Purchase channel": df["channel"],
    }
    # label spend quartiles with the actual dollar ranges
    q = pd.qcut(df["history"], 4, duplicates="drop")
    qmap = {f"Quartile {i + 1}": f"${iv.left:,.0f}-{iv.right:,.0f}" for i, iv in enumerate(q.cat.categories)}
    order = {"Months since last purchase": ["1-2", "3-5", "6-8", "9-12"],
             "Past-year spend": [f"Quartile {i}" for i in range(1, 5)],
             "New customer": ["No", "Yes"], "Bought men's items": ["No", "Yes"],
             "Bought women's items": ["No", "Yes"], "Area type": ["Rural", "Suburban", "Urban"],
             "Purchase channel": ["Phone", "Web", "Multichannel"]}
    observed = []
    for feat, col in groups.items():
        for lvl in order[feat]:
            mk = (col == lvl).values
            row = diff_row(feat, qmap.get(lvl, lvl), y[mk], t[mk])
            if row:
                observed.append(row)

    learned = {}
    n = len(y)
    k = int(0.2 * n)
    sd = ds.X.std().replace(0, 1)
    for m in UPLIFT_MODELS:
        o = rank_order(scores[m])
        top, bot = ds.X.iloc[o[:k]], ds.X.iloc[o[-k:]]
        rows = []
        for c in ds.X.columns:
            rows.append({"feature": c, "top": float(top[c].mean()), "bottom": float(bot[c].mean()),
                         "smd": float((top[c].mean() - bot[c].mean()) / sd[c])})
        learned[m] = rows
    return {"observed": observed, "learned": learned, "columns": list(ds.X.columns)}


def synthetic_validation(ds, scores, ev):
    truth = ds.truth
    n = len(truth)
    segs = ["Persuadable", "Sure Thing", "Lost Cause", "Sleeping Dog"]
    seg_rows = []
    for s in segs:
        mk = (truth["segment"] == s).values
        seg_rows.append({
            "name": s, "share": float(mk.mean()), "true_uplift": float(truth["tau"][mk].mean() * 100),
            "base_rate": float(truth["base"][mk].mean() * 100),
            "estimated": {m: float(scores[m][mk].mean() * 100) for m in UPLIFT_MODELS},
        })
    tau = truth["tau"].values
    true_curves, comp = {}, {}
    k20 = int(0.2 * n)
    for m in ALL_RANKINGS:
        o = ev["orders"][m]
        cum = np.concatenate([[0], np.cumsum(tau[o])])
        true_curves[m] = r(cum[np.round(GRID / 100 * n).astype(int)], 2)
        comp[m] = {s: float((truth["segment"].values[o[:k20]] == s).mean()) for s in segs}
    rho = {m: float(spearmanr(scores[m], tau)[0]) for m in UPLIFT_MODELS}
    return {"segments": seg_rows, "true_curves": true_curves, "true_random": r(tau.mean() * n * GRID / 100, 2),
            "top20_composition": comp, "spearman_true": rho}


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def shuffle_rows(ds, seed=SEED):
    """Random (seeded) row order, so ties in model scores are broken randomly, not by file order."""
    perm = np.random.default_rng(seed).permutation(len(ds.y))
    ds.X = ds.X.iloc[perm].reset_index(drop=True)
    ds.y = ds.y.iloc[perm].reset_index(drop=True)
    ds.t = ds.t.iloc[perm].reset_index(drop=True)
    if ds.raw is not None:
        ds.raw = ds.raw.iloc[perm].reset_index(drop=True)
    if ds.truth is not None:
        ds.truth = ds.truth.iloc[perm].reset_index(drop=True)
    return ds


def run(ds, folds=5, boot=None, seed=SEED):
    t0 = time.time()
    ds = shuffle_rows(ds, seed)
    n = len(ds.y)
    boot = boot if boot is not None else (200 if n <= 150_000 else 30)
    print(f"[{ds.key}] n={n:,}  treated={ds.t.mean():.1%}  outcome rate={ds.y.mean():.2%}", flush=True)
    scores = oof_scores(ds, folds, seed)
    ev = evaluate(ds, scores, boot, seed)
    bal = balance_check(ds, seed)

    y, t = ds.y.values, ds.t.values
    out = {
        "key": ds.key, "label": ds.label, "status": "ok", "source": ds.source, "description": ds.description,
        "n": int(n), "treated_share": float(t.mean()), "outcome": ds.outcome, "unit": ds.unit,
        "rates": {"control": float(y[t == 0].mean() * 100), "treated": float(y[t == 1].mean() * 100),
                  "ate_pts": float(ev["ate"] * 100)},
        "total_effect": float(ev["ate"] * n),
        "grid": GRID.tolist(),
        "rankings": ALL_RANKINGS, "uplift_models": UPLIFT_MODELS, "naive": NAIVE,
        "curves": {m: r(ev["curves"][m], 2) for m in ALL_RANKINGS},
        "random": r(ev["random"], 2),
        "bands": ev["bands"], "delta_ci": ev["delta"],
        "metrics": ev["metrics"], "best_model": ev["best"],
        "agreement": ev["agreement"], "balance": bal,
        "named_features": ds.named_features,
        "run": {"folds": folds, "bootstrap": boot, "seed": seed, "sklift": sklift.__version__,
                "curve_check_max_abs_diff": ev["curve_check"], "seconds": round(time.time() - t0, 1),
                "learner": "HistGradientBoostingClassifier(max_depth=3, lr=0.06, 150 iters, min_leaf=200)"},
        "checks": ds.checks,
        "drivers": hillstrom_drivers(ds, scores) if ds.key == "hillstrom" else None,
        "validation": synthetic_validation(ds, scores, ev) if ds.truth is not None else None,
    }
    print(f"[{ds.key}] best={ev['best']}  curve_check={ev['curve_check']:.2e}  ({out['run']['seconds']}s)", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["synthetic", "hillstrom"],
                    choices=["synthetic", "hillstrom", "criteo"])
    ap.add_argument("--hillstrom-path", default="data/hillstrom.csv")
    ap.add_argument("--criteo-path")
    ap.add_argument("--criteo-fetch", action="store_true", help="(dead) sklift mirror now returns 403")
    ap.add_argument("--sample-frac", type=float, help="random fraction of Criteo rows to keep, e.g. 0.1")
    ap.add_argument("--criteo-target", default="visit", choices=["visit", "conversion"])
    ap.add_argument("--max-rows", type=int)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--boot", type=int)
    ap.add_argument("--out", default="results.json")
    a = ap.parse_args()

    out_path = Path(a.out)
    results = json.loads(out_path.read_text()) if out_path.exists() else {"datasets": {}}
    results.setdefault("datasets", {})

    for key in a.datasets:
        if key == "synthetic":
            ds = load_synthetic()
        elif key == "hillstrom":
            ds = load_hillstrom(a.hillstrom_path)
        else:
            ds = load_criteo(a.criteo_path, a.criteo_fetch, a.criteo_target, a.max_rows, a.sample_frac)
        results["datasets"][key] = run(ds, a.folds, a.boot)
        out_path.write_text(json.dumps(results, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
