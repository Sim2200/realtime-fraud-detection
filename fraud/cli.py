"""The pipeline, one stage per subcommand. Every stage writes JSON under results/.

    python -m fraud.cli data        download + time split summary       -> results/split.json
    python -m fraud.cli train       4 models x 3 imbalance strategies   -> results/baselines.json, best model artifact
    python -m fraud.cli evaluate    anomaly, clustering, calibration, cost, SHAP, 5-seed stability
    python -m fraud.cli privacy     DP-SGD MLP at several epsilons       -> results/privacy.json
    python -m fraud.cli onnx        ONNX / int8 / Core ML + latency      -> results/efficiency.json
    python -m fraud.cli drift       PSI on features and scores           -> results/drift.json
    python -m fraud.cli all

--sample 0.05 runs every stage on a time-contiguous 5% prefix (CI).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np

from . import data as D
from . import metrics as M

RESULTS = Path("results")
FIG = RESULTS / "figures"
ARTIFACTS = RESULTS / "artifacts"
SEEDS = (0, 1, 2, 3, 4)


def save(name: str, obj: dict) -> None:
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / name).write_text(json.dumps(obj, indent=2, default=float))
    print(f"wrote results/{name}")


def load_json(name: str) -> dict:
    return json.loads((RESULTS / name).read_text())


def get_split(sample: float) -> D.Split:
    df = D.load()
    if sample < 1.0:
        df = D.sample(df, sample)
    return D.time_split(df)


# ---------------------------------------------------------------- stages

def stage_data(a) -> None:
    s = get_split(a.sample)
    save("split.json", {"dataset": "OpenML 1597 creditcard", "sample_fraction": a.sample, "features": D.FEATURES,
                        "split": "time-ordered 70/10/20, no shuffling", **s.summary()})


def stage_train(a) -> None:
    from . import models as Mo

    s = get_split(a.sample)
    rows, best, artifacts = [], None, {}
    for name in Mo.MODELS:
        for strategy in Mo.STRATEGIES:
            t = Mo.train(name, strategy, s.x_train, s.y_train, s.x_val, s.y_val, seed=0, n_jobs=a.jobs)
            val_score, test_score = t.score(s.x_val), t.score(s.x_test)
            thr = M.best_f1_threshold(s.y_val, val_score)  # threshold chosen on validation only
            rep = M.full_report(s.y_test, test_score, threshold=thr)
            row = {"model": name, "strategy": strategy, "fit_seconds": t.fit_seconds, "n_train_rows": t.n_train,
                   "val_pr_auc": round(M.pr_auc(s.y_val, val_score), 4), **rep, **t.extra}
            rows.append(row)
            print(f"{name:14s} {strategy:8s} val PR-AUC {row['val_pr_auc']:.4f}  test PR-AUC {row['pr_auc']:.4f}  "
                  f"R@0.1%FPR {row['recall_at_fpr_0.1pct']:.3f}  fit {t.fit_seconds}s", flush=True)
            artifacts[(name, strategy)] = t
            if best is None or row["val_pr_auc"] > best["val_pr_auc"]:
                best = row
    # Best model is chosen on VALIDATION PR-AUC (test is never used to pick).
    bt = artifacts[(best["model"], best["strategy"])]
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    joblib.dump(bt.model, ARTIFACTS / "best_model.joblib")
    np.save(ARTIFACTS / "best_val_scores.npy", bt.score(s.x_val))
    np.save(ARTIFACTS / "best_test_scores.npy", bt.score(s.x_test))
    save("baselines.json", {"rows": rows, "best": best, "selection": "highest validation PR-AUC",
                            "threshold_rule": "max F1 on validation", "cost_model": M.CostModel().__dict__})


def stage_evaluate(a) -> None:
    import shap
    from sklearn.isotonic import IsotonicRegression
    from . import anomaly as A
    from . import clustering as C
    from . import models as Mo

    s = get_split(a.sample)
    base = load_json("baselines.json")
    best_model = joblib.load(ARTIFACTS / "best_model.joblib")
    val_score, test_score = np.load(ARTIFACTS / "best_val_scores.npy"), np.load(ARTIFACTS / "best_test_scores.npy")
    out: dict = {"best": {"model": base["best"]["model"], "strategy": base["best"]["strategy"]}}

    # -- anomaly detection on legitimate training rows only
    legit = s.x_train[s.y_train == 0]
    iforest, if_secs = A.isolation_forest(legit, seed=0)
    ae = A.AutoEncoderDetector(seed=0, epochs=a.ae_epochs).fit(legit)
    out["anomaly"] = {
        "isolation_forest": {"fit_seconds": if_secs, **M.full_report(s.y_test, A.isolation_scores(iforest, s.x_test))},
        "autoencoder": {"fit_seconds": ae.fit_seconds, "epochs": a.ae_epochs, "final_train_mse": round(ae.history[-1], 5),
                        **M.full_report(s.y_test, ae.score(s.x_test))},
        "best_supervised_for_reference": {"pr_auc": base["best"]["pr_auc"], "recall_at_fpr_0.1pct": base["best"]["recall_at_fpr_0.1pct"]},
    }
    print("anomaly:", {k: v["pr_auc"] for k, v in out["anomaly"].items()})

    # -- clustering on the autoencoder embedding
    z_train, z_val, z_test = ae.embed(s.x_train), ae.embed(s.x_val), ae.embed(s.x_test)
    km = C.fit_kmeans(z_train, k=a.clusters, seed=0)
    out["clustering"] = {"k": a.clusters, "embedding": "autoencoder bottleneck (8-d)",
                         "train": C.cluster_report(km, z_train, s.y_train), "test": C.cluster_report(km, z_test, s.y_test)}
    # does cluster id help the best model?
    name, strategy = base["best"]["model"], base["best"]["strategy"]
    aug = lambda x, z: np.hstack([x, C.one_hot_cluster(km, z)])  # noqa: E731
    t_aug = Mo.train(name, strategy, aug(s.x_train, z_train), s.y_train, aug(s.x_val, z_val), s.y_val, seed=0, n_jobs=a.jobs)
    out["clustering"]["as_feature"] = {
        "without": {"val_pr_auc": base["best"]["val_pr_auc"], "test_pr_auc": base["best"]["pr_auc"]},
        "with_cluster_onehot": {"val_pr_auc": round(M.pr_auc(s.y_val, t_aug.score(aug(s.x_val, z_val))), 4),
                                "test_pr_auc": round(M.pr_auc(s.y_test, t_aug.score(aug(s.x_test, z_test))), 4)},
    }
    print("clustering top:", out["clustering"]["test"][:3], "| as feature:", out["clustering"]["as_feature"])

    # -- calibration (isotonic on validation), evaluated on test
    iso = IsotonicRegression(out_of_bounds="clip").fit(val_score, s.y_val)
    cal_test = iso.predict(test_score)
    out["calibration"] = {"method": "isotonic, fit on validation",
                          "brier_raw": round(M.brier(s.y_test, test_score), 6), "brier_calibrated": round(M.brier(s.y_test, cal_test), 6),
                          "pr_auc_after": round(M.pr_auc(s.y_test, cal_test), 4),
                          "reliability_raw": M.reliability(s.y_test, test_score), "reliability_calibrated": M.reliability(s.y_test, cal_test)}
    joblib.dump(iso, ARTIFACTS / "calibrator.joblib")

    # -- cost-based threshold: chosen on validation, reported on test
    cost = M.CostModel()
    cv = M.expected_cost_curve(s.y_val, cal_test if False else iso.predict(val_score), cost)
    thr_cost = cv["threshold"]
    out["cost"] = {"cost_model": cost.__dict__, "threshold_from_validation": round(thr_cost, 6),
                   "test_at_cost_threshold": M.precision_recall_f1(s.y_test, cal_test, thr_cost),
                   "test_curve": M.expected_cost_curve(s.y_test, cal_test, cost)["curve"][::8],
                   "test_cost_per_10k_at_threshold": round(next(r["cost_per_10k"] for r in M.expected_cost_curve(s.y_test, cal_test, cost, 400)["curve"]
                                                              if r["threshold"] >= thr_cost), 2)}
    n_test = len(s.y_test)
    out["cost"]["flag_nothing_cost_per_10k"] = round(int(s.y_test.sum()) * cost.missed_fraud_cost / n_test * 10_000, 2)
    out["cost"]["flag_everything_cost_per_10k"] = round(cost.review_cost * 10_000, 2)

    # -- SHAP for the best model (tree explainer on a test sample)
    idx = np.random.default_rng(0).choice(len(s.x_test), size=min(2000, len(s.x_test)), replace=False)
    explainer = shap.TreeExplainer(best_model) if name in ("xgboost", "lightgbm", "random_forest") else shap.LinearExplainer(best_model[-1], s.x_train[:5000])
    sv = explainer.shap_values(s.x_test[idx])
    sv = sv[1] if isinstance(sv, list) else (sv[..., 1] if sv.ndim == 3 else sv)
    mean_abs = np.abs(sv).mean(axis=0)
    order = np.argsort(-mean_abs)
    out["shap"] = {"sample_rows": int(len(idx)), "top_features": [{"feature": D.FEATURES[i], "mean_abs_shap": round(float(mean_abs[i]), 4)} for i in order[:12]]}
    _plot_shap(sv, s.x_test[idx], order[:15])

    # -- stability over 5 seeds for the best (model, strategy)
    prs = []
    for seed in SEEDS:
        t = Mo.train(name, strategy, s.x_train, s.y_train, s.x_val, s.y_val, seed=seed, n_jobs=a.jobs)
        prs.append(M.pr_auc(s.y_test, t.score(s.x_test)))
    out["stability"] = {"seeds": list(SEEDS), "test_pr_auc": [round(p, 4) for p in prs],
                        "mean": round(float(np.mean(prs)), 4), "std": round(float(np.std(prs)), 4)}
    print("stability:", out["stability"])
    _plot_pr_curves(s, base, test_score, A.isolation_scores(iforest, s.x_test), ae.score(s.x_test))
    _plot_cost(out["cost"]["test_curve"], thr_cost)
    _plot_reliability(out["calibration"])
    save("evaluation.json", out)


def stage_privacy(a) -> None:
    from . import privacy as P

    s = get_split(a.sample)
    rows = []
    for eps in [None, 8.0, 3.0, 1.0, 0.5]:
        clf = P.MLPClassifier(epsilon=eps, epochs=a.dp_epochs, seed=0).fit(s.x_train, s.y_train)
        score = clf.score(s.x_test)
        rows.append({"epsilon_target": eps if eps is not None else "inf", "epsilon_spent": clf.epsilon_spent,
                     "delta": clf.delta, "noise_multiplier": clf.noise_multiplier, "fit_seconds": clf.fit_seconds,
                     **M.full_report(s.y_test, score)})
        print(f"epsilon {rows[-1]['epsilon_target']}: PR-AUC {rows[-1]['pr_auc']:.4f}  R@1%FPR {rows[-1]['recall_at_fpr_1pct']:.3f}", flush=True)
        if eps is None:
            joblib.dump({"scaler": clf.scaler, "state_dict": clf.plain_module().state_dict()}, ARTIFACTS / "mlp_plain.joblib")
    save("privacy.json", {"mechanism": "DP-SGD (Opacus, RDP accountant), per-example clipping at 1.0, delta = 1/n_train",
                          "model": "MLP 30-64-32-1, class-weighted BCE, Adam", "epochs": a.dp_epochs, "rows": rows})
    _plot_privacy(rows)


def stage_onnx(a) -> None:
    import platform
    import torch
    from . import export as E
    from . import privacy as P

    s = get_split(a.sample)
    base = load_json("baselines.json")
    best_model = joblib.load(ARTIFACTS / "best_model.joblib")
    n = s.x_test.shape[1]
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    budget_ms = 10.0
    rows = []

    tree_path = E.export_tree_onnx(best_model, n, ARTIFACTS / "best_tree.onnx")
    rows.append({"artifact": f"{base['best']['model']} ONNX fp32", **E.benchmark_onnx(tree_path, s.x_test),
                 "pr_auc": round(M.pr_auc(s.y_test, E.onnx_scores(tree_path, s.x_test)), 4)})

    mlp = joblib.load(ARTIFACTS / "mlp_plain.joblib")
    module = P.MLP(n)
    module.load_state_dict(mlp["state_dict"])
    module.eval()
    mlp_path = E.export_mlp_onnx(module, mlp["scaler"], n, ARTIFACTS / "mlp_fp32.onnx")
    rows.append({"artifact": "MLP ONNX fp32", **E.benchmark_onnx(mlp_path, s.x_test),
                 "pr_auc": round(M.pr_auc(s.y_test, E.onnx_scores(mlp_path, s.x_test)), 4)})
    q_path = E.quantize_onnx_int8(mlp_path, ARTIFACTS / "mlp_int8.onnx")
    rows.append({"artifact": "MLP ONNX int8 (dynamic)", **E.benchmark_onnx(q_path, s.x_test),
                 "pr_auc": round(M.pr_auc(s.y_test, E.onnx_scores(q_path, s.x_test)), 4)})

    if platform.system() == "Darwin" and not a.skip_coreml:
        cm = E.export_coreml(module, mlp["scaler"], n, ARTIFACTS / "mlp_coreml")
        if cm is not None:
            sub = s.x_test[np.random.default_rng(0).choice(len(s.x_test), size=min(3000, len(s.x_test)), replace=False)]
            sub_y = None
            for cu in ("CPU_ONLY", "ALL"):
                rows.append({"artifact": f"MLP Core ML ({cu})", **E.benchmark_coreml(cm, s.x_test, cu)})
            # accuracy check on a subsample (per-row predict is slow)
            idx = np.random.default_rng(1).choice(len(s.x_test), size=min(20000, len(s.x_test)), replace=False)
            rows[-2]["pr_auc_on_20k_sample"] = round(M.pr_auc(s.y_test[idx], E.coreml_scores(cm, s.x_test[idx])), 4)
            rows[-2]["onnx_fp32_pr_auc_same_sample"] = round(M.pr_auc(s.y_test[idx], E.onnx_scores(mlp_path, s.x_test[idx])), 4)
    for r in rows:
        r["meets_budget_p95_10ms"] = bool(r["p95_ms"] <= budget_ms)
    save("efficiency.json", {"hardware": E.hardware(), "threads": 1, "budget": "p95 <= 10 ms per single transaction",
                             "torch": torch.__version__, "rows": rows})
    for r in rows:
        print(f"{r['artifact']:28s} {r['size_kb']:9.1f} KB  p50 {r['p50_ms']:.3f} ms  p95 {r['p95_ms']:.3f} ms  "
              f"PR-AUC {r.get('pr_auc', r.get('pr_auc_on_20k_sample', float('nan')))}")


def stage_drift(a) -> None:
    from . import drift as Dr

    s = get_split(a.sample)
    test_score = np.load(ARTIFACTS / "best_test_scores.npy")
    val_score = np.load(ARTIFACTS / "best_val_scores.npy")
    best_model = joblib.load(ARTIFACTS / "best_model.joblib")
    train_score = best_model.predict_proba(s.x_train)[:, 1]
    # later windows: validation, first half of test, second half of test
    half = len(s.x_test) // 2
    windows = {"validation": (s.x_val, val_score), "test_first_half": (s.x_test[:half], test_score[:half]),
               "test_second_half": (s.x_test[half:], test_score[half:])}
    out = {"reference": "training window", "bins": "train deciles", "thresholds": {"stable": "<0.1", "moderate": "0.1-0.25", "significant": ">0.25"}}
    for w, (x, sc) in windows.items():
        feats = Dr.feature_drift(s.x_train, x, D.FEATURES)
        out[w] = {"score_psi": round(Dr.psi(train_score, sc), 4), "score_status": Dr.label(Dr.psi(train_score, sc)),
                  "features_flagged": [f for f in feats if f["status"] != "stable"], "top_features": feats[:8]}
        print(w, "score PSI", out[w]["score_psi"], "flagged:", [f["feature"] for f in out[w]["features_flagged"]])
    save("drift.json", out)


# ---------------------------------------------------------------- figures

def _style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "savefig.facecolor": "#fcfcfb",
                         "font.family": "sans-serif", "axes.grid": True, "grid.color": "#e1e0d9", "axes.spines.top": False,
                         "axes.spines.right": False, "axes.edgecolor": "#c3c2b7", "text.color": "#0b0b0b",
                         "axes.labelcolor": "#52514e", "xtick.color": "#898781", "ytick.color": "#898781",
                         "legend.frameon": False, "axes.titleweight": "bold"})
    FIG.mkdir(parents=True, exist_ok=True)
    return plt


COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#4a3aa7"]


def _plot_pr_curves(s, base, best_score, if_score, ae_score):
    from sklearn.metrics import precision_recall_curve
    plt = _style()
    fig, ax = plt.subplots(figsize=(6.5, 5))
    for (label, sc), c in zip([(f"{base['best']['model']} ({base['best']['strategy']})", best_score),
                               ("Isolation Forest", if_score), ("Autoencoder", ae_score)], COLORS):
        p, r, _ = precision_recall_curve(s.y_test, sc)
        ax.plot(r, p, color=c, lw=2, label=f"{label}: PR-AUC {M.pr_auc(s.y_test, sc):.3f}")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision"); ax.set_title("Precision-recall on the test window", loc="left")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=1)
    fig.savefig(FIG / "pr_curves.png", dpi=150, bbox_inches="tight"); plt.close(fig)


def _plot_cost(curve, thr):
    plt = _style()
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.plot([c["threshold"] for c in curve], [c["cost_per_10k"] for c in curve], color=COLORS[0], lw=2)
    ax.axvline(thr, color=COLORS[1], lw=1.2, ls=(0, (4, 3)))
    ax.annotate(f"chosen on validation: {thr:.3f}", (thr, max(c["cost_per_10k"] for c in curve)), xytext=(6, -12),
                textcoords="offset points", fontsize=8, color="#52514e")
    ax.set_xlabel("Score threshold"); ax.set_ylabel("Expected cost per 10,000 transactions ($)")
    ax.set_title("Cost-based threshold (test window)", loc="left")
    fig.savefig(FIG / "cost_curve.png", dpi=150, bbox_inches="tight"); plt.close(fig)


def _plot_reliability(cal):
    plt = _style()
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], color="#c3c2b7", lw=1, ls=(0, (4, 3)))
    for key, label, c in (("reliability_raw", f"raw (Brier {cal['brier_raw']:.5f})", COLORS[1]),
                          ("reliability_calibrated", f"isotonic (Brier {cal['brier_calibrated']:.5f})", COLORS[0])):
        pts = cal[key]
        ax.plot([p["mean_pred"] for p in pts], [p["observed"] for p in pts], marker="o", ms=5, lw=2, color=c, label=label)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Mean predicted probability (log)"); ax.set_ylabel("Observed fraud rate (log)")
    ax.set_title("Reliability (quantile bins)", loc="left"); ax.legend(loc="upper left")
    fig.savefig(FIG / "reliability.png", dpi=150, bbox_inches="tight"); plt.close(fig)


def _plot_privacy(rows):
    plt = _style()
    fig, ax = plt.subplots(figsize=(6.5, 4))
    xs = [r["epsilon_spent"] if r["epsilon_spent"] else 60 for r in rows]
    ax.plot(xs, [r["pr_auc"] for r in rows], marker="o", ms=6, lw=2, color=COLORS[0])
    for x, r in zip(xs, rows):
        ax.annotate("no DP" if r["epsilon_target"] == "inf" else f"ε={r['epsilon_spent']:.1f}", (x, r["pr_auc"]), xytext=(6, 4),
                    textcoords="offset points", fontsize=8, color="#52514e")
    ax.set_xscale("log"); ax.set_xlabel("Privacy budget ε (log; right-most point is no DP)"); ax.set_ylabel("Test PR-AUC")
    ax.set_title("Privacy / utility trade-off (DP-SGD MLP)", loc="left")
    fig.savefig(FIG / "privacy_tradeoff.png", dpi=150, bbox_inches="tight"); plt.close(fig)


def _plot_shap(sv, x, order):
    plt = _style()
    fig, ax = plt.subplots(figsize=(6.5, 5))
    names = [D.FEATURES[i] for i in order][::-1]
    vals = np.abs(sv).mean(axis=0)[order][::-1]
    ax.barh(names, vals, color=COLORS[0])
    ax.set_xlabel("mean |SHAP value|"); ax.set_title("Feature importance of the best model (SHAP)", loc="left")
    ax.grid(axis="y", visible=False)
    fig.savefig(FIG / "shap_summary.png", dpi=150, bbox_inches="tight"); plt.close(fig)


# ---------------------------------------------------------------- main

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["data", "train", "evaluate", "privacy", "onnx", "drift", "all"])
    ap.add_argument("--sample", type=float, default=1.0)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--ae-epochs", type=int, default=20)
    ap.add_argument("--dp-epochs", type=int, default=8)
    ap.add_argument("--clusters", type=int, default=12)
    ap.add_argument("--skip-coreml", action="store_true")
    a = ap.parse_args()
    stages = {"data": stage_data, "train": stage_train, "evaluate": stage_evaluate, "privacy": stage_privacy,
              "onnx": stage_onnx, "drift": stage_drift}
    for name in (list(stages) if a.stage == "all" else [a.stage]):
        t0 = time.perf_counter()
        print(f"== {name}", flush=True)
        stages[name](a)
        print(f"== {name} done in {time.perf_counter() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
