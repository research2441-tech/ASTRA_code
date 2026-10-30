"""Aggregates run results into the numbers used by the manuscript (work/summary.json)."""
import os
import json, numpy as np
from scipy.stats import wilcoxon, friedmanchisquare, rankdata

W = os.environ.get("ASTRA_WORK", os.path.dirname(os.path.abspath(__file__)))
METRICS = ["macro_f1", "pr_auc", "roc_auc", "mcc", "bal_acc", "brier", "nll", "ece"]
LOWER = {"brier", "nll", "ece"}


def holm(ps):
    ps = np.asarray(ps, float); o = np.argsort(ps); G = len(ps); adj = np.empty(G); run = 0
    for i, j in enumerate(o):
        run = max(run, min(1.0, (G - i) * ps[j])); adj[j] = run
    return adj


def rank_biserial(diff):
    d = np.asarray(diff); d = d[d != 0]
    if len(d) == 0:
        return 0.0
    r = rankdata(np.abs(d)); wp, wm = r[d > 0].sum(), r[d < 0].sum()
    return float((wp - wm) / (wp + wm))


def boot_ci(diff, B=10000, seed=0):
    rng = np.random.default_rng(seed); d = np.asarray(diff)
    m = rng.choice(d, (B, len(d)), replace=True).mean(1)
    return [float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))]


def summarize(runs, ref="ASTRA", methods=None):
    methods = methods or [k for k in runs[0] if not k.startswith("_")]
    out = {"methods": {}, "tests": {}}
    for m in methods:
        out["methods"][m] = {k: [float(np.mean([r[m][k] for r in runs])), float(np.std([r[m][k] for r in runs], ddof=1))]
                             for k in METRICS}
    for k in ("macro_f1", "pr_auc", "nll", "ece", "mcc"):
        ps, rows = [], []
        for m in methods:
            if m == ref:
                continue
            a = np.array([r[ref][k] for r in runs]); b = np.array([r[m][k] for r in runs])
            diff = a - b if k not in LOWER else b - a  # positive = ASTRA better
            p = wilcoxon(a, b).pvalue if np.any(a != b) else 1.0
            ps.append(p); rows.append([m, float(diff.mean()), boot_ci(diff), rank_biserial(diff), float(p),
                                       int((diff > 0).sum())])
        adj = holm(ps)
        out["tests"][k] = [dict(method=r[0], mean_diff=r[1], ci=r[2], r_rb=r[3], p=r[4], p_holm=float(a), wins=r[5])
                           for r, a in zip(rows, adj)]
    return out


def main():
    S = {}
    for ds in ("botsim", "tweepfake"):
        try:
            runs = json.load(open(f"{W}/results_{ds}.json"))
        except FileNotFoundError:
            continue
        S[ds] = summarize(runs)
        S[ds]["n_runs"] = len(runs)
        S[ds]["router"] = {k: [float(np.mean([r["_router_params"][k] for r in runs])), float(np.std([r["_router_params"][k] for r in runs]))]
                           for k in runs[0]["_router_params"]}
        S[ds]["mean_weights"] = {k: float(np.mean([r["_mean_weights"][k] for r in runs])) for k in runs[0]["_mean_weights"]}
        S[ds]["shapvar_weights"] = {k: float(np.mean([r["_shapvar_weights"][k] for r in runs])) for k in runs[0]["_shapvar_weights"]}
        S[ds]["stack_coef"] = {k: float(np.mean([r["_stack_coef"][k] for r in runs])) for k in runs[0]["_stack_coef"]}
        S[ds]["H1"] = {m: {k: [float(np.mean([r["_H1"][m][k] for r in runs if m in r["_H1"]])),
                               float(np.std([r["_H1"][m][k] for r in runs if m in r["_H1"]]))]
                           for k in ("auc_instab", "auc_shift", "auc_lowconf", "err_rate")} for m in runs[0]["_H1"]}
        S[ds]["concept_stab"] = {m: [float(np.mean([r["_concept_stab"][m][i] for r in runs])) for i in (0, 1)]
                                 for m in runs[0]["_concept_stab"]}
        S[ds]["concept_share"] = {c: float(np.mean([r["_concept_share"][c] for r in runs])) for c in runs[0]["_concept_share"]}
        if "_moderation" in runs[0]:
            S[ds]["moderation"] = {k: {kk: [float(np.nanmean([r["_moderation"][k][kk] for r in runs])),
                                            float(np.nanstd([r["_moderation"][k][kk] for r in runs]))]
                                       for kk in runs[0]["_moderation"][k]} for k in runs[0]["_moderation"]}
            # paired test of deferral (efficiency of risk control) ASTRA vs Stacking at alpha = 0.02
            for a in ("0.01", "0.02", "0.05", "0.1", "0.01|g0.02", "0.02|g0.02"):
                x = np.array([r["_moderation"][f"ASTRA|{a}"]["defer"] + r["_moderation"][f"ASTRA|{a}"]["msr"] for r in runs])
                y = np.array([r["_moderation"][f"Stacking|{a}"]["defer"] + r["_moderation"][f"Stacking|{a}"]["msr"] for r in runs])
                S[ds]["moderation"][f"_test_defer_plus_msr|{a}"] = dict(p=float(wilcoxon(x, y).pvalue) if np.any(x != y) else 1.0,
                                                                         mean_diff=float((x - y).mean()))
        # Friedman over the combiners
        comb = ["ASTRA", "Stacking", "Stacking-aug", "TTA-Stacking", "Gated-MoE", "KNORA-E", "Equal-vote", "Accuracy-weighted",
                "SHAP-variance trust", "Best-expert", "Concat-LightGBM", "MLP (concat)"]
        X = np.array([[r[m]["nll"] for m in comb] for r in runs])
        S[ds]["friedman_nll"] = dict(stat=float(friedmanchisquare(*X.T).statistic), p=float(friedmanchisquare(*X.T).pvalue),
                                     mean_rank={m: float(v) for m, v in zip(comb, rankdata(X, axis=1).mean(0))})
        X = np.array([[-r[m]["macro_f1"] for m in comb] for r in runs])
        S[ds]["friedman_f1"] = dict(stat=float(friedmanchisquare(*X.T).statistic), p=float(friedmanchisquare(*X.T).pvalue),
                                    mean_rank={m: float(v) for m, v in zip(comb, rankdata(X, axis=1).mean(0))})
    json.dump(S, open(f"{W}/summary.json", "w"), indent=1)
    return S


if __name__ == "__main__":
    S = main()
    for ds, s in S.items():
        print("=====", ds, s["n_runs"])
        for m, v in sorted(s["methods"].items(), key=lambda kv: kv[1]["nll"][0]):
            print(f"{m:22s} F1 {v['macro_f1'][0]:.4f}±{v['macro_f1'][1]:.4f} PR {v['pr_auc'][0]:.4f} AUC {v['roc_auc'][0]:.4f} "
                  f"MCC {v['mcc'][0]:.4f} NLL {v['nll'][0]:.4f} ECE {v['ece'][0]:.4f} Brier {v['brier'][0]:.4f}")
        for t in s["tests"]["nll"]:
            print("  nll vs", t["method"], round(t["mean_diff"], 4), round(t["p_holm"], 4), t["wins"])
        for t in s["tests"]["macro_f1"]:
            print("  f1 vs", t["method"], round(t["mean_diff"], 4), round(t["p_holm"], 4), t["wins"])
        print("router", {k: round(v[0], 3) for k, v in s["router"].items()})
        print("weights", {k: round(v, 3) for k, v in s["mean_weights"].items()})
        print("H1", {m: {k: round(v[0], 3) for k, v in d.items()} for m, d in s["H1"].items()})
        print("concept stab", s["concept_stab"]); print("share", s["concept_share"])
        if "moderation" in s:
            for k, v in s["moderation"].items():
                if "|0.02" in k or "|0.05" in k:
                    print(" ", k, {kk: round(vv[0], 4) for kk, vv in v.items()} if not k.startswith("_") else v)
