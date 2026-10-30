"""Result figures (grayscale, no overlapping legends). Reads work/*.json and preds pickles."""
import os
import json, numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

W = os.environ.get("ASTRA_WORK", os.path.dirname(os.path.abspath(__file__)))
plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 7, "axes.linewidth": 0.6, "xtick.major.width": 0.6,
                     "ytick.major.width": 0.6, "axes.spines.top": False, "axes.spines.right": False,
                     "legend.frameon": False, "mathtext.fontset": "dejavuserif"})
GR = ["#000000", "#444444", "#777777", "#AAAAAA", "#CCCCCC"]
HATCH = ["", "////", "....", "xxxx", "\\\\\\\\"]
MK = ["o", "s", "^", "D", "v", "P", "X"]
S = json.load(open(f"{W}/summary.json"))
SHORT = {"ASTRA": "ASTRA", "Stacking": "Stacking", "Stacking-aug": "Stacking (aug.)", "TTA-Stacking": "TTA-Stack", "Gated-MoE": "Gated-MoE", "KNORA-E": "KNORA-E",
         "Equal-vote": "Equal vote", "Accuracy-weighted": "Acc.-weighted", "SHAP-variance trust": "SHAP-var. trust",
         "Best-expert": "Best expert", "Concat-LightGBM": "Concat-LGBM", "MLP (concat)": "MLP (concat)"}


def fig_main():
    comb = list(SHORT)
    fig, axes = plt.subplots(2, 2, figsize=(7.16, 3.9))
    for col, ds in enumerate(["botsim", "tweepfake"]):
        M = S[ds]["methods"]
        order = sorted(comb, key=lambda m: -M[m]["macro_f1"][0])
        for row, (met, lab) in enumerate([("macro_f1", "Macro-F1"), ("nll", "NLL (lower is better)")]):
            ax = axes[row, col]
            y = np.arange(len(order))[::-1]
            mu = [M[m][met][0] for m in order]; sd = [M[m][met][1] for m in order]
            ax.errorbar(mu, y, xerr=sd, fmt="none", ecolor="#555555", elinewidth=0.7, capsize=1.8)
            for yi, m, v in zip(y, order, mu):
                ax.plot(v, yi, marker="D" if m == "ASTRA" else "o", ms=4.2 if m == "ASTRA" else 3.2,
                        mfc="black" if m == "ASTRA" else "white", mec="black", mew=0.7)
            ax.set_yticks(y); ax.set_yticklabels([SHORT[m] for m in order], fontsize=6.2)
            ax.set_xlabel(lab, fontsize=7); ax.grid(axis="x", color="#DDDDDD", lw=0.4)
            lo, hi = min(np.array(mu) - np.array(sd)), max(np.array(mu) + np.array(sd))
            pad = 0.08 * (hi - lo); ax.set_xlim(lo - pad, hi + pad)
            ax.set_title(f"{'BotSim-24 (Reddit)' if ds == 'botsim' else 'TweepFake (X)'}", fontsize=7.2)
    fig.tight_layout(w_pad=2.0, h_pad=1.2)
    fig.savefig("fig2_main.png", dpi=300); plt.close(fig)


def fig_h1():
    fig, axes = plt.subplots(2, 1, figsize=(3.45, 3.6))
    for ax, ds in zip(axes, ["botsim", "tweepfake"]):
        H = S[ds]["H1"]; ex = list(H)
        x = np.arange(len(ex)); w = 0.26
        for i, (k, lab) in enumerate([("auc_instab", r"instability $1-s_m$"), ("auc_shift", r"shift $\delta_m$"),
                                       ("auc_lowconf", r"low confidence $-|z_m|$")]):
            v = [H[m][k][0] for m in ex]; e = [H[m][k][1] for m in ex]
            ax.bar(x + (i - 1) * w, v, w, yerr=e, color=["#222222", "#888888", "white"][i], edgecolor="black", lw=0.6,
                   hatch=["", "", "////"][i], error_kw=dict(lw=0.6, capsize=1.5), label=lab)
        ax.axhline(0.5, color="black", lw=0.6, ls=(0, (3, 2)))
        ax.set_xticks(x); ax.set_xticklabels(ex, fontsize=6.4); ax.set_ylim(0.3, 1.0)
        ax.set_ylabel("AUROC for expert error", fontsize=6.8)
        ax.set_title("BotSim-24" if ds == "botsim" else "TweepFake", fontsize=7.2)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=3, fontsize=5.6, bbox_to_anchor=(0.5, 1.0), columnspacing=0.8, handlelength=1.4)
    fig.tight_layout(rect=(0, 0, 1, 0.93), h_pad=1.0)
    fig.savefig("fig3_h1.png", dpi=300); plt.close(fig)


def fig_mod():
    M = S["botsim"]["moderation"]; alphas = [0.01, 0.02, 0.05, 0.1]; xs = np.arange(4)
    meths = ["ASTRA", "ASTRA+esc", "Stacking", "Stacking-aug", "SHAP-variance trust"]
    labs = ["ASTRA", "ASTRA + escalation", "Stacking", "Stacking (aug.)", "SHAP-var. trust"]
    lss = ["-", (0, (3, 1.5)), "-", (0, (3, 1.5)), "-"]
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.2))
    for i, (m, lab) in enumerate(zip(meths, labs)):
        def g(k, j=0): return np.array([M[f"{m}|{a}"][k][j] for a in alphas])
        kw = dict(marker=MK[i], ms=3.2, color=GR[min(i, 3)], lw=0.8, ls=lss[i], mfc="white" if i % 2 else GR[min(i, 3)], label=lab)
        axes[0].errorbar(xs + (i - 2) * 0.05, g("frr"), yerr=g("frr", 1), capsize=1.5, **kw)
        axes[1].plot(xs, g("defer"), **kw)
        axes[2].plot(xs, g("msr"), **kw)
    axes[0].plot(xs, alphas, color="black", lw=0.7, ls=(0, (1, 1.5)), marker="_", ms=8, label=r"tolerance $\alpha$")
    for ax, t in zip(axes, ["wrongful-removal rate", "deferral rate", "missed-inauthentic rate"]):
        ax.set_xlabel(r"tolerance $\alpha$ (with $\gamma=0.10$)", fontsize=6.6); ax.set_ylabel(t, fontsize=6.8)
        ax.set_xticks(xs); ax.set_xticklabels(["0.01", "0.02", "0.05", "0.10"], fontsize=6.2)
        ax.grid(color="#E5E5E5", lw=0.4)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=6, fontsize=6.0, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.86), w_pad=1.6)
    fig.savefig("fig4_moderation.png", dpi=300); plt.close(fig)


def fig_attacks():
    A = json.load(open(f"{W}/results_attacks.json"))
    meths = ["ASTRA", "ASTRA-clean-fit", "Stacking", "Stacking-aug", "Gated-MoE"]
    labs = ["ASTRA", "ASTRA (clean fit)", "Stacking", "Stacking (aug.)", "Gated-MoE"]
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.35), gridspec_kw=dict(width_ratios=[1, 1, 1.35]))
    rates = [0, 0.05, 0.15, 0.30, 0.50]; keys = ["clean", "homo-0.05", "homo-0.15", "homo-0.30", "homo-0.50"]
    lss = ["-", (0, (3, 1.5)), "-", (0, (3, 1.5)), "-"]
    for ax, ds in zip(axes[:2], ["botsim", "tweepfake"]):
        for i, (m, lab) in enumerate(zip(meths, labs)):
            ax.plot(rates, [A[ds][k][m]["auroc"] for k in keys], marker=MK[i], ms=3.2, lw=0.8, color=GR[min(i, 3)],
                    ls=lss[i], mfc="white" if i % 2 else GR[min(i, 3)], label=lab)
        ax.set_xlabel("homoglyph rate", fontsize=6.8); ax.set_ylabel("AUROC (attacked vs. genuine)", fontsize=6.6)
        ax.set_title("BotSim-24" if ds == "botsim" else "TweepFake", fontsize=7.2); ax.grid(color="#E5E5E5", lw=0.4)
        lo = min(A[ds][k][m]["auroc"] for k in keys for m in meths); ax.set_ylim(lo - 0.02, min(1.0, max(A[ds][k][m]["auroc"] for k in keys for m in meths) + 0.01))
    ax = axes[2]; names = list(A["botsim"]["clean"]["_weights"])
    conds = ["clean", "homo-0.30", "leet-0.30", "mimic", "homo-0.15+mimic"]
    cl = ["clean", "homo.\n0.30", "leet\n0.30", "mimic", "mimic\n+ homo."]
    bottom = np.zeros(len(conds)); fills = ["#000000", "#555555", "#888888", "#BBBBBB", "#FFFFFF", "#DDDDDD"]
    hat = ["", "", "", "", "////", "...."]
    for j, n in enumerate(names):
        v = np.array([A["botsim"][c]["_weights"][n] for c in conds])
        ax.bar(np.arange(len(conds)), v, 0.58, bottom=bottom, color=fills[j], edgecolor="black", lw=0.5, hatch=hat[j], label=n)
        bottom += v
    ax.set_xticks(np.arange(len(conds))); ax.set_xticklabels(cl, fontsize=5.5)
    ax.set_ylabel("mean routing weight", fontsize=6.8); ax.set_ylim(0, 1); ax.set_title("ASTRA weights (BotSim-24)", fontsize=7.2)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=6.0, handlelength=1.2)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=5, fontsize=6.3, bbox_to_anchor=(0.42, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.86), w_pad=1.4)
    fig.savefig("fig5_attacks.png", dpi=300); plt.close(fig)


def fig_router():
    P = pd.read_pickle(f"{W}/preds_botsim_seed0.pkl")
    names = P["names"]; Wt = P["W"]; npv = np.expm1(P["nprev"])
    bins = [(-0.5, 0.5, "0"), (0.5, 3.5, "1-3"), (3.5, 15.5, "4-15"), (15.5, 63.5, "16-63"), (63.5, 1e9, "64+")]
    fig, axes = plt.subplots(2, 1, figsize=(3.45, 3.9), gridspec_kw=dict(height_ratios=[1.1, 1]))
    ax = axes[0]
    for j, n in enumerate(names):
        v = [Wt[(npv > lo) & (npv <= hi), j].mean() for lo, hi, _ in bins]
        ax.plot(np.arange(len(bins)), v, marker=MK[j], ms=3.2, lw=0.8, color=GR[min(j, 4)] if j < 5 else "#000000",
                ls="-" if j < 3 else (0, (3, 1.5)), mfc="white" if j % 2 else "black", label=n)
    ax.set_xticks(np.arange(len(bins))); ax.set_xticklabels([b[2] for b in bins], fontsize=6.4)
    ax.set_xlabel("prior events of the account", fontsize=6.8); ax.set_ylabel("mean routing weight", fontsize=6.8)
    ax.grid(color="#E5E5E5", lw=0.4)
    hh, ll = ax.get_legend_handles_labels()
    fig.legend(hh, ll, loc="upper center", bbox_to_anchor=(0.5, 1.0), fontsize=5.8, ncol=3, columnspacing=1.2, handlelength=2.0)
    ax = axes[1]
    share = S["botsim"]["concept_share"]; c = list(share); v = [share[k] for k in c]
    o = np.argsort(v)
    ax.barh(np.arange(len(c)), np.array(v)[o], color="#777777", edgecolor="black", lw=0.5)
    ax.set_yticks(np.arange(len(c))); ax.set_yticklabels(np.array(c)[o], fontsize=6.3)
    ax.set_xlabel("share of absolute concept attribution", fontsize=6.6); ax.grid(axis="x", color="#E5E5E5", lw=0.4)
    fig.tight_layout(rect=(0, 0, 1, 0.91), h_pad=1.2)
    fig.savefig("fig6_router.png", dpi=300); plt.close(fig)


def fig_sens():
    D = pd.DataFrame(json.load(open(f"{W}/results_sensitivity.json")))
    groups = [("$K$", ["K=1", "K=2", "base"], ["1", "2", "4"]),
              ("$k$", ["k=10", "base", "k=200"], ["10", "50", "200"]),
              ("RBO $r$ and $p$", ["r=5", "base", "r=20", "p=0.8", "p=0.95"], ["r=5", "r=10", "r=20", "p=.8", "p=.95"])]
    fig, axes = plt.subplots(1, 3, figsize=(3.45, 1.75), gridspec_kw=dict(width_ratios=[0.8, 0.8, 1.6]), sharey=True)
    lo, hi = D.nll.min(), D.nll.max(); pad = 0.15 * (hi - lo)
    for ax, (t, keys, labs) in zip(axes, groups):
        g = D.groupby("setting").nll
        mu = [g.mean()[k] for k in keys]; mn = [g.min()[k] for k in keys]; mx = [g.max()[k] for k in keys]
        x = np.arange(len(keys))
        ax.errorbar(x, mu, yerr=[np.array(mu) - np.array(mn), np.array(mx) - np.array(mu)], fmt="o-", ms=3.4,
                    color="black", lw=0.8, capsize=1.8, mfc="white")
        ax.set_xticks(x); ax.set_xticklabels(labs, fontsize=5.2, rotation=0 if len(keys) < 4 else 30); ax.set_xlabel(t, fontsize=5.6)
        if ax is axes[0]: ax.set_ylabel("NLL", fontsize=6.2)
        ax.tick_params(axis="y", labelsize=5.2)
        ax.set_ylim(lo - pad, hi + pad); ax.grid(color="#E5E5E5", lw=0.4)
    fig.tight_layout(w_pad=0.6)
    fig.savefig("fig7_sensitivity.png", dpi=300); plt.close(fig)


if __name__ == "__main__":
    import sys
    for f in sys.argv[1:] or ["main", "h1", "mod", "attacks", "router", "sens"]:
        globals()[f"fig_{f}"](); print("fig", f)
