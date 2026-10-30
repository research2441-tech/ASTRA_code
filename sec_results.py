"""Front matter, Section V (results), VI (discussion), VII (limitations), VIII (conclusion), back matter.
Every number is read from the result files at build time."""
import os
import json, numpy as np, pandas as pd
from scipy.special import expit

W = os.environ.get("ASTRA_WORK", os.path.dirname(os.path.abspath(__file__)))
S = json.load(open(f"{W}/summary.json"))
A = json.load(open(f"{W}/results_attacks.json"))
L = json.load(open(f"{W}/results_logo.json"))
T = json.load(open(f"{W}/results_transfer.json"))
import os
PREVIEW = not (os.path.exists(f"{W}/results_latency.json") and os.path.exists(f"{W}/results_sensitivity.json"))
SE = pd.DataFrame(json.load(open(f"{W}/results_sensitivity.json"))) if os.path.exists(f"{W}/results_sensitivity.json") \
    else pd.DataFrame([dict(setting="base", nll=float("nan"), macro_f1=float("nan"))])
LAT = json.load(open(f"{W}/results_latency.json")) if os.path.exists(f"{W}/results_latency.json") else \
    {k: float("nan") for k in ["perturbation_ms", "gpt2_orig_ms", "gpt2_pert_ms", "experts_router_full_ms", "astra_total_ms",
                               "stacking_total_ms"]}
B, TF = S["botsim"], S["tweepfake"]


def f(x, d=3):
    return f"{x:.{d}f}"


def ms(ds, m, k, d=3):
    v = S[ds]["methods"][m][k]
    return f"{v[0]:.{d}f}±{v[1]:.{d}f}"


def mv(ds, m, k):
    return S[ds]["methods"][m][k][0]


def test(ds, k, m):
    for t in S[ds]["tests"][k]:
        if t["method"] == m:
            return t
    return None


def att(ds, cond, m, k="auroc"):
    return A[ds][cond][m][k]


LOGO = pd.DataFrame([dict(fam=r["_family"], m=m, auc=r[m]["roc_auc"], f1=r[m]["macro_f1"]) for r in L
                     for m in r if not m.startswith("_")])
lg = LOGO.groupby(["m", "fam"]).auc.mean().unstack()
FWD = T["forward"]; REV = T["reverse"]


def tmean(tgt, m, k):
    return float(np.mean([r[tgt][m][k] for r in FWD if k in r[tgt][m]]))


def rmean(m, k):
    return float(np.mean([r[m][k] for r in REV]))


MAST_FRR = float(np.mean([r["chirper+mastodon"]["ASTRA"]["frr"] for r in FWD]))
CHIRP_REC = float(np.mean([r["chirper+mastodon"]["ASTRA"]["recall_removed"] for r in FWD]))
TF_AUC = float(np.mean([r["tweepfake"]["ASTRA"]["roc_auc"] for r in FWD]))
TF_FRR = float(np.mean([r["tweepfake"]["ASTRA"]["frr"] for r in FWD]))
TF_REC = float(np.mean([r["tweepfake"]["ASTRA"]["recall_removed"] for r in FWD]))
GUAR_HELD = MAST_FRR <= 0.02 and TF_FRR <= 0.02


MODK = B["moderation"]


def mod(m, a, k, g=None):
    key = f"{m}|{a}" + (f"|g{g}" if g else "")
    return MODK[key][k][0]


# ======================================================================== front matter
TITLE = ("Attribution-Stability Trust Routing with Stress-Augmented Fitting and Conformal Moderation-Risk Control "
         "for Detecting LLM-Driven Inauthentic Content")


def front(d, R):
    d.title(TITLE, "[First Author], [Second Author], [Third Author], and [Corresponding Author]",
            "[Department, Institution, City, Country]  |  Corresponding author: [email]  |  ORCID: [0000-0000-0000-0000]")
    d.abstract(
        "Inauthentic accounts operated by large language models write fluent posts, and detectors that combine text, "
        "behaviour and interaction evidence must decide how much to trust each source on each post, explain every removal, "
        "and keep wrongful removals of genuine users rare. This paper tests whether the stability of exact feature "
        "attributions under cheap meaning-preserving perturbations can serve as that per-post trust signal. ASTRA routes "
        "six experts (lexical, semantic, language-model authorship, stylistic, behavioural and interaction) with weights "
        "driven by local reliability, attribution stability and prediction shift, fits the router on clean and perturbed "
        "posts, decomposes each decision exactly into seven policy concepts, and turns probabilities into remove, defer "
        "or keep actions with class-conditional conformal thresholds. On account-disjoint partitions of BotSim-24, whose "
        f"inauthentic accounts are GPT-4o-mini agents, ASTRA reached macro-F1 {mv('botsim', 'ASTRA', 'macro_f1'):.3f} and "
        f"PR-AUC {mv('botsim', 'ASTRA', 'pr_auc'):.3f}, far above global SHAP-variance trust "
        f"({mv('botsim', 'SHAP-variance trust', 'macro_f1'):.3f}), but stacking was slightly better calibrated. Attribution "
        f"instability predicted expert errors with AUROC of only {min(v['auc_instab'][0] for v in B['H1'].values()):.2f} to "
        f"{max(v['auc_instab'][0] for v in B['H1'].values()):.2f}, and routing weights barely moved under attack. Fitting "
        "on perturbed posts did matter: under behavioural mimicry with homoglyphs, AUROC rose from "
        f"{att('botsim', 'homo-0.15+mimic', 'ASTRA-clean-fit'):.3f} to {att('botsim', 'homo-0.15+mimic', 'ASTRA'):.3f}. "
        f"The conformal rule kept wrongful removals at {100 * mod('ASTRA', '0.02', 'frr'):.1f}% for a 2% tolerance in "
        + ("distribution and on unseen platforms, but there it removed few inauthentic posts because ranking quality "
           f"collapsed (AUROC {TF_AUC:.2f} on X)." if GUAR_HELD else
           "distribution but exceeded it after a change of platform."),
        "Bot detection; content moderation; conformal risk control; explanation stability; LLM-generated text; "
        "mixture of experts; SHAP")


# ======================================================================== results
MAIN_METHODS = ["ASTRA", "Stacking", "Stacking-aug", "TTA-Stacking", "Gated-MoE", "KNORA-E", "Equal-vote",
                "Accuracy-weighted", "SHAP-variance trust", "Best-expert", "Concat-LightGBM", "MLP (concat)"]
NICE = {"ASTRA": "ASTRA", "Stacking": "Stacking", "Stacking-aug": "Stacking, stress-augmented", "TTA-Stacking": "TTA-Stacking",
        "Gated-MoE": "Gated-MoE", "KNORA-E": "KNORA-E", "Equal-vote": "Equal vote", "Accuracy-weighted": "Accuracy-weighted",
        "SHAP-variance trust": "SHAP-variance trust", "Best-expert": "Best single expert (LEX)",
        "Concat-LightGBM": "LightGBM on all views", "MLP (concat)": "MLP on all views"}


def mark(ds, m, k):
    t = test(ds, k, m)
    if t is None or t["p_holm"] >= 0.05:
        return ""
    return " *" if t["mean_diff"] > 0 else " †"


def results(d, R):
    d.h1("RESULTS")
    d.h2("Main Benchmark Comparison")
    rows = []
    for m in MAIN_METHODS:
        rows.append([NICE[m]] + [ms("botsim", m, k) + mark("botsim", m, k) for k in ("macro_f1", "pr_auc", "mcc", "nll", "ece")]
                    + [ms("tweepfake", m, k) + mark("tweepfake", m, k) for k in ("macro_f1", "pr_auc", "nll")])
    d.table("Account-disjoint comparison of combiners using the same six experts (mean ± SD over ten partitions)",
            [[("", 1), ("BotSim-24 (Reddit, GPT-4o-mini agents)", 5), ("TweepFake (X)", 3)],
             ["Method", "Macro-F1", "PR-AUC", "MCC", "NLL", "ECE", "Macro-F1", "PR-AUC", "NLL"]],
            rows, widths=[1.9, 1, 1, 1, 1, 1, 1, 1, 1], label="tab:main", full=True, bold_rows=(0,), size=13,
            header_rows=2,
            note="* ASTRA significantly better than the baseline on that metric; † ASTRA significantly worse (two-sided "
                 "Wilcoxon signed-rank test over the ten paired runs, Holm-adjusted $p<0.05$). Higher is better for macro-F1, "
                 "PR-AUC and MCC; lower is better for NLL and ECE. TweepFake uses the four text experts only.")
    sa, ss = mv("botsim", "ASTRA", "nll"), mv("botsim", "Stacking", "nll")
    d.p(f"{{ref:tab:main}} and {{ref:fig:main}} compare ASTRA with eleven alternatives built on the same experts. On BotSim-24 the "
        f"strong combiners are separated by less than one point of macro-F1. ASTRA reached {ms('botsim', 'ASTRA', 'macro_f1')} "
        f"macro-F1 and {ms('botsim', 'ASTRA', 'pr_auc')} PR-AUC; stacking reached {ms('botsim', 'Stacking', 'macro_f1')} and "
        f"{ms('botsim', 'Stacking', 'pr_auc')}, and its negative log-likelihood was lower ({f(ss, 4)} against {f(sa, 4)}). "
        f"The difference in log-likelihood had the same sign in all ten partitions and survives the Holm correction, so the "
        "honest reading is that on in-distribution data the router is slightly worse calibrated than an unregularised "
        "logistic meta-learner. Section V-B traces this cost to stress-augmented fitting.")
    d.p(f"The gaps to the weaker combiners are large and consistent. Relative to equal voting, ASTRA lowered the NLL from "
        f"{f(mv('botsim', 'Equal-vote', 'nll'), 3)} to {f(sa, 3)} and the ECE from {f(mv('botsim', 'Equal-vote', 'ece'), 3)} to "
        f"{f(mv('botsim', 'ASTRA', 'ece'), 3)}. Relative to the global SHAP-variance trust score, the combination rule that "
        f"motivated this study, macro-F1 rose from {f(mv('botsim', 'SHAP-variance trust', 'macro_f1'))} to "
        f"{f(mv('botsim', 'ASTRA', 'macro_f1'))} and the NLL fell by a factor of "
        f"{mv('botsim', 'SHAP-variance trust', 'nll') / sa:.1f}. The inverse-variance rule put an average weight of "
        f"{B['shapvar_weights']['LEX']:.2f} on the lexical expert, because its top attributions are many small, evenly sized "
        "n-gram contributions, and it therefore ignored the behavioural and interaction evidence almost entirely. Dynamic "
        "selection by KNORA-E and the two monolithic models trained on all views were also significantly worse than ASTRA "
        f"on every BotSim-24 metric in {{ref:tab:main}}. The best single expert, LEX, reached {f(mv('botsim', 'Best-expert', 'macro_f1'))} "
        "macro-F1, so combining views removed roughly two thirds of its errors.")
    d.p(f"TweepFake is much harder at the account level. Every method stayed between {f(min(mv('tweepfake', m, 'macro_f1') for m in MAIN_METHODS), 2)} "
        f"and {f(max(mv('tweepfake', m, 'macro_f1') for m in MAIN_METHODS), 2)} macro-F1, the standard deviations across "
        f"folds were about {f(np.mean([S['tweepfake']['methods'][m]['macro_f1'][1] for m in MAIN_METHODS]), 2)}, and no difference "
        "between ASTRA and the strong combiners was significant. With only 40 accounts, each fold tests on eight of them, "
        "so the variation reflects which accounts are held out more than which combiner is used.")
    d.figure("fig2_main.png", "Macro-F1 and negative log-likelihood of the combiners (mean ± SD over ten account-disjoint runs). "
             "ASTRA is drawn as a filled diamond. The horizontal axes differ between panels.", label="fig:main", full=True)

    d.h2("The Stability Hypothesis and Ablation")
    h1 = B["H1"]; th = TF["H1"]
    d.p("Hypothesis H1 states that a low attribution stability marks posts on which an expert is wrong. {ref:fig:h1} tests "
        "it directly. On BotSim-24 the AUROC of $1-s_m$ for predicting the expert's own error ranged from "
        f"{min(v['auc_instab'][0] for v in h1.values()):.2f} to {max(v['auc_instab'][0] for v in h1.values()):.2f} across the "
        "six experts, and the prediction shift did no better "
        f"({min(v['auc_shift'][0] for v in h1.values()):.2f} to {max(v['auc_shift'][0] for v in h1.values()):.2f}). The "
        "expert's own confidence was far more informative, with AUROC between "
        f"{min(v['auc_lowconf'][0] for v in h1.values()):.2f} and {max(v['auc_lowconf'][0] for v in h1.values()):.2f}. "
        f"TweepFake showed the same ordering (instability {min(v['auc_instab'][0] for v in th.values()):.2f} to "
        f"{max(v['auc_instab'][0] for v in th.values()):.2f}, confidence {min(v['auc_lowconf'][0] for v in th.values()):.2f} "
        f"to {max(v['auc_lowconf'][0] for v in th.values()):.2f}). H1 is therefore not supported in the form tested here: "
        "the perturbation bank changes the reasons of every expert on almost every post, correct or not, and what changes "
        "little is the ranking of features that already dominate when the expert is confident.")
    d.figure("fig3_h1.png", "Does an unstable explanation flag an expert error? AUROC of three per-post signals for predicting "
             "whether each expert misclassifies a test post (mean ± SD over ten runs). The dashed line marks chance.",
             label="fig:h1", full=False)
    rp = B["router"]
    abl = [("ASTRA (full)", "ASTRA"), ("without stability $s_m$", "ASTRA-no-s"), ("without shift $\\delta_m$", "ASTRA-no-d"),
           ("without reliability $\\rho_m$", "ASTRA-no-rho"), ("reliability only", "ASTRA-rho-only"),
           ("static weights (no signals)", "Static-learned"), ("clean-fit router (no augmentation)", "ASTRA-clean-fit"),
           ("Stacking", "Stacking"), ("Stacking, stress-augmented", "Stacking-aug")]
    rows = []
    for name, m in abl:
        def a(cond):
            return f(att("botsim", cond, m)) if m in A["botsim"][cond] else "n/a"
        rows.append([name, ms("botsim", m, "nll"), ms("botsim", m, "macro_f1"), ms("tweepfake", m, "nll"),
                     a("mimic"), a("homo-0.15+mimic"), a("leet-0.30"), f(lg.loc[m, "gpt2"])])
    d.table("Ablation of the router and of stress-augmented fitting",
            [[("", 1), ("Clean data, 10 runs", 3), ("BotSim-24 attacks, AUROC", 3), ("Held-out", 1)],
             ["Variant", "BotSim NLL", "BotSim F1", "TweepFake NLL", "mimicry", "mimicry + homoglyph", "leetspeak 0.30", "GPT-2 bots"]],
            rows, widths=[2.0, 1.05, 1.05, 1.05, 0.9, 1.05, 0.95, 0.9], label="tab:ablation", full=True, size=13, header_rows=2,
            note="Attack columns use partition 0 with 600 attacked inauthentic posts against all genuine test posts; leetspeak "
                 "is not part of the perturbation bank. The held-out column is the mean AUROC over three TweepFake runs in which "
                 "all GPT-2 bot accounts were excluded from training. n/a: variant not run under attack.")
    d.p(f"{{ref:tab:ablation}} isolates the components. The fitted router coefficients were small: averaged over the ten "
        f"BotSim-24 partitions, $a={rp['a'][0]:.2f}$ for reliability, $b={rp['b'][0]:.2f}$ for stability and "
        f"$c={rp['c'][0]:.2f}$ for shift, per standard deviation of each signal, while the static biases ranged from "
        f"{f(min(v[0] for k, v in rp.items() if k.startswith('beta')), 2)} "
        f"to {f(max(v[0] for k, v in rp.items() if k.startswith('beta')), 2)}. The weights therefore varied little from post "
        f"to post, and removing any single signal, or all of them, changed the BotSim-24 NLL by at most "
        f"{max(abs(mv('botsim', m, 'nll') - sa) for m in ['ASTRA-no-s', 'ASTRA-no-d', 'ASTRA-no-rho', 'ASTRA-rho-only', 'Static-learned']):.4f}. "
        "None of these differences was significant after correction. The per-post signals, in short, did not earn their "
        "weight on clean data.")
    d.p("Stress-augmented fitting is the component that mattered. Fitting the same router on clean posts only gave a better "
        f"clean NLL ({f(mv('botsim', 'ASTRA-clean-fit', 'nll'), 4)} against {f(sa, 4)}), which is the calibration cost noted "
        "above, but under behavioural mimicry it lost "
        f"{att('botsim', 'mimic', 'ASTRA') - att('botsim', 'mimic', 'ASTRA-clean-fit'):.3f} AUROC and under mimicry combined "
        f"with homoglyphs it lost {att('botsim', 'homo-0.15+mimic', 'ASTRA') - att('botsim', 'homo-0.15+mimic', 'ASTRA-clean-fit'):.3f}. "
        "The same augmentation applied to stacking produced the same benefit "
        f"({f(att('botsim', 'homo-0.15+mimic', 'Stacking'))} to {f(att('botsim', 'homo-0.15+mimic', 'Stacking-aug'))} on the "
        "combined attack), which confirms that the gain belongs to the fitting set and not to routing. The static-weight "
        f"variant, fitted on the same augmented set, matched ASTRA within {abs(att('botsim', 'homo-0.15+mimic', 'ASTRA') - att('botsim', 'homo-0.15+mimic', 'Static-learned')):.3f} AUROC on every attack.")

    d.h2("Robustness to Evasion")
    d.p("{ref:fig:attacks} and {ref:tab:ablation} report the evasion experiments. Character-level attacks barely moved the "
        "ranking quality on BotSim-24, because the behavioural and interaction experts are untouched: at a homoglyph rate of "
        f"0.5 every strong combiner kept AUROC above {min(att('botsim', 'homo-0.50', m) for m in ['ASTRA', 'Stacking', 'Stacking-aug', 'Gated-MoE']):.3f}. "
        "They did lower the share of attacked posts that crossed the removal threshold calibrated at $\\alpha=0.02$, from "
        f"{100 * att('botsim', 'clean', 'ASTRA', 'removed'):.1f}% to {100 * att('botsim', 'homo-0.50', 'ASTRA', 'removed'):.1f}% for ASTRA and from "
        f"{100 * att('botsim', 'clean', 'Stacking', 'removed'):.1f}% to {100 * att('botsim', 'homo-0.50', 'Stacking', 'removed'):.1f}% for stacking. "
        "The held-out leetspeak family behaved like the homoglyphs. On TweepFake, where only text is available, the same "
        f"attacks hurt more: under leetspeak at 0.3 the AUROC of ASTRA fell from {f(att('tweepfake', 'clean', 'ASTRA'))} to "
        f"{f(att('tweepfake', 'leet-0.30', 'ASTRA'))}, whereas both stacking variants were unaffected or even improved "
        f"(stress-augmented stacking moved from {f(att('tweepfake', 'clean', 'Stacking-aug'))} to "
        f"{f(att('tweepfake', 'leet-0.30', 'Stacking-aug'))}). Because the attacks touch only inauthentic posts, a visible "
        "character trace can itself become evidence for a combiner that weights the character-level expert heavily, so a "
        "rising AUROC under attack should not be read as robustness in general.")
    d.figure("fig5_attacks.png", "Evasion on partition 0. Left and centre: AUROC of attacked inauthentic posts against genuine "
             "test posts as the homoglyph rate grows. Right: mean ASTRA routing weight per expert on BotSim-24 under four "
             "conditions; the weights hardly change.", label="fig:attacks", full=True)
    d.p("Behavioural mimicry, in which each attacked bot inherits the history and reply footprint of a genuine test account, is "
        "the decisive case. Stacking fitted on clean posts dropped to "
        f"{f(att('botsim', 'mimic', 'Stacking'))} AUROC and the input-gated mixture to {f(att('botsim', 'mimic', 'Gated-MoE'))}, "
        f"whereas ASTRA kept {f(att('botsim', 'mimic', 'ASTRA'))} and removed {100 * att('botsim', 'mimic', 'ASTRA', 'removed'):.0f}% "
        f"of the attacked posts against {100 * att('botsim', 'mimic', 'Stacking', 'removed'):.0f}% for clean stacking. When mimicry "
        f"was combined with homoglyphs, ASTRA reached {f(att('botsim', 'homo-0.15+mimic', 'ASTRA'))} and clean stacking "
        f"{f(att('botsim', 'homo-0.15+mimic', 'Stacking'))}. The global SHAP-variance weighting was nearly immune to mimicry "
        f"({f(att('botsim', 'mimic', 'SHAP-variance trust'))}) for the wrong reason: it never used the context views in the first "
        "place. The right panel of {ref:fig:attacks} explains why routing did not help: the mean weights of the six experts "
        "were within two points of their clean values under every attack, so the router never shifted trust towards the views "
        "that the attack had left intact.")

    d.h2("Generalisation to Unseen Generators and Platforms")
    fam_rows = []
    for name, m in [("ASTRA", "ASTRA"), ("Static weights", "Static-learned"), ("Stacking", "Stacking"),
                    ("Stacking, stress-augmented", "Stacking-aug"), ("Gated-MoE", "Gated-MoE"), ("Equal vote", "Equal-vote"),
                    ("LightGBM on all views", "Concat-LightGBM"), ("SHAP-variance trust", "SHAP-variance trust"),
                    ("STYLE expert alone", "Expert:STYLE"), ("LEX expert alone", "Expert:LEX")]:
        fam_rows.append([name] + [f(lg.loc[m, fam]) for fam in ("gpt2", "rnn", "others")])
    d.table("AUROC on TweepFake when one generator family is held out of training (mean of three runs)",
            [["Method", "GPT-2", "RNN", "Other"]], fam_rows, widths=[2.2, 0.9, 0.9, 0.9], label="tab:logo")
    d.p(f"{{ref:tab:logo}} holds out each generator family of TweepFake in turn. All methods lost ground relative to the "
        "in-distribution folds, most of all on GPT-2 text, the only family produced by a transformer language model. Here the "
        f"convex weighting of the router family generalised better than stacking ({f(lg.loc['ASTRA', 'gpt2'])} and "
        f"{f(lg.loc['Static-learned', 'gpt2'])} against {f(lg.loc['Stacking', 'gpt2'])}), presumably because a softmax over "
        "experts cannot assign the extreme negative coefficients a logistic meta-learner can learn for a family-specific "
        f"expert. The single STYLE expert was nevertheless the best detector of unseen families ({f(lg.loc['Expert:STYLE', 'gpt2'])} "
        f"on GPT-2 and {f(lg.loc['Expert:STYLE', 'others'])} on the other generators), which says that surface cues such as "
        "emoji, hashtags and capitalisation transfer across generators better than lexical or language-model statistics do.")
    tr_rows = []
    for name, m in [("ASTRA", "ASTRA"), ("Static weights", "Static-learned"), ("Stacking", "Stacking"),
                    ("Stacking, stress-augmented", "Stacking-aug"), ("Gated-MoE", "Gated-MoE"), ("Equal vote", "Equal-vote")]:
        tr_rows.append([name, f(tmean("botsim-test", m, "roc_auc")), f(tmean("tweepfake", m, "roc_auc")),
                        f(tmean("chirper+mastodon", m, "roc_auc")), f(100 * tmean("chirper+mastodon", m, "frr"), 1),
                        f(100 * tmean("chirper+mastodon", m, "recall_removed"), 1), f(rmean(m, "roc_auc"))])
    d.table("Cross-platform transfer with the four text experts (mean of three runs)",
            [[("", 1), ("Trained on BotSim-24", 5), ("Trained on X", 1)],
             ["Method", "BotSim AUROC", "TweepFake AUROC", "Chirper vs. Mastodon AUROC", "Mastodon removed (%)",
              "Chirper removed (%)", "BotSim AUROC"]],
            tr_rows, widths=[1.7, 1, 1, 1.2, 1.1, 1.1, 1], label="tab:transfer", full=True, size=13, header_rows=2,
            note="Removal thresholds were calibrated on BotSim-24 accounts at $\\alpha=0.02$, $\\gamma=0.10$. Mastodon posts are "
                 "presumed legitimate, so their removal rate estimates wrongful moderation; Chirper.ai posts are all written by "
                 "LLM agents.")
    d.p(f"{{ref:tab:transfer}} moves the text-only system to other platforms. Trained on Reddit agents, ASTRA ranked TweepFake "
        f"tweets at AUROC {f(tmean('tweepfake', 'ASTRA', 'roc_auc'))}, below chance: the detector found the genuine tweets "
        "more machine-like than the tweets of GPT-2, RNN and Markov bots, whose style differs from that of an instruction-tuned "
        f"agent. The reverse direction was no better ({f(rmean('ASTRA', 'roc_auc'))}), and every combiner behaved the same way, "
        "so generator and platform shift together erase the signal rather than any particular combination rule. Chirper.ai agents "
        f"were separated from Mastodon posts at AUROC {f(tmean('chirper+mastodon', 'ASTRA', 'roc_auc'))}, but this pair differs "
        "in topic and register as well as in authorship, so the figure is an upper bound on what transfers. "
        + (f"With thresholds calibrated on BotSim-24 for $\\alpha=0.02$, {f(100 * MAST_FRR, 2)}% of the presumed-legitimate "
           f"Mastodon posts and {f(100 * TF_FRR, 2)}% of the genuine TweepFake tweets would have been removed, so the bound on "
           "wrongful removals happened to hold on both platforms. It held for an unhelpful reason: the transferred scores "
           "were low for nearly everything, and only "
           f"{f(100 * CHIRP_REC, 1)}% of the Chirper.ai posts and {f(100 * TF_REC, 1)}% of the TweepFake bot tweets were "
           "removed. The guarantee in {ref:eq:guarantee} rests on exchangeability between calibration and deployment "
           "accounts, which does not hold here; a conservative outcome is what one should hope for in that case, but "
           "nothing in the method ensures it." if GUAR_HELD else
           f"The moderation guarantee did not survive the move: with thresholds calibrated on BotSim-24 for "
           f"$\\alpha=0.02$, {f(100 * MAST_FRR, 2)}% of the presumed-legitimate Mastodon posts and {f(100 * TF_FRR, 2)}% of "
           "the genuine TweepFake tweets would have been removed. Exchangeability between calibration and deployment "
           "accounts is what the bound in {ref:eq:guarantee} rests on, and this result shows what happens when it is "
           "assumed away."))

    d.h2("Sensitivity and Efficiency")
    bse = SE[SE.setting == "base"].set_index("seed").nll
    dl = {s: (SE[SE.setting == s].set_index("seed").nll - bse).dropna() for s in SE.setting.unique()}
    sig_max = max(abs(dl[s]).max() for s in dl if s[0] in "rpk" and s != "base")
    d.p(f"{{ref:fig:sens}} varies the hyperparameters of the trust signals on two partitions. Changing the list length $r$, "
        f"the RBO persistence $p$ or the neighbourhood size $k$ moved the NLL of ASTRA by at most {sig_max:.4f} on either "
        f"partition, against a difference of {abs(bse.diff().iloc[-1]):.4f} between the two partitions themselves; with "
        "router coefficients this small, the exact definition of the signals hardly matters. The number of perturbations "
        f"$K$ behaved differently: a single perturbation per post lowered the NLL by {-dl['K=1'].mean():.4f} and two by "
        f"{-dl['K=2'].mean():.4f} on average, on both partitions. Fewer perturbed copies give clean posts more weight in "
        "the stress-augmented fitting set, which helps clean calibration; whether the smaller bank keeps the gains under "
        "attack was not tested, so $K=4$ is retained for the reported results.")
    d.figure("fig7_sensitivity.png", "Sensitivity of ASTRA on BotSim-24 to the number of perturbations $K$, the neighbourhood size "
             "$k$, and the list length $r$ and persistence $p$ of rank-biased overlap (two partitions; markers show the mean, "
             "bars the range).", label="fig:sens", full=False)
    lat_rows = [["Perturbation bank (4 text variants)", f(LAT["perturbation_ms"], 1)],
                ["GPT-2 pass, original post", f(LAT["gpt2_orig_ms"], 1)],
                ["GPT-2 passes, 4 perturbed variants", f(LAT["gpt2_pert_ms"], 1)],
                ["Experts, attributions, signals, router", f(LAT["experts_router_full_ms"], 1)],
                ["ASTRA total", f(LAT["astra_total_ms"], 1)],
                ["Stacking total (no perturbation)", f(LAT["stacking_total_ms"], 1)]]
    d.table("Measured cost per post on two CPU cores (batch of 400 BotSim-24 posts)", [["Stage", "ms per post"]], lat_rows,
            widths=[2.6, 1], label="tab:latency", bold_rows=(4,))
    d.p(f"{{ref:tab:latency}} reports the measured cost. ASTRA needed {f(LAT['astra_total_ms'], 0)} ms per post on two CPU "
        f"cores, against {f(LAT['stacking_total_ms'], 0)} ms for stacking on the same experts; the four extra language-model "
        "passes account for most of the difference, as the complexity analysis of Section III-I predicted. Given the ablation, "
        "an operator who wants the robustness benefit can fit a static or stacked combiner on stress-augmented data and skip "
        "the per-post perturbation at inference.")

    d.h2("Statements of Reasons and Moderation-Risk Control")
    cs = B["concept_stab"]
    d.p(f"Explanations were checked at the level operators see. Holding each post's weights fixed, the top concept of ASTRA "
        f"stayed the same under the perturbation bank for {100 * cs['ASTRA'][0]:.1f}% of test posts, against "
        f"{100 * cs['Equal-vote'][0]:.1f}% for equal voting, and the concept ranking had an RBO of {cs['ASTRA'][1]:.3f} against "
        f"{cs['Equal-vote'][1]:.3f}. The SHAP-variance weighting scored higher ({100 * cs['SHAP-variance trust'][0]:.1f}%) only "
        "because almost all of its attribution is topical content. Across BotSim-24 test posts, topical content accounted for "
        f"{100 * B['concept_share']['Topical content']:.0f}% of the absolute concept attribution of ASTRA, coordinated interaction "
        f"for {100 * B['concept_share']['Coordinated interaction']:.0f}% and automated cadence for "
        f"{100 * B['concept_share']['Automated cadence']:.0f}%; link redirection contributed nothing because neither corpus "
        "relies on links. {ref:fig:router} shows that the weights hardly depend on how much history an account has: even for "
        "a first post, when the behavioural features are missing, the behavioural expert kept a weight close to its average, "
        "another sign that the router behaves almost like a static combiner.")
    d.figure("fig6_router.png", "Top: mean ASTRA weight per expert as a function of the number of prior events of the account "
             "(BotSim-24, partition 0). Bottom: share of absolute concept attribution over the test posts.", label="fig:router", full=False)
    d.p(f"{{ref:fig:mod}} and {{ref:tab:mod}} evaluate the decision rule. For every combiner the observed wrongful-removal rate "
        f"followed the tolerance closely: for ASTRA it was {f(100 * mod('ASTRA', '0.01', 'frr'), 2)}%, "
        f"{f(100 * mod('ASTRA', '0.02', 'frr'), 2)}% and {f(100 * mod('ASTRA', '0.05', 'frr'), 2)}% of genuine posts at "
        "$\\alpha$ of 1%, 2% and 5%. Averaged per genuine account, which is the quantity the one-post-per-account calibration "
        f"targets, the rate at $\\alpha=0.05$ was {f(100 * mod('ASTRA', '0.05', 'frr_acc'), 2)}%, at the nominal level within the "
        "sampling error of ten partitions with about 290 genuine calibration accounts each. Because the two classes are well "
        "separated, a missed-inauthentic tolerance of $\\gamma=0.10$ placed the keep threshold above the removal threshold and "
        "no post was deferred; tightening it to $\\gamma=0.02$ cut the missed rate from "
        f"{f(100 * mod('ASTRA', '0.02', 'msr'), 2)}% to {f(100 * mod('ASTRA', '0.02', 'msr', '0.02'), 2)}% at the price of deferring "
        f"{f(100 * mod('ASTRA', '0.02', 'defer', '0.02'), 2)}% of posts. The global SHAP-variance weighting needed "
        f"{f(100 * mod('SHAP-variance trust', '0.02', 'defer', '0.02'), 1)}% deferral for the same pair of tolerances, which is what "
        "a poorly separated score costs under risk control. Escalating removals whose routed stability fell below the 10th "
        f"calibration percentile sent {f(100 * mod('ASTRA+esc', '0.02', 'defer'), 1)}% of posts to review and lowered the "
        f"wrongful-removal rate to {f(100 * mod('ASTRA+esc', '0.02', 'frr'), 2)}% without changing the missed rate, as the "
        "argument in Section III-G requires; given H1, it acts mostly as a random audit sample rather than a targeted one. {ref:tab:sor} gives three statements of reasons produced by "
        "the completeness identity {ref:eq:complete}.")
    d.figure("fig4_moderation.png", "Conformal moderation on BotSim-24 for $\\gamma=0.10$ (mean ± SD over ten runs). Left: observed "
             "wrongful-removal rate on genuine posts against the tolerance $\\alpha$; the dashed line is $y=\\alpha$. Centre and "
             "right: deferral rate and rate of inauthentic posts kept without review.", label="fig:mod", full=True)
    mrows = []
    for name, m in [("ASTRA", "ASTRA"), ("ASTRA + escalation", "ASTRA+esc"), ("Stacking", "Stacking"),
                    ("Stacking, stress-augmented", "Stacking-aug"), ("SHAP-variance trust", "SHAP-variance trust")]:
        for a, g in [("0.02", None), ("0.02", "0.02")]:
            mrows.append([name if (a, g) == ("0.02", None) else "", f"{a} / {g or '0.10'}",
                          f(100 * mod(m, a, "frr", g), 2), f(100 * mod(m, a, "frr_acc", g), 2), f(100 * mod(m, a, "msr", g), 2),
                          f(100 * mod(m, a, "defer", g), 2), f(100 * mod(m, a, "prec_remove", g), 1)])
    d.table("Conformal moderation on BotSim-24 test accounts (percent, mean of ten runs)",
            [["Combiner", "$\\alpha$ / $\\gamma$", "Wrongful removal (posts)", "Wrongful removal (accounts)", "Missed",
              "Deferred", "Removal precision"]],
            mrows, widths=[1.7, 0.9, 1.1, 1.1, 0.8, 0.8, 0.9], label="tab:mod", full=True, size=13)
    P = pd.read_pickle(f"{W}/preds_botsim_seed0.pkl")
    p = expit(P["preds"]["ASTRA"]); y = P["y"]; C = P["C"]
    CONCEPTS = ['Synthetic authorship', 'Automated cadence', 'Coordinated interaction', 'Link redirection',
                'Engagement bait', 'Stylometric form', 'Topical content']
    picks = [int(np.where((p > 0.95) & (y == 1) & (P["kind"] == "post"))[0][0]),
             int(np.where((p > 0.95) & (y == 1) & (P["kind"] == "comment"))[0][0]),
             int(np.where((p > 0.5) & (y == 0))[0][0])]
    sr = []
    for i in picks:
        o = np.argsort(-C[i])[:2]
        txt = P["text"][i]
        txt = (txt[:118] + "...") if len(txt) > 121 else txt
        sr.append([txt, "inauthentic" if y[i] else "genuine", f(p[i]),
                   f"{CONCEPTS[o[0]]} ({C[i, o[0]]:+.2f}); {CONCEPTS[o[1]]} ({C[i, o[1]]:+.2f})"])
    d.table("Statements of reasons generated from concept attributions for three BotSim-24 test posts",
            [["Post text", "True label", "$\\hat{p}$", "Two largest positive concept contributions (log-odds)"]],
            sr, widths=[2.9, 0.8, 0.5, 1.9], label="tab:sor", full=True, size=13, align=["l", "c", "c", "l"],
            note="The third post is a false positive above the 0.5 threshold; at $\\alpha=0.02$ its probability falls below "
                 "$\\hat{\\lambda}_R$ and it is not removed. Each statement also lists the automated nature of the decision and the "
                 "observable facts behind each concept.")


# ======================================================================== discussion, limitations, conclusion
def discussion(d, R):
    d.h1("DISCUSSION")
    d.h2("Interpretation of Findings")
    d.p("The central question was whether the stability of an expert's reasons is a usable trust signal. On the evidence here "
        "it is not, at least when stability is measured against cheap perturbations of the input. The perturbation bank "
        "reshuffles the attribution lists of all experts on most posts, so instability is close to a constant property of "
        "each expert rather than a property of the post. Confidence, which the literature on selective prediction already "
        "uses {cite:chow1970optimum,geifman2017selective}, carried most of the information about where an expert fails. Once "
        "the signals were standardised and the router could have used them, it chose not to: the fitted coefficients were "
        "small and the ablations without them were statistically indistinguishable.")
    d.p("What did help was mundane. Showing the combiner perturbed versions of its fitting posts taught it that the behavioural "
        "and interaction experts remain informative when text is manipulated, and that lesson transferred to an attack family "
        "it had never seen and to wholesale mimicry of a genuine account's behaviour. The benefit came from the fitting set, "
        "not from the routing rule, since augmented stacking gained as much. The convex form of the router did show one "
        "advantage over stacking, better transfer to an unseen generator family, which fits the intuition that bounded "
        "weights cannot over-commit to an expert that happens to separate the training generators.")
    d.h2("Comparison with Prior Literature")
    d.p("The weak link between instability and error agrees with reports that explanations change under perturbations that do "
        "not change predictions {cite:ghorbani2019fragile,dombrowski2019manipulated} and that explainers disagree among "
        "themselves {cite:krishna2024disagreement}: if explanations move for reasons unrelated to correctness, their movement "
        "cannot be expected to track correctness. The drop of detectors under homoglyphs and leetspeak on TweepFake matches "
        "the evasion results for machine-text detectors {cite:creo2025silverspeak,krishna2023paraphrasing}, and the collapse "
        "of clean-fit combiners under behavioural mimicry echoes the warning that human-mimicking bots defeat detectors "
        "built on earlier bots {cite:cresci2017paradigm,feng2024botsay}.")
    d.h2("Practical and Scientific Implications")
    d.p("For a moderation team the results suggest a simple recipe: combine views with a convex or stacked combiner fitted "
        "on perturbed copies of held-out labelled posts, calibrate class-conditional thresholds on one post per account, and "
        "use the exact concept decomposition as the statement of reasons. The per-post perturbation machinery can be dropped "
        "at inference, which cuts the cost to that of a single language-model pass. Scientifically, the study shows that letting explanation "
        "stability decide trust can be tested cleanly when attributions are exact and the router is readable.")
    d.h2("Failure Cases")
    d.p("Three failures stand out. First, the combined attack still removed most of the benefit of the context views: even "
        f"with augmentation, AUROC fell to {f(att('botsim', 'homo-0.15+mimic', 'ASTRA'))}, and only "
        f"{100 * att('botsim', 'homo-0.15+mimic', 'ASTRA', 'removed'):.0f}% of the attacked posts crossed the removal threshold. "
        "Second, the risk-control bound is only as good as its exchangeability assumption. "
        + ("Across platforms the wrongful-removal rate happened to stay below the tolerance, but only because the "
           "transferred scores were uniformly low, so the system missed most inauthentic posts there. " if GUAR_HELD else
           "Across platforms the wrongful-removal rate exceeded the tolerance. ")
        + "Third, on TweepFake no combiner was reliable at the account "
        "level, and the third statement in {ref:tab:sor} shows the kind of genuine comment, short and politically charged, "
        "that the topical-content concept misreads.")


def limitations(d, R):
    d.h1("LIMITATIONS")
    d.p("The inauthentic accounts of BotSim-24 come from a simulator driven by one model, GPT-4o-mini. Their behaviour and "
        "interaction traces were generated by the simulator, and although simulator fields were removed, the separability of "
        "the context views may be higher than on a live platform. TweepFake uses generators that predate modern "
        "instruction-tuned models, and its 40 accounts make account-level estimates noisy. Bluesky data with moderation labels "
        "could not be obtained within the network constraints of this study, and the Mastodon posts carry no bot label, so "
        "the decentralised-platform evaluation is limited to presumed-legitimate traffic. The FediData release includes "
        "manually labelled Mastodon bots, and evaluating on them is the most direct next step.")
    d.p("The language model is GPT-2 small quantised to 8-bit weights and truncated to 48 tokens, chosen so that every "
        "experiment could run on a CPU; stronger encoders would raise the text experts and might change the balance of the "
        "router. The perturbation bank uses WordNet synonyms rather than LLM paraphrase, so the strongest realistic text "
        "attack, a fluent rewrite, was not tested. Attack experiments use a single partition and 600 attacked posts, so their "
        "differences should be read as indicative. Finally, the conformal guarantee assumes that accounts are exchangeable "
        "between calibration and deployment; weighted conformal methods {cite:tibshirani2019covariate,barber2023beyond} "
        "could relax this when the shift can be estimated, and were not evaluated here.")


def conclusion(d, R):
    d.h1("CONCLUSION")
    d.p("This paper tested whether the stability of exact attributions under cheap perturbations can "
        "decide how much to trust each expert on each post. ASTRA implemented the idea with six experts, an eleven-parameter "
        "router, an exact decomposition of every decision into seven policy concepts and class-conditional conformal "
        f"thresholds. On BotSim-24 it reached {f(mv('botsim', 'ASTRA', 'macro_f1'))} macro-F1, far above global "
        f"explanation-weighted voting ({f(mv('botsim', 'SHAP-variance trust', 'macro_f1'))}), and kept wrongful removals near the "
        "chosen tolerance. The stability signal itself, however, predicted expert errors barely better than chance and did "
        "not move the routing weights under attack. The gains under evasion and behavioural mimicry came from fitting the "
        "combiner on perturbed posts, and the convex form of the router helped on an unseen generator family. The principal limitation is the reliance on a simulated botnet and "
        "on exchangeability between calibration and deployment accounts; across platforms the detector lost most of its "
        "ranking power even where the removal bound was respected. The next step is to evaluate on labelled bots from decentralised "
        "services and to replace the fixed thresholds with calibration that adapts to platform shift.")


def backmatter(d, R):
    d.h1("DATA AVAILABILITY", numbered=False)
    d.p("All corpora are public: BotSim-24 and its LLM-selection data (github.com/QQQQQQBY/BotSim), TweepFake "
        "(github.com/tizfa/tweepfake_deepfake_text_detection), Masquerade-23 (github.com/Litsay/Masquerade-23) and the "
        "FediData request files (github.com/FDUDataNET/FediData). The sampled item lists and all derived features can be "
        "regenerated with the released code.")
    d.h1("CODE AVAILABILITY", numbered=False)
    d.p("The complete pipeline (data preparation, perturbation bank, GPT-2 feature extraction, experts, router, risk control, "
        "experiments, statistics and figures) is provided as a single flat folder accompanying this manuscript, together with "
        "the result files from which every number in the paper was produced.")
    d.h1("ETHICS STATEMENT", numbered=False)
    d.p("Only public research datasets were used; no new data were collected from users and no account was contacted. "
        "Account identifiers were used only to form partitions and are not reported.")
    d.h1("DECLARATIONS", numbered=False)
    d.p("Author contributions (CRediT), conflicts of interest and acknowledgments: [to be completed by the authors].")
