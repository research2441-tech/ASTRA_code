"""E1-E3, E6-E8: main comparison, ablations, conformal moderation-risk control, H1 analysis.

Usage: python run_main.py botsim|tweepfake [n_runs]
Writes work/results_<dataset>.json and work/preds_<dataset>_seed0.pkl
"""
import os
import sys, json, time, numpy as np, pandas as pd
from scipy.special import expit, softmax
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import roc_auc_score
from astra_lib import *

W = os.environ.get("ASTRA_WORK", os.path.dirname(os.path.abspath(__file__)))


def load(dataset):
    texts = pd.read_pickle(f"{W}/texts.pkl")
    f = np.load(f"{W}/gpt2_feats.npz")
    if dataset == "botsim":
        items = pd.read_pickle(f"{W}/botsim_items.pkl")
    else:
        items = pd.read_pickle(f"{W}/text_items.pkl"); items = items[items.dataset == dataset]
    sub = texts[texts.item.isin(set(items.item))]
    return Views(items, sub.reset_index(drop=True), f["auth"][sub.index.values], f["emb"][sub.index.values],
                 has_context=(dataset == "botsim"))


def account_split(items, seed, fr=(0.5, 0.15, 0.15, 0.2)):
    rng = np.random.default_rng(seed)
    acc = items.groupby("account").label.first()
    parts = [[], [], [], []]
    for lab in (0, 1):
        a = acc[acc == lab].index.values.copy(); rng.shuffle(a)
        cut = np.cumsum([int(round(x * len(a))) for x in fr[:-1]])
        for k, s in enumerate(np.split(a, cut)):
            parts[k].extend(s)
    return [np.where(items.account.isin(set(p)))[0] for p in parts]


def tweep_splits(items, n_rep=2, k=5):
    acc = items.groupby("account").agg(label=("label", "first"), gen=("gen", "first"))
    out = []
    for rep in range(n_rep):
        rng = np.random.default_rng(100 + rep)
        folds = [[] for _ in range(k)]
        for g, grp in acc.groupby("gen"):
            a = grp.index.values.copy(); rng.shuffle(a)
            for i, x in enumerate(a):
                folds[(i + rep) % k].append(x)
        for f in range(k):
            te_acc = set(folds[f]); rest = acc.drop(list(te_acc))
            rf_acc = set()
            for lab, grp in rest.groupby("label"):
                a = grp.index.values.copy(); rng.shuffle(a); rf_acc |= set(a[: max(1, int(round(0.3 * len(a))))])
            tr_acc = set(rest.index) - rf_acc
            idx = lambda s: np.where(items.account.isin(s))[0]
            out.append((idx(tr_acc), idx(rf_acc), np.array([], int), idx(te_acc)))
    return out


def mats(sig, names, key):
    return np.stack([sig[m][key] for m in names], 1)


def fit_gate(U, Z, y, l2=1e-2):
    d, M = U.shape[1], Z.shape[1]
    def unpack(th):
        return th[: d * M].reshape(d, M), th[d * M: d * M + M], th[-2], th[-1]
    def loss(th):
        A, b, tau, g0 = unpack(th); Wt = softmax(U @ A + b, axis=1); z = g0 + tau * (Wt * Z).sum(1)
        return -loglik(z, y).mean() + l2 * (th[: d * M] ** 2).sum()
    th = minimize(loss, np.r_[np.zeros(d * M + M), 1.0, 0.0], method="L-BFGS-B").x
    def pred(Uq, Zq):
        A, b, tau, g0 = unpack(th); Wt = softmax(Uq @ A + b, axis=1); return g0 + tau * (Wt * Zq).sum(1), Wt
    return pred


def knora_e(U_rf, Z_rf, y_rf, U_q, Z_q, kmax=7):
    nn = NearestNeighbors(n_neighbors=kmax).fit(U_rf); _, ind = nn.kneighbors(U_q)
    corr = ((Z_rf > 0).astype(int) == y_rf[:, None])
    out = np.zeros(len(U_q))
    for i in range(len(U_q)):
        sel = None
        for k in range(kmax, 0, -1):
            ok = corr[ind[i, :k]].all(0)
            if ok.any():
                sel = ok; break
        out[i] = Z_q[i].mean() if sel is None else Z_q[i, sel].mean()
    return out


def shapvar_weights(bank, sig_rf, topk=10):
    """MEAH-E-style global trust: inverse variance of the top-k mean |SHAP| values of each expert."""
    w = []
    for m in bank.names:
        phi = sig_rf[m]["phi"]
        mabs = np.asarray(abs(phi).mean(0)).ravel()
        top = np.sort(mabs)[::-1][:topk]
        w.append(1.0 / (np.var(top) + 1e-12))
    w = np.array(w); return w / w.sum()


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6); return np.log(p / (1 - p))


def concept_sums(bank, idx):
    """Per expert, per version: concept-summed attributions [n, K]."""
    out = {}
    for m in bank.names:
        mod = bank.models[m]; cm = bank.concept_of[m]; lst = []
        for v in bank.V.versions(m)[: 1 + bank.K]:
            phi = mod.attrib(bank.V.X(m, v, idx, bank.tfidf)); C = np.zeros((len(idx), len(CONCEPTS)))
            for k in range(len(CONCEPTS)):
                cols = np.where(cm == k)[0]
                if len(cols):
                    C[:, k] = np.asarray(phi[:, cols].sum(1)).ravel()
            lst.append(C)
        out[m] = lst
    return out


def concept_stability(csum, names, Wt, tau):
    """Top-1 concept agreement and concept-rank RBO between original and perturbed inputs (weights fixed per post)."""
    K = len(csum[names[0]]) - 1
    sys0 = sum(tau * Wt[:, i:i + 1] * csum[m][0] for i, m in enumerate(names))
    top1, r = [], []
    for k in range(1, K + 1):
        sysk = sum(tau * Wt[:, i:i + 1] * csum[m][min(k, len(csum[m]) - 1)] for i, m in enumerate(names))
        top1.append(np.argmax(np.abs(sys0), 1) == np.argmax(np.abs(sysk), 1))
        r.append([rbo(topr_signed(sys0[i], 7), topr_signed(sysk[i], 7), 0.9) for i in range(len(sys0))])
    return float(np.mean(top1)), float(np.mean(r))


def run(dataset, split, seed, V, save=False, K=4, r=10, k_nn=50):
    tr, rf, ca, te = split
    I = V.items; y = I.label.values; acc = I.account.values
    t0 = time.time()
    bank = ExpertBank(V, seed=seed, r=r, K=K).fit(tr)
    names = bank.names
    sig_rf, sig_te = bank.signals(rf, pairwise=True), bank.signals(te)
    sig_ca = bank.signals(ca) if len(ca) else None
    t_sig = time.time() - t0
    Z = {"rf": mats(sig_rf, names, "z"), "te": mats(sig_te, names, "z")}
    S = {"rf": mats(sig_rf, names, "s"), "te": mats(sig_te, names, "s")}
    D = {"rf": mats(sig_rf, names, "d"), "te": mats(sig_te, names, "d")}
    if sig_ca:
        Z["ca"], S["ca"], D["ca"] = mats(sig_ca, names, "z"), mats(sig_ca, names, "s"), mats(sig_ca, names, "d")
    U_rf, pca, sc = routing_space(V, rf, pca=PCA(16, random_state=0).fit(V.emb["orig"][tr]))
    U = {"rf": U_rf, "te": routing_space(V, te, pca, sc)[0]}
    if sig_ca: U["ca"] = routing_space(V, ca, pca, sc)[0]
    ll_rf = loglik(Z["rf"], y[rf][:, None])
    R = {"rf": local_reliability(U["rf"], acc[rf], ll_rf, U["rf"], acc[rf], k=k_nn),
         "te": local_reliability(U["rf"], acc[rf], ll_rf, U["te"], k=k_nn)}
    if sig_ca: R["ca"] = local_reliability(U["rf"], acc[rf], ll_rf, U["ca"], k=k_nn)
    yrf, yte = y[rf], y[te]
    res, preds = {}, {}
    Za, Ra, Sa, Da, ya = augmented_router_data(V, bank, sig_rf, rf, R["rf"], pca, sc, U["rf"], acc[rf], ll_rf, k_nn=k_nn)

    # ---------------- ASTRA and ablations
    variants = {"ASTRA": (1, 1, 1), "ASTRA-no-s": (1, 0, 1), "ASTRA-no-d": (1, 1, 0), "ASTRA-no-rho": (0, 1, 1),
                "ASTRA-rho-only": (1, 0, 0), "Static-learned": (0, 0, 0), "ASTRA-clean-fit": (1, 1, 1)}
    routers = {}
    for nm, u in variants.items():
        if nm == "ASTRA-clean-fit":
            ro = Router(*map(bool, u)).fit(Z["rf"], R["rf"], S["rf"], D["rf"], yrf)
        else:
            ro = Router(*map(bool, u)).fit(Za, Ra, Sa, Da, ya)
        routers[nm] = ro
        zt, Wt = ro.predict(Z["te"], R["te"], S["te"], D["te"]); preds[nm] = zt
        if nm == "ASTRA":
            W_te = Wt
    # ---------------- baselines on the same experts
    i_best = int(np.argmin(-ll_rf.mean(0)))
    preds["Best-expert"] = Z["te"][:, i_best]
    preds["Equal-vote"] = Z["te"].mean(1)
    accw = ((Z["rf"] > 0).astype(int) == yrf[:, None]).mean(0); accw = accw / accw.sum()
    preds["Accuracy-weighted"] = logit((expit(Z["te"]) * accw).sum(1))
    sw = shapvar_weights(bank, sig_rf)
    preds["SHAP-variance trust"] = logit((expit(Z["te"]) * sw).sum(1))
    st = LogisticRegression(C=1.0, max_iter=2000).fit(Z["rf"], yrf); preds["Stacking"] = st.decision_function(Z["te"])
    sta = LogisticRegression(C=1.0, max_iter=2000).fit(Za, ya); preds["Stacking-aug"] = sta.decision_function(Z["te"])
    preds["KNORA-E"] = knora_e(U["rf"], Z["rf"], yrf, U["te"], Z["te"])
    gate = fit_gate(U["rf"], Z["rf"], yrf); preds["Gated-MoE"] = gate(U["te"], Z["te"])[0]
    zp_rf = np.stack([np.c_[sig_rf[m]["z"], sig_rf[m]["z_pert"]].mean(1) for m in names], 1)
    zp_te = np.stack([np.c_[sig_te[m]["z"], sig_te[m]["z_pert"]].mean(1) for m in names], 1)
    stp = LogisticRegression(C=1.0, max_iter=2000).fit(zp_rf, yrf); preds["TTA-Stacking"] = stp.decision_function(zp_te)
    # monolithic models on concatenated dense views (train + router-fit data)
    trrf = np.r_[tr, rf]
    pca64 = PCA(64, random_state=0).fit(V.emb["orig"][tr])
    def dense(idx):
        blocks = [V.auth["orig"][idx], V.style["orig"][idx], pca64.transform(V.emb["orig"][idx])]
        if V.has_context: blocks += [V.beh["orig"][idx], V.gr["orig"][idx]]
        return np.concatenate(blocks, 1)
    Xa, Xt = dense(trrf), dense(te)
    preds["Concat-LightGBM"] = TreeExpert(seed=seed, n=400, leaves=31).fit(Xa, y[trrf]).logit(Xt)
    scm = StandardScaler().fit(np.nan_to_num(Xa))
    mlp = MLPClassifier((256, 64), alpha=1e-3, early_stopping=True, random_state=seed, max_iter=300)
    mlp.fit(scm.transform(np.nan_to_num(Xa)), y[trrf])
    preds["MLP (concat)"] = logit(mlp.predict_proba(scm.transform(np.nan_to_num(Xt)))[:, 1])
    for i, m in enumerate(names):
        preds[f"Expert:{m}"] = Z["te"][:, i]
    for k, z in preds.items():
        res[k] = metrics(z, yte)
    res["_router_params"] = routers["ASTRA"].params(names)
    res["_mean_weights"] = {m: float(W_te[:, i].mean()) for i, m in enumerate(names)}
    res["_shapvar_weights"] = {m: float(sw[i]) for i, m in enumerate(names)}
    res["_stack_coef"] = {m: float(st.coef_[0, i]) for i, m in enumerate(names)}
    res["_time_signals_s"] = t_sig

    # ---------------- H1: does attribution instability flag expert errors?
    h1 = {}
    for i, m in enumerate(names):
        err = ((Z["te"][:, i] > 0).astype(int) != yte).astype(int)
        if 0 < err.sum() < len(err):
            h1[m] = dict(auc_instab=roc_auc_score(err, -S["te"][:, i]), auc_shift=roc_auc_score(err, D["te"][:, i]),
                         auc_lowconf=roc_auc_score(err, -np.abs(Z["te"][:, i])), err_rate=float(err.mean()))
    res["_H1"] = h1

    # ---------------- concept stability (weights fixed per post)
    cs = {m: sig_te[m]["csum"] for m in names}
    tau = routers["ASTRA"].params(names)["tau"]
    res["_concept_stab"] = {
        "ASTRA": concept_stability(cs, names, W_te, tau),
        "Equal-vote": concept_stability(cs, names, np.full_like(W_te, 1.0 / len(names)), 1.0),
        "SHAP-variance trust": concept_stability(cs, names, np.tile(sw, (len(te), 1)), 1.0)}
    C_te = concept_attribution(bank, sig_te, W_te, tau)
    res["_concept_share"] = {c: float(np.abs(C_te[:, k]).sum() / np.abs(C_te).sum()) for k, c in enumerate(CONCEPTS)}

    # ---------------- conformal moderation-risk control
    if sig_ca:
        rng = np.random.default_rng(seed)
        sub = one_per_account(acc[ca], rng)
        mod = {}
        z_ca = {"ASTRA": routers["ASTRA"].predict(Z["ca"], R["ca"], S["ca"], D["ca"])[0],
                "Stacking": st.decision_function(Z["ca"]), "Stacking-aug": sta.decision_function(Z["ca"]),
                "Equal-vote": Z["ca"].mean(1),
                "SHAP-variance trust": logit((expit(Z["ca"]) * sw).sum(1))}
        W_ca = routers["ASTRA"].predict(Z["ca"], R["ca"], S["ca"], D["ca"])[1]
        stab_ca, stab_te = (W_ca * S["ca"]).sum(1), (W_te * S["te"]).sum(1)
        legit_acc = acc[te][yte == 0]
        for nm, zc in z_ca.items():
            pc, pt = expit(zc[sub]), expit(preds[nm])
            for alpha, gam in [(0.01, 0.10), (0.02, 0.10), (0.05, 0.10), (0.10, 0.10), (0.01, 0.02), (0.02, 0.02), (0.02, 0.05), (0.05, 0.02)]:
                lr, lk = crc_thresholds(pc, y[ca][sub], alpha, gam)
                for esc in ((False, True) if nm == "ASTRA" else (False,)):
                    tau_s = np.quantile(stab_ca[sub], 0.10) if esc else None
                    dec = moderate(pt, lr, lk, stab_te if esc else None, tau_s)
                    frr_acc = pd.Series((dec[yte == 0] == 2).astype(float)).groupby(legit_acc).mean().mean()
                    mod[f"{nm}{'+esc' if esc else ''}|{alpha}" + ("" if gam == 0.10 else f"|g{gam}")] = dict(
                        frr=float(np.mean(dec[yte == 0] == 2)), frr_acc=float(frr_acc),
                        msr=float(np.mean(dec[yte == 1] == 0)), defer=float(np.mean(dec == 1)),
                        remove=float(np.mean(dec == 2)), prec_remove=float(yte[dec == 2].mean()) if (dec == 2).any() else np.nan,
                        lam_R=float(lr), lam_K=float(lk), n0=int((y[ca][sub] == 0).sum()), n1=int((y[ca][sub] == 1).sum()))
        res["_moderation"] = mod
    if save:
        pd.to_pickle(dict(names=names, te=te, y=yte, acc=acc[te], preds=preds, W=W_te, S=S["te"], D=D["te"],
                          R=R["te"], Z=Z["te"], C=C_te, gen=I.gen.values[te], text=V.text["orig"][te],
                          kind=I.kind.values[te] if "kind" in I else None,
                          nprev=I.beh_log_nprev.values[te] if V.has_context else None,
                          router=routers["ASTRA"].params(names), sw=sw),
                     f"{W}/preds_{dataset}_seed{seed}.pkl")
    return res


if __name__ == "__main__":
    ds = sys.argv[1]; n_runs = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    V = load(ds)
    splits = [account_split(V.items, s) for s in range(n_runs)] if ds == "botsim" else tweep_splits(V.items)[:n_runs]
    allres = []
    for s, sp in enumerate(splits):
        t = time.time(); r = run(ds, sp, s, V, save=(s == 0)); allres.append(r)
        print(f"[{ds}] run {s} {time.time() - t:.0f}s ASTRA F1={r['ASTRA']['macro_f1']:.3f} PR={r['ASTRA']['pr_auc']:.3f} "
              f"Stack F1={r['Stacking']['macro_f1']:.3f} PR={r['Stacking']['pr_auc']:.3f} NLL A/S={r['ASTRA']['nll']:.3f}/{r['Stacking']['nll']:.3f}",
              flush=True)
        json.dump(allres, open(f"{W}/results_{ds}.json", "w"), default=float)
