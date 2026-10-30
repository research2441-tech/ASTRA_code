"""E3 sensitivity, E4 robustness (evasion attacks), E5 held-out generator, E10 cross-platform transfer, E6 latency.

Usage: python run_extra.py attacks|logo|transfer|sensitivity|latency
"""
import os
import sys, os, json, time, numpy as np, pandas as pd
from scipy.special import expit
from sklearn.metrics import roc_auc_score
from astra_lib import *
from run_main import load, account_split, tweep_splits, metrics, W, run
from system import System
from perturb import Glove, perturb_all, p_homo, p_syn, p_leet
from gpt2_onnx import Scorer
from extract_features import auth_vector

METHODS = ["ASTRA", "ASTRA-clean-fit", "Stacking", "Stacking-aug", "Equal-vote", "SHAP-variance trust", "Gated-MoE", "Static-learned"]


def gpt2_views(items, glove, scorer, has_context, seed=0):
    rows = []
    for k, r in enumerate(items.itertuples()):
        rows.append((r.item, "orig", r.text))
        for name, txt in zip(PERTS_TEXT, perturb_all(r.text, seed=seed + k, glove=glove)):
            rows.append((r.item, name, txt))
    texts = pd.DataFrame(rows, columns=["item", "version", "text"])
    out = scorer.score(texts.text.tolist(), bs=16)
    auth = np.stack([auth_vector(o) for o in out]); emb = np.stack([o["emb"] for o in out]).astype(np.float16)
    return Views(items.reset_index(drop=True), texts, auth, emb, has_context)


def attacks():
    glove, scorer = Glove(), Scorer(48)
    res = {}
    for ds in ("botsim", "tweepfake"):
        V = load(ds); I = V.items
        if ds == "botsim":
            tr, rf, ca, te = account_split(I, 0)
        else:
            tr, rf, ca, te = tweep_splits(I)[0]
        sysA = System(seed=0).fit(V, tr, rf, ca if len(ca) else None)
        rng = np.random.default_rng(7)
        te_b = te[I.label.values[te] == 1]; te_h = te[I.label.values[te] == 0]
        att = rng.choice(te_b, size=min(600, len(te_b)), replace=False)
        clean_h = sysA.score(V, te_h); clean_b = sysA.score(V, att)
        base = I.iloc[att].copy().reset_index(drop=True)
        donors = I.iloc[rng.choice(te_h, size=len(att))].reset_index(drop=True)
        specs = [("clean", None, False), ("homo-0.05", 0.05, False), ("homo-0.15", 0.15, False), ("homo-0.30", 0.30, False),
                 ("homo-0.50", 0.50, False), ("syn-0.25", ("syn", 0.25), False), ("syn-0.50", ("syn", 0.5), False),
                 ("leet-0.15", ("leet", 0.15), False), ("leet-0.30", ("leet", 0.30), False)]
        if ds == "botsim":
            specs += [("mimic", None, True), ("homo-0.15+mimic", 0.15, True)]
        res[ds] = {}
        for name, txt_att, mimic in specs:
            if name == "clean":
                sc_b = clean_b
            else:
                it = base.copy(); it["item"] = [f"att_{name}_{i}" for i in range(len(it))]
                if txt_att is not None:
                    rr = np.random.default_rng(11)
                    if isinstance(txt_att, tuple) and txt_att[0] == "leet":
                        it["text"] = [p_leet(t, rr, rate=txt_att[1]) for t in it.text]
                    elif isinstance(txt_att, tuple):
                        it["text"] = [p_syn(t, rr, glove, rate=txt_att[1]) for t in it.text]
                    else:
                        it["text"] = [p_homo(t, rr, rate=txt_att) for t in it.text]
                if mimic:
                    cols = [c for c in it.columns if c.startswith("beh_") or c.startswith("gr_")]
                    it[cols] = donors[cols].values
                if txt_att is None and mimic:
                    # text unchanged: reuse original GPT-2 features
                    Vq = Views(it, V_text_rows(V, att, it.item.values), *V_feats(V, att), ds == "botsim")
                else:
                    cache = f"{W}/attcache_{ds}_{name}.pkl"
                    if os.path.exists(cache):
                        cz = pd.read_pickle(cache); Vq = Views(it, cz["texts"], cz["auth"], cz["emb"], ds == "botsim")
                    else:
                        Vq = gpt2_views(it, glove, scorer, ds == "botsim", seed=1000)
                        tx = pd.DataFrame([(i, v, Vq.text[v][j]) for j, i in enumerate(it.item) for v in ["orig"] + PERTS_TEXT],
                                          columns=["item", "version", "text"])
                        au = np.stack([Vq.auth[v][j] for j in range(len(it)) for v in ["orig"] + PERTS_TEXT])
                        em = np.stack([Vq.emb[v][j] for j in range(len(it)) for v in ["orig"] + PERTS_TEXT]).astype(np.float16)
                        pd.to_pickle(dict(texts=tx, auth=au, emb=em), cache)
                sc_b = sysA.score(Vq, np.arange(len(it)))
            out = {}
            for m in METHODS:
                zb, zh = sc_b["z"][m], clean_h["z"][m]
                yy = np.r_[np.ones(len(zb)), np.zeros(len(zh))]
                out[m] = dict(recall=float(np.mean(zb > 0)), auroc=float(roc_auc_score(yy, np.r_[zb, zh])))
                if sysA.thr is None and hasattr(sysA, "cal"):
                    lr, lk = sysA.thresholds(m, 0.02)
                    dec = moderate(expit(zb), lr, lk)
                    out[m].update(removed=float(np.mean(dec == 2)), deferred=float(np.mean(dec == 1)),
                                  kept=float(np.mean(dec == 0)))
            out["_weights"] = {mm: float(sc_b["W"][:, i].mean()) for i, mm in enumerate(sysA.names)}
            out["_stab"] = {mm: float(sc_b["S"][:, i].mean()) for i, mm in enumerate(sysA.names)}
            res[ds][name] = out
            print(ds, name, {m: round(out[m]["recall"], 3) for m in METHODS}, flush=True)
            json.dump(res, open(f"{W}/results_attacks.json", "w"))


def V_text_rows(V, idx, new_ids):
    rows = []
    for j, i in enumerate(idx):
        for v in ["orig"] + PERTS_TEXT:
            rows.append((new_ids[j], v, V.text[v][i]))
    return pd.DataFrame(rows, columns=["item", "version", "text"])


def V_feats(V, idx):
    auth, emb = [], []
    for i in idx:
        for v in ["orig"] + PERTS_TEXT:
            auth.append(V.auth[v][i]); emb.append(V.emb[v][i])
    return np.stack(auth), np.stack(emb).astype(np.float16)


def logo():
    V = load("tweepfake"); I = V.items
    acc = I.groupby("account").agg(label=("label", "first"), gen=("gen", "first"))
    out = []
    for fam in ("gpt2", "rnn", "others"):
        for rep in range(3):
            rng = np.random.default_rng(200 + rep)
            hum = acc[acc.gen == "human"].index.values.copy(); rng.shuffle(hum)
            h_tr, h_rf, h_te = hum[:10], hum[10:13], hum[13:]
            bots = acc[(acc.label == 1) & (acc.gen != fam)].index.values.copy(); rng.shuffle(bots)
            nb = max(1, int(round(0.25 * len(bots))))
            b_rf, b_tr = bots[:nb], bots[nb:]
            b_te = acc[acc.gen == fam].index.values
            ix = lambda a: np.where(I.account.isin(set(a)))[0]
            split = (ix(np.r_[h_tr, b_tr]), ix(np.r_[h_rf, b_rf]), np.array([], int), ix(np.r_[h_te, b_te]))
            r = run("tweepfake-logo", split, rep, V)
            r["_family"] = fam; r["_rep"] = rep; out.append(r)
            print("LOGO", fam, rep, round(r["ASTRA"]["macro_f1"], 3), round(r["Stacking"]["macro_f1"], 3), flush=True)
            json.dump(out, open(f"{W}/results_logo.json", "w"), default=float)


def transfer():
    TEXT = ["LEX", "SEM", "AUTH", "STYLE"]
    Vb = load("botsim"); Vt = load("tweepfake")
    items = pd.read_pickle(f"{W}/text_items.pkl")
    texts = pd.read_pickle(f"{W}/texts.pkl"); f = np.load(f"{W}/gpt2_feats.npz")
    ext = items[items.dataset.isin(["chirper", "mastodon"])]
    dup = set(ext.item[ext.item.duplicated()])
    ext = ext[~ext.item.isin(dup)].reset_index(drop=True)  # truncated Chirper handles can collide; drop ambiguous ids
    sub = texts[texts.item.isin(set(ext.item))]
    Ve = Views(ext, sub.reset_index(drop=True), f["auth"][sub.index.values], f["emb"][sub.index.values], False)
    Vb.has_context = False
    res = []
    for seed in range(3):
        tr, rf, ca, te = account_split(Vb.items, seed)
        s = System(seed=seed, experts=TEXT).fit(Vb, tr, rf, ca)
        rec = {"seed": seed}
        for tgt, Vq in (("botsim-test", None), ("tweepfake", Vt), ("chirper+mastodon", Ve)):
            if Vq is None:
                Vq, idx = Vb, te
            else:
                idx = np.arange(len(Vq.items))
            sc = s.score(Vq, idx); yy = Vq.items.label.values[idx]
            d = {}
            for m in METHODS + [f"Expert:{e}" for e in TEXT]:
                d[m] = metrics(sc["z"][m], yy)
                if m in s.cal:
                    lr, lk = s.thresholds(m, 0.02); dec = moderate(expit(sc["z"][m]), lr, lk)
                    d[m].update(frr=float(np.mean(dec[yy == 0] == 2)), msr=float(np.mean(dec[yy == 1] == 0)),
                                defer=float(np.mean(dec == 1)), recall_removed=float(np.mean(dec[yy == 1] == 2)))
            d["_weights"] = {e: float(sc["W"][:, i].mean()) for i, e in enumerate(TEXT)}
            rec[tgt] = d
            print("transfer", seed, tgt, round(d["ASTRA"]["roc_auc"], 3), round(d["Stacking"]["roc_auc"], 3),
                  d["ASTRA"].get("frr"), flush=True)
        res.append(rec)
        json.dump(res, open(f"{W}/results_transfer.json", "w"), default=float)
    # reverse direction: TweepFake -> BotSim (text views only)
    rev = []
    for seed in range(3):
        tr, rf, _, _ = tweep_splits(Vt.items)[seed]
        s = System(seed=seed, experts=TEXT).fit(Vt, tr, rf)
        tb = np.arange(len(Vb.items)); sc = s.score(Vb, tb)
        rev.append({m: metrics(sc["z"][m], Vb.items.label.values) for m in METHODS})
        print("reverse", seed, round(rev[-1]["ASTRA"]["roc_auc"], 3), flush=True)
    json.dump(dict(forward=res, reverse=rev), open(f"{W}/results_transfer.json", "w"), default=float)


def sensitivity():
    """ASTRA (stress-augmented fit) under different r, p, K, k; two account partitions."""
    V = load("botsim"); I = V.items; y = I.label.values
    out = []
    settings = [("base", {}), ("r=5", dict(r=5)), ("r=20", dict(r=20)), ("p=0.8", dict(rbo_p=0.8)), ("p=0.95", dict(rbo_p=0.95)),
                ("K=1", dict(K=1)), ("K=2", dict(K=2)), ("k=10", dict(k_nn=10)), ("k=200", dict(k_nn=200))]
    for seed in range(2):
        tr, rf, ca, te = account_split(I, seed)
        for name, kw in settings:
            t = time.time()
            s = System(seed=seed, **kw).fit(V, tr, rf)
            sc = s.score(V, te)
            m = metrics(sc["z"]["ASTRA"], y[te]); m.update(setting=name, seed=seed, sec=time.time() - t,
                                                           wsd=float(sc["W"].std(0).mean()), **s.router.params(s.names))
            out.append(m); print("sens", seed, name, round(m["macro_f1"], 4), round(m["nll"], 4), round(time.time() - t), flush=True)
            json.dump(out, open(f"{W}/results_sensitivity.json", "w"), default=float)


def latency():
    """Per-item wall-clock cost of each stage on 2 CPU threads (batch of 400 BotSim test items)."""
    V = load("botsim"); I = V.items
    tr, rf, ca, te = account_split(I, 0)
    s = System(seed=0).fit(V, tr, rf)
    idx = te[:400]; n = len(idx); glove, scorer = Glove(), Scorer(48)
    texts = list(V.text["orig"][idx])
    t = time.time(); pert = [perturb_all(x, k, glove) for k, x in enumerate(texts)]; t_pert = (time.time() - t) / n
    t = time.time(); scorer.score(texts, bs=16); t_gpt = (time.time() - t) / n
    t = time.time(); scorer.score([p for ps in pert for p in ps], bs=16); t_gpt_p = (time.time() - t) / n
    s.K = 0; s.bank.K = 0
    t = time.time(); s.score(V, idx); t_exp0 = (time.time() - t) / n
    s.K = 4; s.bank.K = 4
    t = time.time(); s.score(V, idx); t_exp4 = (time.time() - t) / n
    res = dict(perturbation_ms=1e3 * t_pert, gpt2_orig_ms=1e3 * t_gpt, gpt2_pert_ms=1e3 * t_gpt_p,
               experts_router_no_pert_ms=1e3 * t_exp0, experts_router_full_ms=1e3 * t_exp4)
    res["astra_total_ms"] = res["perturbation_ms"] + res["gpt2_orig_ms"] + res["gpt2_pert_ms"] + res["experts_router_full_ms"]
    res["stacking_total_ms"] = res["gpt2_orig_ms"] + res["experts_router_no_pert_ms"]
    print(res); json.dump(res, open(f"{W}/results_latency.json", "w"))


if __name__ == "__main__":
    {"attacks": attacks, "logo": logo, "transfer": transfer, "sensitivity": sensitivity, "latency": latency}[sys.argv[1]]()
