"""ASTRA core: experts with exact additive attributions, attribution-stability and local-reliability
signals, the trust router, policy-concept attribution and conformal moderation-risk control.

Notation follows the manuscript: expert m outputs a logit z_m(x) = phi_{m,0} + sum_j phi_{m,j}(x);
the router returns weights w_m(x) and the decision logit z(x) = gamma_0 + tau * sum_m w_m(x) z_m(x).
"""
import os
import re, numpy as np, pandas as pd, lightgbm as lgb
from scipy.optimize import minimize
from scipy.special import expit, log_softmax, softmax
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import (f1_score, average_precision_score, roc_auc_score, matthews_corrcoef,
                             balanced_accuracy_score)

PERTS_TEXT = ["syn", "homo", "typo", "link"]
URL = re.compile(r"https?://\S+|www\.\S+")
SHORTENERS = ("bit.ly", "tinyurl", "t.co/", "ow.ly", "is.gd", "buff.ly", "rebrand.ly", "goo.gl")
CTA = re.compile(r"\b(click|follow|free|win|giveaway|subscribe|buy|deal|offer|limited|now|link|dm|join|check out|"
                 r"sign up|earn|promo|discount|retweet|rt)\b", re.I)
EMOJI = re.compile("[\U0001F300-\U0001FAFF☀-➿]")

STYLE_NAMES = ["st_log_chars", "st_log_words", "st_n_url", "st_shortener", "st_n_hashtag", "st_n_mention",
               "st_n_emoji", "st_upper_ratio", "st_digit_ratio", "st_punct_ratio", "st_n_exclam", "st_n_quest",
               "st_ttr", "st_mean_wlen", "st_cta", "st_nonascii_ratio", "st_char_runs"]
BEH_NAMES = ["beh_log_nprev", "beh_hour_now", "beh_since_last", "beh_med_gap", "beh_cv_gap", "beh_burst",
             "beh_hour_ent", "beh_frac_comment", "beh_n_subs", "beh_top_sub", "beh_fast_frac", "beh_night_frac",
             "beh_span_days"]
GR_NAMES = ["gr_out_int", "gr_out_distinct", "gr_in_int", "gr_in_distinct", "gr_repeat", "gr_recip",
            "gr_cotarget_accounts", "gr_cotarget_max", "gr_cotarget_ratio", "gr_reply_to_post"]

# ---- policy-aligned concepts (Section III-E) ----
CONCEPTS = ["Synthetic authorship", "Automated cadence", "Coordinated interaction", "Link redirection",
            "Engagement bait", "Stylometric form", "Topical content"]


def style_vector(t):
    w = t.split()
    n = max(len(t), 1)
    letters = [c for c in t if c.isalpha()]
    return np.array([
        np.log1p(len(t)), np.log1p(len(w)), len(URL.findall(t)), float(any(s in t for s in SHORTENERS)),
        t.count("#"), t.count("@"), len(EMOJI.findall(t)),
        sum(c.isupper() for c in letters) / max(len(letters), 1), sum(c.isdigit() for c in t) / n,
        sum((not c.isalnum()) and (not c.isspace()) for c in t) / n, t.count("!"), t.count("?"),
        len(set(x.lower() for x in w)) / max(len(w), 1), np.mean([len(x) for x in w]) if w else 0.0,
        len(CTA.findall(t)), sum(ord(c) > 127 for c in t if c.isalpha()) / max(len(letters), 1),
        len(re.findall(r"(.)\1{2,}", t))], dtype=np.float32)


def style_concept(name):
    if name in ("st_n_url", "st_shortener"):
        return 3
    if name in ("st_n_hashtag", "st_n_mention", "st_n_emoji", "st_upper_ratio", "st_n_exclam", "st_n_quest",
                "st_cta", "st_nonascii_ratio", "st_char_runs", "st_punct_ratio"):
        return 4
    return 5


def lex_concept(tok):
    t = tok.strip().lower()
    if any(s in t for s in ("http", "www", ".com", "t.co", "bit.ly", "://")):
        return 3
    if t.startswith("#") or t.startswith("@") or CTA.search(t):
        return 4
    return 6


# =============================================================== experts
class LinearExpert:
    """Logistic regression; attribution phi_j = w_j * x_j against an all-zero (empty / mean) reference."""
    def __init__(self, C=1.0, dense=True):
        self.C, self.dense = C, dense

    def fit(self, X, y):
        if self.dense:
            self.sc = StandardScaler().fit(X); X = self.sc.transform(X)
        self.clf = LogisticRegression(C=self.C, max_iter=3000, solver="liblinear" if not self.dense else "lbfgs")
        self.clf.fit(X, y); self.w = self.clf.coef_.ravel(); self.b = float(self.clf.intercept_[0])
        return self

    def _t(self, X):
        return self.sc.transform(X) if self.dense else X

    def logit(self, X):
        return np.asarray(self._t(X) @ self.w).ravel() + self.b

    def attrib(self, X):
        Xt = self._t(X)
        return Xt.multiply(self.w).tocsr() if sparse.issparse(Xt) else Xt * self.w


class TreeExpert:
    """LightGBM; attribution by exact TreeSHAP in log-odds space (pred_contrib)."""
    def __init__(self, seed=0, n=300, leaves=15):
        self.p = dict(objective="binary", learning_rate=0.05, num_leaves=leaves, min_child_samples=20,
                      subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0, verbose=-1, seed=seed,
                      num_threads=2)
        self.n = n

    def fit(self, X, y):
        self.m = lgb.train(self.p, lgb.Dataset(X, y), num_boost_round=self.n); return self

    def logit(self, X):
        return self.m.predict(X, raw_score=True)

    def attrib(self, X):
        return self.m.predict(X, pred_contrib=True)[:, :-1]


def topr_signed(phi_row, r):
    """Top-r features by |phi| with sign, as an ordered list of (index, sign)."""
    if sparse.issparse(phi_row):
        idx, val = phi_row.indices, phi_row.data
    else:
        val = np.asarray(phi_row).ravel(); idx = np.arange(len(val))
    if len(val) == 0:
        return []
    o = np.argsort(-np.abs(val))[:r]
    return [(int(idx[i]), int(np.sign(val[i]))) for i in o if val[i] != 0]


def rbo(a, b, p=0.9):
    """Rank-biased overlap of two top-r lists (extrapolated, Webber et al.)."""
    if not a and not b:
        return 1.0
    r = max(len(a), len(b)); sa, sb = set(), set(); s = 0.0
    for d in range(1, r + 1):
        if d <= len(a): sa.add(a[d - 1])
        if d <= len(b): sb.add(b[d - 1])
        s += (p ** (d - 1)) * len(sa & sb) / d
    x_r = len(sa & sb)
    return float((1 - p) * s + (p ** r) * x_r / r)


# =============================================================== view assembly
class Views:
    """Builds the per-expert matrices for original and perturbed versions of each item."""
    def __init__(self, items, texts, auth, emb, has_context):
        self.items = items.reset_index(drop=True)
        self.has_context = has_context
        key = texts.item.values + "|" + texts.version.values
        pos = pd.Series(np.arange(len(texts)), index=key)
        self.rows = {v: pos.reindex(self.items.item.values + "|" + v).values for v in ["orig"] + PERTS_TEXT}
        self.text = {v: texts.text.values[self.rows[v]] for v in self.rows}
        self.auth = {v: auth[self.rows[v]] for v in self.rows}
        self.emb = {v: emb[self.rows[v]].astype(np.float32) for v in self.rows}
        self.style = {v: np.stack([style_vector(t) for t in self.text[v]]) for v in self.rows}
        if has_context:
            I = self.items
            self.beh = {"orig": I[BEH_NAMES].values.astype(np.float32)}
            self.gr = {"orig": I[GR_NAMES].values.astype(np.float32)}
            for k in range(4):
                self.beh[f"p{k}"] = I[[f"{c}__p{k}" for c in BEH_NAMES]].values.astype(np.float32)
                self.gr[f"p{k}"] = I[[f"{c}__p{k}" for c in GR_NAMES]].values.astype(np.float32)

    def expert_names(self):
        return ["LEX", "SEM", "AUTH", "STYLE"] + (["BEH", "GRAPH"] if self.has_context else [])

    def versions(self, m):
        return ["orig"] + (PERTS_TEXT if m in ("LEX", "SEM", "AUTH", "STYLE") else [f"p{k}" for k in range(4)])

    def X(self, m, v, idx, tfidf=None):
        if m == "LEX":
            return tfidf.transform(self.text[v][idx])
        if m == "SEM":
            return self.emb[v][idx]
        if m == "AUTH":
            return self.auth[v][idx]
        if m == "STYLE":
            return self.style[v][idx]
        if m == "BEH":
            return self.beh[v][idx]
        if m == "GRAPH":
            return self.gr[v][idx]


class LexVectorizer:
    def __init__(self):
        self.w = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=40000, sublinear_tf=True, lowercase=True)
        self.c = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=3, max_features=40000,
                                 sublinear_tf=True, lowercase=False)

    def fit(self, t):
        self.w.fit(t); self.c.fit(t)
        self.names = list(self.w.get_feature_names_out()) + list(self.c.get_feature_names_out())
        return self

    def transform(self, t):
        return sparse.hstack([self.w.transform(t), self.c.transform(t)]).tocsr()


# =============================================================== expert bank
class ExpertBank:
    def __init__(self, V, seed=0, experts=None, r=10, rbo_p=0.9, K=4):
        self.V, self.seed, self.r, self.p, self.K = V, seed, r, rbo_p, K
        self.names = experts or V.expert_names()

    def fit(self, tr):
        V, y = self.V, self.V.items.label.values[tr]
        self.tfidf = LexVectorizer().fit(V.text["orig"][tr]) if "LEX" in self.names else None
        self.models = {}
        for m in self.names:
            X = V.X(m, "orig", tr, self.tfidf)
            if m == "LEX":
                mod = LinearExpert(C=4.0, dense=False)
            elif m == "SEM":
                mod = LinearExpert(C=0.05, dense=True)
            else:
                mod = TreeExpert(seed=self.seed)
            self.models[m] = mod.fit(X, y)
        self.feature_names = {"LEX": self.tfidf.names if self.tfidf else [],
                              "SEM": [f"emb_{i}" for i in range(768)], "AUTH": list(AUTH_NAMES),
                              "STYLE": STYLE_NAMES, "BEH": BEH_NAMES, "GRAPH": GR_NAMES}
        self.concept_of = {m: self._concepts(m) for m in self.names}
        return self

    def _concepts(self, m):
        fn = self.feature_names[m]
        if m == "LEX": return np.array([lex_concept(t) for t in fn])
        if m == "SEM": return np.full(len(fn), 6)
        if m == "AUTH": return np.full(len(fn), 0)
        if m == "STYLE": return np.array([style_concept(t) for t in fn])
        if m == "BEH": return np.full(len(fn), 1)
        if m == "GRAPH": return np.full(len(fn), 2)

    def _csum(self, m, phi):
        cm = self.concept_of[m]; C = np.zeros((phi.shape[0], len(CONCEPTS)))
        for k in range(len(CONCEPTS)):
            cols = np.where(cm == k)[0]
            if len(cols):
                C[:, k] = np.asarray(phi[:, cols].sum(1)).ravel()
        return C

    def signals(self, idx, versions_override=None, pairwise=False):
        """Logits, attributions (orig), attribution stability s_m and prediction shift d_m for items idx.
        With pairwise=True the stability and shift of every perturbed version relative to its siblings are also
        returned (used for stress-augmented router fitting)."""
        out = {}
        for m in self.names:
            mod = self.models[m]
            vers = self.V.versions(m)[: 1 + self.K]
            X0 = self.V.X(m, "orig", idx, self.tfidf)
            z0 = mod.logit(X0); phi0 = mod.attrib(X0)
            tops0 = [topr_signed(phi0[i], self.r) for i in range(len(idx))]
            all_tops, all_z = [tops0], [z0]
            S = np.zeros(len(idx)); D = np.zeros(len(idx)); zs = [z0]; cs = [self._csum(m, phi0)]; S_all, D_all = [], []
            for v in vers[1:]:
                Xv = self.V.X(m, v, idx, self.tfidf)
                zv = mod.logit(Xv); phv = mod.attrib(Xv); zs.append(zv); cs.append(self._csum(m, phv))
                topsv = [topr_signed(phv[i], self.r) for i in range(len(idx))]
                if pairwise:
                    all_tops.append(topsv); all_z.append(zv)
                sv = np.array([rbo(tops0[i], topsv[i], self.p) for i in range(len(idx))])
                S += sv; D += np.abs(zv - z0); S_all.append(sv); D_all.append(np.abs(zv - z0))
            nv = max(len(vers) - 1, 1)
            S_ver = D_ver = None
            if pairwise and len(all_tops) > 1:
                V_ = len(all_tops); n = len(idx)
                RB = np.ones((V_, V_, n))
                for u in range(V_):
                    for v in range(u + 1, V_):
                        RB[u, v] = RB[v, u] = [rbo(all_tops[u][i], all_tops[v][i], self.p) for i in range(n)]
                Zv = np.stack(all_z)  # [V, n]
                S_ver = np.stack([(RB[v].sum(0) - 1) / (V_ - 1) for v in range(V_)], 1)
                D_ver = np.stack([np.abs(Zv[v] - Zv).sum(0) / (V_ - 1) for v in range(V_)], 1)
            out[m] = dict(z=z0, phi=phi0, s=S / nv, d=D / nv, z_pert=np.stack(zs[1:], 1) if len(zs) > 1 else None,
                          phi0=getattr(mod, "b", None), csum=cs,
                          S_all=np.stack(S_all, 1) if S_all else None, D_all=np.stack(D_all, 1) if D_all else None,
                          S_ver=S_ver, D_ver=D_ver)
        return out


AUTH_NAMES = ["au_mean_lp", "au_std_lp", "au_q10_lp", "au_q90_lp", "au_mean_ent", "au_xppl_ratio",
              "au_mean_logrank", "au_top1", "au_top10", "au_max_surprise", "au_ntok"]


# =============================================================== local reliability
def routing_space(V, idx, pca=None, sc=None, version="orig"):
    E = V.emb[version][idx]
    if pca is None:
        pca = PCA(16, random_state=0).fit(E)
    U = pca.transform(E)
    if V.has_context:
        ctx = np.c_[V.items.beh_log_nprev.values[idx], np.log1p(V.items.gr_out_int.values[idx])]
        U = np.c_[U, 3.0 * ctx]
    if sc is None:
        sc = StandardScaler().fit(U)
    return sc.transform(U), pca, sc


def loglik(z, y):
    return y * np.log(expit(z) + 1e-7) + (1 - y) * np.log(1 - expit(z) + 1e-7)


def local_reliability(U_ref, acc_ref, ll_ref, U_q, acc_q=None, k=50):
    """rho_m(x): mean log-likelihood of each expert over the k nearest reference items (other accounts)."""
    extra = 0 if acc_q is None else int(pd.Series(acc_ref).value_counts().iloc[0])
    nn = NearestNeighbors(n_neighbors=min(k + extra + 5, len(U_ref))).fit(U_ref)
    dist, ind = nn.kneighbors(U_q)
    out = np.zeros((len(U_q), ll_ref.shape[1])); glob = ll_ref.mean(0)
    for i in range(len(U_q)):
        sel = ind[i]
        if acc_q is not None:
            sel = sel[acc_ref[sel] != acc_q[i]]
        sel = sel[:k]
        out[i] = ll_ref[sel].mean(0) if len(sel) else glob
    return out


# =============================================================== router
class Router:
    """w_m(x) = softmax_m(beta_m + a*rho_m(x) + b*s_m(x) - c*d_m(x)); z = g0 + tau * sum_m w_m z_m."""
    def __init__(self, use_rho=True, use_s=True, use_d=True, l2=1e-3):
        self.use = (use_rho, use_s, use_d); self.l2 = l2

    def _unpack(self, th, M):
        beta = th[:M]; a, b, c, tau, g0 = th[M:M + 5]
        a, b, c = a * self.use[0], b * self.use[1], c * self.use[2]
        return beta, a, b, c, tau, g0

    def weights(self, th, Z, R, S, D):
        beta, a, b, c, tau, g0 = self._unpack(th, Z.shape[1])
        return softmax(beta + a * R + b * S - c * D, axis=1)

    def _loss(self, th, Z, R, S, D, y):
        beta, a, b, c, tau, g0 = self._unpack(th, Z.shape[1])
        W = softmax(beta + a * R + b * S - c * D, axis=1)
        z = g0 + tau * (W * Z).sum(1)
        return -loglik(z, y).mean() + self.l2 * (th ** 2).sum()

    def _std(self, R, S, D):
        """Per-expert standardisation of the trust signals with router-fit statistics."""
        return tuple((X - mu) / sd for X, (mu, sd) in zip((R, S, D), self.norm))

    def fit(self, Z, R, S, D, y):
        self.norm = [(X.mean(0), X.std(0) + 1e-6) for X in (R, S, D)]
        R, S, D = self._std(R, S, D)
        M = Z.shape[1]; th0 = np.r_[np.zeros(M), 1.0, 1.0, 0.1, 1.0, 0.0]
        best = None
        for init in (th0, np.r_[np.zeros(M), 0.0, 0.0, 0.0, 1.0, 0.0], np.r_[np.zeros(M), 0.5, 0.5, 0.5, 1.0, 0.0]):
            r = minimize(self._loss, init, args=(Z, R, S, D, y), method="L-BFGS-B")
            if best is None or r.fun < best.fun:
                best = r
        self.th = best.x; return self

    def predict(self, Z, R, S, D):
        R, S, D = self._std(R, S, D)
        beta, a, b, c, tau, g0 = self._unpack(self.th, Z.shape[1])
        W = self.weights(self.th, Z, R, S, D)
        return g0 + tau * (W * Z).sum(1), W

    def params(self, names):
        beta, a, b, c, tau, g0 = self._unpack(self.th, len(names))
        d = {f"beta_{n}": float(v) for n, v in zip(names, beta)}
        d.update(a=float(a), b=float(b), c=float(c), tau=float(tau), g0=float(g0)); return d


# =============================================================== concept attribution
def concept_attribution(bank, sig, W, tau, idx_local=None):
    """C_k(x) = tau * sum_m w_m(x) sum_{j in G_{m,k}} phi_{m,j}(x). Returns [n, K]."""
    n = W.shape[0]; C = np.zeros((n, len(CONCEPTS)))
    for mi, m in enumerate(bank.names):
        phi = sig[m]["phi"]; cm = bank.concept_of[m]
        for k in range(len(CONCEPTS)):
            cols = np.where(cm == k)[0]
            if len(cols) == 0:
                continue
            part = phi[:, cols].sum(1)
            part = np.asarray(part).ravel()
            C[:, k] += tau * W[:, mi] * part
    return C


# =============================================================== conformal moderation-risk control
def crc_thresholds(p_cal, y_cal, alpha, gamma):
    """lambda_R: smallest threshold with ((#legit >= lambda) + 1)/(n0 + 1) <= alpha  (wrongful removal);
       lambda_K: largest threshold with ((#bots < lambda) + 1)/(n1 + 1) <= gamma     (missed inauthentic)."""
    p0 = np.sort(p_cal[y_cal == 0]); p1 = np.sort(p_cal[y_cal == 1]); n0, n1 = len(p0), len(p1)
    k0 = int(np.floor(alpha * (n0 + 1) - 1))  # number of legit allowed at or above lambda_R
    lam_R = 1.0 + 1e-9 if k0 < 0 else (p0[n0 - k0 - 1] + 1e-12 if k0 < n0 else 0.0)
    k1 = int(np.floor(gamma * (n1 + 1) - 1))  # number of bots allowed below lambda_K
    lam_K = 0.0 if k1 < 0 else (p1[k1] if k1 < n1 else 1.0)
    return lam_R, lam_K


def moderate(p, lam_R, lam_K, stab=None, tau_s=None):
    """Three-way decision: 2 = remove, 1 = defer to human review, 0 = keep."""
    dec = np.where(p >= lam_R, 2, np.where(p < lam_K, 0, 1))
    if stab is not None and tau_s is not None:
        dec = np.where((dec == 2) & (stab < tau_s), 1, dec)
    return dec


def one_per_account(acc, rng):
    s = pd.Series(np.arange(len(acc)), index=acc)
    return np.array([rng.choice(g.values) for _, g in s.groupby(level=0)])


# =============================================================== metrics
def ece(p, y, bins=15):
    e = 0.0; edges = np.linspace(0, 1, bins + 1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & (p < hi) if hi < 1 else (p >= lo) & (p <= hi)
        if m.any():
            e += m.mean() * abs(p[m].mean() - y[m].mean())
    return e


def metrics(z, y):
    p = expit(z); yh = (p >= 0.5).astype(int)
    return dict(macro_f1=f1_score(y, yh, average="macro"), pr_auc=average_precision_score(y, p),
                roc_auc=roc_auc_score(y, p), mcc=matthews_corrcoef(y, yh), bal_acc=balanced_accuracy_score(y, yh),
                brier=float(np.mean((p - y) ** 2)), nll=float(-loglik(z, y).mean()), ece=ece(p, y))


def augmented_router_data(V, bank, sig, idx, R0, pca, sc, U_ref, acc_ref, ll_ref, k_nn=50):
    """Stress-augmented router-fit set: the clean posts plus each of their K perturbed versions, with signals of a
    perturbed version measured against its sibling versions (Section III-E)."""
    names = bank.names; y = V.items.label.values[idx]; acc = V.items.account.values[idx]
    Z0 = np.stack([sig[m]["z"] for m in names], 1); S0 = np.stack([sig[m]["s"] for m in names], 1)
    D0 = np.stack([sig[m]["d"] for m in names], 1)
    Zs, Ss, Ds, Rs, ys = [Z0], [S0], [D0], [R0], [y]
    K = sig[names[0]]["z_pert"].shape[1]
    for k in range(K):
        Zs.append(np.stack([sig[m]["z_pert"][:, k] for m in names], 1))
        Ss.append(np.stack([sig[m]["S_ver"][:, k + 1] for m in names], 1))
        Ds.append(np.stack([sig[m]["D_ver"][:, k + 1] for m in names], 1))
        Uk = routing_space(V, idx, pca, sc, version=PERTS_TEXT[k])[0]
        Rs.append(local_reliability(U_ref, acc_ref, ll_ref, Uk, acc, k=k_nn)); ys.append(y)
    return np.concatenate(Zs), np.concatenate(Rs), np.concatenate(Ss), np.concatenate(Ds), np.concatenate(ys)
