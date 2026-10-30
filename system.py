"""A fitted ASTRA system (plus the ensemble baselines on the same experts) that can score any item set.
Used by the attack, held-out-generator, transfer, sensitivity and latency experiments."""
import numpy as np, pandas as pd
from scipy.special import expit
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from astra_lib import *
from run_main import mats, shapvar_weights, logit, fit_gate


def rspace(V, idx, pca, sc, ctx):
    U = pca.transform(V.emb["orig"][idx])
    if ctx:
        U = np.c_[U, 3.0 * np.c_[V.items.beh_log_nprev.values[idx], np.log1p(V.items.gr_out_int.values[idx])]]
    return U if sc is None else sc.transform(U)


class System:
    def __init__(self, seed=0, experts=None, K=4, r=10, k_nn=50, rbo_p=0.9):
        self.seed, self.experts, self.K, self.r, self.k, self.p = seed, experts, K, r, k_nn, rbo_p

    def fit(self, V, tr, rf, ca=None):
        self.V = V; y = V.items.label.values; acc = V.items.account.values
        self.bank = ExpertBank(V, seed=self.seed, experts=self.experts, r=self.r, K=self.K, rbo_p=self.p).fit(tr)
        self.names = self.bank.names; self.ctx = "BEH" in self.names
        sig = self.bank.signals(rf, pairwise=True)
        Z, S, D = mats(sig, self.names, "z"), mats(sig, self.names, "s"), mats(sig, self.names, "d")
        self.pca = PCA(16, random_state=0).fit(V.emb["orig"][tr])
        U = rspace(V, rf, self.pca, None, self.ctx); self.sc = StandardScaler().fit(U); self.U_rf = self.sc.transform(U)
        self.acc_rf, self.y_rf = acc[rf], y[rf]
        self.ll_rf = loglik(Z, self.y_rf[:, None])
        R = local_reliability(self.U_rf, self.acc_rf, self.ll_rf, self.U_rf, self.acc_rf, k=self.k)
        Za, Ra, Sa, Da, ya = augmented_router_data(V, self.bank, sig, rf, R, self.pca, self.sc, self.U_rf, self.acc_rf,
                                                   self.ll_rf, k_nn=self.k)
        self.router = Router().fit(Za, Ra, Sa, Da, ya)
        self.router_clean = Router().fit(Z, R, S, D, self.y_rf)
        self.static = Router(False, False, False).fit(Za, Ra, Sa, Da, ya)
        self.stack = LogisticRegression(C=1.0, max_iter=2000).fit(Z, self.y_rf)
        self.stack_aug = LogisticRegression(C=1.0, max_iter=2000).fit(Za, ya)
        self.sw = shapvar_weights(self.bank, sig)
        self.gate = fit_gate(self.U_rf, Z, self.y_rf)
        self.thr = None
        if ca is not None and len(ca):
            out = self.score(V, ca)
            rng = np.random.default_rng(self.seed); sub = one_per_account(acc[ca], rng)
            self.cal = {k: (expit(v[sub]), y[ca][sub]) for k, v in out["z"].items()}
            self.stab_cal = out["stab"][sub]
        return self

    def score(self, Vq, idx):
        old = self.bank.V; self.bank.V = Vq
        try:
            sig = self.bank.signals(idx)
        finally:
            self.bank.V = old
        Z, S, D = mats(sig, self.names, "z"), mats(sig, self.names, "s"), mats(sig, self.names, "d")
        U = rspace(Vq, idx, self.pca, self.sc, self.ctx)
        R = local_reliability(self.U_rf, self.acc_rf, self.ll_rf, U, k=self.k)
        zA, Wt = self.router.predict(Z, R, S, D)
        z = {"ASTRA": zA, "ASTRA-clean-fit": self.router_clean.predict(Z, R, S, D)[0],
             "Static-learned": self.static.predict(Z, R, S, D)[0], "Stacking": self.stack.decision_function(Z),
             "Stacking-aug": self.stack_aug.decision_function(Z),
             "Equal-vote": Z.mean(1), "SHAP-variance trust": logit((expit(Z) * self.sw).sum(1)),
             "Gated-MoE": self.gate(U, Z)[0]}
        for i, m in enumerate(self.names):
            z[f"Expert:{m}"] = Z[:, i]
        return dict(z=z, W=Wt, S=S, D=D, R=R, Z=Z, stab=(Wt * S).sum(1), sig=sig)

    def thresholds(self, method, alpha, gamma=0.10):
        p, yy = self.cal[method]; return crc_thresholds(p, yy, alpha, gamma)
