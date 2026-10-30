"""Runs the perturbation bank and frozen GPT-2 over every text (original + 4 perturbations).

Outputs work/texts.pkl (item, version, text) and work/gpt2_feats.npz with
  auth[N,11]: token-statistics authorship features, emb[N,768] (float16) mean-pooled hidden states.
"""
import os
import sys, time, numpy as np, pandas as pd
from perturb import Glove, perturb_all, PERTS
from gpt2_onnx import Scorer

W = os.environ.get("ASTRA_WORK", os.path.dirname(os.path.abspath(__file__)))
AUTH_NAMES = ["au_mean_lp", "au_std_lp", "au_q10_lp", "au_q90_lp", "au_mean_ent", "au_xppl_ratio",
              "au_mean_logrank", "au_top1", "au_top10", "au_max_surprise", "au_ntok"]


def auth_vector(o):
    lp, ent, rank = o["lp"], o["ent"], o["rank"]
    return np.array([lp.mean(), lp.std(), np.quantile(lp, 0.1), np.quantile(lp, 0.9), ent.mean(),
                     (-lp.mean()) / (ent.mean() + 1e-6), np.log1p(rank).mean(), (rank == 0).mean(),
                     (rank < 10).mean(), (-lp).max(), np.log1p(len(lp))], dtype=np.float32)


if __name__ == "__main__":
    b = pd.read_pickle(f"{W}/botsim_items.pkl")[["item", "text"]]
    t = pd.read_pickle(f"{W}/text_items.pkl")[["item", "text"]]
    items = pd.concat([b, t], ignore_index=True)
    import os
    t0 = time.time()
    RESUME = os.path.exists(f"{W}/gpt2_feats_partial.npz") and os.path.exists(f"{W}/texts.pkl") and "--resume" in sys.argv
    rows = []
    glove = None if RESUME else Glove()
    if RESUME:
        texts = pd.read_pickle(f"{W}/texts.pkl")
    else:
        for k, r in enumerate(items.itertuples()):
            rows.append((r.item, "orig", r.text))
            for name, txt in zip(PERTS, perturb_all(r.text, seed=k, glove=glove)):
                rows.append((r.item, name, txt))
        texts = pd.DataFrame(rows, columns=["item", "version", "text"])
        texts.to_pickle(f"{W}/texts.pkl")
    print("perturbed", len(texts), round(time.time() - t0, 1), "s", flush=True)
    sc = Scorer(max_len=48)
    N = len(texts); auth = np.zeros((N, len(AUTH_NAMES)), np.float32); emb = np.zeros((N, 768), np.float16)
    CH = 2048; start = 0
    if RESUME:
        pz = np.load(f"{W}/gpt2_feats_partial.npz"); start = int(pz["done"]); auth[:start] = pz["auth"][:start]; emb[:start] = pz["emb"][:start]
        print("resuming at", start, flush=True)
    for st in range(start, N, CH):
        out = sc.score(texts.text.values[st: st + CH].tolist(), bs=16)
        for j, o in enumerate(out):
            auth[st + j] = auth_vector(o); emb[st + j] = o["emb"].astype(np.float16)
        print(f"{st + CH}/{N} {time.time() - t0:.0f}s", flush=True)
        if (st // CH) % 10 == 0:
            np.savez(f"{W}/gpt2_feats_partial.npz", auth=auth, emb=emb, done=st + CH)
    np.savez(f"{W}/gpt2_feats.npz", auth=auth, emb=emb, names=np.array(AUTH_NAMES))
    print("done", time.time() - t0)
