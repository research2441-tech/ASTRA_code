"""GPT-2 (ONNX) scorer: one forward pass returns token log-probabilities and the final hidden states."""
import os
import json, os, regex as re, numpy as np, onnx, onnxruntime as ort
from functools import lru_cache

MD = os.environ.get("ASTRA_MODELS", os.path.join(os.path.dirname(os.path.abspath(__file__)), "models"))

def bytes_to_unicode():
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]; n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b); cs.append(256 + n); n += 1
    return dict(zip(bs, [chr(c) for c in cs]))

class BPE:
    def __init__(self):
        self.enc = json.load(open(f"{MD}/gp/GPT2/encoder.json"))
        merges = open(f"{MD}/gp/GPT2/vocab.bpe", encoding="utf-8").read().split("\n")[1:-1]
        self.ranks = {tuple(m.split()): i for i, m in enumerate(merges)}
        self.b2u = bytes_to_unicode()
        self.pat = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
        self.cache = {}
    def bpe(self, tok):
        if tok in self.cache: return self.cache[tok]
        word = tuple(tok)
        while len(word) > 1:
            pairs = {(word[i], word[i + 1]) for i in range(len(word) - 1)}
            best = min(pairs, key=lambda p: self.ranks.get(p, 1e12))
            if best not in self.ranks: break
            a, b = best; nw = []; i = 0
            while i < len(word):
                if i < len(word) - 1 and word[i] == a and word[i + 1] == b:
                    nw.append(a + b); i += 2
                else:
                    nw.append(word[i]); i += 1
            word = tuple(nw)
        self.cache[tok] = word
        return word
    def encode(self, text):
        ids = []
        for t in self.pat.findall(text):
            t = "".join(self.b2u[b] for b in t.encode("utf-8"))
            ids.extend(self.enc[p] for p in self.bpe(t) if p in self.enc)
        return ids

def build_session():
    path = f"{MD}/gpt2_hidden_int8.onnx"
    if not os.path.exists(f"{MD}/gpt2_hidden.onnx"):
        m = onnx.load(f"{MD}/gpt2-lm-head-10.onnx")
        vi = onnx.helper.make_tensor_value_info("3085", onnx.TensorProto.FLOAT, None)
        m.graph.output.insert(1, vi)
        onnx.save(m, f"{MD}/gpt2_hidden.onnx")
    if not os.path.exists(path):
        from onnxruntime.quantization import quantize_dynamic, QuantType
        quantize_dynamic(f"{MD}/gpt2_hidden.onnx", path, weight_type=QuantType.QInt8)
    so = ort.SessionOptions(); so.intra_op_num_threads = 2
    s = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
    outs = [o.name for o in s.get_outputs()][:2]
    return s, outs

class Scorer:
    EOS = 50256
    def __init__(self, max_len=48):
        self.bpe = BPE(); self.max_len = max_len
        self.sess, self.outs = build_session()
    def score(self, texts, bs=16):
        """Returns per-text dict of token statistics and a 768-d mean-pooled hidden embedding."""
        res = []
        enc = [([self.EOS] + self.bpe.encode(t))[: self.max_len] for t in texts]
        order = np.argsort([len(e) for e in enc])
        out = [None] * len(texts)
        for st in range(0, len(enc), bs):
            idx = order[st: st + bs]; L = max(len(enc[i]) for i in idx); L = max(L, 2)
            arr = np.full((len(idx), 1, L), self.EOS, dtype=np.int64)
            for r, i in enumerate(idx): arr[r, 0, : len(enc[i])] = enc[i]
            logits, hid = self.sess.run(self.outs, {"input1": arr})
            logits = logits.reshape(len(idx), L, -1); hid = np.asarray(hid).reshape(len(idx), L, -1)
            for r, i in enumerate(idx):
                n = len(enc[i]); ids = enc[i]
                if n < 2:
                    out[i] = dict(lp=np.zeros(1), ent=np.zeros(1), rank=np.zeros(1), emb=hid[r, :1].mean(0)); continue
                lg = logits[r, : n - 1].astype(np.float64)
                lg = lg - lg.max(1, keepdims=True); lse = np.log(np.exp(lg).sum(1))
                logp = lg - lse[:, None]
                tgt = np.array(ids[1:n])
                lp = logp[np.arange(n - 1), tgt]
                p = np.exp(logp); ent = -(p * logp).sum(1)
                rank = (logp > lp[:, None]).sum(1)
                out[i] = dict(lp=lp, ent=ent, rank=rank, emb=hid[r, 1:n].mean(0))
        return out
