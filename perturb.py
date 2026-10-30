"""Semantics-preserving perturbation bank for text views.

  syn   : embedding-constrained synonym substitution (WordNet candidates, GloVe-Twitter cosine >= 0.5)
  homo  : homoglyph substitution (Latin letters -> visually identical Cyrillic/Greek code points)
  typo  : adjacent-character swaps inside words (DeepWordBug-style)
  link  : shortener/URL rotation plus benign hashtag/emoji padding
"""
import os
import re, gzip, numpy as np
from nltk.corpus import wordnet as wn, stopwords
import nltk

SW = set(stopwords.words("english"))
HOMO = {"a": "а", "c": "с", "e": "е", "o": "о", "p": "р", "x": "х", "y": "у", "i": "і", "s": "ѕ", "j": "ј",
        "A": "А", "B": "В", "C": "С", "E": "Е", "H": "Н", "K": "К", "M": "М", "O": "О", "P": "Р", "T": "Т", "X": "Х"}
SHORT = ["bit.ly", "tinyurl.com", "t.co", "ow.ly", "is.gd", "buff.ly", "rebrand.ly"]
PAD = ["#news", "#update", "#today", "#community", "#thoughts", "#trending", "🙂", "👍", "🔥", "✨", "📌"]
URL = re.compile(r"https?://\S+|www\.\S+")
POSMAP = {"N": wn.NOUN, "V": wn.VERB, "J": wn.ADJ, "R": wn.ADV}


class Glove:
    def __init__(self, path=os.path.join(os.environ.get("ASTRA_MODELS", "models"), "glove-twitter-100.gz"), limit=400000):
        self.idx, vecs = {}, []
        with gzip.open(path, "rt", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i == 0 and len(line.split()) == 2:
                    continue
                p = line.rstrip().split(" ")
                if len(p) != 101:
                    continue
                self.idx[p[0]] = len(vecs); vecs.append(np.asarray(p[1:], dtype=np.float32))
                if len(vecs) >= limit:
                    break
        V = np.stack(vecs); self.V = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-9)

    def cos(self, a, b):
        ia, ib = self.idx.get(a), self.idx.get(b)
        if ia is None or ib is None:
            return -1.0
        return float(self.V[ia] @ self.V[ib])


_syn_cache = {}


def synonyms(w, pos, glove, thr=0.5):
    key = (w, pos)
    if key in _syn_cache:
        return _syn_cache[key]
    cands = set()
    for s in wn.synsets(w, pos=POSMAP.get(pos[0])):
        for l in s.lemmas():
            n = l.name().replace("_", " ").lower()
            if n != w and " " not in n and n.isalpha():
                cands.add(n)
    out = sorted([c for c in cands if glove.cos(w, c) >= thr], key=lambda c: -glove.cos(w, c))[:5]
    _syn_cache[key] = out
    return out


def _match_case(src, dst):
    if src.isupper():
        return dst.upper()
    if src[:1].isupper():
        return dst[:1].upper() + dst[1:]
    return dst


def p_syn(text, rng, glove, rate=0.25):
    toks = re.findall(r"\S+|\s+", text)
    words = [(i, t) for i, t in enumerate(toks) if re.fullmatch(r"[A-Za-z]{4,}", t) and t.lower() not in SW]
    if not words:
        return text
    tags = nltk.pos_tag([t for _, t in words])
    elig = [(i, t, p) for (i, t), (_, p) in zip(words, tags) if p[0] in POSMAP]
    rng.shuffle(elig)
    budget = max(1, int(round(rate * len(words))))
    for i, t, p in elig:
        if budget == 0:
            break
        s = synonyms(t.lower(), p, glove)
        if s:
            toks[i] = _match_case(t, s[int(rng.integers(len(s)))]); budget -= 1
    return "".join(toks)


def p_homo(text, rng, rate=0.15):
    ch = list(text)
    pos = [i for i, c in enumerate(ch) if c in HOMO and not URL.search(text[max(0, i - 30):i + 1] or "")]
    k = max(1, int(round(rate * len(pos)))) if pos else 0
    for i in rng.choice(pos, size=k, replace=False) if k else []:
        ch[i] = HOMO[ch[i]]
    return "".join(ch)


def p_typo(text, rng, rate=0.15):
    toks = re.findall(r"\S+|\s+", text)
    idx = [i for i, t in enumerate(toks) if re.fullmatch(r"[A-Za-z]{4,}", t)]
    k = max(1, int(round(rate * len(idx)))) if idx else 0
    for i in rng.choice(idx, size=k, replace=False) if k else []:
        w = list(toks[i]); j = int(rng.integers(1, len(w) - 2)); w[j], w[j + 1] = w[j + 1], w[j]
        toks[i] = "".join(w)
    return "".join(toks)


def p_link(text, rng):
    def rot(_m):
        slug = "".join(rng.choice(list("abcdefghijkmnpqrstuvwxyz23456789"), size=7))
        return f"https://{SHORT[int(rng.integers(len(SHORT)))]}/{slug}"
    t = URL.sub(rot, text)
    pads = list(rng.choice(PAD, size=int(rng.integers(1, 3)), replace=False))
    return (t + " " + " ".join(pads)).strip()


PERTS = ["syn", "homo", "typo", "link"]


def perturb_all(text, seed, glove):
    rng = np.random.default_rng(seed)
    return [p_syn(text, rng, glove), p_homo(text, rng), p_typo(text, rng), p_link(text, rng)]


LEET = {"a": "4", "e": "3", "i": "1", "o": "0", "s": "5", "t": "7", "A": "4", "E": "3", "I": "1", "O": "0", "S": "5", "T": "7"}


def p_leet(text, rng, rate=0.15):
    """Held-out evasion family (not in the bank): digit-for-letter substitution."""
    ch = list(text)
    pos = [i for i, c in enumerate(ch) if c in LEET]
    k = int(round(rate * len(pos)))
    for i in (rng.choice(pos, size=k, replace=False) if k else []):
        ch[i] = LEET[ch[i]]
    return "".join(ch)
