"""Builds the item tables used in every experiment.

Datasets
  botsim   : BotSim-24 (Reddit; 1,907 real accounts, 1,000 GPT-4o-mini agent bots). Post/comment level.
  tweepfake: TweepFake (X; 17 human accounts, 23 generative bot accounts). Tweet level.
  chirper  : Masquerade-23 English Chirper.ai LLM-bot posts (transfer positives).
  mastodon : FediData Mastodon posts (transfer, presumed legitimate).

For botsim, behaviour and interaction features are computed causally: only events strictly
before the item timestamp are visible. Simulation artefacts (score, upvote ratio, number of
comments, identifiers) are never used.
"""
import os
import json, re, glob, os, numpy as np, pandas as pd
from collections import defaultdict

RAW = os.environ.get("ASTRA_DATA", os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_raw"))
OUT = os.environ.get("ASTRA_WORK", os.path.dirname(os.path.abspath(__file__)))
os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(0)
CAP_BOTSIM, CAP_TWEEP = 8, 250


def clean(t):
    t = str(t).replace("\r", " ").replace("\n", " ")
    return re.sub(r"\s+", " ", t).strip()


# ---------------------------------------------------------------- BotSim-24
def load_botsim():
    u = pd.read_csv(f"{RAW}/botsim/BotSim-24-Dataset/Users.csv")
    d = json.load(open(f"{RAW}/botsim/BotSim-24-Dataset/user_post_comment.json"))
    lab = {uid: int(i >= 1907) for i, uid in enumerate(u.user_id.astype(str))}
    ev = []  # all events of all accounts
    for a, v in d.items():
        for p in v["posts"]:
            ev.append(dict(account=a, eid="s_" + p["submission_id"], kind="post", text=clean(p["posts"]),
                           time=pd.Timestamp(p["created_utc"]).value / 1e9, sub=p["subreddit"], parent=None))
        for lvl in ("comment_1", "comment_2"):
            for c in v[lvl]:
                par = c["parent_id"]
                par = ("s_" if par.startswith("t3_") else "c_") + par[3:]
                ev.append(dict(account=a, eid="c_" + c["comment_id"], kind="comment", text=clean(c["comment_body"]),
                               time=pd.Timestamp(c["created_utc"]).value / 1e9, sub=c["subreddit"], parent=par))
    ev = pd.DataFrame(ev).sort_values("time").reset_index(drop=True)
    ev["label"] = ev.account.map(lab)
    author = dict(zip(ev.eid, ev.account))
    ev["target"] = ev.parent.map(author)  # author of the item replied to (None if outside the corpus)
    return ev, lab


def behaviour_feats(times, kinds, subs, t_now):
    """Causal temporal-behaviour features from prior events (times strictly < t_now)."""
    n = len(times)
    f = dict(beh_log_nprev=np.log1p(n), beh_hour_now=(t_now % 86400) / 3600.0)
    if n == 0:
        for k in ["beh_since_last", "beh_med_gap", "beh_cv_gap", "beh_burst", "beh_hour_ent", "beh_frac_comment",
                  "beh_n_subs", "beh_top_sub", "beh_fast_frac", "beh_night_frac", "beh_span_days"]:
            f[k] = np.nan
        return f
    t = np.asarray(times)
    f["beh_since_last"] = np.log1p(t_now - t[-1])
    f["beh_span_days"] = (t_now - t[0]) / 86400.0
    gaps = np.diff(np.append(t, t_now))
    mu, sd = gaps.mean(), gaps.std()
    f["beh_med_gap"] = np.log1p(np.median(gaps))
    f["beh_cv_gap"] = sd / (mu + 1e-9)
    f["beh_burst"] = (sd - mu) / (sd + mu + 1e-9)
    hrs = ((t % 86400) // 3600).astype(int)
    p = np.bincount(hrs, minlength=24) / n
    f["beh_hour_ent"] = float(-(p[p > 0] * np.log(p[p > 0])).sum())
    f["beh_night_frac"] = float(np.mean((hrs >= 0) & (hrs < 6)))
    f["beh_frac_comment"] = float(np.mean(np.asarray(kinds) == "comment"))
    vc = pd.Series(subs).value_counts()
    f["beh_n_subs"] = len(vc)
    f["beh_top_sub"] = vc.iloc[0] / n
    f["beh_fast_frac"] = float(np.mean(gaps < 120))
    return f


def graph_feats(a, t_now, out_edges, in_edges, repliers_of):
    """Causal interaction-graph features. out_edges[a] = list of (time, target, parent_item)."""
    oe = [e for e in out_edges.get(a, []) if e[0] < t_now]
    ie = [e for e in in_edges.get(a, []) if e[0] < t_now]
    tg = [e[1] for e in oe]
    src = [e[1] for e in ie]
    f = dict(gr_out_int=len(oe), gr_out_distinct=len(set(tg)), gr_in_int=len(ie), gr_in_distinct=len(set(src)))
    f["gr_repeat"] = len(oe) / (len(set(tg)) + 1e-9) if oe else np.nan
    f["gr_recip"] = len(set(tg) & set(src)) / (len(set(tg)) + 1e-9) if tg else np.nan
    # co-targeting: other accounts that replied to the same items before t_now
    co = defaultdict(int)
    for (tt, _, pit) in oe:
        for (t2, b) in repliers_of.get(pit, []):
            if b != a and t2 < t_now:
                co[b] += 1
    f["gr_cotarget_accounts"] = np.log1p(len(co))
    f["gr_cotarget_max"] = max(co.values()) if co else 0
    f["gr_cotarget_ratio"] = (max(co.values()) / len(oe)) if (co and oe) else np.nan
    f["gr_reply_to_post"] = np.mean([p.startswith("s_") for (_, _, p) in oe]) if oe else np.nan
    return f


def build_botsim():
    ev, lab = load_botsim()
    ev = ev[ev.text.str.len() >= 3]
    ev = ev[~ev.text.isin(["[deleted]", "[removed]"])].reset_index(drop=True)
    by_acc = {a: g for a, g in ev.groupby("account")}
    out_edges, in_edges, repliers_of = defaultdict(list), defaultdict(list), defaultdict(list)
    for r in ev.itertuples():
        if r.parent is not None and isinstance(r.target, str) and r.target != r.account:
            out_edges[r.account].append((r.time, r.target, r.parent))
            in_edges[r.target].append((r.time, r.account, r.parent))
        if r.parent is not None:
            repliers_of[r.parent].append((r.time, r.account))
    rows = []
    for a, g in by_acc.items():
        g = g.sort_values("time")
        take = g.sample(n=min(CAP_BOTSIM, len(g)), random_state=int(rng.integers(1e9)))
        T, K, S = g.time.values, g.kind.values, g["sub"].values
        for r in take.itertuples():
            m = T < r.time
            row = dict(item=r.eid, dataset="botsim", account=a, label=lab[a], text=r.text, time=r.time, kind=r.kind,
                       gen="gpt-4o-mini" if lab[a] else "human")
            row.update(behaviour_feats(T[m], K[m], S[m], r.time))
            row.update(graph_feats(a, r.time, out_edges, in_edges, repliers_of))
            # ---- behaviour perturbations: log-normal jitter of prior inter-event gaps
            for k, sig in enumerate([0.25, 0.5, 0.75, 1.0]):
                tp = T[m]
                if len(tp):
                    gaps = np.diff(np.append(tp, r.time))[::-1]
                    gaps = gaps * np.exp(rng.normal(0, sig, len(gaps)))
                    tp = r.time - np.cumsum(gaps)[::-1]
                fb = behaviour_feats(tp, K[m], S[m], r.time)
                row.update({f"{kk}__p{k}": vv for kk, vv in fb.items()})
            # ---- graph perturbations: camouflage replies to random human-authored items
            for k, rate in enumerate([0.25, 0.5, 1.0, 2.0]):
                oe = list(out_edges.get(a, []))
                n_prev = sum(e[0] < r.time for e in oe)
                n_add = int(rng.poisson(rate * max(n_prev, 1))) + 1
                cand = human_items[rng.integers(0, len(human_items), n_add)]
                add = [(r.time - rng.uniform(60, 30 * 86400), human_author[c], c) for c in cand]
                oe2 = dict(out_edges); oe2[a] = oe + add
                fg = graph_feats(a, r.time, oe2, in_edges, repliers_of)
                row.update({f"{kk}__p{k}": vv for kk, vv in fg.items()})
            rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- TweepFake
def build_tweepfake():
    t = pd.concat([pd.read_csv(f"{RAW}/tweepfake_deepfake_text_detection/data/splits/{s}.csv", sep=";")
                   for s in ["train", "validation", "test"]])
    t["text"] = t.text.map(clean)
    t = t[t.text.str.len() >= 3]
    out = []
    for a, g in t.groupby("screen_name"):
        g = g.sample(n=min(CAP_TWEEP, len(g)), random_state=int(rng.integers(1e9)))
        for i, r in enumerate(g.itertuples()):
            out.append(dict(item=f"tf_{a}_{i}", dataset="tweepfake", account=a, label=int(r._3 == "bot"),
                            text=r.text, gen=r.class_type))
    return pd.DataFrame(out)


# ---------------------------------------------------------------- transfer sets
def build_chirper(n_per_user=3):
    out = []
    for line in open(f"{RAW}/Masquerade-23/SDUR/chirper_user_en.json"):
        if not line.strip():
            continue
        u = json.loads(line)
        tw = [clean(x) for x in (u.get("tweet_content") or []) if x and len(clean(x)) > 10]
        for i, x in enumerate(tw[:n_per_user]):
            out.append(dict(item=f"ch_{u['user_herf'][-12:]}_{i}", dataset="chirper", account=u["user_herf"],
                            label=1, text=x, gen="chirper-llm"))
    return pd.DataFrame(out)


def build_mastodon(n=1500):
    from nltk.corpus import stopwords
    sw = set(stopwords.words("english"))
    rows = []
    for f in sorted(glob.glob(f"{RAW}/FediData/dataset/batch_files/*_request_mapping.json")):
        m = json.load(open(f))
        for k, v in m.items():
            x = clean(v.get("tweet", ""))
            w = re.findall(r"[a-zA-Z']+", x.lower())
            if len(w) >= 6 and np.mean([z in sw for z in w]) > 0.25:
                rows.append(dict(item=f"md_{v['instance']}_{v['tweet_index']}", dataset="mastodon",
                                 account=v["instance"], label=0, text=x[:600], gen="human-presumed"))
    df = pd.DataFrame(rows).drop_duplicates("text")
    return df.sample(n=min(n, len(df)), random_state=0)


if __name__ == "__main__":
    ev, lab = load_botsim()
    hum = ev[(ev.label == 0)]
    human_items = hum.eid.values
    human_author = dict(zip(hum.eid, hum.account))
    b = build_botsim(); print("botsim", b.shape, b.label.mean(), b.account.nunique())
    tf = build_tweepfake(); print("tweepfake", tf.shape, tf.label.mean(), tf.account.nunique())
    ch = build_chirper(); print("chirper", ch.shape)
    md = build_mastodon(); print("mastodon", md.shape, md.account.nunique())
    b.to_pickle(f"{OUT}/botsim_items.pkl")
    pd.concat([tf, ch, md], ignore_index=True).to_pickle(f"{OUT}/text_items.pkl")
