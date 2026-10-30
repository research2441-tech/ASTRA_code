"""Fig. 1: ASTRA architecture (grayscale). Every arrow is attached to box edges and checked:
no box overlap (with clearance), arrow endpoints on the declared source/target edges, no arrow
segment passing through a third box, and every text label kept inside its own box."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams.update({"font.family": "DejaVu Serif", "mathtext.fontset": "dejavuserif"})
W, H = 196, 100
fig = plt.figure(figsize=(7.16, 7.16 * H / W), dpi=300)
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")
B, TEXTS = {}, []


def box(key, x, y, w, h, title, lines=(), fill="#FFFFFF", fs=6.0, lfs=5.0, ls="-", gap=3.3, frame=True):
    B[key] = (x, y, w, h)
    if frame:
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.2", lw=0.8, ec="black", fc=fill, ls=ls))
    n = len(lines)
    block = (n * gap)
    ty = y + h / 2 + block / 2
    t = ax.text(x + w / 2, ty, title, ha="center", va="center", fontsize=fs, fontweight="bold"); TEXTS.append((key, t))
    for i, l in enumerate(lines):
        t = ax.text(x + w / 2, ty - gap * (i + 1), l, ha="center", va="center", fontsize=lfs); TEXTS.append((key, t))


def edge(key, side, f=0.5):
    x, y, w, h = B[key]
    return {"L": (x, y + f * h), "R": (x + w, y + f * h), "T": (x + f * w, y + h), "B": (x + f * w, y)}[side]


ARROWS = []


def arrow(src, sside, dst, dside, sf=0.5, df=0.5, label=None, seg=0, dashed=False, via=None, loff=(0, 1.8), lfs=5.0):
    p0, p1 = edge(src, sside, sf), edge(dst, dside, df)
    pts = [p0] + (via or []) + [p1]
    ARROWS.append((src, dst, pts))
    style = dict(color="black", lw=0.8, ls=(0, (3, 2)) if dashed else "-")
    for a, b in zip(pts[:-2], pts[1:-1]):
        ax.plot([a[0], b[0]], [a[1], b[1]], **style)
    ax.add_patch(FancyArrowPatch(pts[-2], pts[-1], arrowstyle="-|>", mutation_scale=7, shrinkA=0, shrinkB=0, **style))
    if label:
        a, b = pts[seg], pts[seg + 1]
        t = ax.text((a[0] + b[0]) / 2 + loff[0], (a[1] + b[1]) / 2 + loff[1], label, ha="center", va="center", fontsize=lfs,
                    bbox=dict(fc="white", ec="none", pad=0.3))
        TEXTS.append((None, t))


# ---------------------------------------------------------------- legend
ax.plot([2, 9], [96, 96], color="black", lw=0.8); ax.text(10.5, 96, "inference path", fontsize=5.2, va="center")
ax.plot([36, 43], [96, 96], color="black", lw=0.8, ls=(0, (3, 2)))
ax.text(44.5, 96, "fitting path (held-out labelled accounts only)", fontsize=5.2, va="center")
# ---------------------------------------------------------------- inputs
box("post", 2, 66, 27, 13, "Incoming post", [r"text $x$, author $a$, time $t$"], fill="#EDEDED")
box("hist", 2, 46, 27, 13, "Account history", [r"events of $a$ before $t$"], fill="#EDEDED")
box("graph", 2, 26, 27, 13, "Interaction graph", [r"replies of $a$ before $t$"], fill="#EDEDED")
# ---------------------------------------------------------------- perturbation bank
box("pert", 38, 24, 30, 57, r"Perturbation bank $\mathcal{T}$",
    ["text views:", "synonym swap", "homoglyphs", "char. transposition", "link rotation, padding", "",
     "context views:", "gap jitter", "camouflage replies"], fill="#F7F7F7", gap=3.6)
# ---------------------------------------------------------------- experts
ex = [("LEX", "TF-IDF, logistic"), ("SEM", "GPT-2 states, logistic"), ("AUTH", "token statistics, GBDT"),
      ("STYLE", "surface cues, GBDT"), ("BEH", "causal cadence, GBDT"), ("GRAPH", "interaction, GBDT")]
for i, (k, d) in enumerate(ex):
    box(k, 80, 71 - i * 9.4, 30, 7.8, k, [d], fs=5.6, lfs=4.8, gap=3.0)
top, bot = 71 + 7.8, 71 - 5 * 9.4
B["exgrp"] = (77, bot - 2.5, 36, top - bot + 5.0)
ax.add_patch(FancyBboxPatch(B["exgrp"][:2], B["exgrp"][2], B["exgrp"][3], boxstyle="round,pad=0,rounding_size=1.5",
                            lw=0.6, ec="black", fc="none", ls=(0, (1, 1.5))))
t = ax.text(95, top + 5.2, r"Experts $f_m$ give logit $z_m$ and attribution $\phi_m$", ha="center", va="center", fontsize=5.4)
TEXTS.append((None, t))
# ---------------------------------------------------------------- trust signals
box("rho", 122, 62, 30, 12, r"Local reliability $\rho_m$", [r"$k$-NN log-likelihood"], fs=5.6, lfs=4.9)
box("stab", 122, 45, 30, 12, r"Attribution stability $s_m$", [r"top-$r$ RBO under $\mathcal{T}$"], fs=5.6, lfs=4.9)
box("shift", 122, 28, 30, 12, r"Prediction shift $\delta_m$", [r"mean $|z_m-\tilde{z}_{m,k}|$"], fs=5.6, lfs=4.9)
# ---------------------------------------------------------------- router, concepts, moderation
box("router", 160, 56, 26, 22, "Trust router", [r"$w_m=\mathrm{softmax}_m(\cdot)$", r"$z=\gamma_0+\tau\sum_m w_m z_m$"],
    fill="#DDDDDD", fs=6.0, lfs=4.8, gap=3.6)
box("concept", 160, 28, 26, 18, "Concept layer", [r"$C_k=\tau\sum_m w_m\sum_{j\in G_{m,k}}\phi_{m,j}$", "seven policy concepts"],
    fs=6.0, lfs=4.3, gap=3.8)
box("crc", 112, 3, 74, 15, "Conformal moderation-risk control",
    [r"remove if $p\geq\hat\lambda_R$, keep if $p<\hat\lambda_K$, otherwise defer;",
     r"removals with routed stability $S<\tau_S$ are escalated to review"], fill="#DDDDDD", fs=6.0, lfs=4.9)
box("rf", 122, 84, 64, 10, "Router-fit split, clean + perturbed", [r"$\min_\theta\,\mathrm{NLL}(\theta)$ on $\mathcal{D}_R^{+}$ by L-BFGS"],
    ls=(0, (3, 2)), fs=5.8, lfs=4.9, gap=3.2)
box("cal", 38, 4, 64, 11, "Calibration split (one post per account)", [r"thresholds $\hat\lambda_R$ and $\hat\lambda_K$"],
    ls=(0, (3, 2)), fs=5.8, lfs=4.9, gap=3.2)

# ---------------------------------------------------------------- arrows: inference path
arrow("post", "R", "pert", "L", df=(72.5 - 24) / 57)
arrow("hist", "R", "pert", "L", df=(52.5 - 24) / 57)
arrow("graph", "R", "pert", "L", df=(32.5 - 24) / 57)
for i, (k, _) in enumerate(ex):
    yk = 71 - i * 9.4 + 3.9
    arrow("pert", "R", k, "L", sf=(yk - 24) / 57, df=0.5,
          label=(r"$x,\ \tilde{x}_k$" if i == 0 else None), loff=(0, 2.0))
gy = lambda y: (y - B["exgrp"][1]) / B["exgrp"][3]
arrow("exgrp", "R", "rho", "L", sf=gy(68), df=0.5, label=r"$z_m$", loff=(0, 1.9))
arrow("exgrp", "R", "stab", "L", sf=gy(51), df=0.5, label=r"$\phi_m,\ \tilde{\phi}_{m,k}$", loff=(0, 2.0))
arrow("exgrp", "R", "shift", "L", sf=gy(34), df=0.5, label=r"$z_m,\ \tilde{z}_{m,k}$", loff=(0, 2.0))
arrow("exgrp", "R", "router", "L", sf=gy(76.6), df=(76.6 - 56) / 22, label=r"$z_m$", seg=0, loff=(13.5, 1.7))
arrow("rho", "R", "router", "L", df=(68 - 56) / 22)
arrow("stab", "R", "router", "L", df=(62.5 - 56) / 22, via=[(156, 51), (156, 62.5)])
arrow("shift", "R", "router", "L", df=(59 - 56) / 22, via=[(158, 34), (158, 59)])
arrow("router", "B", "concept", "T", label=r"$w_m$", loff=(3.0, 0))
arrow("exgrp", "R", "concept", "L", sf=gy(25.0), df=(29.8 - 28) / 18, via=[(156, 25.0), (156, 29.8)],
      label=r"$\phi_m$", seg=0, loff=(0, 1.8))
arrow("concept", "B", "crc", "T", sf=0.5, df=(173 - 112) / 74, label="statement of reasons", loff=(-13.5, 0), lfs=4.6)
arrow("router", "R", "crc", "R", sf=(61.5 - 56) / 22, df=(10.5 - 3) / 15, via=[(190, 61.5), (190, 10.5)],
      label=r"$p=\sigma(z)$", seg=1, loff=(0, 14.0))
# ---------------------------------------------------------------- arrows: fitting path (dashed)
arrow("rf", "B", "rho", "T", sf=(128 - 122) / 64, df=6 / 30, dashed=True, label="neighbour bank", loff=(10.5, 2.4))
arrow("rf", "B", "router", "T", sf=(176 - 122) / 64, df=16 / 26, dashed=True, label=r"$-\nabla_\theta\mathrm{NLL}$",
      loff=(-8.0, 0))
arrow("cal", "R", "crc", "L", dashed=True)

# ---------------------------------------------------------------- checks
def inside(pt, b, tol):
    x, y, w, h = b
    return x + tol < pt[0] < x + w - tol and y + tol < pt[1] < y + h - tol


def seg_hits(a, b, bx):
    return any(inside((a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])), bx, 0.4) for t in [i / 300 for i in range(1, 300)])


keys = [k for k in B if k != "exgrp"]
for i in range(len(keys)):
    for j in range(i + 1, len(keys)):
        a, b = B[keys[i]], B[keys[j]]
        sep = (a[0] + a[2] + 1.5 <= b[0] or b[0] + b[2] + 1.5 <= a[0] or a[1] + a[3] + 1.5 <= b[1] or b[1] + b[3] + 1.5 <= a[1])
        assert sep, f"boxes overlap or touch: {keys[i]} {keys[j]}"
ex_keys = [k for k, _ in ex]
for src, dst, pts in ARROWS:
    for a, b in zip(pts[:-1], pts[1:]):
        for k in list(B):
            if k in (src, dst) or (src == "exgrp" and k in ex_keys) or (k == "exgrp" and dst in ex_keys):
                continue
            assert not seg_hits(a, b, B[k]), f"arrow {src}->{dst} crosses box {k}"
    assert all(0 <= p[0] <= W and 0 <= p[1] <= H for p in pts)
fig.canvas.draw()
r = fig.canvas.get_renderer(); inv = ax.transData.inverted()
for key, t in TEXTS:
    bb = inv.transform(t.get_window_extent(r).get_points())
    (x0, y0), (x1, y1) = bb
    assert 0.5 <= x0 and x1 <= W - 0.5 and 0.5 <= y0 and y1 <= H - 0.5, f"text out of canvas: {t.get_text()}"
    if key:
        bx = B[key]
        assert bx[0] + 0.6 <= x0 and x1 <= bx[0] + bx[2] - 0.6 and bx[1] + 0.4 <= y0 and y1 <= bx[1] + bx[3] - 0.4, \
            f"text overflows box {key}: {t.get_text()}"
    for k, bx in B.items():
        if k in (key, "exgrp") or key is None and False:
            continue
        if key is None:
            ov = not (x1 < bx[0] or bx[0] + bx[2] < x0 or y1 < bx[1] or bx[1] + bx[3] < y0)
            assert not ov or k == "exgrp", f"free label '{t.get_text()}' overlaps box {k}"
fig.savefig("fig1_architecture.png", dpi=300)
print("ok", len(B), "boxes", len(ARROWS), "arrows")
