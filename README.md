# ASTRA: attribution-stability trust routing with stress-augmented fitting and conformal moderation-risk control

Code and result files for the manuscript *Attribution-Stability Trust Routing with Stress-Augmented Fitting and
Conformal Moderation-Risk Control for Detecting LLM-Driven Inauthentic Content*.

Everything runs on a CPU. The reported experiments used two cores and 7.7 GB of memory.

## Data (all public, fetched from GitHub)

| Corpus | Repository | Use |
|---|---|---|
| BotSim-24 (Reddit, GPT-4o-mini agents) | github.com/QQQQQQBY/BotSim | training and test, all six experts |
| TweepFake (X) | github.com/tizfa/tweepfake_deepfake_text_detection | training and test, text experts |
| Masquerade-23 (Chirper.ai LLM agents) | github.com/Litsay/Masquerade-23 | transfer target |
| FediData (Mastodon) | github.com/FDUDataNET/FediData | transfer target (presumed legitimate) |

Clone them into `data_raw/` (or the folder named by `ASTRA_DATA`) (`botsim`, `tweepfake_deepfake_text_detection`, `Masquerade-23`, `FediData`).

## Models and resources

- GPT-2 small ONNX (`gpt2-lm-head-10.onnx`, onnx/models on GitHub LFS) plus the BPE files `encoder.json` and
  `vocab.bpe` (github.com/graykode/gpt-2-Pytorch, folder `GPT2`). `gpt2_onnx.py` exposes the final hidden state and
  quantises the model to int8 on first use.
- GloVe Twitter 100-d (gensim-data GitHub release `glove-twitter-100`).
- NLTK WordNet, stopwords and the averaged perceptron tagger.

Set `ASTRA_MODELS` to the folder holding these files.

## Pipeline

```
python prep_data.py              # items, causal behaviour/graph features, context perturbations
python extract_features.py       # text perturbation bank + GPT-2 statistics and embeddings (resumable: --resume)
python run_main.py botsim 10     # E1-E3, E7, E8: main comparison, ablations, H1, concepts, conformal moderation
python run_main.py tweepfake 10
python run_extra.py attacks      # E4: homoglyph, synonym, leetspeak (held out), mimicry, combined
python run_extra.py logo         # E5: held-out generator families (TweepFake)
python run_extra.py transfer     # E10: cross-platform transfer
python run_extra.py sensitivity  # E3: r, p, K, k
python run_extra.py latency      # E6
python analyze.py                # statistics (Wilcoxon, Holm, rank-biserial, bootstrap CIs) -> summary.json
python make_figs.py              # result figures
python fig_arch.py               # Fig. 1 with automatic overlap and arrow checks
```

## Files

- `astra_lib.py`: experts with exact attributions, attribution stability (RBO), local reliability, router with
  standardised signals, stress-augmented fitting data, concept attribution, conformal thresholds and metrics.
- `system.py`: a fitted system that scores any item set (used by the attack, transfer, sensitivity and latency runs).
- `perturb.py`: perturbation bank (synonym, homoglyph, typo, link) and the held-out leetspeak attack.
- `gpt2_onnx.py`: byte-level BPE tokenizer and ONNX scorer.
- `results_*.json`, `summary.json`, `paper_numbers.json`: every number reported in the manuscript.
- `make_paper.py` with `builder.py`, `omml.py`, `sec_*.py` and `refs_all.json`: regenerates the Word manuscript from the
  result files (`python make_paper.py ASTRA_manuscript.docx`); all equations are written as native Word (OMML) equations.

Figure files and manuscript numbers: `fig1_architecture` = Fig. 1, `fig2_main` = Fig. 2, `fig3_h1` = Fig. 3,
`fig5_attacks` = Fig. 4, `fig7_sensitivity` = Fig. 5, `fig6_router` = Fig. 6, `fig4_moderation` = Fig. 7.

All scripts sit in one folder. Paths default to that folder; set `ASTRA_WORK` (intermediate and result files),
`ASTRA_DATA` (raw corpora) and `ASTRA_MODELS` (GPT-2, GloVe) to point elsewhere.

## Notes

- Simulator fields of BotSim-24 (post score, upvote ratio, comment count, identifiers) are never used.
- All partitions are account-disjoint; conformal calibration keeps one post per account.
