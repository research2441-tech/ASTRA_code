"""Section IV: data and experimental protocol."""


def data_protocol(d, R):
    d.h1("DATA AND EXPERIMENTAL PROTOCOL")
    d.h2("Datasets")
    d.p("Four public sources are used, chosen so that every claim can be tested on a corpus whose inauthentic content was "
        "actually produced by generative models. {ref:tab:data} lists their characteristics.")
    d.p("**BotSim-24** {cite:qiao2025botsim} contains one year of Reddit activity (20 June 2023 to 19 June 2024) from 1,907 "
        "genuine accounts collected through the platform API and 1,000 agent bots driven by GPT-4o-mini that post, comment "
        "and reply inside the same threads while pursuing disinformation goals. It is the only corpus among those available "
        "that provides text, timestamps and reply structure for LLM-operated accounts, so it supports all six experts. Posts "
        "and comments with fewer than three characters or with deleted bodies were removed, and up to eight items were sampled "
        "per account, giving 19,840 items from 2,906 accounts (one genuine account had no usable text); 34.9% of the items "
        "come from bots. Several fields are produced by the simulator rather than by users, namely post score, upvote ratio, "
        "comment count and identifiers, and they differ systematically between the two classes. They were excluded from every "
        "model, because a detector that read them would be detecting the simulator.")
    d.p("**TweepFake** {cite:fagni2021tweepfake} contains tweets from 17 human accounts and 23 bot accounts on X whose text "
        "was produced by GPT-2, by recurrent networks, by Markov chains and by other generators. Up to 250 tweets were sampled "
        "per account, giving 7,164 tweets. The corpus has no timestamps or graph, so only the four text experts apply. With 40 "
        "accounts it is small at the account level, and the evaluation protocol below treats it accordingly.")
    d.p("Two further sources are used only as unseen target domains. **Masquerade-23** {cite:li2023masquerade} provides posts "
        "of LLM-driven accounts on Chirper.ai, a social platform populated entirely by language-model agents; three posts were "
        "taken from each of the 500 English accounts in its account-record release; after removing ten posts whose truncated "
        "identifiers collided, 1,453 inauthentic posts from 496 accounts remain. **FediData** "
        "{cite:gao2025fedidata} provides public Mastodon posts from 24 instances of the fediverse; 1,500 English posts from 19 "
        "instances were sampled. The Mastodon posts carry no bot label, and some instances relay news feeds automatically, so "
        "they are treated as presumed-legitimate traffic and used only to estimate how often a detector trained elsewhere would "
        "remove posts on a decentralised service.")
    d.table("Characteristics of the datasets used in this study",
            [["Dataset", "Platform", "Items", "Accounts (genuine / inauthentic)", "Inauthentic generator", "Views", "Role"]],
            [["BotSim-24", "Reddit", "19,840", "1,906 / 1,000", "GPT-4o-mini agents", "text, history, graph", "train, test"],
             ["TweepFake", "X (Twitter)", "7,164", "17 / 23", "GPT-2, RNN, Markov, others", "text", "train, test, held-out generator"],
             ["Masquerade-23", "Chirper.ai", "1,453", "0 / 496", "LLM agents", "text", "transfer target"],
             ["FediData", "Mastodon", "1,500", "19 instances / 0", "none (presumed)", "text", "transfer target"]],
            widths=[1.3, 1.1, 0.8, 1.6, 1.8, 1.4, 1.6], label="tab:data", full=True,
            note="Items are posts, comments or tweets. Counts after cleaning and per-account sampling. FediData posts are "
                 "unlabelled and treated as presumed legitimate.")

    d.h2("Preprocessing and Data Partitioning")
    d.p("Text was whitespace-normalised and otherwise left untouched, since case, emoji, links and unusual characters are "
        "evidence. The frozen language model is GPT-2 small {cite:radford2019gpt2} exported to ONNX and dynamically quantised to "
        "8-bit integer weights; each text is truncated to 48 tokens after a beginning-of-text token, the 768-dimensional final "
        "hidden states are mean-pooled for SEM, and eleven token statistics are taken for AUTH (mean, standard deviation and "
        "10th/90th percentiles of the token log-probability, mean predictive entropy, the ratio of log-perplexity to entropy, "
        "mean log-rank, top-1 and top-10 rates, maximum surprisal and length). The ratio of log-perplexity to entropy follows "
        "the reasoning of perplexity-normalised detectors {cite:hans2024binoculars,bao2024fastdetectgpt}, computed here with "
        "a single model.")
    d.p("BEH features are computed only from events strictly earlier than the item: number of prior events, time since the "
        "last one, median and coefficient of variation of the inter-event gaps, burstiness, hour-of-day entropy, night-time "
        "share, share of comments, number and concentration of communities, share of gaps under two minutes and span of "
        "activity. GRAPH features use the reply edges that existed before the item: in- and out-interactions and distinct "
        "partners, repeat ratio, reciprocity, share of replies to submissions, and co-targeting (how many other accounts had "
        "replied to the same items, and the largest overlap with a single account). Label-dependent neighbourhood features were "
        "deliberately avoided: in BotSim-24 genuine users never reply to bots, because their histories predate the simulation, "
        "so any feature counting known-bot partners would again detect the simulator. Of the BotSim-24 items, 7.8% have no "
        "prior event, so the cold-start case is present and visible to the router through the history length.")
    d.p("The perturbation bank uses $K=4$ transformations per view. Synonym substitution replaces up to 25% of the eligible "
        "content words with WordNet {cite:miller1995wordnet} synonyms of the same part of speech whose GloVe-Twitter "
        "{cite:pennington2014glove} cosine similarity to the original is at least 0.5; the homoglyph transform replaces 15% of "
        "replaceable letters with Cyrillic look-alikes {cite:boucher2022bad}; the typo transform swaps adjacent characters "
        "in 15% of words {cite:gao2018deepwordbug}; and the link transform rotates URL shorteners and appends one or two benign "
        "hashtags or emoji. For BEH the prior gaps are multiplied by log-normal noise with standard deviation 0.25, 0.5, 0.75 "
        "and 1.0, and for GRAPH between one and three times the existing number of replies are added towards random genuine "
        "threads. The synonym transform is not perfectly meaning-preserving (WordNet occasionally picks a wrong sense), which "
        "is also true of published word-substitution attacks {cite:jin2020textfooler} and is accepted here because the bank "
        "is meant to mimic cheap evasion, not to paraphrase faithfully.")
    d.p("Robustness was tested on the first BotSim-24 partition and the first TweepFake fold with 600 randomly chosen "
        "inauthentic test posts, ranked against all genuine test posts. The text attacks were homoglyph substitution at rates "
        "of 0.05 to 0.50, synonym substitution at 0.25 and 0.50, and leetspeak (digits for letters) at 0.15 and 0.30. Leetspeak "
        "is absent from the perturbation bank and serves as a held-out attack family; the homoglyph and synonym attacks share "
        "a family with the bank but are up to three times stronger. Behavioural mimicry replaced the history and interaction "
        "features of each attacked bot, together with their perturbed versions, by those of a randomly drawn genuine test "
        "account, and a combined attack applied mimicry and homoglyphs at 0.15. Generalisation was tested by holding each "
        "TweepFake generator family out of training (three repeats per family, with ten genuine accounts for training, three "
        "for router fitting and four for testing), and by transferring the four text experts from BotSim-24 to TweepFake and to "
        "the Chirper.ai and Mastodon posts, and from TweepFake to BotSim-24 (three repeats each).")
    d.p("All partitions are made at the account level, stratified by class. For BotSim-24, ten random partitions assign 50% "
        "of the accounts to expert training, 15% to router fitting, 15% to calibration and 20% to testing. For TweepFake, "
        "five-fold cross-validation over accounts is repeated twice, with folds stratified by generator family, and the "
        "non-test accounts of each fold are split 70/30 into expert training and router fitting; there is no calibration split "
        "because three or four genuine test accounts per fold cannot support account-level conformal guarantees. The same "
        "partitions are used for every method, so all comparisons are paired.")

    d.h2("Baselines")
    d.p("The baselines answer one question: given the same six experts, is per-post trust routing better than other ways "
        "of combining them? Seven combiners use exactly the same expert outputs. Best-expert selects the expert with the "
        "lowest router-fit loss. Equal-vote averages logits. Accuracy-weighted voting weights experts by router-fit accuracy. "
        "SHAP-variance trust gives each expert one global weight, the inverse "
        "variance of its top-10 mean absolute SHAP values {cite:lundberg2017shap} on the router-fit split; it rewards experts "
        "whose leading attributions are evenly sized and stands for the global, explanation-derived weighting that ASTRA "
        "replaces. Stacking fits a logistic meta-learner on the expert logits {cite:wolpert1992stacked}. KNORA-E "
        "keeps the experts that are correct on all of the nearest router-fit neighbours {cite:ko2008knora}. Gated-MoE learns a "
        "softmax gate from the routing space, the input-driven gating used by mixture-of-experts bot detectors "
        "{cite:liu2023botmoe}. Stress-augmented stacking fits the same meta-learner on $\\mathcal{D}_R^{+}$, so that the effect of augmentation can "
        "be separated from that of routing, and the clean-fit router is reported as an ablation. TTA-Stacking applies stacking to logits averaged over the original and perturbed inputs, which "
        "uses the same perturbation budget as ASTRA but for test-time averaging rather than for trust. Two monolithic models, "
        "LightGBM {cite:ke2017lightgbm} and a two-layer perceptron, are trained on the concatenation of all dense views "
        "(AUTH, STYLE, BEH, GRAPH and a 64-dimensional PCA of the GPT-2 embedding) using both the expert-training and "
        "router-fit accounts, i.e. with more labelled data than the experts receive. Each expert is also reported alone.")

    d.h2("Implementation Details")
    d.p("All experiments ran on two cores of an Intel Xeon processor at 2.8 GHz with 7.7 GB of memory and no GPU, using Python "
        "3.13, scikit-learn 1.9.1 {cite:pedregosa2011sklearn}, LightGBM 4.7.0, ONNX Runtime 1.29.0 and SciPy 1.18.1. "
        "{ref:tab:impl} gives the settings. None of them was tuned on test accounts; the router has no tunable "
        "hyperparameters beyond $\\lambda$, and its sensitivity to $r$, $p$, $K$ and $k$ is reported in Section V-E.")
    d.table("Implementation settings",
            [["Component", "Setting"]],
            [["LEX", "word 1-2-grams and character 2-5-grams, 40,000 each, sublinear TF; logistic, $C=4$"],
             ["SEM", "GPT-2 (int8) mean hidden state, 768-d, standardised; logistic, $C=0.05$"],
             ["AUTH, STYLE, BEH, GRAPH", "LightGBM, 300 rounds, 15 leaves, learning rate 0.05, subsample 0.8"],
             ["Stability", "$K=4$, $r=10$, $p=0.9$"],
             ["Local reliability", "$k=50$, 16-d PCA routing space, same-account exclusion"],
             ["Router", "$M+5$ parameters, L-BFGS, three starts, $\\lambda=10^{-3}$"],
             ["Risk control", "$\\alpha\\in\\{0.01,0.02,0.05,0.10\\}$, $\\gamma=0.10$, $\\tau_S$ = 10th percentile"],
             ["Runs", "BotSim-24: 10 account partitions; TweepFake: 2 x 5 account folds"]],
            widths=[1.15, 2.85], label="tab:impl", align=["l", "l"])

    d.h2("Evaluation Metrics")
    d.p("Class balance differs between corpora, so threshold-free and imbalance-aware metrics are reported alongside "
        "macro-F1: the area under the precision-recall curve (average precision) for the inauthentic class, ROC-AUC, balanced "
        "accuracy and the Matthews correlation coefficient,")
    d.eq(r"\mathrm{MCC}=\frac{TP\cdot TN-FP\cdot FN}{\sqrt{(TP+FP)(TP+FN)(TN+FP)(TN+FN)}}", "eq:mcc")
    d.p("which is high only when both classes are predicted well {cite:chicco2020mcc}. Probability quality is measured by the "
        "negative log-likelihood, the Brier score and the expected calibration error over $B=15$ equal-width bins $\\mathcal{B}_b$,")
    d.eq(r"\mathrm{ECE}=\sum_{b=1}^{B}\frac{\left|\mathcal{B}_b\right|}{n}\left|\frac{1}{\left|\mathcal{B}_b\right|}\sum_{i\in\mathcal{B}_b}y_i-\frac{1}{\left|\mathcal{B}_b\right|}\sum_{i\in\mathcal{B}_b}\hat{p}_i\right|", "eq:ece")
    d.p("following {cite:guo2017calibration}. Moderation is evaluated by the wrongful-removal rate on legitimate posts, the "
        "missed rate on inauthentic posts, the deferral rate and the precision of removals; the wrongful-removal rate is also "
        "averaged per legitimate account, so that prolific accounts do not dominate. Explanation stability is measured by the "
        "agreement of the top concept and by the concept-ranking RBO between each post and its perturbed versions, with the "
        "routing weights held at their values for the original post.")

    d.h2("Statistical Analysis")
    d.p("Each method is evaluated on the same ten partitions per corpus, and results are reported as mean and standard "
        "deviation. Paired differences between ASTRA and each baseline are tested with the two-sided Wilcoxon signed-rank test "
        "{cite:wilcoxon1945individual}, which makes no normality assumption and is appropriate for ten paired runs {cite:demsar2006statistical}. The "
        "family of comparisons against all baselines is corrected with Holm's step-down procedure {cite:holm1979simple}: with the "
        "$G$ raw p-values sorted as $p_{(1)}\\le\\dots\\le p_{(G)}$, the adjusted values are")
    d.eq(r"\tilde{p}_{(g)}=\max_{h\le g}\ \min\left\{1,\ (G-h+1)\,p_{(h)}\right\}", "eq:holm")
    d.p("and differences are called significant when $\\tilde{p}<0.05$. Effect sizes are given as the matched-pairs "
        "rank-biserial correlation")
    d.eq(r"r_{\mathrm{rb}}=\frac{W^{+}-W^{-}}{W^{+}+W^{-}}", "eq:rb")
    d.p("where $W^{+}$ and $W^{-}$ are the sums of the signed ranks of positive and negative differences, and 95% confidence "
        "intervals of mean differences are obtained with 10,000 bootstrap resamples of the runs {cite:efron1993bootstrap}. With ten "
        "runs the smallest attainable two-sided Wilcoxon p-value is 0.002, so a Holm-adjusted value below 0.05 requires a "
        "consistent direction in nearly every run. Hypothesis H1, that attribution instability flags expert errors, is "
        "tested per expert by the area under the ROC curve of $1-s_m$ for predicting the expert's own error on test posts, "
        "compared with the same area for the expert's low confidence $-|z_m|$.")
