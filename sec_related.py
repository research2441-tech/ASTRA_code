"""Section II: related work. Every reference in the bibliography is cited here."""


def related(d, R):
    d.h1("RELATED WORK")
    d.h2("Feature-Based Detection of Social Spam and Bots")
    d.p("The first generation of detectors treated spam and bot accounts as a supervised classification problem over hand-built "
        "features. Honeypot accounts collected content polluters over months and showed that link use, posting behaviour and "
        "follower dynamics separate them from ordinary users {cite:lee2011seven}, while the entropy of posting intervals "
        "distinguished human, bot and cyborg accounts {cite:chu2012detecting}. Large feature sets covering user, network, "
        "temporal, content and sentiment signals later supported population estimates of automated activity on X "
        "{cite:varol2017online}, and recurrent models combined tweet text with account metadata {cite:kudugunta2018deep}. "
        "Surveys of the period describe a steady arms race in which every published feature was soon imitated "
        "{cite:ferrara2016rise,cresci2020decade}, and the observation that a new generation of spambots mimicked genuine "
        "users well enough to evade both moderation and published detectors {cite:cresci2017paradigm} moved the field towards "
        "group-level and adversarial reasoning.")
    d.p("Generalisation became the central concern of this line. Lightweight metadata models with carefully selected training "
        "subsets transfer better across collections {cite:yang2020scalable,yang2022botometer101}, and ensembles of classifiers "
        "specialised to individual bot classes detect novel bot types more reliably than a single model "
        "{cite:sayyadiharikandeh2020detection}. Evaluation protocols that hold out whole botnets exposed how poorly classifiers "
        "trained on known bots transfer to unseen ones {cite:echeverria2018lobo}, and simple decision rules on a handful of "
        "features were found to solve popular benchmarks almost perfectly because of collection and labelling artefacts "
        "{cite:hays2023simplistic}. These two findings motivate the account-disjoint partitions and the explicit removal of "
        "simulator fields in the present study. Recent spam-specific systems for X still follow the feature-engineering "
        "template: tuned gradient boosting with SHAP explanations {cite:shevtsov2022explainable}, neuro-fuzzy inference with "
        "gradient-based sampling {cite:citlak2025lightanfis}, and majority-vote ensembles of recurrent and autoencoder models "
        "{cite:bilgen2024egma}. They report high accuracy on their corpora, yet the explanation, where one exists, is produced "
        "after the decision and plays no part in it.")

    d.h2("Graph, Multimodal and Mixture-of-Experts Detectors")
    d.p("Benchmarks with follow relations {cite:feng2021twibot20,feng2022twibot22} made graph learning the dominant approach. "
        "Relational graph convolution over a heterogeneous follow graph {cite:feng2021botrgcn} and relational graph "
        "transformers that model heterogeneous influence {cite:feng2022rgt} outperform feature-based detectors on these "
        "benchmarks, and later models let text and graph exchange information {cite:lei2023bic}, encode hierarchical community "
        "structure through structural entropy {cite:yang2024sebot}, or follow the graph as it evolves {cite:he2024botdgt}. When "
        "the graph is unavailable at deployment, its knowledge can be distilled into a language model {cite:cai2024lmbot}, and "
        "calibrated multi-model detectors have been used to estimate the share of bots in whole communities "
        "{cite:tan2023botpercent}. Graph detectors are not immune to manipulation: injecting new bot nodes near a target evades "
        "graph neural detectors in a black-box setting {cite:wang2023brother}, which is the graph analogue of the camouflage "
        "replies in the perturbation bank of this study.")
    d.p("Mixture-of-experts designs are the closest relatives of ASTRA. A community-aware mixture of modality-specific experts "
        "fuses metadata, text and network experts and generalises across benchmarks {cite:liu2023botmoe}; mixtures of "
        "heterogeneous LLM-based experts reach strong accuracy with little labelled data {cite:feng2024botsay}; and mixture "
        "modules combined with graph networks have been proposed for LLM-driven bots specifically {cite:duan2025llmbotguard}. "
        "In all of them the gate is a learned function of the input representation. Such a gate can learn which modality is "
        "usually informative for a region of the input space, but nothing in its objective asks whether an expert's reasons "
        "for a particular post would survive a trivial rewrite of that post, and the gate weights are themselves unexplained.")

    d.h2("LLM-Driven Inauthentic Accounts and Machine-Generated Text")
    d.p("Generative models changed the cost structure of inauthentic activity. Commentaries and reviews describe cheap, "
        "scalable synthetic content, synthetic identities and targeted persuasion {cite:menczer2023addressing,ferrara2023chatgpt,"
        "ferrara2024genai}, and a survey experiment found model-written propaganda about as persuasive as the real thing once "
        "lightly curated {cite:goldstein2024persuasive}. A coordinated network of more than a thousand accounts on X was found "
        "posting ChatGPT-written content that contemporary text classifiers did not flag {cite:yang2024anatomy}; a platform "
        "populated entirely by LLM agents showed stronger individual camouflage alongside group-level regularities "
        "{cite:li2023masquerade}; and an LLM-powered botnet simulator released a corpus on which detectors that work on older "
        "datasets lose much of their accuracy {cite:qiao2025botsim}. LLM-guided manipulation of bot accounts also reduces the "
        "performance and calibration of existing detectors {cite:feng2024botsay}. Detection work for this generation combines "
        "content and structure channels {cite:duan2026botdmm}, fuses language-model representations with behaviour and "
        "machine-text detector outputs {cite:wang2026tracebot}, or uses explainability to analyse misclassifications of "
        "generative bots {cite:lopezjoya2025dissecting}.")
    d.p("A parallel literature asks whether a given text was written by a machine. Early work on X released tweets from bots "
        "that imitated real users with Markov chains, recurrent networks and GPT-2 {cite:fagni2021tweepfake,radford2019gpt2}, "
        "and token-rank statistics under a language model helped humans spot generated text {cite:gehrmann2019gltr}. Zero-shot "
        "detectors exploit the curvature of the log-probability surface {cite:mitchell2023detectgpt,bao2024fastdetectgpt} or "
        "contrast perplexity with cross-perplexity {cite:hans2024binoculars}, supervised detectors build features from weaker "
        "language models {cite:verma2024ghostbuster} or train against an adversarial paraphraser {cite:hu2023radar}, and "
        "watermarking embeds a detectable signal at generation time {cite:kirchenbauer2023watermark}. Surveys and benchmarks "
        "converge on one conclusion {cite:wu2025survey,wang2024m4,dugan2024raid}: detectors degrade on unseen generators, "
        "domains and decoding settings, and a multilingual benchmark across five platforms shows that the choice of training "
        "platform matters {cite:macko2025multisocial}. The authorship features of AUTH borrow the perplexity reasoning of these "
        "detectors, but they are used as one expert among several precisely because they are fragile.")
    d.p("That fragility is well documented. Paraphrasing collapses the accuracy of zero-shot detectors at low false-positive "
        "rates {cite:krishna2023paraphrasing}, and homoglyph substitution drives the agreement of AI-text detectors to chance "
        "{cite:creo2025silverspeak}. The same edits come from the adversarial NLP literature: synonym substitution guided by "
        "embeddings {cite:jin2020textfooler,li2020bertattack}, character-level and word-level edits that keep text readable "
        "{cite:li2019textbugger,gao2018deepwordbug,ebrahimi2018hotflip}, visual look-alike characters "
        "{cite:eger2019viper} and imperceptible Unicode manipulations {cite:boucher2022bad}. Lexical substitution of this kind "
        "draws its candidates from lexical databases and embedding neighbourhoods {cite:miller1995wordnet,pennington2014glove}, "
        "which is how the synonym transform of the present bank is built.")

    d.h2("Explanations and Their Stability")
    d.p("Additive feature attributions unify several explanation methods under the Shapley axioms {cite:lundberg2017shap}, and "
        "exact polynomial-time algorithms exist for tree ensembles {cite:lundberg2020treeshap}. Local surrogates "
        "{cite:ribeiro2016lime} and path-integrated gradients {cite:sundararajan2017axiomatic} cover other model classes. "
        "Several results show that such explanations can be unstable when the prediction is not: small label-preserving "
        "perturbations change attribution maps drastically {cite:ghorbani2019fragile,dombrowski2019manipulated}, and a "
        "classifier can hide biased behaviour from perturbation-based explainers {cite:slack2020fooling}. Practitioners also "
        "face disagreement between explainers on the same prediction {cite:krishna2024disagreement}, and open benchmarks now "
        "score stability alongside faithfulness {cite:agarwal2022openxai}. Comparing two explanations requires a similarity "
        "for top-weighted rankings that need not share elements, for which rank-biased overlap was designed "
        "{cite:webber2010rbo}. Concept-level explanations offer a coarser vocabulary that people can act on, either by testing "
        "the sensitivity of a model to a human-defined concept {cite:kim2018tcav} or by predicting concepts first "
        "{cite:koh2020cbm}. ASTRA takes a third route: concepts are fixed groupings of exact attributions, so the concept "
        "explanation inherits completeness from the attributions rather than from an auxiliary model. The literature treats "
        "instability as a defect of explanations to be measured or repaired; it has not been used as a signal about which "
        "model to trust for a given input.")

    d.h2("Combining Experts")
    d.p("Ensembles reduce variance and bias by combining diverse learners {cite:breiman2001random,chen2016xgboost,"
        "ke2017lightgbm}. Stacking learns the combination from held-out predictions {cite:wolpert1992stacked}, and mixtures of "
        "experts learn an input-dependent gate {cite:jacobs1991adaptive}, scaled to sparse gating with thousands of experts "
        "{cite:shazeer2017outrageously,fedus2022switch}. Dynamic classifier selection chooses experts per test point from their "
        "competence on nearby validation points {cite:woods1997combination,ko2008knora}, and its taxonomy of regions of "
        "competence and selection criteria {cite:cruz2018dynamic} is the closest precedent for the local reliability signal "
        "of ASTRA. Such combiners are typically fitted with standard libraries and quasi-Newton solvers "
        "{cite:pedregosa2011sklearn,liu1989}. Competence in this literature is measured by accuracy on neighbours; the stability "
        "of an expert's reasoning under perturbation has not been used as a competence criterion.")

    d.h2("Calibration, Conformal Risk Control and Deferral")
    d.p("Modern classifiers are often miscalibrated, and temperature scaling is a strong post-hoc remedy "
        "{cite:guo2017calibration}. Abstention has a long history, from the optimal error-reject trade-off "
        "{cite:chow1970optimum} to selective classifiers with guaranteed risk {cite:geifman2017selective} and models that learn "
        "when to defer to a human {cite:madras2018predict,mozannar2020consistent}. Conformal prediction supplies "
        "distribution-free guarantees under exchangeability {cite:vovk2005algorithmic,angelopoulos2023gentle}; label-conditional "
        "(Mondrian) validity {cite:vovk2012conditional} and class-specific error control "
        "{cite:sadinle2019least,ding2023classconditional} match the asymmetric costs of moderation, and adaptive scores improve "
        "conditional behaviour {cite:romano2020classification}. Risk-controlling prediction sets and conformal risk control "
        "extend the guarantee from coverage to any monotone loss {cite:bates2021rcps,angelopoulos2024crc}, and weighted variants "
        "handle known covariate shift or bound the loss of validity without exchangeability "
        "{cite:tibshirani2019covariate,barber2023beyond}. ASTRA uses the simplest member of this family, a pair of "
        "class-conditional thresholds, and adds an escalation rule that preserves the guarantee by construction.")

    d.h2("Moderation, Regulation and Decentralised Platforms")
    d.p("Automated moderation raises problems of opacity, fairness and accountability that accuracy alone does not address "
        "{cite:gorwa2020algorithmic}, and survey evidence shows that removals fall disproportionately on some groups of users "
        "{cite:haimson2021disproportionate}. The Digital Services Act turned part of this concern into law by requiring a "
        "statement of reasons for every restriction {cite:eu2022dsa}; analyses of the resulting transparency database find real "
        "gains together with inconsistent and partly uninformative statements {cite:kaushal2024automated,trujillo2025dsa}. "
        "Moderation is also moving to decentralised services. Mastodon concentrates users on a few instances "
        "{cite:raman2019mastodon} whose administrators moderate with limited resources {cite:anaobi2023admins}, and a recent "
        "multimodal Mastodon dataset includes manually labelled bots {cite:gao2025fedidata}. Bluesky separates hosting, feeds and "
        "labelling services {cite:kleppmann2024bluesky}, measurement studies and public datasets describe its growth "
        "{cite:balduf2024bluesky,failla2024bluesky}, and user-level blocking has been modelled with explainable classifiers "
        "{cite:bono2026blocking}. In such settings a detector trained on one platform is often deployed on another, which is "
        "why this study measures how the moderation guarantee behaves under platform shift.")

    d.h2("Evaluation Practice")
    d.p("For comparisons over repeated partitions, rank-based tests are recommended over parametric ones "
        "{cite:demsar2006statistical,wilcoxon1945individual}, with step-down correction for families of comparisons "
        "{cite:holm1979simple} and resampling intervals for effect sizes {cite:efron1993bootstrap}. For imbalanced binary "
        "problems the Matthews correlation coefficient is more informative than accuracy or F1 {cite:chicco2020mcc}. These "
        "conventions are followed in Section IV.")

    d.h2("Closest Methods and Research Gap")
    d.table("Comparison with the closest methods",
            [["Ref.", "Problem", "Method", "Data", "Strength", "Limitation", "Difference from ASTRA"]],
            [["{cite:liu2023botmoe}", "bot detection", "community-aware MoE over metadata, text, graph", "TwiBot-20/22, Cresci-15",
              "cross-benchmark generalisation", "input-driven gate; no decision guarantee", "gate uses reliability and attribution stability; risk control"],
             ["{cite:feng2024botsay}", "LLM-era bots", "mixture of LLM-based experts", "TwiBot-20/22", "few labels",
              "evasion by LLM rewriting reduces accuracy and calibration", "stability-aware trust; calibrated three-way action"],
             ["{cite:duan2025llmbotguard}", "LLM-driven bots", "features, MoE and GNN", "LLM-bot data", "targets LLM bots",
              "global combination; no explanation guarantee", "exact, complete concept attribution"],
             ["{cite:wang2026tracebot}", "LLM-driven bots", "LM representations, behaviour, AIGC detector outputs", "two LLM-bot sets",
              "high accuracy", "fused by a learned classifier; no abstention", "per-post routing; bounded wrongful removal"],
             ["{cite:sayyadiharikandeh2020detection}", "novel bots", "specialised classifiers, max-rule vote", "Botometer collections",
              "unseen bot classes", "fixed combination rule", "learned, signal-driven weights"],
             ["{cite:shevtsov2022explainable}", "bot detection", "XGBoost with SHAP", "election tweets", "transparent features",
              "explanation not used in decision", "explanation stability drives trust"],
             ["{cite:ko2008knora}", "classifier combination", "dynamic ensemble selection", "UCI data", "per-sample competence",
              "accuracy-only competence", "likelihood plus stability competence"],
             ["{cite:angelopoulos2024crc}", "risk control", "conformal control of monotone loss", "vision, NLP", "finite-sample bound",
              "no explanation or escalation", "class-conditional thresholds with stability escalation"]],
            widths=[0.55, 1.0, 1.55, 1.0, 1.15, 1.6, 1.75], label="tab:closest", full=True, size=13,
            align=["c", "l", "l", "l", "l", "l", "l"])
    d.p("{ref:tab:closest} sets the closest methods side by side. Taken together, the literature offers accurate detectors that combine several views of an account, a mature body of "
        "work showing that both machine-text detectors and their explanations break under cheap perturbations, and "
        "distribution-free tools for bounding decision errors. What is missing is their combination at the level of a single "
        "post. Existing combiners weight experts globally or through an input-driven gate, so an expert whose attribution is "
        "reshuffled by a homoglyph or a synonym swap keeps its full say in the decision; explanations are computed after the "
        "fact and need not account for the decision actually taken; and decisions are made at a fixed threshold with no bound "
        "on the rate at which genuine users lose their posts. The research gap addressed here is therefore the absence of a "
        "per-post trust mechanism that uses the stability of each expert's exact attributions together with its local "
        "reliability, whose explanation is complete for the deployed decision, and whose actions carry a finite-sample bound "
        "on wrongful moderation under account-level exchangeability.")
