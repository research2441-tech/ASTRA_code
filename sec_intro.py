"""Section I: introduction."""


def intro(d, R):
    d.h1("INTRODUCTION")
    d.p("Inauthentic posting used to be cheap only when it was crude. Spam accounts on X repeated templates, pushed links "
        "and kept machine-like schedules, and detectors learned to recognise all three {cite:lee2011seven,chu2012detecting}. "
        "Large language models remove that trade-off. A network of more than a thousand accounts on X was found posting "
        "ChatGPT-written text that contemporary classifiers did not flag {cite:yang2024anatomy}, simulators now run hundreds "
        "of LLM agents that comment inside real discussion threads {cite:qiao2025botsim}, and rewriting bot content with an "
        "LLM lowers both the accuracy and the calibration of published detectors {cite:feng2024botsay}. The same campaigns are "
        "not tied to one platform: Mastodon instances are moderated by volunteer administrators with limited resources, and "
        "Bluesky lets third parties run their own moderation services {cite:anaobi2023admins,kleppmann2024bluesky}. Meanwhile the cost of a wrong removal has become explicit: the Digital "
        "Services Act obliges providers to send a statement of reasons for every restriction they impose {cite:eu2022dsa}, and "
        "audits of the resulting statements show that many say little about why a decision was taken "
        "{cite:kaushal2024automated,trujillo2025dsa}.")
    d.p("Three properties make the problem hard. First, text alone is a weak signal. Detectors of machine-generated text lose "
        "most of their power under paraphrase {cite:krishna2023paraphrasing}, homoglyph substitution reduces them to chance "
        "{cite:creo2025silverspeak}, and they fail on unseen generators and domains {cite:wang2024m4,dugan2024raid}. Second, the "
        "signals that do help are uneven. Posting cadence and interaction structure expose automation {cite:varol2017online,"
        "feng2021botrgcn}, but they are absent for a new account and can be imitated by an operator who knows they are watched "
        "{cite:cresci2017paradigm}. Third, the decision has to be explained and its errors bounded, and both requirements sit "
        "uneasily with the evidence that post-hoc explanations change under label-preserving perturbations "
        "{cite:ghorbani2019fragile,slack2020fooling}.")
    d.p("Current systems address parts of this. Feature-based and graph-based detectors combine metadata, text and network "
        "information {cite:yang2020scalable,feng2022rgt,lei2023bic}; mixture-of-experts designs learn which view to rely on "
        "{cite:liu2023botmoe,duan2025llmbotguard}; LLM-era detectors add language-model representations and machine-text "
        "detector outputs {cite:duan2026botdmm,wang2026tracebot}; and explainable spam classifiers attach SHAP explanations to "
        "their outputs {cite:shevtsov2022explainable,citlak2025lightanfis}. On their own benchmarks these systems are accurate, "
        "and the mixture designs in particular adapt to the modality that carries the most information.")
    d.p("Their common weakness is the way trust is assigned. Whether an ensemble uses fixed votes, a stacked meta-learner or a "
        "gate computed from the input, the weight of an expert on a given post does not depend on whether that expert's reasons "
        "for the post are stable. A character-level expert can keep its score after a homoglyph attack while the n-grams "
        "behind that score are replaced, and the combined decision gives it the same voice as before. Explanations, where "
        "they exist, are computed after the decision and need not add up to it, and the final action is a hard label at a "
        "threshold of 0.5 with no stated bound on how often genuine users are removed. Evaluation practice adds a fourth "
        "problem: when posts of the same account appear in training and test data, or when simulator fields leak the label, "
        "reported accuracy says little about deployment {cite:echeverria2018lobo,hays2023simplistic}.")
    d.p("The gap studied in this paper is the lack of a per-post trust mechanism that (i) uses the stability of each expert's "
        "exact attributions under cheap, meaning-preserving perturbations together with its local reliability, (ii) produces "
        "an explanation that is complete for the decision actually taken, and (iii) maps the result to actions whose rate of "
        "wrongful removal is bounded in finite samples. The difficulty is that these requirements interact. A trust signal "
        "based on explanations is only meaningful if the explanations are exact; a complete explanation of a combined "
        "decision requires the combination to be additive in the same units as the attributions; and a guarantee on removals "
        "must survive whatever extra caution the trust signal adds.")
    d.p("ASTRA (Attribution-STable trust Routing with risk-controlled Abstention) is built around these constraints. Six experts "
        "read lexical, semantic, authorship, stylistic, behavioural and interaction views of a post, and each is restricted "
        "to a model class with exact additive attributions in log-odds. A perturbation bank applies synonym substitution, "
        "homoglyphs, character transpositions, link rotation, timing jitter and camouflage replies, and for every expert the "
        "router measures how much the top of the attribution list changes, how much the logit moves, and how well the expert "
        "has done on similar posts from other accounts. An eleven-parameter router turns these signals into weights, the "
        "weighted attributions are grouped into seven policy concepts that add up exactly to the decision, and two "
        "class-conditional conformal thresholds turn the probability into remove, defer or keep, with removals whose "
        "explanation is unstable escalated to review. The router is fitted on clean and perturbed versions of held-out posts, "
        "so that it sees experts fail before it is deployed. All experiments use account-disjoint partitions on BotSim-24, "
        "where the inauthentic accounts are GPT-4o-mini agents, and on TweepFake from X, with Chirper.ai and Mastodon posts "
        "as unseen target platforms.")
    d.p("The results support only part of this design. The combined system is far more accurate and better "
        "calibrated than global explanation-weighted voting and keeps wrongful removals at the chosen tolerance in "
        "distribution. The stability signal itself, however, turns out to be a poor indicator of expert error, and the router "
        "learns to rely on it very little. Robustness to evasion and to behavioural mimicry comes instead from fitting the "
        "combiner on perturbed posts, a step that benefits stacking just as much, while the bounded weights of the router "
        "help on a generator family excluded from training.")

    d.h2("Research Objectives")
    d.p("The study is organised around four questions.")
    d.bullets([
        "RQ1 (hypothesis H1): Does low attribution stability of an expert on a post indicate that the expert is wrong on that "
        "post, beyond what its own confidence indicates?",
        "RQ2: Does per-post routing on stability and local reliability give better-calibrated and more accurate decisions than "
        "global, stacked and input-gated combinations of the same experts?",
        "RQ3: Do class-conditional conformal thresholds keep wrongful removals of genuine users within the chosen tolerance, "
        "how much review do they require, and what does stability-based escalation cost?",
        "RQ4: How do the router and its guarantees behave under evasion attacks, unseen generator families and a change of "
        "platform?"])

    d.h2("Contributions")
    d.bullets([
        "A per-post trust signal derived from explanation stability, defined as the rank-biased overlap between the signed "
        "top-$r$ exact attributions of an expert on a post and on its perturbed versions, together with a direct test of "
        "whether it identifies expert errors (Section V-B).",
        "A parsimonious router whose eleven readable coefficients weigh local reliability, attribution stability and "
        "prediction shift, fitted on stress-augmented data, with a completeness identity that splits every routed decision "
        "exactly into seven policy concepts for use as a statement of reasons (Sections III-D to III-F and V-F).",
        "A moderation rule with class-conditional conformal thresholds, account-level calibration and an escalation step that "
        "provably preserves the bound on wrongful removals, evaluated in distribution and after a change of platform "
        "(Sections III-G, V-D and V-F).",
        "An evaluation protocol for LLM-era inauthentic content that removes simulator artefacts, partitions by account and "
        "tests homoglyph, synonym and held-out leetspeak evasion, behavioural mimicry, held-out generator families and "
        "cross-platform transfer, with paired non-parametric statistics; its results separate the contribution of routing "
        "from that of stress-augmented fitting (Sections IV and V)."])
    d.p("Section II reviews related work. Section III presents the method, Section IV the data and protocol, and Section V "
        "the results. Section VI discusses them, Section VII lists limitations and Section VIII concludes.")
