"""Section III: methodology (equations are native OMML via the builder)."""


def methodology(d, R):
    d.h1("PROPOSED METHODOLOGY")
    d.h2("Problem Formulation")
    d.p("A moderation system on X, Reddit or a federated service receives a stream of posts. Each post is written by an account "
        "$a$ at time $t$ and carries its text $u$. The decision about the post may also use what the platform already knows about "
        "the account at that moment, and nothing later. The input is therefore")
    d.eq(r"\mathbf{x}=\left(u,\ \mathcal{H}_a(t),\ \mathcal{G}_a(t)\right),\quad \mathcal{H}_a(t)=\left\{e\in\mathcal{E}_a:\ t_e<t\right\}", "eq:input")
    d.p("where $\\mathcal{E}_a$ is the event log of the account (posts and comments), $\\mathcal{H}_a(t)$ its causal history, and "
        "$\\mathcal{G}_a(t)$ the reply interactions involving $a$ that occurred before $t$. The label $y\\in\\{0,1\\}$ marks "
        "whether the post was produced by an inauthentic, LLM-operated or automated account ($y=1$) or by a genuine user ($y=0$). "
        "The system must return one of three actions, remove, defer to a human reviewer, or keep, and the operator fixes two "
        "tolerances in advance: the wrongful-moderation rate $\\alpha$, which is the share of legitimate posts that may be removed "
        "automatically, and the missed-inauthentic rate $\\gamma$, which is the share of inauthentic posts that may be kept "
        "without review. Both are stated as conditional risks,")
    d.eq(r"R_{\mathrm{FR}}(d)=\Pr\left[d(\mathbf{X})=\mathrm{remove}\mid Y=0\right]\le\alpha,\qquad R_{\mathrm{MS}}(d)=\Pr\left[d(\mathbf{X})=\mathrm{keep}\mid Y=1\right]\le\gamma", "eq:risk")
    d.p("Every removal must also come with a statement of reasons that names the facts behind it, because Article 17 of the "
        "Digital Services Act requires one for each restriction a provider imposes {cite:eu2022dsa}. The formulation therefore has "
        "three coupled requirements: an accurate score, an explanation that stays the same when the post is rephrased in ways "
        "that do not change its meaning, and a decision rule whose error rates are controlled by construction rather than by "
        "tuning on the test data.")

    d.h2("System Overview")
    d.p("{ref:fig:arch} shows the information flow. The post and its context pass through a perturbation bank "
        "$\\mathcal{T}=\\{T_1,\\dots,T_K\\}$ that produces $K$ variants of every view. Six experts read different views of "
        "the same post. LEX applies logistic regression to word and character TF-IDF features; SEM applies logistic regression "
        "to mean-pooled hidden states of a frozen GPT-2; AUTH applies gradient-boosted trees to token-level statistics of the same "
        "language model (log-probability moments, predictive entropy, token rank); STYLE models surface cues such as links, "
        "hashtags, emoji, capitalisation and non-ASCII letters; BEH reads the cadence of the causal history; and GRAPH reads "
        "the reply structure, including how many other accounts targeted the same threads. Each expert returns a logit and an "
        "exact additive attribution. Three trust signals are then computed per expert and per post, the router turns them into "
        "weights, the concept layer aggregates the weighted attributions into policy categories, and the risk-control stage maps "
        "the routed probability to an action.")
    d.figure("fig1_architecture.png",
             "Architecture of ASTRA. Solid arrows are used at inference time for every post. Dashed arrows are used only when "
             "the router parameters are fitted (gradient of the negative log-likelihood on held-out labelled accounts) and when "
             "the two moderation thresholds are calibrated. The perturbation bank acts on the text views and on the two context "
             "views separately, so each expert is probed only with changes that are meaningful for its own input.",
             label="fig:arch", full=True)

    d.h2("Experts with Exact Additive Attributions")
    d.p("Routing on attributions is only defensible if the attributions are exact, because an approximate explanation can drift "
        "for reasons that have nothing to do with the model. ASTRA therefore restricts the expert family to models whose "
        "Shapley attributions can be computed exactly and quickly. Every expert $m$ satisfies the local-accuracy identity")
    d.eq(r"z_m(\mathbf{x})=\phi_{m,0}+\sum_{j=1}^{d_m}\phi_{m,j}(\mathbf{x})", "eq:additive")
    d.p("where $z_m$ is the log-odds output, $d_m$ the number of input features of the expert and $\\phi_{m,0}$ its base value. "
        "For the two logistic experts the attribution of feature $j$ is the product of the weight and the feature value measured "
        "against a reference point $\\mu_j$ (the training mean for dense features and zero, i.e. an empty document, for TF-IDF "
        "features),")
    d.eq(r"\phi_{m,j}(\mathbf{x})=w_{m,j}\left(\psi_{m,j}(\mathbf{x})-\mu_j\right)", "eq:linshap")
    d.p("with $\\psi_{m,j}$ the $j$-th feature map of the expert. For the four tree experts the attribution is the Shapley value "
        "of the tree ensemble,")
    d.eq(r"\phi_{m,j}(\mathbf{x})=\sum_{S\subseteq\mathcal{F}_m\setminus\{j\}}\frac{|S|!\,\left(d_m-|S|-1\right)!}{d_m!}\left[v_{\mathbf{x}}\left(S\cup\{j\}\right)-v_{\mathbf{x}}(S)\right]", "eq:shapley")
    d.p("where $\\mathcal{F}_m$ is the feature set and $v_{\\mathbf{x}}(S)$ the expected log-odds when only the features in $S$ "
        "are fixed to their values in $\\mathbf{x}$. The tree algorithm of {cite:lundberg2020treeshap} evaluates this sum exactly in "
        "polynomial time. Because both families are additive in log-odds, the attributions of different experts live on the "
        "same scale and can be pooled without rescaling, which is the property the concept layer relies on.")

    d.h2("Attribution Stability and Local Reliability")
    d.p("The perturbation bank generates $\\tilde{\\mathbf{x}}_k=T_k(\\mathbf{x})$ for $k=1,\\dots,K$. For the four text "
        "experts the transformations are embedding-constrained synonym substitution, homoglyph substitution, adjacent-character "
        "transposition, and link rotation with benign padding; for BEH they are log-normal jitter of the inter-event gaps with "
        "increasing spread, and for GRAPH they are camouflage replies to randomly chosen threads of genuine users. None of these "
        "should change what the post is, so a trustworthy expert should keep both its decision and the reasons for it. To "
        "compare reasons, the attribution vector is reduced to the ordered list of its $r$ largest entries, each tagged with its "
        "sign,")
    d.eq(r"\pi_m^{r}(\mathbf{x})=\left(\left(j_1,\mathrm{sgn}\,\phi_{m,j_1}\right),\dots,\left(j_r,\mathrm{sgn}\,\phi_{m,j_r}\right)\right),\quad \left|\phi_{m,j_1}\right|\ge\dots\ge\left|\phi_{m,j_r}\right|", "eq:topr")
    d.p("so that a feature whose influence flips direction counts as a different reason. Two such lists are compared with "
        "rank-biased overlap {cite:webber2010rbo}, which weights agreement at the top of the list more heavily than agreement in "
        "the tail and is defined for lists that do not share all their elements:")
    d.eq(r"\mathrm{RBO}_p(A,B)=(1-p)\sum_{q=1}^{r}p^{\,q-1}\frac{\left|A_{1:q}\cap B_{1:q}\right|}{q}+p^{\,r}\frac{\left|A_{1:r}\cap B_{1:r}\right|}{r}", "eq:rbo")
    d.p("The attribution stability of expert $m$ on the post is the average agreement over the bank, and the prediction shift "
        "is the average change of its logit:")
    d.eq(r"s_m(\mathbf{x})=\frac{1}{K}\sum_{k=1}^{K}\mathrm{RBO}_p\left(\pi_m^{r}(\mathbf{x}),\ \pi_m^{r}(\tilde{\mathbf{x}}_k)\right)", "eq:stab")
    d.eq(r"\delta_m(\mathbf{x})=\frac{1}{K}\sum_{k=1}^{K}\left|z_m(\mathbf{x})-z_m(\tilde{\mathbf{x}}_k)\right|", "eq:shift")
    d.p("The two quantities answer different questions. A homoglyph attack, for example, may leave the character-level logit "
        "almost unchanged while replacing the n-grams that carry it, which lowers $s_m$ but not $\\delta_m$. Neither signal "
        "says whether the expert is accurate for this kind of post, so a third, label-based signal is added. Let "
        "$\\mathcal{N}_k(\\mathbf{x})$ be the $k$ nearest neighbours of the post in a routing space (a 16-dimensional PCA "
        "projection of the frozen GPT-2 embedding plus, when available, the logarithm of the account's history length and "
        "interaction count), searched within a labelled router-fit set and excluding posts of the same account. The local "
        "reliability is the mean log-likelihood of the expert on those neighbours,")
    d.eq(r"\rho_m(\mathbf{x})=\frac{1}{k}\sum_{i\in\mathcal{N}_k(\mathbf{x})}\left[y_i\log\sigma\left(z_m(\mathbf{x}_i)\right)+(1-y_i)\log\left(1-\sigma\left(z_m(\mathbf{x}_i)\right)\right)\right]", "eq:rho")
    d.p("with $\\sigma$ the logistic function. Using the log-likelihood instead of the local accuracy of dynamic classifier "
        "selection {cite:woods1997combination,ko2008knora} makes the signal sensitive to confident mistakes, which are exactly the cases that "
        "produce wrongful removals. Including the history length in the routing space lets the router learn that BEH and GRAPH "
        "are unreliable for a first post and become informative once the account has a track record.")

    d.h2("Trust Router")
    d.p("The router converts the three signals into a convex weighting of the experts. With parameters "
        "$\\theta=(\\beta_1,\\dots,\\beta_M,a,b,c,\\tau,\\gamma_0)$ the routing score and the weights are")
    d.eq(r"\eta_m(\mathbf{x})=\beta_m+a\,\rho_m(\mathbf{x})+b\,s_m(\mathbf{x})-c\,\delta_m(\mathbf{x}),\qquad w_m(\mathbf{x})=\frac{\exp\left(\eta_m(\mathbf{x})\right)}{\sum_{l=1}^{M}\exp\left(\eta_l(\mathbf{x})\right)}", "eq:router")
    d.p("and the routed decision logit is a temperature-scaled mixture of the expert logits,")
    d.eq(r"z(\mathbf{x})=\gamma_0+\tau\sum_{m=1}^{M}w_m(\mathbf{x})\,z_m(\mathbf{x}),\qquad \hat{p}(\mathbf{x})=\sigma\left(z(\mathbf{x})\right)", "eq:mix")
    d.p("Before entering {ref:eq:router}, each signal is standardised per expert with its mean and standard deviation on "
        "the router-fit split, for example $\\tilde{s}_m(\\mathbf{x})=(s_m(\\mathbf{x})-\\mu_{s,m})/\\sigma_{s,m}$, and likewise "
        "for $\\rho_m$ and $\\delta_m$. Without this step the raw signals differ in spread by an order of magnitude across "
        "experts, and the ridge penalty below would shrink the coefficient of the narrowest signal towards zero for reasons "
        "of scale alone. After standardisation, $a$, $b$ and $c$ measure how strongly the router reacts when an expert is "
        "unusually reliable, stable or volatile on a post compared with its own typical behaviour. The parameters are shared "
        "across experts except for the static biases $\\beta_m$, so the router has only $M+5$ free parameters (11 for six "
        "experts). A gating network that reads the raw input, as in mixtures of experts {cite:jacobs1991adaptive,"
        "shazeer2017outrageously,liu2023botmoe}, can fit the router-fit split better but has no reason to prefer a stable "
        "expert over an unstable one, and its weights are themselves hard to explain; here the fitted values of $a$, $b$ and "
        "$c$ are reported and show directly how much the deployed system rewards reliability and stability.")
    d.p("A router fitted only on clean posts never sees an expert fail under evasion, so it has no way to learn how the "
        "trust signals behave when that happens. The router-fit set is therefore augmented with the perturbed versions of "
        "its own posts. Writing $\\tilde{\\mathbf{x}}_0=\\mathbf{x}$, the stability of a perturbed version is measured "
        "against its siblings,")
    d.eq(r"s_m(\tilde{\mathbf{x}}_k)=\frac{1}{K}\sum_{v=0,\,v\neq k}^{K}\mathrm{RBO}_p\left(\pi_m^{r}(\tilde{\mathbf{x}}_k),\ \pi_m^{r}(\tilde{\mathbf{x}}_v)\right),\qquad k=1,\dots,K", "eq:sibling")
    d.p("with the prediction shift defined in the same leave-one-out manner and the local reliability recomputed from the "
        "perturbed embedding. The stress-augmented set $\\mathcal{D}_R^{+}=\\{(\\tilde{\\mathbf{x}}_k,y):(\\mathbf{x},y)\\in"
        "\\mathcal{D}_R,\\ k=0,\\dots,K\\}$ is $K+1$ times larger than $\\mathcal{D}_R$, and the parameters minimise the "
        "regularised negative log-likelihood over it,")
    d.eq(r"\mathcal{L}(\theta)=-\frac{1}{\left|\mathcal{D}_R^{+}\right|}\sum_{(\mathbf{x}_i,y_i)\in\mathcal{D}_R^{+}}\left[y_i\log\hat{p}(\mathbf{x}_i)+(1-y_i)\log\left(1-\hat{p}(\mathbf{x}_i)\right)\right]+\lambda\left\Vert\theta\right\Vert_2^2", "eq:nll")
    d.p("with $\\lambda=10^{-3}$. The objective is smooth, so L-BFGS {cite:liu1989} is run from three starting points and "
        "the solution with the lowest loss is kept. Because augmentation changes what the combiner is fitted on, any "
        "combiner can use it; Section V therefore also reports stacking fitted on $\\mathcal{D}_R^{+}$, so that the effect "
        "of augmentation is not credited to routing. The perturbations used for fitting are mild (Section IV-B), and the "
        "robustness tests use stronger versions of them together with an attack family that never appears in the bank.")

    d.h2("Policy-Concept Attribution and Statements of Reasons")
    d.p("Because every expert is additive in log-odds and the mixture in {ref:eq:mix} is linear in the expert logits for a "
        "fixed post, the routed decision decomposes exactly over the features of all experts. Each feature is assigned to one of "
        "seven concepts aligned with platform-manipulation policies: synthetic authorship (AUTH), automated cadence (BEH), "
        "coordinated interaction (GRAPH), link redirection (URL and shortener features, link n-grams), engagement bait "
        "(hashtags, mentions, emoji, calls to action, capitalisation, non-ASCII letters), stylometric form (length, lexical "
        "diversity, word length) and topical content (SEM and the remaining n-grams). With $G_{m,k}$ the features of expert $m$ "
        "assigned to concept $k$, the concept attribution is")
    d.eq(r"C_k(\mathbf{x})=\tau\sum_{m=1}^{M}w_m(\mathbf{x})\sum_{j\in G_{m,k}}\phi_{m,j}(\mathbf{x})", "eq:concept")
    d.p("and substituting {ref:eq:additive} into {ref:eq:mix} gives the completeness identity")
    d.eq(r"z(\mathbf{x})=\gamma_0+\tau\sum_{m=1}^{M}w_m(\mathbf{x})\,\phi_{m,0}+\sum_{k=1}^{7}C_k(\mathbf{x})", "eq:complete")
    d.p("so the concepts account for the whole deviation of the decision from its baseline, with no residual. The statement of "
        "reasons attached to a removal lists the concepts with the largest positive $C_k$, the observable facts behind them "
        "(for instance the number of prior replies to the same thread by other accounts), and the fact that the decision was "
        "automated. The identity holds for the weights actually used on that post; it does not claim that the weights "
        "themselves are explained, and the router parameters are reported separately for that reason. The routed stability "
        "of the explanation is summarised as")
    d.eq(r"S(\mathbf{x})=\sum_{m=1}^{M}w_m(\mathbf{x})\,s_m(\mathbf{x})", "eq:routedstab")
    d.p("which is high when most of the weight sits on experts whose reasons survive the perturbation bank.")

    d.h2("Conformal Moderation-Risk Control")
    d.p("The routed probability is converted into an action with two thresholds calibrated on a held-out calibration split. "
        "Posts of one account are not exchangeable with each other, so the calibration set keeps one randomly chosen post per "
        "account; under the assumption that accounts are sampled independently from the same population, the calibration "
        "scores of each class are then exchangeable with the score of a new post from that class {cite:vovk2005algorithmic,vovk2012conditional}. "
        "Let $\\mathcal{I}_0$ and $\\mathcal{I}_1$ index the $n_0$ legitimate and $n_1$ inauthentic calibration posts. The "
        "removal and keep thresholds are")
    d.eq(r"\hat{\lambda}_{R}=\inf\left\{\lambda:\ \frac{1}{n_0+1}\left(1+\sum_{i\in\mathcal{I}_0}\mathbf{1}\left[\hat{p}_i\ge\lambda\right]\right)\le\alpha\right\}", "eq:lamR")
    d.eq(r"\hat{\lambda}_{K}=\sup\left\{\lambda:\ \frac{1}{n_1+1}\left(1+\sum_{i\in\mathcal{I}_1}\mathbf{1}\left[\hat{p}_i<\lambda\right]\right)\le\gamma\right\}", "eq:lamK")
    d.p("and the action, including the escalation of removals whose explanation is unstable, is")
    d.eq(r"d(\mathbf{x})=\begin{cases}\mathrm{remove}, & \hat{p}(\mathbf{x})\ge\hat{\lambda}_R\ \text{and}\ S(\mathbf{x})\ge\tau_S\\ \mathrm{keep}, & \hat{p}(\mathbf{x})<\hat{\lambda}_K\\ \mathrm{defer}, & \text{otherwise}\end{cases}", "eq:decision")
    d.p("where $\\tau_S$ is the 10th percentile of $S$ on the calibration posts. Each threshold is a class-conditional "
        "(Mondrian) conformal quantile, and the finite-sample argument of conformal risk control {cite:angelopoulos2024crc,"
        "angelopoulos2023gentle} gives, for a new legitimate post exchangeable with the legitimate calibration posts,")
    d.eq(r"\Pr\left[\hat{p}(\mathbf{X}_{n+1})\ge\hat{\lambda}_R\mid Y_{n+1}=0\right]\le\alpha", "eq:guarantee")
    d.p("and the symmetric statement for $\\hat{\\lambda}_K$ and $\\gamma$. Two properties follow. First, the escalation rule "
        "can only move posts from remove to defer, so the removal set shrinks and both bounds in {ref:eq:risk} continue to "
        "hold without recalibration; stability-based escalation costs reviewer time but never validity. Second, the guarantee "
        "is marginal over calibration draws and is tied to exchangeability; it says nothing under a shift of platform or "
        "generator, and Section V-D examines what happens to it in that case. When $\\hat{\\lambda}_K>\\hat{\\lambda}_R$ the "
        "two bands overlap, no post is deferred by the score alone, and the decision reduces to the removal threshold.")

    d.h2("Training and Inference Algorithms")
    d.p("{ref:alg:train} and {ref:alg:infer} summarise the procedure. Accounts, not posts, are the unit of every split, which "
        "keeps the expert-training, router-fit, calibration and test sets disjoint at the level that matters for leakage.")
    d.algorithm("Fitting ASTRA", [
        "**Input:** labelled posts split by account into $\\mathcal{D}_{E}$, $\\mathcal{D}_R$, $\\mathcal{D}_C$; bank $\\mathcal{T}$; $r$, $p$, $k$, $\\alpha$, $\\gamma$",
        "**Output:** experts $f_{1:M}$, router $\\theta$, thresholds $\\hat{\\lambda}_R$, $\\hat{\\lambda}_K$, $\\tau_S$",
        "for $m=1$ to $M$ do",
        "  fit $f_m$ on $\\mathcal{D}_E$ (logistic or GBDT)",
        "#end for",
        "for each post $\\mathbf{x}\\in\\mathcal{D}_R\\cup\\mathcal{D}_C$ do",
        "  build $\\tilde{\\mathbf{x}}_k=T_k(\\mathbf{x})$ for $k=1,\\dots,K$",
        "  compute $z_m$, $\\phi_m$ for $\\mathbf{x}$ and each $\\tilde{\\mathbf{x}}_k$; get $s_m$ by {ref:eq:stab}, $\\delta_m$ by {ref:eq:shift}",
        "#end for",
        "compute $\\rho_m$ on $\\mathcal{D}_R$ by {ref:eq:rho}, excluding same-account neighbours",
        "build $\\mathcal{D}_R^{+}$ with sibling signals {ref:eq:sibling}; standardise all signals per expert",
        "$\\theta\\leftarrow\\arg\\min\\mathcal{L}(\\theta)$ by L-BFGS from three starts {ref:eq:nll}",
        "keep one post per account of $\\mathcal{D}_C$; compute $\\hat{p}$ and $S$",
        "set $\\hat{\\lambda}_R$, $\\hat{\\lambda}_K$ by {ref:eq:lamR}-{ref:eq:lamK} and $\\tau_S$ as the 10th percentile of $S$",
        "**return** $f_{1:M}$, $\\theta$, $\\hat{\\lambda}_R$, $\\hat{\\lambda}_K$, $\\tau_S$"], label="alg:train")
    d.algorithm("Moderating one post", [
        "**Input:** post $\\mathbf{x}$ with causal context; fitted system",
        "**Output:** action $d(\\mathbf{x})$, probability $\\hat{p}$, statement of reasons",
        "build $\\tilde{\\mathbf{x}}_{1:K}$ and score all experts on the $K+1$ inputs",
        "compute $s_m$, $\\delta_m$ and $\\rho_m$ (neighbours from $\\mathcal{D}_R$)",
        "compute $w_m$ by {ref:eq:router} and $\\hat{p}$ by {ref:eq:mix}",
        "compute $C_{1:7}$ by {ref:eq:concept} and $S$ by {ref:eq:routedstab}",
        "assign $d(\\mathbf{x})$ by {ref:eq:decision}",
        "if $d(\\mathbf{x})=\\mathrm{remove}$ then",
        "  emit the concepts with the largest positive $C_k$ as the statement of reasons",
        "#end if",
        "**return** $d(\\mathbf{x})$, $\\hat{p}$, statement"], label="alg:infer")

    d.h2("Complexity Analysis")
    d.p("Let $c_m$ be the cost of one forward pass of expert $m$, $a_m$ the cost of its attribution, $n_R$ the size of the "
        "router-fit set and $\\kappa$ the cost of the shared language-model pass that feeds SEM and AUTH. Inference for one post "
        "costs")
    d.eq(r"\mathcal{O}\left((K+1)\left[\kappa+\sum_{m=1}^{M}\left(c_m+a_m\right)\right]+M K r+k\log n_R\right)", "eq:complexity")
    d.p("where the term $MKr$ is the list comparison and $k\\log n_R$ the neighbour search with a tree index. The linear "
        "attributions cost as much as a forward pass, TreeSHAP costs $\\mathcal{O}(TLD^2)$ for $T$ trees of $L$ leaves and depth "
        "$D$ {cite:lundberg2020treeshap}, and the router itself is $\\mathcal{O}(M)$. In practice the $K+1$ language-model passes "
        "dominate, so the perturbation bank multiplies the dominant cost by roughly $K+1$; Section V-E reports measured "
        "latencies and the accuracy lost when $K$ is reduced. Memory is dominated by the router-fit index, "
        "$\\mathcal{O}(n_R\\,d_U)$ with $d_U\\le18$ routing dimensions, and by the TF-IDF vocabulary.")
