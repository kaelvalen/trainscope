# trainscope — Where we are going

This is not a task list. It answers a different question: **a year from now,
why would someone open trainscope, and what would they find?** As of v1.8.0,
all three phases below have shipped in some form; the sections mark what is
done and what is still open.

## The promise today, clarified

"When a loss spike hits, you know *that* it happened — trainscope tells you
*why*." That sentence is in the README and it is correct. Since 1.0, that
promise has grown from **a single event inside a single run** to the
researcher's real day: you try ten runs over a week, two blow up, eight do
not, and the real question is **"why did these two blow up and not the
others?"** — answered now by the Runs view (clusters, common-fate bands,
counterexamples) and by `trainscope report` from the shell.

Trainscope is a case file, not a single case.

## A three-phase future

Three phases, each building on the previous one; none must wait for another,
but there is a natural order.

### Phase 1 — "Why this run, not the others?"

Trainscope's current single-run post-mortem gains its real power in
multi-run comparison: put ten runs side by side and ask "which
hyperparameter combination contributes most to exploding?" This is **not** a
W&B-style sweep dashboard — they ask "which run gave the best result"; we
ask "which runs exploded and what is the common cause". The same forensics
identity, extended from one run to many.

Concretely: a researcher wakes up, looks at the 6 runs they tried overnight,
and trainscope tells them "every run with lr=5e-4 and above has a gradient
explosion around step 40; none at lr=1e-4" — without opening each run and
comparing by hand.

**Status (v1.8.0).** Phase 1's body has shipped. The Runs view lists runs
side by side, compares them (divergence point, config diff, common cause),
clusters them by signal signature, reports each cluster's discriminant
config traits and common-fate loss band, and pairs any exploding run with
its nearest stable run via `GET /api/counterexample`. `trainscope report
--runs <root>` produces the same cluster analysis from the shell. What
remains open is depth, not foundation: finer-grained attribution (which
single hyperparameter separates a cluster from its counterexample) and
automated sweep suggestions are natural follow-ups.

### Phase 2 — "Are you asking the right question for this architecture?"

Today trainscope asks every model the same questions: did the loss explode,
did gradients explode, did weights shift. These are universal questions,
valid for every architecture. But modern architectures (MoE,
routing-based hybrids) have their own pathology classes — an expert never
being selected, routing collapsing — which appear in a different signal
*before* the loss spike.

In this phase trainscope stops being a "generic loss monitor" and becomes "a
tool that knows how this architecture breaks". This is the only place that
intersects your own research interests (addressor routing,
memory-augmented architectures) — and no competitor (W&B, MLflow,
TensorBoard) goes there, because they prefer to stay architecture-agnostic.

Concretely: you train a MoE, the loss never spikes but the model quality is
poor — trainscope tells you "routing has concentrated on expert 1 since
step 200; the other three experts are idle", a problem visible in no loss
curve.

**Empirical status (v1.3.0).** The expert-collapse claim has been tested the
same way the CUSUM claim was, via `scripts/verify_expert_collapse_signal.py`
(mini Mixtral-style MoE, 4 experts top-2, wikitext-2). Result is **positive
but more specific than the original phrasing**:

- **Routing concentration is a real early warning**: in the organic
  LR-ramp divergence, max-expert share exceeded 0.85 durably 4–12 steps
  (mean 7.7) before the loss exploded, in 3/3 seeds; the stable control
  produced zero collapses in 3/3 seeds.
- **"Dead expert" is NOT a signal**: one of four experts dropping below
  2% share happens in the stable control too (top-2-of-4 routing
  naturally neglects an expert), so "expert 3 has not been selected since
  step 200" is normal behavior, not a pathology. Phase 2 detectors must
  key on *concentration* (single-expert dominance), not on per-expert
  abandonment.

This decides what v1.4.0 must build: an expert-utilization detector that
measures routing concentration drift, and a UI panel showing per-expert
share over time.

**Empirical status (v1.5.0 prep).** The addressor-collapse claim — the
memory-augmented sibling — has been tested the same way via
`scripts/verify_addressor_collapse_signal.py` (mini memory-augmented
transformer, 16 soft-addressed slots, wikitext-2). Result is **positive**:

- **Addressor concentration is a real early warning**: in the organic
  LR-ramp divergence, mean max-slot addressing share exceeded 0.6 durably
  7–11 steps (mean 9.3) before the loss exploded, in 3/3 seeds; the stable
  control produced zero collapses in 3/3 seeds (max share 0.24–0.32 — the
  model keeps using the bank diffusely).
- **Threshold rationale**: 0.6 is not proportional to slot count; it sits
  ~2x above the measured healthy ceiling of the signal (control max share
  0.32), exactly like MoE's 0.85 sat above its control's 0.74. Both
  thresholds follow "control max + margin", validated by running the
  control first.
- **Dead-slot signal rejected (measured, not assumed)**: a slot staying
  below 2% mean weight occurs in *every* step of both conditions (140/140
  stable, 87–90/88–91 ramp steps) — 16 slots make one slot near-idle
  structurally. Same finding as MoE's dead-expert: abandonment is normal,
  concentration is the signal.

This is the second architecture-class signal verified with the same
methodology. The verified signal is now production code (v1.5.0): the
`addressor_concentration_drift` detector joins the family (default
threshold 0.6 — the experiment's "control max + margin" rule), and the
Routing & addressing view renders per-slot shares for addressor blocks
alongside per-expert shares for routers. The detector keys on slot
concentration, not on unused slots.

**Signal ordering (v1.6.0, corrected in v1.7.0).** The four verified signals were measured in the
*same* organic run (a hybrid MoE+memory transformer under the LR ramp) via
`scripts/verify_signal_ordering.py`. The original v1.6.0 run reported a consistent order
(kurtosis → gradient norm → concentration → loss CUSUM) across 3/3 seeds, but that measurement
used **weight** kurtosis (kurtosis of the attention projection weights) — a different physical
quantity than the **activation** kurtosis that production records as `act_kurtosis` and that the
original `verify_kurtosis_early_warning.py` experiment measured. Re-running with the correct
activation metric (v1.7.0): the order is **NOT consistent** across seeds (3 distinct orders in
3/3 seeds). The one stable finding: **loss CUSUM always fires last** (lead 7–10, 3/3 seeds). The
relative order of kurtosis, gradient norm, and routing concentration varies by seed, so there is
no single mechanical cascade to anchor a UI "primary alarm". Implication: do NOT promote any
signal to primary-alarm status; the Spike Inspector should present all signals as independent
evidence, and the earlier "kurtosis first" UI implication is withdrawn.

**Attention collapse (v1.8.0, in progress).** The third candidate architecture
class — attention concentration/uniformization (lazy-head / rank collapse) —
has its verification script (`scripts/verify_attention_collapse_signal.py`),
built with the same stable-control vs LR-ramp methodology. Two candidate
statistics are measured per step (worst head across blocks): max attention
weight and normalized attention entropy, each with a durable 3-step crossing
rule. This script has **not yet been calibrated**: its thresholds follow the
"control max + margin" rule, but the control's observed ceiling has not been
measured in this repo yet, so the script prints the ceilings for calibration.
Until it runs to a result, the attention signal is an experiment, not a claim;
if the result is negative, it joins the rejection list below as measured-and-
rejected.

### Phase 3 — "Can trainscope defend itself?"

This is not a feature but a maturity test: is 1.0's stability promise really
kept? A year from now, does `pip install trainscope==1.4.0` still work with a
config written against 1.0? This phase is not about adding anything new; it
is about maintaining the discipline of keeping what we promised — every new
feature (Phase 1, Phase 2) must stay faithful to the rules in the Stability
scope (Arrow additions are free, breaking Python API changes require a major
release).

This phase never ends; it runs continuously in the background — the question
to ask on every PR.

**Status (v1.8.0).** The stability promise is now pinned in code, not just
documented: legacy-run compatibility tests (`tests/test_compat.py`), Arrow
schema-evolution tests (`tests/test_schema_evolution.py`), a frozen plugin
contract (`tests/test_plugin_contract.py`), and config round-trip equivalence
tests (`tests/test_config.py`). The remaining Phase 3 work is discipline: keep
the suites green and extend them whenever a new Arrow stream or config field
lands.

## What we will not do, deliberately

A product plan matters as much for what it rejects as for what it lists.
Two directions were rejected early, and two more have been added since the
first three phases shipped. All are reviewed against the current state of
the product (as of v1.8.0), not the original vision draft:

- **Becoming CV fleet monitoring ("inferencescope").** That is a shift from
  a training-time tool to an inference-time one — it breaks the identity.
  Trainscope stays training's detective; something else can exist another
  day, under another name, as a separate project, but it never enters
  trainscope's roadmap. *Still rejected:* the multi-run and detector work
  since 1.1 made trainscope more of a training forensics platform, not
  closer to inference serving.
- **Trying to be W&B/MLflow's general-purpose dashboard.** Phase 1
  (multi-run comparison) may look similar to them, but trainscope will never
  focus on "which run gave the best result" — it stays focused on "why did
  this run behave this way". A case file, not a metrics chart. *Still
  rejected:* the comparison and clustering features added since 1.2
  strengthened the "why" framing (divergence points, common causes, signal
  signatures) rather than drifting toward ranking sweeps.
- **Promoting any single signal to "primary alarm" status.** The v1.6.0
  signal-ordering experiment initially suggested a mechanical cascade
  (kurtosis always first), which would have made one signal the headline
  of the Spike Inspector. Re-measured with the correct activation-kurtosis
  metric (v1.7.0), the order is not consistent across seeds — signals are
  independent indicators, and CUSUM firing last is the only stable
  finding. *Rejected on evidence:* no signal gets a privileged position;
  all are presented as independent evidence.
- **Adding a third architecture class without a question it answers.**
  With three signal types verified and the ordering result showing signals
  are independent, a fourth signal (e.g. state-space/Mamba) would not
  automatically add explanatory power. Such an addition is only justified
  by a concrete failure mode that existing signals cannot see — not by
  architecture novelty alone.

## What is next

The three phases have all shipped in some form; what remains is their open
edges, in priority order:

1. **Finish the attention-collapse calibration (Phase 2).** Run
   `scripts/verify_attention_collapse_signal.py` against real wikitext-2
   data, calibrate its thresholds against the control's observed ceiling,
   and promote the attention signal to a production detector + UI panel
   (mirroring the MoE/addressor path) — or reject it as measured-and-not-a-
   signal. This is the only remaining empirical gate.
2. **Phase 1 depth: attribute clusters to a single cause.** The Runs view
   already reports discriminant config traits and counterexamples; the
   natural next step is making the attribution sharper — a ranked list of
   "this field is what separates this cluster from its nearest stable run".
3. **Phase 3 discipline.** Keep the stability suites green and extend them
   whenever a new Arrow stream or config field lands. This never ends.
