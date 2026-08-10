# trainscope — Where we are going

This is not a task list. The roadmap has two engineering options (A/C) that
remain open and will be revisited when needed. This document answers a
different question: **a year from now, why would someone open trainscope, and
what would they find?**

## The promise today, clarified

"When a loss spike hits, you know *that* it happened — trainscope tells you
*why*." That sentence is already in the README and it is correct — but today
trainscope keeps that promise **for a single event inside a single run**. You
can rewind and inspect the moment the loss exploded. That is good, but a
researcher's real day is not one spike: you try ten runs over a week, two
blow up, eight do not, and the real question is: **"why did these two blow up
and not the others?"**

Trainscope cannot answer that question today. It does post-mortem for one
event, like a single detective — but real research accumulates a case file,
not a single case.

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

**Signal ordering (v1.6.0).** The four verified signals were measured in
the *same* organic run (a hybrid MoE+memory transformer under the LR ramp)
via `scripts/verify_signal_ordering.py`. The order is **consistent across
3/3 seeds**: activation kurtosis fires first (lead 29–36 steps), then
gradient norm (22–24), then routing concentration (8–22), and loss CUSUM
last (7–10). The earlier single-run observation that kurtosis precedes
CUSUM was therefore not a coincidence — there is a mechanical cascade:
activation distribution degrades first, gradients grow, routing
concentrates, and only then does the loss CUSUM fire. Implication for the
UI: the Spike Inspector's cascade narrative should present signals in this
order (kurtosis as the primary alarm, the rest as confirming evidence).

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

## What we will not do, deliberately

A product plan matters as much for what it rejects as for what it lists.
Two tempting but wrong directions came up in this conversation:

- **Becoming CV fleet monitoring ("inferencescope").** That is a shift from
  a training-time tool to an inference-time one — it breaks the identity.
  Trainscope stays training's detective; something else can exist another
  day, under another name, as a separate project, but it never enters
  trainscope's roadmap.
- **Trying to be W&B/MLflow's general-purpose dashboard.** Phase 1
  (multi-run comparison) may look similar to them, but trainscope will never
  focus on "which run gave the best result" — it stays focused on "why did
  this run behave this way". A case file, not a metrics chart.

## What is next

Phase 1 and Phase 2 do not block each other; either can start independently.
But Phase 1 materializes faster (on top of the existing UI, like adding a new
"runs" view), while Phase 2 is more research-heavy (first we must verify
which MoE/routing signals truly give early warning — the same way the CUSUM
9-11 step claim was proven, a new empirical claim must be proven).
