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
poor — trainscope tells you "expert 3 has not been selected since step 200",
a problem visible in no loss curve.

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
