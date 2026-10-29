# trainscope UI redesign

## Why

The owner's verdict on the current UI: "rezalet ... ai slop". It is the generic generated-dashboard look: near-black background with an acid-green accent, every number in its own rounded card with an icon, tracked-out all-caps eyebrows ("RUN OVERVIEW", "WORKSPACE"), a grid-pattern backdrop, and Plotly's toolbar on every chart. It reads as a template, not as a debugging instrument somebody cared about.

The target, in the owner's words, is "Linear ve Vercel dashboard gibi bir şey". In the mockups the owner picked direction **A** (a Vercel-style run report) plus the **layer × step heatmap** from direction C ("C'deki ... layer çok güzel tablosu"). The approved composite is `.superpowers/brainstorm/1641638-1793230815/content/merged.html`.

## What success looks like

- Nobody opening the UI would describe it as AI-generated. It looks quiet, dense and deliberate, in the way Vercel's deployment page and Linear do.
- The Overview answers the product's core question in one screen without scrolling at 1440×900: *what moved first before this spike, and by how many steps?*
- Charts stay fast on long runs (100k+ steps, 300+ parameter tensors).

## Goals

1. A new app shell: a light top bar with a run switcher in the breadcrumb, and tabs below it.
2. A new **Overview** page, which replaces "Timeline". It is the composite from the mockup.
3. A **runs index** page (multi-run mode) in the same visual language.
4. Every other tab restyled with the same tokens and components: Layers, Spikes, Routing, Compare, Replay. Cards, icons-as-decoration, eyebrows and neon go everywhere, not just on the Overview.
5. A **layer heatmap** component, plus the server endpoint that feeds it.
6. Light and dark themes from one token set. Light is the default and dark follows `prefers-color-scheme`.

## Non-goals (this project)

- Rethinking the information architecture of Layers, Spikes, Compare and Replay. They get the new look and chart component, not new layouts. Each can be its own follow-up.
- A ⌘K command palette. The mockup's search box becomes the run switcher instead.
- New analysis or detection logic. Every number on screen comes from data trainscope already records or computes (`run_signal_signature`, cluster reports, spike windows).

## Anti-goals (a failure even if it works)

- Any of the old tells coming back: neon-on-black, a rounded card per number, icons inside stat tiles, ALL-CAPS labels, gradient washes, grid-pattern backgrounds, staggered fade-in animations, Plotly's modebar.
- Color used for decoration. Color means *spike*, *threshold/warning*, or a *series identity*, and nothing else.
- A one-off spike that squashes the loss axis so the whole pre-spike history becomes a flat line.

## Visual language

Taken from the approved mockup.

- **Palette, light theme:** page `#fafafa`, surface `#ffffff`, border `#eaeaea`, ink `#171717`, muted `#6b6b6b`, spike/critical red `#e5484d`, series blue `#0a6cff` (grad norm). Warmup band: 3.5% black. Spike band: 10% red.
- **Palette, dark theme:** its own steps from the same hues, validated against the dark surface. It is not an automatic inversion. Run the dataviz palette validator (`validate_palette.js`) for both modes before shipping.
- **Type:** Geist Sans, self-hosted (the `geist` npm package), because the UI must work offline. Use tabular numerals for every number. One family. Layer names are set in the same sans.
- **Structure:** hairline borders separate regions. White panels sit on a light grey page. Numbers go in one segmented strip, not in cards. Status uses a small dot plus a word. Buttons: one primary (ink fill) and secondary (bordered).
- **Polish floor:** follow the make-interfaces-feel-better checklist. No `transition: all`. Hit areas of at least 40px. Visible focus rings. `text-wrap: balance` on titles. Concentric radii on nested panels.

## Shell

- **Top bar:** logo mark, then `trainscope / <run name> ▾`. The run name opens a popover listing runs with their status dot. Next comes a count pill ("4 runs"), and on the right a live indicator when the WebSocket is connected.
- **Tabs:** Overview · Layers · Spikes · Routing · Compare · Replay.
  - Routing appears only when the run has MoE data.
  - Compare appears only in multi-run mode.
- **Old views map to new tabs:** Timeline → Overview, Layer drill-down → Layers, Spike inspector → Spikes, Routing & addressing → Routing, Diff view → Compare, Replay → Replay. Runs → runs index (the breadcrumb's "trainscope" link).
- **Hash routing stays.** The new tab ids are `#overview`, `#layers`, `#spikes`, `#routing`, `#compare`, `#replay`. The 1.9.0 ids (`#timeline`, `#diff`, …) are accepted as aliases so existing links keep working.
- **Keyboard:** the number keys 1–6 switch tabs and `?` opens help, as today.

## Overview page

Two columns at ≥1280px: main is flexible, the side column is 340px. Top to bottom:

1. **Header:** run name, then a status line. The status is a dot plus a word plus detail, e.g. "Diverged at step 124, 129 steps recorded". Actions on the right: *Compare* (secondary) and *Replay from step N* (primary, where N is the replay start the Replay tab would suggest).
2. **Stat strip:** one segmented row of four: final loss, grad norm (latest), first warning (earliest signal crossing step, or "none"), spike window ("113–128", or "No spike").
3. **Loss and gradient norm panel:** two stacked charts that share the x axis and the cursor.
   - Warmup band (detector warming up).
   - Spike band(s).
   - A dashed vertical line at the first warning step, labelled at the bottom.
   - Grad norm shows its 3σ limit as a dashed line.
   - **Off-scale rule:** if a spike exists, the loss y-max is 1.25 × the maximum loss before the first spike. The line is clipped at the top and a red marker reads "`<peak>` at step `<n>`, off scale".
   - Dragging across the chart selects a step range, which zooms both charts and the heatmap. Double-click resets the zoom.
4. **Gradient norm by layer** (the heatmap, below).
5. **Side column:**
   - "Signals before the spike": a table with Signal (and where it fired, e.g. layer name), Crossed (step) and Lead (steps before detection). It comes from the existing signal-signature analysis. Where that analysis doesn't name a layer for a signal, the second line is omitted rather than guessed.
   - "Other runs" (multi-run only): status dot, name, and a short outcome ("diverged 125", "160 steps").

**Several spikes:** the Overview is anchored on the first spike. The strip reads "3 spikes, first 113–128", every spike window gets a band, and leads in the side table are measured to the first spike. Later spikes are reached from the Spikes tab.

**Live runs:** the Overview follows the existing WebSocket step events. Charts append points as they arrive. The heatmap matrix is re-fetched at most every 5 s while steps are arriving.

**Run status vocabulary:** Training (live, no spike yet), Stable (finished, no spike), Spiked (a spike was detected and the run recovered), Diverged. My call on the definition of Diverged: the last recorded loss is non-finite, or still above 10× the pre-spike median loss.

**Empty and stable runs:**
- No spike: the strip shows "No spike", the side table reads "No signal crossed its threshold", and the chart has no bands.
- No data yet: one line, "Waiting for the first step from `<run dir>`", plus the command that writes runs.

## Layer heatmap

- **Rows** are parameter tensors in model order.
  - With more than 24 rows, rows group by module path up to the first numbered block (e.g. `blocks.3`). A collapsed group shows the max of its children and can be expanded.
  - A toggle switches to sorting by first crossing, so the earliest-heating layer sits on top.
- **Columns** are step buckets aligned with the chart's x axis: the same left label column (132px) and the same plot width. A bucket holds the max over its steps. There are at most ~600 buckets across the visible range.
- **Value** is how far each layer's `grad_l2_norm` sits above its own baseline: a robust z = (x − median) / (1.4826 × MAD).
  - The baseline window is the steps before the first spike, or the first half of the run if there is no spike.
  - Values are clipped to [0, 6].
  - The color ramp is a single hue, from the surface color to deep red, validated in both themes.
- **Overlays:** the first warning step is a dashed line and the spike window has a red outline, matching the chart.
- **Interaction:**
  - Hover shows a tooltip (layer, step range, value, z) and the shared cursor.
  - Clicking a cell opens Layers on that layer, scrolled to that step.
  - Zoom follows the chart's selected range.
- **Rendering:** canvas, not SVG, because 300 rows × 600 buckets is too many DOM nodes.
- **Accessibility:** an `aria-label` summary names the first three layers to cross 3σ and their steps. Identity never depends on color alone, because every cell has a tooltip and rows are labelled.

**Data:** a new endpoint, `GET /api/layers/matrix?metric=grad_l2_norm&buckets=600&start=&end=`, returns `{layers: [...], bucket_starts: [...], values: [[...]], z: [[...]], baseline: {start, end}}`.
- It is computed server-side from the per-layer Arrow files.
- It is cached by file mtime, like `/api/layers/ranked`.
- The HTTP API is documented as a non-public contract, so this is an additive internal change.

## Charts

Plotly is replaced by **uPlot** for line charts. uPlot is a small canvas library that is fast on 100k+ points and has built-in cursor sync across charts, which is what gives the shared cursor between the loss chart, the grad-norm chart and the heatmap.
- Every Plotly use goes through `frontend/src/components/Chart.jsx` (plus spike-band helpers in `utils/spikeCluster.js`), so the swap is contained to that wrapper and its callers.
- This drops the ~3.5MB Plotly bundle.
- Charts follow the dataviz rules: one y-axis per chart (never dual axis), 2px lines, recessive grid, a legend for 2+ series, and a crosshair tooltip.

This is my call. The alternative was restyling Plotly, but its look and modebar are part of what reads as generic, and it cannot share a cursor with a canvas heatmap cheaply.

## Responsive

- **≥1280px:** two columns.
- **1024–1279px:** the side column moves below the main column.
- **<1024px:** single column, and the heatmap scrolls horizontally inside its panel. There is no horizontal page scroll at any width.

## Constraints

- Stack stays as is: React 18 + Vite + Tailwind + vitest. Tailwind colors move to CSS variables, so both themes come from one token file. `theme.js` `COLORS` becomes those variables.
- Offline: no runtime CDN or font requests.
- Existing frontend tests (40) and the Python suite keep passing. New tests cover:
  - the heatmap bucketing and z-score utility (vitest)
  - `/api/layers/matrix` (pytest: bucketing, baseline window, caching, a run with no spike)
  - hash aliases
- README screenshots are regenerated from `trainscope demo` after the redesign.

## Left to the builder

- Exact spacing scale, radii values and the dark palette steps, within the rules above.
- Component boundaries and file layout under `frontend/src`.
- uPlot plugin details for bands, the off-scale marker and the selection zoom.
- Copy for loading and error states, following the voice above: plain, specific, no apologies.

## Calls I made (for the owner to check)

1. Plotly → uPlot, and canvas for the heatmap.
2. Light by default, dark via system preference.
3. ⌘K palette dropped; the search box becomes the run switcher.
4. Old hash ids kept as aliases.
5. The "Diverged" definition (non-finite, or >10× the pre-spike median at the last step).
6. Heatmap grouping above 24 rows, and the robust-z baseline window.
7. Other tabs get the new look but keep their current layouts for now.
8. With several spikes the Overview anchors on the first. Live heatmap refresh is throttled to 5 s.
