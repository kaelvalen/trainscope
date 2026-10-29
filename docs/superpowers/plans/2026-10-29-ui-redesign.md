# UI Redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the generic dark dashboard with the approved Vercel/Linear-style UI: a light shell with tabs, an Overview built around loss/grad charts plus an aligned layer heatmap, and every other tab in the same language.

**Architecture:**
- **Backend.** Two additive endpoints feed the Overview:
  - `/api/layers/matrix` (per-layer robust-z buckets, built by a pure Python function)
  - `/api/signals` (the existing `run_signal_signature` for the current run)
- **Frontend.**
  - The shell, tokens and chart layer are rebuilt.
  - Plotly is replaced by uPlot behind two small chart components, and a canvas heatmap shares the chart's x axis and cursor.
  - Views keep their data logic and move onto the new primitives.

**Tech Stack:** FastAPI + pyarrow (backend). React 18, Vite 6, Tailwind 3, vitest, uPlot, the `geist` font package (frontend).

**Spec:** `docs/superpowers/specs/2026-10-29-ui-redesign-design.md`. The approved visual reference is `.superpowers/brainstorm/1641638-1793230815/content/merged.html` (local, git-excluded). Open it next to the spec.

## Global Constraints

- **Light palette:** page `#fafafa`, surface `#ffffff`, border `#eaeaea`, ink `#171717`, muted `#6b6b6b`, spike/critical red `#e5484d`, series blue `#0a6cff`. Warmup band is 3.5% black, spike band 10% red.
- **Dark palette:** its own steps from the same hues, validated against the dark surface. Never an automatic inversion. Light is the default; dark follows `prefers-color-scheme`.
- **Type:** Geist Sans, self-hosted, one family. `font-variant-numeric: tabular-nums` on every number.
- **Offline:** no runtime CDN or font requests.
- **Banned:**
  - neon-on-black
  - a rounded card per number
  - icons inside stat tiles
  - ALL-CAPS labels
  - gradient washes
  - grid-pattern backgrounds
  - staggered fade-in animations
  - Plotly's modebar
  - `transition: all`
- **Color only means:** spike, threshold/warning, or series identity.
- **Interaction floor:** hit areas ≥ 40px, visible focus rings, `text-wrap: balance` on titles.
- **Charts:**
  - One y-axis per chart (no `yaxis2`).
  - 2px lines, recessive grid.
  - A legend whenever there are 2+ series.
  - A crosshair tooltip.
- **Hash tab ids:** `#overview #layers #spikes #routing #compare #replay`. Aliases from 1.9.0: `#timeline`→overview, `#diff`→compare, `#moe`→routing, `#runs`→runs index.
- **Layout breakpoints:** two columns at ≥1280px (side column 340px). Side column below main at 1024–1279px. One column below 1024px. No horizontal page scroll at any width.
- **Verification:** existing frontend tests and the Python suite must keep passing. Run `cd frontend && npm test && npm run lint && npm run build`, then `PYTHONPATH=. pytest tests -q`.
- **Commits:** no `Co-Authored-By` or other Claude trailers (repo rule).

## Review Focus

1. **A run with no layer files, or layers without `grad_l2_norm`.** The matrix endpoint returns `layers: []` and the heatmap shows "No layer metrics recorded for this run", with no crash. Test: Task 1, Task 2 and Task 5.
2. **Non-finite loss values (NaN/inf) in a diverged run.** Charts draw a gap, not a crash or a NaN axis. The off-scale marker reads "non-finite at step N" when the peak is not finite. Test: Task 4.
3. **Runs shorter than the bucket count, or with 0–1 steps.**
   - The matrix uses one bucket per step when steps < buckets.
   - With 0 steps the Overview shows the waiting state.
   - With 1 step the charts render without division by zero.
   Test: Task 1, Task 4 and Task 8.
4. **Spikes arriving after first render on a live run.** The strip, the bands and the status update from `spikeEvents` without a reload. Test: Task 8.
5. **Unknown or legacy hash on load.** `#diff` opens Compare, `#garbage` opens Overview, and the URL is rewritten to the canonical id. Test: Task 6.

---

### Task 1: Layer matrix builder

**Files:**
- Create: `trainscope/layer_matrix.py`
- Test: `tests/test_layer_matrix.py`

**Interfaces:**
- Produces:
  - `build_layer_matrix(series: dict[str, list[tuple[int, float | None]]], *, buckets: int = 600, start: int | None = None, end: int | None = None, first_spike: int | None = None) -> dict`
  - It returns `{"layers": list[str], "bucket_starts": list[int], "values": list[list[float | None]], "z": list[list[float | None]], "baseline": {"start": int, "end": int} | None}`.
  - `natural_key(name: str) -> list` is also produced; it sorts `blocks.2` before `blocks.10`.

**Rules (from the spec):**
- Rows come in `natural_key` order.
- Buckets split `[start, end]` (defaults: min and max step present) into `min(buckets, n_steps)` contiguous ranges. A bucket's value is the max of the finite values in it, or `None` if there are none.
- **Baseline:** steps `< first_spike` when given, else the first half of the observed steps. z = (x − median) / (1.4826 × MAD) over the layer's baseline values, clipped to [0, 6], and computed per bucket from the bucket's value.
- If MAD is 0: z = 0 when x ≤ median, else 6.
- A layer with no finite values gets all-`None` rows.

- [ ] **Step 1: Write the failing tests**

```python
from trainscope.layer_matrix import build_layer_matrix, natural_key

def _ramp(n, onset=None):
    return [(s, 1.0 + 0.01 * (s % 3) + (0.5 * (s - onset) if onset is not None and s >= onset else 0.0)) for s in range(n)]

def test_rows_are_naturally_sorted():
    m = build_layer_matrix({"blocks.10.w": _ramp(4), "blocks.2.w": _ramp(4), "emb": _ramp(4)})
    assert m["layers"] == ["blocks.2.w", "blocks.10.w", "emb"]
    assert sorted(["b.10", "b.2"], key=natural_key) == ["b.2", "b.10"]

def test_one_bucket_per_step_when_short():
    assert build_layer_matrix({"a": _ramp(5)}, buckets=600)["bucket_starts"] == [0, 1, 2, 3, 4]

def test_bucket_takes_max_and_respects_range():
    m = build_layer_matrix({"a": [(s, float(s)) for s in range(100)]}, buckets=10, start=20, end=59)
    assert m["bucket_starts"][0] == 20 and len(m["bucket_starts"]) == 10
    assert m["values"][0][0] == 23.0

def test_baseline_is_pre_spike_and_z_flags_onset():
    m = build_layer_matrix({"a": _ramp(60, onset=40)}, buckets=60, first_spike=45)
    assert m["baseline"] == {"start": 0, "end": 44}
    assert m["z"][0][10] < 3
    assert m["z"][0][50] == 6.0

def test_no_spike_uses_first_half():
    assert build_layer_matrix({"a": _ramp(40)}, buckets=40)["baseline"] == {"start": 0, "end": 19}

def test_nonfinite_values_become_gaps():
    m = build_layer_matrix({"a": [(0, float("nan")), (1, None), (2, float("inf"))], "b": _ramp(3)})
    assert m["values"][0] == [None, None, None] and m["z"][0] == [None, None, None]

def test_empty_input():
    assert build_layer_matrix({})["layers"] == []
```

- [ ] **Step 2: Run them and confirm they fail.** Run `PYTHONPATH=. pytest tests/test_layer_matrix.py -q`. Expected: `ModuleNotFoundError: trainscope.layer_matrix`.
- [ ] **Step 3: Implement `build_layer_matrix` and `natural_key` in `trainscope/layer_matrix.py`.** Use the stdlib only (`statistics.median`, `math.isfinite`). `natural_key` splits on `(\d+)` and converts the digit runs to int.
- [ ] **Step 4: Run them again and confirm they pass.** Same command. Expected: 7 passed.
- [ ] **Step 5: Commit.** `git add trainscope/layer_matrix.py tests/test_layer_matrix.py && git commit -m "Add per-layer robust-z matrix builder for the layer heatmap"`

### Task 2: Matrix and signals endpoints

**Files:**
- Modify: `trainscope/ui/server.py`. Add the routes before `@app.get("/api/layers/{layer_name:path}")` at ~line 895, so `matrix` is not captured as a layer name.
- Modify: `frontend/src/api.js` (add the fetchers next to `fetchLayersRanked`).
- Test: `tests/test_server.py`

**Interfaces:**
- Consumes: `build_layer_matrix` (Task 1), `run_signal_signature` and `get_spike_steps` (`trainscope/analysis.py`), `_cache` (`_TTLCache`, in the server), `_decode_layer_name`.
- Produces:
  - `GET /api/layers/matrix?metric=grad_l2_norm&buckets=600&start=&end=` returns the Task 1 dict.
    - `metric` must be one of `grad_l2_norm`, `weight_l2_norm`, `grad_max_abs`, `act_kurtosis`; anything else gets a 400.
    - `buckets` is limited to 1..2000.
    - `first_spike` is `min(get_spike_steps(rp))` or None.
    - The cache key includes the metric, buckets, start, end and the max mtime of `layers/*.arrow`.
  - `GET /api/signals` returns `run_signal_signature(rp)`.
  - `api.js`: `fetchLayerMatrix({ metric = 'grad_l2_norm', buckets = 600, start, end } = {})` and `fetchSignals()`.

- [ ] **Step 1: Write the failing tests** in `tests/test_server.py`, reusing the existing `run_dir`/`client` fixtures:

```python
def test_layer_matrix(client):
    body = client.get("/api/layers/matrix?buckets=5").json()
    assert set(body) == {"layers", "bucket_starts", "values", "z", "baseline"}
    assert body["layers"] and len(body["values"]) == len(body["layers"])

def test_layer_matrix_rejects_unknown_metric(client):
    assert client.get("/api/layers/matrix?metric=hist_counts").status_code == 400

def test_layer_matrix_without_layers(tmp_path):
    run = tmp_path / "run"; run.mkdir(); (run / "meta.json").write_text("{}")
    with TestClient(create_app(run)) as c:
        assert c.get("/api/layers/matrix").json()["layers"] == []

def test_signals(client):
    assert {"fired", "first", "first_steps", "explosion_step"} <= set(client.get("/api/signals").json())
```

- [ ] **Step 2: Run them and confirm they fail.** Run `PYTHONPATH=. pytest tests/test_server.py -q -k "matrix or signals"`. Expected: 404 / assertion failures.
- [ ] **Step 3: Implement the two routes and the two fetchers.** Read each layer file with `read_arrow` and map it to `(row["step"], row.get(metric))`. If `create_app`'s signature differs from `create_app(run)`, follow the existing fixture.
- [ ] **Step 4: Run the full server test file.** Run `PYTHONPATH=. pytest tests/test_server.py -q`. Expected: all pass.
- [ ] **Step 5: Commit.** `git commit -am "Serve layer matrix and run signals for the Overview"`

### Task 3: Tokens, font and theme

**Files:**
- Create: `frontend/src/styles/tokens.css`
- Modify: `frontend/src/styles/index.css` (replace it wholesale: import the tokens, font-face, base element styles, focus ring; drop the old shell, card, grid and glow rules)
- Modify: `frontend/tailwind.config.js`, `frontend/src/theme.js`, `frontend/package.json` (add `geist`)
- Test: `frontend/src/__tests__/theme.test.js`

**Interfaces:**
- Produces:
  - CSS variables: `--page --surface --border --ink --muted --red --red-band --warm-band --blue --amber`, plus `--radius-panel: 10px` and `--radius-control: 8px`. Defined under `:root`, redefined under `@media (prefers-color-scheme: dark)`.
  - Tailwind colors `page surface border ink muted red blue amber`, mapped to those variables.
  - `theme.js`:
    - `CHART_COLORS = { loss: 'var(--ink)', gradNorm: 'var(--blue)', weight: 'var(--muted)', kurtosis: 'var(--amber)', spike: 'var(--red)' }`
    - `readToken(name: string): string`, which resolves a CSS variable via `getComputedStyle(document.documentElement)` (canvas code needs real colors).
    - `DARK_LAYOUT` and `PLOT_CONFIG` stay exported, untouched, until Task 12 deletes them with `Chart.jsx`. The build stays green while views migrate.

- [ ] **Step 1: Write the failing test.** Assert that `readToken('--ink')` returns the value set via `document.documentElement.style.setProperty('--ink', '#171717')`, and that `CHART_COLORS.spike === 'var(--red)'`.
- [ ] **Step 2: Run it and confirm it fails.** Run `cd frontend && npx vitest run src/__tests__/theme.test.js`. Expected: FAIL (`readToken` not exported).
- [ ] **Step 3: Install and wire the font.** Run `npm install geist`. Find the variable woff2 with `ls node_modules/geist/dist/fonts/geist-sans/` and reference it from `@font-face` in `index.css`. Set Tailwind `fontFamily.sans = ['Geist', 'system-ui', 'sans-serif']` and drop the JetBrains Mono default.
- [ ] **Step 4: Write `tokens.css` with the Global Constraints light values, then choose and validate the dark steps.**
  - Validate both with `node "$(ls -d /tmp/claude-1000/bundled-skills/*/*/dataviz | head -1)/scripts/validate_palette.js" "<ink>,<blue>,<red>,<amber>" --mode light`, and the same with `--mode dark` using the dark values.
  - Fix any FAIL before moving on.
- [ ] **Step 5: Run the tests, then build.** Run `npx vitest run && npm run build`. Expected: pass and build OK.
- [ ] **Step 6: Commit.** `git commit -am "Add light/dark design tokens and self-hosted Geist"`

### Task 4: uPlot chart components

**Files:**
- Create: `frontend/src/components/charts/LineChart.jsx`, `frontend/src/components/charts/BarChart.jsx`, `frontend/src/components/charts/chartOptions.js`
- Modify: `frontend/package.json` (add `uplot`)
- Test: `frontend/src/__tests__/chartOptions.test.js`

**Interfaces:**
- Consumes: `CHART_COLORS` and `readToken` (Task 3).
- Produces:
  - `<LineChart x={number[]} series={[{label, values: (number|null)[], color, width?=2, dash?, points?}]} height={number} bands={[{from, to, kind: 'spike'|'warmup'}]} markers={[{at, label?, kind: 'warning'|'spike'}]} limit={{value, label}?} yMax={number?} syncKey={string?} xRange={[from, to]?} onSelectRange={([from, to]) => void?} onResetRange={() => void?} />`
  - `<BarChart x={(number|string)[]} series={[...same shape]} height bands? />`
  - In `chartOptions.js`:
    - `lossYMax(steps: number[], loss: (number|null)[], firstSpike: number|null): number|null` returns 1.25 × the max finite loss at steps before `firstSpike`, or null when there is no spike or no pre-spike data.
    - `offScaleMarker(steps, values, yMax): {at: number, text: string}|null`. `at` is the first step whose value exceeds `yMax` or is non-finite. `text` is `"<peak, 1 decimal> at step <peak step>, off scale"`, or `"non-finite at step <n>"` when that point is non-finite.
    - `buildLineOptions(props, size: {width}): uPlot.Options` is pure.
- **Behavior:**
  - Non-finite values become `null` (gaps).
  - The cursor syncs through `uPlot.sync(syncKey)`.
  - Drag-select calls `onSelectRange` and does not zoom internally (the parent owns `xRange`). Double-click calls `onResetRange`.
  - The tooltip is an absolutely-positioned div showing the step and each series value with tabular nums.
  - Bands, markers, limit and the off-scale marker are drawn in a `draw` hook with `readToken` colors.
  - Width follows the container via `ResizeObserver`.

- [ ] **Step 1: Write the failing tests:**

```js
import { lossYMax, offScaleMarker, buildLineOptions } from '../components/charts/chartOptions.js'
test('lossYMax is 1.25x the pre-spike max', () => {
  expect(lossYMax([0, 1, 2, 3], [4, 2, 3, 60], 3)).toBeCloseTo(5)
  expect(lossYMax([0, 1], [4, 2], null)).toBeNull()
})
test('offScaleMarker reports peak or non-finite', () => {
  expect(offScaleMarker([0, 1, 2], [1, 9, 63.6], 5)).toEqual({ at: 1, text: '63.6 at step 2, off scale' })
  expect(offScaleMarker([0, 1], [1, NaN], 5).text).toBe('non-finite at step 1')
  expect(offScaleMarker([0, 1], [1, 2], 5)).toBeNull()
})
test('options: one y scale, 2px lines, single-step safe', () => {
  const o = buildLineOptions({ x: [0], series: [{ label: 'Loss', values: [Infinity], color: 'var(--ink)' }], height: 200 }, { width: 600 })
  expect(Object.keys(o.scales)).toEqual(['x', 'y'])
  expect(o.series[1].width).toBe(2)
})
```

- [ ] **Step 2: Run them and confirm they fail.** Run `npx vitest run src/__tests__/chartOptions.test.js`.
- [ ] **Step 3: Implement.** Run `npm install uplot`, then write the three files (bars via `uPlot.paths.bars`). Import `uplot/dist/uPlot.min.css` once and override its fonts and colors with tokens.
- [ ] **Step 4: Run the tests again; expect them to pass.**
- [ ] **Step 5: Commit.** `git commit -am "Add uPlot LineChart/BarChart with bands, markers and synced cursor"`

### Task 5: Layer heatmap

**Files:**
- Create: `frontend/src/components/charts/LayerHeatmap.jsx`, `frontend/src/utils/layerMatrix.js`
- Test: `frontend/src/__tests__/layerMatrix.test.js`

**Interfaces:**
- Consumes: the Task 2 response shape, `readToken` (Task 3), and the shared `syncKey` cursor (Task 4).
- Produces, in `layerMatrix.js`:
  - `groupRows(matrix, { maxRows = 24, expanded = new Set() }) -> [{ key, label, isGroup, children?: string[], z: (number|null)[], values: (number|null)[] }]`. Grouping applies only when `matrix.layers.length > maxRows`. The group key is the path up to and including the first numeric segment (`blocks.3.attn.qkv.weight` → `blocks.3`); names without a numeric segment stay ungrouped. A group's z and values are the per-bucket max of its children; expanded groups also emit their children.
  - `firstCrossings(rows, bucketStarts, threshold = 3) -> Map<key, number|null>`
  - `sortRows(rows, mode: 'model'|'first', crossings)` puts null crossings last.
  - `zColor(z, surfaceRgb: [r,g,b], redRgb: [r,g,b]) -> string` is a linear ramp with `t = clamp(z / 6)`; `null` gives the surface.
  - `heatmapSummary(rows, crossings) -> string` is the aria-label, listing at most three rows. Example: "First above 3σ: blocks.1.mlp.fc2 at step 98, blocks.1.mlp.fc1 at step 101, blocks.1.attn.qkv at step 104". With no crossings it reads "No layer crossed 3σ".
- Produces: `<LayerHeatmap matrix labelWidth={132} plotPadRight={10} warningStep spikeWindow={{from, to}?} syncKey xRange onCellClick={(layer, step) => void} />`
  - Canvas rendering, scaled by `devicePixelRatio`.
  - Columns use the same x→px mapping as the charts: the same left label width and right pad, and `xRange` when set.
  - Overlays: a dashed line at `warningStep` and a red outline around the spike window.
  - A hover tooltip shows layer, step range, value and z.
  - A "Sort by first crossing" toggle; group rows expand on click.
  - Empty matrix shows "No layer metrics recorded for this run".

- [ ] **Step 1: Write the failing tests:**
  - 30 synthetic rows (`blocks.{0..14}.w` and `.b`) group into 15 rows, with group z equal to the child max.
  - No grouping at ≤ 24 rows.
  - `firstCrossings` returns the first bucket start with z ≥ 3.
  - `sortRows('first')` order, with nulls last.
  - The exact `heatmapSummary` strings above.
  - `zColor(0)` equals the surface rgb and `zColor(6)` the red rgb.
  - `groupRows` of an empty matrix returns `[]`.
- [ ] **Step 2: Run them and confirm they fail.** Run `npx vitest run src/__tests__/layerMatrix.test.js`.
- [ ] **Step 3: Implement `layerMatrix.js`, then `LayerHeatmap.jsx`.**
- [ ] **Step 4: Run the tests again; expect them to pass.**
- [ ] **Step 5: Commit.** `git commit -am "Add canvas layer heatmap aligned with the chart axis"`

### Task 6: App shell and primitives

**Files:**
- Create:
  - `frontend/src/components/shell/TopBar.jsx` (logo mark, breadcrumb `trainscope / <run> ▾` whose popover lists `runs` with status dots and calls `switchRun`, the "N runs" pill, live indicator from `liveStatus`)
  - `frontend/src/components/shell/Tabs.jsx`
  - `frontend/src/components/ui/Panel.jsx` (`<Panel title actions>`: surface, hairline border, header row)
  - `frontend/src/components/ui/StatStrip.jsx` (`items=[{label, value}]`, one segmented row)
  - `frontend/src/components/ui/StatusDot.jsx` (`kind`)
- Modify:
  - `frontend/src/navigation.js`: `NAV_ITEMS = [{id, label, shortcut}]` for overview, layers, spikes, routing, compare, replay. Export `HASH_ALIASES` and `resolveTab(hash: string, visibleIds: string[]) -> {view: string, canonical: string}`. `runs` is a non-tab view reached from the breadcrumb.
  - `frontend/src/App.jsx`:
    - The layout is TopBar, then Tabs, then the view.
    - Routing is visible only when `moeData.length > 0`; Compare only when `runs.length > 1`.
    - Keys `1`–`6` select visible tabs in order, `r` refreshes, `?` opens help.
    - The canonical hash is written with `replaceState`.
  - `frontend/src/components/ui/Button.jsx`, `Badge.jsx`: restyle. `Button` gets `variant: 'primary'|'secondary'`, is 32px tall, and has a ≥ 40px hit area.
- Test: `frontend/src/__tests__/navigation.test.js`

**Interfaces:**
- Produces: `resolveTab`, `HASH_ALIASES`, `NAV_ITEMS`, and `Panel`, `StatStrip`, `StatusDot`, `Button`. `Sidebar`, `Navbar`, `PageHeader`, `StatCard`, `ChartCard` and `Card` stop being used by `App.jsx`; they are deleted in Task 12 once no view imports them.

- [ ] **Step 1: Write the failing tests.** With `ids` = all six:
  - `resolveTab('#diff', ids)` equals `{view: 'compare', canonical: '#compare'}`.
  - `resolveTab('#timeline', ids).view` is `'overview'`.
  - `resolveTab('#garbage', ids).view` is `'overview'`.
  - `resolveTab('#routing', ids.filter(i => i !== 'routing')).view` is `'overview'`.
  - `resolveTab('#runs', ids)` equals `{view: 'runs', canonical: '#runs'}`.
- [ ] **Step 2: Run them and confirm they fail.** Run `npx vitest run src/__tests__/navigation.test.js`.
- [ ] **Step 3: Implement navigation, the shell components and `App.jsx`.** Views render inside a `max-width: 1360px` container with 32px gutters (16px below 640px).
- [ ] **Step 4: Run all tests and lint.** Run `npx vitest run && npm run lint`. Update `RunContext.test.jsx` only where it asserted old nav labels.
- [ ] **Step 5: Commit.** `git commit -am "Replace sidebar shell with top bar, tabs and hash aliases"`

### Task 7: Run status

**Files:**
- Create: `frontend/src/utils/runStatus.js`
- Test: `frontend/src/__tests__/runStatus.test.js`

**Interfaces:**
- Produces: `runStatus({ globalData, spikeEvents, live }) -> { kind: 'waiting'|'training'|'stable'|'spiked'|'diverged', label: string, detail: string }`
- **Rules:**
  - No rows → `waiting`.
  - Diverged when the last loss is non-finite, or > 10 × the median loss of rows before the first spike's `startStep` (or of all rows when there is no spike). Detail: `"Diverged at step <first step meeting the rule>, <n> steps recorded"`.
  - Spiked when spikes exist and the run is not diverged. Detail: `"<k> spike(s), recovered"`.
  - Training when `live` and there are no spikes.
  - Otherwise stable, with detail `"<n> steps, no spike"`.

- [ ] **Step 1: Write the failing tests**, one per rule, including a NaN last loss and the exact detail strings.
- [ ] **Step 2: Run them and confirm they fail.** Run `npx vitest run src/__tests__/runStatus.test.js`.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run them again; expect them to pass.**
- [ ] **Step 5: Commit.** `git commit -am "Add run status vocabulary"`

### Task 8: Overview view

**Files:**
- Create: `frontend/src/views/Overview.jsx`
- Delete: `frontend/src/views/Timeline.jsx`
- Modify: `frontend/src/App.jsx` (view map)
- Test: `frontend/src/__tests__/Overview.test.jsx`. Mock `useRun` and `api.js` the way `Replay.test.jsx` does.

**Interfaces:**
- Consumes:
  - `useRun()`: `globalData, spikeEvents, meta, runs, switchRun, liveStatus`
  - `fetchLayerMatrix` and `fetchSignals` (Task 2)
  - `LineChart`, `lossYMax`, `offScaleMarker` (Task 4)
  - `LayerHeatmap` (Task 5)
  - `Panel`, `StatStrip`, `StatusDot`, `Button` (Task 6)
  - `runStatus` (Task 7)
  - `computeWarmupBand` (existing, `utils/warmup.js`)
- **Layout (top to bottom):**
  1. Header: run name and `StatusDot` with `runStatus().detail`. Actions: *Compare* (secondary, when `runs.length > 1`) and *Open replay* (primary, goes to `#replay`).
  2. `StatStrip`: Final loss, Grad norm, First warning, Spike window.
  3. Panel "Loss and gradient norm":
     - Loss `LineChart` (200px) with the warmup band, spike bands, a warning marker, `yMax = lossYMax(...)` and the off-scale marker.
     - Grad norm `LineChart` (104px) with `limit` at the 3σ line.
     - Both share `syncKey="overview"`, `xRange` and the 132px left gutter.
  4. Panel "Gradient norm by layer" with `LayerHeatmap`.
  5. Side column: Panel "Signals before the spike" (Signal, Crossed, Lead), then Panel "Other runs" (when `runs.length > 1`).
- **Values:**
  - First warning = the min of `signals.first_steps` excluding `loss`; "none" if there is none.
  - Signal labels: `kurtosis` → "Activation kurtosis", `grad_norm` → "Grad norm", `concentration` → "Routing concentration", `loss` → "Loss".
  - Lead = `firstSpike.startStep − crossed`, shown when ≥ 0.
  - There is no second line per signal (the analysis names no layer).
  - The grad-norm 3σ limit is the median + 3 × 1.4826 × MAD of the pre-spike grad norms, which is the same rule as `first_crossing_step`.
- **Spike window cell:** `"No spike"`, `"113–128"`, or `"3 spikes, first 113–128"`.
- **Empty states:**
  - Waiting: "Waiting for the first step from `<meta run name>`", with the hint `trainscope demo`.
  - No signals: "No signal crossed its threshold".
- **Live:** charts follow `globalData`. `fetchLayerMatrix` reruns when `globalData.length` changes, at most once per 5000 ms (a last-fetch ref plus a trailing timeout).
- **Interaction:** drag-select on either chart sets `xRange` for both charts and the heatmap; double-click clears it. A heatmap cell click goes to `#layers?layer=<encoded name>`, and Task 10 reads that param.
- **Note:** the spec's "Replay from step N" ships as "Open replay", because the Replay view suggests no start step today. Say so in the commit message.

- [ ] **Step 1: Write the failing tests:**
  - Waiting state with `globalData: []`.
  - Stable run: the strip shows "No spike" and the side panel "No signal crossed its threshold".
  - Three `spikeEvents`: the strip shows "3 spikes, first 113–128".
  - The signals table is chronological, with leads `15` and `5` for `first_steps {kurtosis: 98, grad_norm: 108, loss: 113}` and first spike start 113.
  - A rerender with an added spike event updates the strip without remount (Review Focus 4).
  - A single-row `globalData` renders without throwing (Review Focus 3).
  - Mock `LineChart` and `LayerHeatmap` as stubs in this test; their behavior is covered in Tasks 4–5.
- [ ] **Step 2: Run them and confirm they fail.** Run `npx vitest run src/__tests__/Overview.test.jsx`.
- [ ] **Step 3: Implement `Overview.jsx` and swap it in for Timeline in `App.jsx`.**
- [ ] **Step 4: Run all tests and the build.** Run `npx vitest run && npm run build`. Expected: pass.
- [ ] **Step 5: Look at it.**
  - Run `trainscope demo --dir /tmp/ts-demo` (serves the UI).
  - Screenshot `#overview` with headless Chrome at 1440×900, 1100×900 and 800×900.
  - Compare against `merged.html`: chart and heatmap columns line up, the off-scale marker is visible, and there is no horizontal page scroll.
- [ ] **Step 6: Commit.** `git commit -am "Add Overview: status, strip, synced charts, layer heatmap, signals"`

### Task 9: Runs index

**Files:**
- Modify: `frontend/src/views/Runs.jsx`. Replace cards and Plotly with `Panel`, a runs table, and `LineChart` for the overlays; the `fill: 'tonexty'` envelopes at lines 217–236 become two lines plus a legend.
- Modify: `frontend/src/__tests__/Runs.test.jsx`

**Interfaces:**
- Consumes: Tasks 4, 6 and 7.
- Produces: the `runs` view. Table columns: Run, Status, Steps, First signal, Lead, Cluster. A row click calls `switchRun(name)` and opens `#overview`.

- [ ] **Step 1: Update `Runs.test.jsx`.** Assert the column headers, each row's status word, and that a row click calls `switchRun`. Run it and confirm it fails against the old view.
- [ ] **Step 2: Rewrite the rendering, keeping the view's data fetching and derived values.**
- [ ] **Step 3: Run `npx vitest run && npm run lint`; expect them to pass.**
- [ ] **Step 4: Commit.** `git commit -am "Restyle runs index as a table on the new primitives"`

### Task 10: Layers and Spikes tabs

**Files:**
- Modify: `frontend/src/views/LayerDrilldown.jsx` (`<Chart>` at 231 line, 272 histogram bar), `frontend/src/views/SpikeInspector.jsx` (214 with `yaxis2` at 237, 272, 294)
- Modify: `frontend/src/utils/spikeCluster.js`. Replace `buildSpikeClusterShapes`/`buildSpikeClusterAnnotations` with `spikeBands(events) -> [{from, to, kind: 'spike'}]`; `groupSpikes` stays unchanged.
- Modify: `frontend/src/__tests__/spikeCluster.test.js`

**Interfaces:**
- Consumes: Tasks 4 and 6.
- Produces:
  - `spikeBands(events)`.
  - SpikeInspector's dual-axis chart becomes two stacked `LineChart`s on one `syncKey`.
  - LayerDrilldown preselects the layer from `?layer=` in the hash (Task 8).

- [ ] **Step 1: Update `spikeCluster.test.js`.** Expect `spikeBands([{startStep: 113, endStep: 128}])` to equal `[{from: 113, to: 128, kind: 'spike'}]`, and drop the shape/annotation tests. Run it and confirm it fails.
- [ ] **Step 2: Implement `spikeBands`, then migrate both views onto `Panel`, `LineChart` and `BarChart`.** Drop the `Card`, `StatCard`, `ChartCard` and `PageHeader` imports.
- [ ] **Step 3: Run `npx vitest run && npm run lint`; expect them to pass.**
- [ ] **Step 4: Commit.** `git commit -am "Move Layers and Spikes tabs to the new charts; split dual-axis chart"`

### Task 11: Compare, Routing and Replay tabs

**Files:**
- Modify: `frontend/src/views/DiffView.jsx` (233 bar), `frontend/src/views/ExpertUtilization.jsx` (103 lines), `frontend/src/views/Replay.jsx` (144 lines plus markers)
- Modify: `frontend/src/__tests__/ExpertUtilization.test.jsx`, `frontend/src/__tests__/Replay.test.jsx`, but only where they assert removed markup

**Interfaces:**
- Consumes: Tasks 4 and 6.
- Replay's skipped-step markers become a `points: true, width: 0` series in `var(--red)`.
- ExpertUtilization colors:
  - Use a fixed categorical order: ink, blue, amber, then two muted tones from `tokens.css`.
  - Validate it in light and dark with the dataviz validator.
  - Fold experts beyond five into "Other". Never cycle colors.

- [ ] **Step 1: Migrate the three views and adjust their tests' markup expectations.** Run `npx vitest run` and confirm it passes.
- [ ] **Step 2: Run `npm run lint && npm run build`; expect them to pass.**
- [ ] **Step 3: Commit.** `git commit -am "Move Compare, Routing and Replay tabs to the new charts"`

### Task 12: Remove Plotly, clean up, document

**Files:**
- Modify: `frontend/package.json` (remove `plotly.js-dist-min` and `react-plotly.js`; remove `lucide-react` if unused), `frontend/src/theme.js` (delete `DARK_LAYOUT`, `PLOT_CONFIG`, `COLORS`)
- Delete: `frontend/src/components/Chart.jsx`, `components/ui/Sidebar.jsx`, `Navbar.jsx`, `PageHeader.jsx`, `StatCard.jsx`, `ChartCard.jsx`, `Card.jsx`, and any CSS no longer referenced
- Modify: `README.md` (screenshot references), `CHANGELOG.md` (`[Unreleased]` → Changed: new UI, Plotly → uPlot, `/api/layers/matrix`, `/api/signals`)
- Replace: `docs/images/runs-view.png`. Add `docs/images/overview.png`, which replaces `timeline-view.png` (delete the old file and update the references).

- [ ] **Step 1: Run the anti-goal checks; each must print nothing:**

```bash
cd frontend
grep -rn "plotly" src package.json
grep -rn "transition: all\|transition-all" src
grep -rnE "\buppercase\b|tracking-\[0\.[1-9]" src
grep -rn "bg-gradient\|linear-gradient" src
```

- [ ] **Step 2: Run the full verification.** Run `npm ci && npm test && npm run lint && npm run build && cd .. && PYTHONPATH=. pytest tests -q`. All must pass. Record the built JS size from the Vite output before (main branch) and after for the CHANGELOG, as measured numbers.
- [ ] **Step 3: Regenerate the screenshots.**
  - With `trainscope demo --dir /tmp/ts-demo` running, capture headless Chrome at 1440×900: `#overview` to `docs/images/overview.png` and `#runs` to `docs/images/runs-view.png`.
  - Open both images and check them against the spec's anti-goals.
- [ ] **Step 4: Commit.** `git commit -am "Remove Plotly and old shell; refresh README screenshots"`
