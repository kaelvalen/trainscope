const BASE = '/api'

export async function fetchMeta() { return (await fetch(`${BASE}/meta`)).json() }
export async function fetchGlobal() { return (await fetch(`${BASE}/global`)).json() }
export async function fetchLayers() { return (await fetch(`${BASE}/layers`)).json() }
export async function fetchLayersRanked(topN = 8) { return (await fetch(`${BASE}/layers/ranked?top_n=${topN}`)).json() }
export async function fetchLayer(name) { return (await fetch(`${BASE}/layers/${encodeURIComponent(name)}`)).json() }
export async function fetchSpikes() { return (await fetch(`${BASE}/spikes`)).json() }
export async function fetchSpike(step) { return (await fetch(`${BASE}/spikes/${step}`)).json() }
export async function fetchSpikeLayerNames(step) { return (await fetch(`${BASE}/spikes/${step}/layers`)).json() }
export async function fetchSpikeLayer(step, name) { return (await fetch(`${BASE}/spikes/${step}/layers/${encodeURIComponent(name)}`)).json() }
export async function fetchDiff(stepA, stepB) { return (await fetch(`${BASE}/diff?step_a=${stepA}&step_b=${stepB}`)).json() }
