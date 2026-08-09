const BASE = '/api'

const DEFAULT_RETRIES = 2
const RETRY_DELAY_MS = 300

/**
 * In-flight request registry. `abortAllRequests()` cancels every active call;
 * individual callers can also cancel their own request via the returned
 * `abort()` function (see `requestWithAbort`).
 */
const activeControllers = new Set()

function register(controller) {
  activeControllers.add(controller)
  controller.signal.addEventListener('abort', () => activeControllers.delete(controller))
}

function unregister(controller) {
  activeControllers.delete(controller)
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms))
}

export class ApiError extends Error {
  constructor(message, status = null) {
    super(message)
    this.name = 'ApiError'
    this.status = status
  }

  /** True when the error represents a client-side mistake (4xx). */
  isClientError() {
    return this.status != null && this.status >= 400 && this.status < 500
  }
}

function validateApiResponse(path, data) {
  if (path === '/meta' || path.startsWith('/manifest') || path.startsWith('/health')) {
    if (!data || typeof data !== 'object') {
      throw new ApiError(`Invalid response from ${path}: expected object`)
    }
    return
  }

  const listEndpoints = ['/global', '/layers', '/spikes', '/layers/ranked', '/runs']
  const basePath = path.split('?')[0]
  if (listEndpoints.includes(basePath)) {
    if (!Array.isArray(data)) {
      throw new ApiError(`Invalid response from ${path}: expected array`)
    }
    return
  }

  // Layer/spike/diff responses are all arrays of row dicts.
  if (Array.isArray(data)) return
  if (data && typeof data === 'object') return
  throw new ApiError(`Invalid response from ${path}: expected array or object`)
}

async function request(path, options = {}) {
  const {
    retries = DEFAULT_RETRIES,
    retryDelay = RETRY_DELAY_MS,
    signal,
    ...fetchOptions
  } = options
  const url = `${BASE}${path}`
  let lastError

  for (let attempt = 0; attempt <= retries; attempt++) {
    const controller = new AbortController()
    register(controller)

    // Honor an external abort signal by aborting our internal controller.
    let externalAbortHandler
    if (signal) {
      externalAbortHandler = () => controller.abort()
      if (signal.aborted) {
        controller.abort()
      } else {
        signal.addEventListener('abort', externalAbortHandler)
      }
    }

    try {
      const response = await fetch(url, { ...fetchOptions, signal: controller.signal })
      if (!response.ok) {
        throw new ApiError(`HTTP ${response.status}: ${response.statusText}`, response.status)
      }
      const data = await response.json()
      validateApiResponse(path, data)
      return data
    } catch (err) {
      lastError = err
      if (err.name === 'AbortError') throw err

      // Do not retry client errors (4xx) except for 408 / 429.
      if (
        err instanceof ApiError &&
        err.isClientError() &&
        err.status !== 408 &&
        err.status !== 429
      ) {
        throw err
      }

      if (attempt < retries) {
        await sleep(retryDelay * 2 ** attempt)
      }
    } finally {
      if (signal && externalAbortHandler) {
        signal.removeEventListener('abort', externalAbortHandler)
      }
      unregister(controller)
    }
  }

  throw lastError
}

/**
 * Make a cancellable request.
 *
 * Returns `{ promise, abort }`. Calling `abort()` aborts the underlying fetch.
 * This is useful in `useEffect` clean-up functions to cancel stale requests.
 */
export function requestWithAbort(path, options = {}) {
  const controller = new AbortController()
  const promise = request(path, { ...options, signal: controller.signal })
  return {
    promise,
    abort: () => controller.abort(),
  }
}

export function abortAllRequests() {
  activeControllers.forEach((c) => {
    if (!c.signal.aborted) c.abort()
  })
  activeControllers.clear()
}

export async function fetchMeta() {
  return request('/meta')
}

export async function fetchGlobal() {
  return request('/global')
}

export async function fetchLayers() {
  return request('/layers')
}

export async function fetchLayersRanked(topN = 8) {
  if (!Number.isInteger(topN) || topN <= 0) {
    throw new ApiError('topN must be a positive integer')
  }
  return request(`/layers/ranked?top_n=${topN}`)
}

export async function fetchLayer(name) {
  if (!name) throw new ApiError('Layer name is required')
  return request(`/layers/${encodeURIComponent(name)}`)
}

export async function fetchSpikes() {
  return request('/spikes')
}

export async function fetchSpike(step) {
  if (!Number.isInteger(step)) throw new ApiError('Step must be an integer')
  return request(`/spikes/${step}`)
}

export async function fetchSpikeLayerNames(step) {
  if (!Number.isInteger(step)) throw new ApiError('Step must be an integer')
  return request(`/spikes/${step}/layers`)
}

export async function fetchSpikeLayer(step, name) {
  if (!Number.isInteger(step)) throw new ApiError('Step must be an integer')
  if (!name) throw new ApiError('Layer name is required')
  return request(`/spikes/${step}/layers/${encodeURIComponent(name)}`)
}

export async function fetchDiff(stepA, stepB) {
  const a = Number(stepA)
  const b = Number(stepB)
  if (!Number.isInteger(a) || !Number.isInteger(b)) {
    throw new ApiError('Both step numbers must be integers')
  }
  return request(`/diff?step_a=${a}&step_b=${b}`)
}

export async function fetchRuns() {
  return request('/runs')
}

export async function fetchCompare(runNames) {
  if (!Array.isArray(runNames) || runNames.length < 2) {
    throw new ApiError('At least two run names are required for comparison')
  }
  return request(`/compare?runs=${encodeURIComponent(runNames.join(','))}`)
}

export async function selectRun(name) {
  if (!name) throw new ApiError('Run name is required')
  return request('/runs/select', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ name }),
  })
}
