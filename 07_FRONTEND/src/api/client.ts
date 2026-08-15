/**
 * API client.
 *
 * One place that knows how to reach the backend, so no component ever builds a
 * URL. The base URL is resolved from the environment at build time and falls
 * back to a SAME-ORIGIN path — never to a hard-coded localhost — so the same
 * bundle works behind any reverse proxy.
 */

export class ApiError extends Error {
  constructor(
    public readonly status: number,
    public readonly errorType: string,
    message: string,
    public readonly context?: Record<string, unknown>,
    public readonly requestId?: string,
  ) {
    super(message)
    this.name = 'ApiError'
  }

  /** A short, human-readable line safe to render in an error panel. */
  get userMessage(): string {
    if (this.status === 503) return 'The forecasting service is not ready yet.'
    if (this.status === 404) return this.message
    if (this.status === 0) return 'Cannot reach the forecasting service.'
    return this.message || 'Something went wrong.'
  }
}

/**
 * Resolution order:
 *   1. VITE_API_BASE_URL          explicit, for split deployments
 *   2. same-origin "/api/v1"      default: frontend served behind the same proxy
 *
 * In development, vite.config.ts proxies /api to the local backend, so the
 * same-origin default works there too.
 */
export const API_BASE: string =
  import.meta.env.VITE_API_BASE_URL?.replace(/\/+$/, '') ??
  '/api/v1'

export interface FetchOptions {
  signal?: AbortSignal
  method?: 'GET' | 'POST'
}

export async function apiFetch<T>(
  path: string,
  params?: Record<string, string | number | boolean | undefined | null>,
  options: FetchOptions = {},
): Promise<T> {
  const qs = new URLSearchParams()
  if (params) {
    for (const [k, v] of Object.entries(params)) {
      if (v !== undefined && v !== null && v !== '') qs.append(k, String(v))
    }
  }
  const query = qs.toString()
  const url = `${API_BASE}${path}${query ? `?${query}` : ''}`

  let res: Response
  try {
    res = await fetch(url, {
      method: options.method ?? 'GET',
      headers: { Accept: 'application/json' },
      signal: options.signal,
    })
  } catch (err) {
    if ((err as Error)?.name === 'AbortError') throw err
    throw new ApiError(0, 'network_error',
      'Cannot reach the forecasting service. Is the API running?')
  }

  if (!res.ok) {
    let body: Record<string, unknown> = {}
    try {
      body = await res.json()
    } catch {
      /* non-JSON error body — fall through to the status text */
    }
    throw new ApiError(
      res.status,
      (body.error as string) ?? 'http_error',
      (body.message as string) ?? res.statusText,
      body.context as Record<string, unknown> | undefined,
      (body.request_id as string) ?? res.headers.get('X-Request-ID') ?? undefined,
    )
  }

  return (await res.json()) as T
}

export const post = <T>(path: string) =>
  apiFetch<T>(path, undefined, { method: 'POST' })
