import { beforeEach, describe, expect, it, vi } from 'vitest'

import { API_BASE, ApiError, apiFetch } from '../api/client'

function mockFetch(status: number, body: unknown, headers: Record<string, string> = {}) {
  const fn = vi.fn().mockResolvedValue({
    ok: status >= 200 && status < 300,
    status,
    statusText: 'x',
    json: async () => body,
    headers: { get: (k: string) => headers[k] ?? null },
  })
  vi.stubGlobal('fetch', fn)
  return fn
}

describe('API base URL', () => {
  it('defaults to a same-origin path, never a hard-coded host', () => {
    expect(API_BASE).toBe('/api/v1')
    expect(API_BASE).not.toMatch(/localhost|127\.0\.0\.1|https?:\/\//)
  })
})

describe('apiFetch', () => {
  beforeEach(() => vi.unstubAllGlobals())

  it('builds a URL from the base and path', async () => {
    const fn = mockFetch(200, { ok: true })
    await apiFetch('/meta/model')
    expect(fn).toHaveBeenCalledWith('/api/v1/meta/model', expect.any(Object))
  })

  it('serialises query parameters and drops empty ones', async () => {
    const fn = mockFetch(200, [])
    await apiFetch('/hierarchy/aggregate', {
      level: 'store', node_id: 'CA_3', history_days: 30,
      empty: '', missing: undefined, nothing: null,
    })
    const url = fn.mock.calls[0][0] as string
    expect(url).toContain('level=store')
    expect(url).toContain('node_id=CA_3')
    expect(url).toContain('history_days=30')
    expect(url).not.toContain('empty')
    expect(url).not.toContain('missing')
    expect(url).not.toContain('nothing')
  })

  it('returns parsed JSON on success', async () => {
    mockFetch(200, { validation_rmse: 2.0929 })
    const out = await apiFetch<{ validation_rmse: number }>('/meta/model')
    expect(out.validation_rmse).toBe(2.0929)
  })

  it('turns a structured backend error into an ApiError', async () => {
    mockFetch(404, {
      error: 'not_found',
      message: "no series for store 'ZZ_9'",
      context: { hint: 'Use search' },
      request_id: 'abc123',
    })
    await expect(apiFetch('/series/ZZ_9/NOPE')).rejects.toMatchObject({
      status: 404,
      errorType: 'not_found',
      requestId: 'abc123',
    })
  })

  it('explains an unavailable service rather than leaking the status code', async () => {
    mockFetch(503, { error: 'service_unavailable', message: 'db missing' })
    try {
      await apiFetch('/hierarchy/levels')
      expect.unreachable()
    } catch (e) {
      expect((e as ApiError).userMessage).toMatch(/not ready/i)
    }
  })

  it('reports a network failure as an unreachable service, not a crash', async () => {
    vi.stubGlobal('fetch', vi.fn().mockRejectedValue(new TypeError('failed')))
    try {
      await apiFetch('/health')
      expect.unreachable()
    } catch (e) {
      expect(e).toBeInstanceOf(ApiError)
      expect((e as ApiError).status).toBe(0)
      expect((e as ApiError).userMessage).toMatch(/cannot reach/i)
    }
  })

  it('survives an error response that is not JSON', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({
      ok: false, status: 500, statusText: 'Internal Server Error',
      json: async () => { throw new Error('not json') },
      headers: { get: () => null },
    }))
    await expect(apiFetch('/boom')).rejects.toBeInstanceOf(ApiError)
  })
})
