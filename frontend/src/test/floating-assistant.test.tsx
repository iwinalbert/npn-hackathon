import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { render, screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { MemoryRouter, Route, Routes } from 'react-router-dom'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import { AppShell } from '../components/layout/AppShell'

const MODEL = {
  model_name: 'Direct+Recursive Tweedie Blend', status: 'FROZEN',
  validation_rmse: 2.0929, validation_mae: 1.0395,
}

function mockApi() {
  vi.stubGlobal('fetch', vi.fn(async (url: string) => {
    const body = String(url).includes('/meta/model')
      ? MODEL
      : { ready: true, degraded: false }
    return {
      ok: true, status: 200, statusText: 'OK',
      json: async () => body,
      headers: { get: () => null },
    }
  }))
}

function wrap(initial = '/forecast') {
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  return render(
    <QueryClientProvider client={qc}>
      <MemoryRouter initialEntries={[initial]}>
        <Routes>
          <Route path="/" element={<AppShell />}>
            <Route index element={<p>OVERVIEW PAGE</p>} />
            <Route path="forecast" element={<p>FORECAST PAGE</p>} />
            <Route path="hierarchy" element={<p>HIERARCHY PAGE</p>} />
            <Route path="assistant" element={<p>ASSISTANT PAGE</p>} />
          </Route>
        </Routes>
      </MemoryRouter>
    </QueryClientProvider>,
  )
}

const floating = () => screen.getByRole('link', { name: /open ai assistant/i })

describe('FloatingAIAssistant', () => {
  beforeEach(() => vi.unstubAllGlobals())

  it('exists and reads "AI"', () => {
    mockApi()
    wrap()
    expect(floating()).toBeInTheDocument()
    expect(floating()).toHaveTextContent('AI')
  })

  it('has an accessible label that says what it opens', () => {
    mockApi()
    wrap()
    expect(floating()).toHaveAccessibleName('Open AI Assistant')
  })

  const tooltip = () =>
    within(floating().closest('.group') as HTMLElement).getByText('AI Assistant')

  it('reveals the "AI Assistant" label on hover', async () => {
    const user = userEvent.setup()
    mockApi()
    wrap()

    expect(tooltip().className).toMatch(/opacity-0/)
    expect(tooltip().className).toMatch(/pointer-events-none/)

    await user.hover(floating())
    expect(tooltip().className).toMatch(/group-hover:opacity-100/)
  })

  it('reveals the same label for keyboard users, not just on hover', async () => {
    const user = userEvent.setup()
    mockApi()
    wrap()

    await user.tab()
    expect(tooltip().className).toMatch(/group-focus-within:opacity-100/)
  })

  it('is keyboard focusable', async () => {
    mockApi()
    wrap()
    floating().focus()
    expect(floating()).toHaveFocus()
  })

  it('navigates to /assistant on click without a full page load', async () => {
    const user = userEvent.setup()
    mockApi()
    wrap('/forecast')
    expect(screen.getByText('FORECAST PAGE')).toBeInTheDocument()

    await user.click(floating())

    await waitFor(() => expect(screen.getByText('ASSISTANT PAGE')).toBeInTheDocument())
    expect(floating()).toHaveAttribute('href', '/assistant')
    expect(floating()).not.toHaveAttribute('target')
  })

  it.each(['/', '/forecast', '/hierarchy'])('is present on %s', (route) => {
    mockApi()
    wrap(route)
    expect(floating()).toBeInTheDocument()
  })

  it('is present on /assistant too, marked as the current page', () => {
    mockApi()
    wrap('/assistant')
    expect(floating()).toBeInTheDocument()
    expect(floating()).toHaveAttribute('aria-current', 'page')
  })

  it('does not claim to be the current page elsewhere', () => {
    mockApi()
    wrap('/forecast')
    expect(floating()).not.toHaveAttribute('aria-current')
  })

  it('does NOT replace the sidebar entry — both must exist', () => {
    mockApi()
    wrap()
    const assistantLinks = screen
      .getAllByRole('link')
      .filter((el) => (el.getAttribute('href') ?? '') === '/assistant')
    expect(assistantLinks.length).toBe(2)

    const sidebar = screen.getByRole('navigation', { name: /main/i })
    expect(within(sidebar).getByText('AI Assistant')).toBeInTheDocument()
    expect(within(sidebar).getByText(/ask about forecasts in plain language/i))
      .toBeInTheDocument()
  })

  it('leaves room below the content so it cannot permanently cover it', () => {
    mockApi()
    const { container } = wrap()
    const main = container.querySelector('main')!
    expect(main.className).toMatch(/pb-\d+/)
  })
})
