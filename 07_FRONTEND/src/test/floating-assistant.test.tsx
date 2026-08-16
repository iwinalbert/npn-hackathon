/**
 * Floating AI Assistant shortcut.
 *
 * The assertions that matter are the ones that keep it a *shortcut*: the
 * sidebar entry must survive, navigation must stay client-side, and the button
 * must be reachable and labelled without a mouse.
 */
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
            <Route path="accuracy" element={<p>ACCURACY PAGE</p>} />
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
    // the visible "AI" alone would not tell a screen-reader user anything
    expect(floating()).toHaveAccessibleName('Open AI Assistant')
  })

  // The sidebar entry also reads "AI Assistant", which is the point of keeping
  // both — so the tooltip has to be found relative to the floating button.
  const tooltip = () =>
    within(floating().closest('.group') as HTMLElement).getByText('AI Assistant')

  it('reveals the "AI Assistant" label on hover', async () => {
    const user = userEvent.setup()
    mockApi()
    wrap()

    // hidden until hovered, and it must not swallow clicks while invisible
    expect(tooltip().className).toMatch(/opacity-0/)
    expect(tooltip().className).toMatch(/pointer-events-none/)

    await user.hover(floating())
    // the reveal is CSS-driven, so assert the mechanism rather than a computed
    // style jsdom does not evaluate: the label sits inside the button's group
    // and carries the group-hover rule.
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
    // an anchor that would leave the SPA is the failure mode this guards
    expect(floating()).toHaveAttribute('href', '/assistant')
    expect(floating()).not.toHaveAttribute('target')
  })

  it.each(['/', '/forecast', '/accuracy'])('is present on %s', (route) => {
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

    // the sidebar one keeps its full label and description
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
