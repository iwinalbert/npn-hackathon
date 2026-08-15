import '@testing-library/jest-dom/vitest'
import { cleanup } from '@testing-library/react'
import { afterEach, vi } from 'vitest'

afterEach(() => {
  cleanup()
  vi.restoreAllMocks()
})

// Recharts measures its container; jsdom reports zero, which makes charts render
// nothing and assertions fail for the wrong reason. Give it a real size.
class ResizeObserverStub {
  observe() {}
  unobserve() {}
  disconnect() {}
}
;(globalThis as unknown as { ResizeObserver: unknown }).ResizeObserver = ResizeObserverStub

Object.defineProperty(HTMLElement.prototype, 'clientWidth', {
  configurable: true, value: 800,
})
Object.defineProperty(HTMLElement.prototype, 'clientHeight', {
  configurable: true, value: 400,
})
