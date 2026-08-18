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

// jsdom does not implement scrollIntoView. The assistant uses it to keep the
// newest answer in view; without a stub the component throws during render.
if (!HTMLElement.prototype.scrollIntoView) {
  HTMLElement.prototype.scrollIntoView = () => {}
}

Object.defineProperty(HTMLElement.prototype, 'clientWidth', {
  configurable: true, value: 800,
})
Object.defineProperty(HTMLElement.prototype, 'clientHeight', {
  configurable: true, value: 400,
})
