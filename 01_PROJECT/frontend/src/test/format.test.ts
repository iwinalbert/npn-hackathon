import { describe, expect, it } from 'vitest'

import {
  combineHistoryAndForecast, compact, humanise, longDate, nf, pct, shortDate, signed,
} from '../lib/format'

describe('number formatting', () => {
  it('formats compact units without losing magnitude', () => {
    expect(compact(1_309_021)).toBe('1.31M')
    expect(compact(189_101)).toBe('189.1K')
    expect(compact(3_331)).toBe('3.3K')
    expect(compact(96)).toBe('96')
    expect(compact(0.63)).toBe('0.63')
  })

  it('keeps signs explicit for deltas', () => {
    expect(signed(12.3)).toBe('+12.30')
    expect(signed(-4)).toBe('-4.00')
    expect(signed(0)).toBe('+0.00')
  })

  it('formats percentages and plain numbers', () => {
    expect(pct(94.7)).toBe('94.7%')
    expect(nf(853720)).toBe('853,720')
  })
})

describe('date formatting', () => {
  it('renders dates in UTC so a forecast day never shifts by timezone', () => {
    expect(shortDate('2016-05-23')).toBe('23 May')
    expect(longDate('2016-05-23')).toBe('23 May 2016')
  })

  it('passes through unparseable input rather than showing "Invalid Date"', () => {
    expect(shortDate('not-a-date')).toBe('not-a-date')
  })
})

describe('humanise', () => {
  it('makes ids readable without discarding them', () => {
    expect(humanise('FOODS_3_090')).toBe('Foods 3 090')
  })
})

describe('combineHistoryAndForecast', () => {
  const history = [
    { date: '2016-05-20', sales: 10 },
    { date: '2016-05-21', sales: 12 },
    { date: '2016-05-22', sales: 8 },
  ]
  const forecast = [
    { date: '2016-05-23', yhat: 9.5, lower: 7, upper: 12 },
    { date: '2016-05-24', yhat: 9.1, lower: 6.5, upper: 11.6 },
  ]

  it('produces one row per day across both periods', () => {
    const out = combineHistoryAndForecast(history, forecast)
    expect(out).toHaveLength(5)
    expect(out.map((p) => p.date)).toEqual([
      '2016-05-20', '2016-05-21', '2016-05-22', '2016-05-23', '2016-05-24',
    ])
  })

  it('never puts an actual on a forecast day', () => {
    const out = combineHistoryAndForecast(history, forecast)
    for (const p of out.filter((x) => x.isForecast)) {
      expect(p.actual).toBeNull()
      expect(p.forecast).not.toBeNull()
    }
  })

  it('never invents a forecast for a historical day, except at the join', () => {
    const out = combineHistoryAndForecast(history, forecast)
    const past = out.filter((x) => !x.isForecast)
    // all but the last historical point carry no forecast value
    expect(past.slice(0, -1).every((p) => p.forecast === null)).toBe(true)
    // the last one is duplicated so the two lines visually connect
    expect(past.at(-1)!.forecast).toBe(8)
  })

  it('emits a band only where the API supplied both bounds', () => {
    const out = combineHistoryAndForecast(history, [
      { date: '2016-05-23', yhat: 9.5, lower: 7, upper: 12 },
      { date: '2016-05-24', yhat: 9.1, lower: null, upper: null },
    ])
    const [withBand, withoutBand] = out.filter((p) => p.isForecast)
    expect(withBand.band).toEqual([7, 12])
    expect(withoutBand.band).toBeNull()
  })

  it('handles an empty history without inventing a join point', () => {
    const out = combineHistoryAndForecast([], forecast)
    expect(out).toHaveLength(2)
    expect(out.every((p) => p.isForecast)).toBe(true)
  })
})
