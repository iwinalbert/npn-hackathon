import { describe, expect, it } from 'vitest'

import {
  buildDemandSeries, compact, humanise, longDate, nf, pct, shortDate, signed,
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

describe('buildDemandSeries', () => {
  const history = [
    { date: '2016-05-18', sales: 15 },
    { date: '2016-05-19', sales: 11 },
    { date: '2016-05-20', sales: 10 },
    { date: '2016-05-21', sales: 12 },
    { date: '2016-05-22', sales: 8 },
  ]
  const backtest = [
    { date: '2016-05-20', actual: 10, predicted: 9.2 },
    { date: '2016-05-21', actual: 12, predicted: 10.4 },
    { date: '2016-05-22', actual: 8, predicted: 9.9 },
  ]
  const forecast = [
    { date: '2016-05-23', yhat: 9.5, lower: 7, upper: 12 },
    { date: '2016-05-24', yhat: 9.1, lower: 6.5, upper: 11.6 },
  ]

  it('produces one date-ordered row per day across all three sources', () => {
    const out = buildDemandSeries({ history, backtest, forecast })
    expect(out.map((p) => p.date)).toEqual([
      '2016-05-18', '2016-05-19', '2016-05-20',
      '2016-05-21', '2016-05-22', '2016-05-23', '2016-05-24',
    ])
  })

  it('marks each day with the only phase its data supports', () => {
    const out = buildDemandSeries({ history, backtest, forecast })
    expect(out.map((p) => p.phase)).toEqual([
      'history', 'history', 'compared', 'compared', 'compared', 'forecast', 'forecast',
    ])
  })

  it('NEVER invents an actual for a day that has not happened', () => {
    const out = buildDemandSeries({ history, backtest, forecast })
    for (const p of out.filter((x) => x.phase === 'forecast')) {
      expect(p.actual).toBeNull()
      expect(p.predicted).not.toBeNull()
    }
  })

  it('NEVER invents a prediction for a day outside the held-out window', () => {
    const out = buildDemandSeries({ history, backtest, forecast })
    for (const p of out.filter((x) => x.phase === 'history')) {
      expect(p.predicted).toBeNull()
      expect(p.actual).not.toBeNull()
    }
  })

  it('carries both real values on the compared days, unchanged', () => {
    const out = buildDemandSeries({ history, backtest, forecast })
    const compared = out.filter((p) => p.phase === 'compared')
    expect(compared.map((p) => p.actual)).toEqual([10, 12, 8])
    expect(compared.map((p) => p.predicted)).toEqual([9.2, 10.4, 9.9])
  })

  it('emits a band only where the API supplied both bounds', () => {
    const out = buildDemandSeries({
      forecast: [
        { date: '2016-05-23', yhat: 9.5, lower: 7, upper: 12 },
        { date: '2016-05-24', yhat: 9.1, lower: null, upper: null },
      ],
    })
    expect(out[0].band).toEqual([7, 12])
    expect(out[1].band).toBeNull()
  })

  it('works from any single source alone', () => {
    expect(buildDemandSeries({ forecast })).toHaveLength(2)
    expect(buildDemandSeries({ history })).toHaveLength(5)
    expect(buildDemandSeries({})).toEqual([])
  })
})
