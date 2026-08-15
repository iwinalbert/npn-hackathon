/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
        // Deep navy-slate, never pure black: cards must separate from the page
        // and charts must stay readable.
        base: '#0B1220',        // page background
        surface: '#131C2E',     // cards
        elevated: '#1B2740',    // hover / raised
        line: '#243149',        // borders
        'line-strong': '#31415F',

        ink: '#E8EDF6',         // primary text
        'ink-muted': '#9AA9C2', // secondary text
        'ink-dim': '#6B7C99',   // tertiary / axis labels

        // One accent for "the model", one for "reality", plus semantic states.
        forecast: '#4EA8F0',    // predictions
        actual: '#C6D2E4',      // observed history
        good: '#3FB98B',
        warn: '#D9A63C',
        bad: '#E0665F',
        accentSoft: '#1E3350',
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', '-apple-system', 'Segoe UI', 'Roboto', 'sans-serif'],
        mono: ['ui-monospace', 'SFMono-Regular', 'Menlo', 'Consolas', 'monospace'],
      },
      fontSize: {
        'metric': ['1.75rem', { lineHeight: '2rem', letterSpacing: '-0.02em' }],
      },
      borderRadius: { card: '0.5rem' },
      transitionDuration: { DEFAULT: '150ms' },
      keyframes: {
        fadeIn: { '0%': { opacity: '0' }, '100%': { opacity: '1' } },
        slideUp: {
          '0%': { opacity: '0', transform: 'translateY(4px)' },
          '100%': { opacity: '1', transform: 'translateY(0)' },
        },
        shimmer: { '100%': { transform: 'translateX(100%)' } },
      },
      animation: {
        fadeIn: 'fadeIn 200ms ease-out',
        slideUp: 'slideUp 200ms ease-out',
      },
    },
  },
  plugins: [],
}
