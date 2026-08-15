import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// The dev proxy exists so the browser talks to a same-origin /api during
// development. In production the API base URL comes from VITE_API_BASE_URL, or
// defaults to same-origin /api/v1 when the frontend is served behind the same
// reverse proxy as the backend. Nothing is hard-coded to localhost at runtime.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/api': {
        target: process.env.VITE_DEV_API_TARGET ?? 'http://127.0.0.1:8000',
        changeOrigin: true,
      },
    },
  },
  // `npm run preview` serves the real production bundle. Proxying /api here too
  // means the built artefact can be exercised against a live API without Docker,
  // which is how the production bundle is verified on this machine.
  preview: {
    port: 4173,
    proxy: {
      '/api': {
        target: process.env.VITE_DEV_API_TARGET ?? 'http://127.0.0.1:8000',
        changeOrigin: true,
      },
    },
  },
  build: {
    outDir: 'dist',
    sourcemap: false,
    rollupOptions: {
      output: {
        // Split the chart library out: it is the largest dependency and is not
        // needed by the first paint of every route.
        manualChunks: {
          react: ['react', 'react-dom', 'react-router-dom'],
          charts: ['recharts'],
          query: ['@tanstack/react-query'],
        },
      },
    },
  },
  test: {
    environment: 'jsdom',
    globals: true,
    setupFiles: ['./src/test/setup.ts'],
    css: false,
  },
})
