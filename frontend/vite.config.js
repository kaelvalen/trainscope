import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/api': {
        target: 'http://localhost:7007',
        changeOrigin: true,
      },
      '/ws': {
        target: 'ws://localhost:7007',
        ws: true,
        changeOrigin: true,
      },
    },
  },
  build: {
    outDir: '../trainscope/ui/static',
    emptyOutDir: true,
    chunkSizeWarningLimit: 2000,
  },
  test: {
    globals: true,
    environment: 'jsdom',
    setupFiles: ['./src/test/setup.js'],
  },
})
