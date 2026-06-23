import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// Vite config — serves the app on localhost:5173 by default.
// The `proxy` block forwards /api and /static requests to the Flask
// backend (localhost:5001) so images (Grad-CAM, segmentation overlays,
// original uploads) are loaded from Flask without CORS issues.
// Port 5001 (not 5000) — port 5000 is occupied by an unrelated process
// on this machine.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/api': {
        target: 'http://localhost:5001',
        changeOrigin: true,
      },
      '/static': {
        target: 'http://localhost:5001',
        changeOrigin: true,
      },
    },
  },
})
