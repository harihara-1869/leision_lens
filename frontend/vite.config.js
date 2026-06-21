import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// Vite config — serves the app on localhost:5173 by default.
// The `proxy` block forwards /api and /static requests to the Flask
// backend (localhost:5000) so images (Grad-CAM, segmentation overlays,
// original uploads) are loaded from Flask without CORS issues.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/api': {
        target: 'http://localhost:5000',
        changeOrigin: true,
      },
      '/static': {
        target: 'http://localhost:5000',
        changeOrigin: true,
      },
    },
  },
})
