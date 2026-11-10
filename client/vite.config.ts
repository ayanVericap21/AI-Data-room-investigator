import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// The backend runs on :8000; proxying /api avoids CORS and keeps the API origin out of the bundle.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: { "/api": "http://localhost:8000" },
  },
});
