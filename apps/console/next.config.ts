import type {NextConfig} from "next";
import {SECURITY_RESPONSE_HEADERS} from "./lib/security-headers.ts";

const nextConfig: NextConfig = {
  // Production builds may run beside an already-serving immutable artifact.
  // A build-specific directory prevents `next build` from mutating files used
  // by the live server.
  distDir: process.env.TEACHLAB_NEXT_DIST_DIR || ".next",
  output: "standalone",
  reactStrictMode: true,
  poweredByHeader: false,
  devIndicators: false,
  async headers() {
    return [
      {source: "/:path*", headers: [...SECURITY_RESPONSE_HEADERS]},
      {
        source: "/teachlab-offline-shell-v1.js",
        headers: [{key: "Cache-Control", value: "public, max-age=31536000, immutable"}],
      },
      {
        source: "/teachlab-sw-v1.js",
        headers: [{key: "Cache-Control", value: "no-cache, no-store, must-revalidate"}],
      },
    ];
  },
  experimental: {
    optimizePackageImports: ["lucide-react"]
  }
};

export default nextConfig;
