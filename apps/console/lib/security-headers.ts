const developmentScriptPolicy = process.env.NODE_ENV === "development"
  ? "script-src 'self' 'unsafe-inline' 'unsafe-eval'"
  : "script-src 'self' 'unsafe-inline'";
const developmentConnectPolicy = process.env.NODE_ENV === "development"
  ? "connect-src 'self' ws: wss:"
  : "connect-src 'self'";

export const CONTENT_SECURITY_POLICY = [
  "default-src 'self'",
  "base-uri 'self'",
  "object-src 'none'",
  "frame-ancestors 'none'",
  "form-action 'self'",
  developmentScriptPolicy,
  developmentConnectPolicy,
  "style-src 'self' 'unsafe-inline'",
  "img-src 'self' data: blob:",
  "font-src 'self' data:",
  "media-src 'self' blob:",
  "worker-src 'self' blob:",
  "frame-src 'none'",
].join("; ");

export const SECURITY_RESPONSE_HEADERS = [
  {key: "Content-Security-Policy", value: CONTENT_SECURITY_POLICY},
  {key: "Cross-Origin-Opener-Policy", value: "same-origin"},
  {key: "Cross-Origin-Resource-Policy", value: "same-origin"},
  {key: "Origin-Agent-Cluster", value: "?1"},
  {key: "Permissions-Policy", value: "camera=(), microphone=(), geolocation=(), display-capture=(), payment=(), usb=(), serial=()"},
  {key: "Referrer-Policy", value: "no-referrer"},
  {key: "X-Content-Type-Options", value: "nosniff"},
  {key: "X-Frame-Options", value: "DENY"},
] as const;
