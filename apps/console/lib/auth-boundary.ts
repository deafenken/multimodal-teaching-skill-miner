export type AuthProvider = "clerk" | "auth0" | "disabled";

export const authProvider = (process.env.NEXT_PUBLIC_AUTH_PROVIDER ?? "disabled") as AuthProvider;

/**
 * Authentication belongs in the BFF/session boundary. The browser may receive
 * an HttpOnly session cookie, but it must never receive GitHub App private keys,
 * installation tokens, model keys, or sandbox credentials.
 */
export const browserCredentialPolicy = {
  githubInstallationTokensInBrowser: false,
  modelKeysInBrowser: false,
  sandboxSecretsInBrowser: false,
  bffSessionCookieOnly: true
} as const;
