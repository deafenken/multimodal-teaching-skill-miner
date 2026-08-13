import {
  Controller,
  Get,
  Headers,
  HttpCode,
  Inject,
  Query,
  Req,
  Res
} from "@nestjs/common";
import type {FastifyReply, FastifyRequest} from "fastify";

import {AppConfigService} from "../config/app-config.service";
import {PublicRoute} from "./public.decorator";
import {SessionCookieService} from "./session-cookie";
import {SessionRevocationService} from "./session-revocation.service";
import {
  OidcAuthorizationFlow,
  OidcAuthorizationFlowError
} from "./oidc-authorization-flow";

type HeaderMap = Record<string, string | string[] | undefined>;

function firstHeader(headers: HeaderMap, name: string): string | undefined {
  const value = headers[name];
  return Array.isArray(value) ? value[0] : value;
}

function cookieValue(header: string | undefined, name: string): string | undefined {
  const matches: string[] = [];
  for (const part of header?.split(";") ?? []) {
    const separator = part.indexOf("=");
    if (separator <= 0 || part.slice(0, separator).trim() !== name) continue;
    try {
      matches.push(decodeURIComponent(part.slice(separator + 1).trim()));
    } catch {
      return undefined;
    }
  }
  return matches.length === 1 ? matches[0] : undefined;
}

@Controller("api/v1/auth/oidc")
export class OidcAuthorizationController {
  constructor(
    @Inject(AppConfigService) private readonly config: AppConfigService,
    @Inject(OidcAuthorizationFlow) private readonly flow: OidcAuthorizationFlow,
    @Inject(SessionCookieService) private readonly cookies: SessionCookieService,
    @Inject(SessionRevocationService) private readonly revocations: SessionRevocationService
  ) {}

  @Get("begin")
  @PublicRoute()
  @HttpCode(303)
  async begin(
    @Headers() headers: HeaderMap,
    @Query("return_to") returnTo: string | undefined,
    @Req() request: FastifyRequest,
    @Res() reply: FastifyReply
  ): Promise<void> {
    try {
      this.assertBeginRequest(headers);
      this.assertExactQuery(request.query, new Set(["return_to"]));
    } catch {
      this.safeFailure(reply, 400, "login_failed");
      return;
    }
    if (this.config.authMode !== "oidc") {
      this.safeFailure(reply, 404, "login_unavailable");
      return;
    }
    try {
      const started = await this.flow.begin(returnTo);
      reply.header("cache-control", "no-store");
      reply.header(
        "set-cookie",
        this.cookies.serializeOidcTransactionCookie(
          started.transactionCookie,
          this.config.oidcTransactionTtlSeconds
        )
      );
      reply.redirect(started.authorizationUrl, 303);
    } catch {
      this.safeFailure(reply, 503, "login_unavailable");
    }
  }

  /**
   * Starts a provider-forced, AAL2 OIDC ceremony. AuthenticationGuard requires
   * an existing session, while the encrypted transaction binds the callback to
   * the account-data-rights purpose; a legacy/login transaction cannot be
   * upgraded by changing query parameters.
   */
  @Get("account-step-up")
  @HttpCode(303)
  async accountStepUp(
    @Headers() headers: HeaderMap,
    @Query("return_to") returnTo: string | undefined,
    @Req() request: FastifyRequest,
    @Res() reply: FastifyReply
  ): Promise<void> {
    try {
      this.assertBeginRequest(headers);
      this.assertExactQuery(request.query, new Set(["return_to"]));
    } catch {
      this.safeFailure(reply, 400, "login_failed");
      return;
    }
    if (this.config.authMode !== "oidc") {
      this.safeFailure(reply, 404, "login_unavailable");
      return;
    }
    try {
      const started = await this.flow.begin(returnTo, "account_data_rights");
      reply.header("cache-control", "no-store");
      reply.header(
        "set-cookie",
        this.cookies.serializeOidcTransactionCookie(
          started.transactionCookie,
          this.config.oidcTransactionTtlSeconds
        )
      );
      reply.redirect(started.authorizationUrl, 303);
    } catch {
      this.safeFailure(reply, 503, "login_unavailable");
    }
  }

  @Get("callback")
  @PublicRoute()
  async callback(
    @Headers() headers: HeaderMap,
    @Query("code") code: string | undefined,
    @Query("state") state: string | undefined,
    @Query("error") providerError: string | undefined,
    @Query("iss") responseIssuer: string | undefined,
    @Req() request: FastifyRequest,
    @Res() reply: FastifyReply
  ): Promise<void> {
    const clearTransaction = this.cookies.serializeOidcTransactionCookie("", 0);
    reply.header("cache-control", "no-store");
    let registeringSession = false;
    try {
      this.assertCallbackRequest(headers);
      this.assertExactQuery(
        request.query,
        new Set(["code", "state", "error", "error_description", "error_uri", "iss"])
      );
      if (this.config.authMode !== "oidc") throw new Error("disabled");
      if (
        providerError !== undefined
        || !code
        || !state
        || (responseIssuer !== undefined && responseIssuer !== this.config.oidcIssuer)
      ) {
        throw new OidcAuthorizationFlowError("Provider rejected login", "callback_invalid");
      }
      const sealed = cookieValue(
        firstHeader(headers, "cookie"),
        this.config.oidcTransactionCookieName
      );
      if (!sealed) {
        throw new OidcAuthorizationFlowError("Missing transaction", "transaction_invalid");
      }
      const completed = await this.flow.complete({code, state, transactionCookie: sealed});
      const minted = this.cookies.mint({...completed.identity, provider: "oidc"});
      registeringSession = true;
      await this.revocations.register(minted.authenticatedSession);
      reply.header("set-cookie", [
        clearTransaction,
        this.cookies.serializeSessionCookie(minted.cookieValue),
        this.cookies.serializeCsrfCookie(minted.csrfToken)
      ]);
      reply.redirect(completed.returnTo, 303);
    } catch {
      this.safeFailure(
        reply,
        registeringSession ? 503 : 400,
        registeringSession ? "login_unavailable" : "login_failed",
        clearTransaction
      );
    }
  }

  private assertBeginRequest(headers: HeaderMap): void {
    const origin = firstHeader(headers, "origin");
    const fetchSite = firstHeader(headers, "sec-fetch-site");
    const fetchMode = firstHeader(headers, "sec-fetch-mode");
    if (
      firstHeader(headers, "host") !== this.config.oidcExpectedHost
      || (origin !== undefined && !this.config.isTrustedOrigin(origin))
      || fetchSite !== "same-origin"
      || !new Set(["navigate", "same-origin"]).has(fetchMode ?? "")
    ) throw new Error("untrusted begin request");
  }

  private assertCallbackRequest(headers: HeaderMap): void {
    const fetchSite = firstHeader(headers, "sec-fetch-site");
    const fetchMode = firstHeader(headers, "sec-fetch-mode");
    if (
      firstHeader(headers, "host") !== this.config.oidcExpectedHost
      || !new Set(["cross-site", "same-site", "same-origin", "none"]).has(fetchSite ?? "")
      || fetchMode !== "navigate"
    ) throw new Error("untrusted callback request");
  }

  private assertExactQuery(query: unknown, allowed: ReadonlySet<string>): void {
    if (!query || typeof query !== "object" || Array.isArray(query)) throw new Error("invalid query");
    for (const [key, value] of Object.entries(query as Record<string, unknown>)) {
      if (!allowed.has(key) || typeof value !== "string" || value.length > 4096) {
        throw new Error("invalid query");
      }
    }
  }

  private safeFailure(
    reply: FastifyReply,
    status: 400 | 404 | 503,
    code: "login_failed" | "login_unavailable",
    clearTransaction?: string
  ): void {
    if (clearTransaction) reply.header("set-cookie", clearTransaction);
    reply.status(status)
      .type("text/html; charset=utf-8")
      .send(`<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>TeachLab 登录</title><body><main><h1>无法完成登录</h1><p>${code === "login_unavailable" ? "组织登录服务暂时不可用，请稍后重试。" : "登录请求无效或已经过期，请返回 TeachLab 重新开始。"}</p></main></body></html>`);
  }
}
