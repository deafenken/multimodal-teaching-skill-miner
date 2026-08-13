import {pipeline} from "node:stream/promises";

import {
  Body,
  Controller,
  ForbiddenException,
  Get,
  Headers,
  HttpCode,
  HttpException,
  Inject,
  Post,
  Req,
  Res
} from "@nestjs/common";
import type {FastifyReply, FastifyRequest} from "fastify";

import type {AuthenticatedPrincipal, AuthProviderPort} from "../auth/auth-provider.port";
import {CurrentPrincipal} from "../auth/current-principal.decorator";
import {PublicRoute} from "../auth/public.decorator";
import {SessionCookieService} from "../auth/session-cookie";
import {AUTH_PROVIDER} from "../platform/tokens";
import {AppConfigService} from "../config/app-config.service";
import {AllowAccountDeletionScope} from "./allow-account-deletion.decorator";
import {AccountDataRightsError} from "./account-data-rights.errors";
import {
  ConfirmAccountDeletionDto,
  PrepareAccountDeletionDto
} from "./account-data-rights.dto";
import {AccountDataRightsService} from "./account-data-rights.service";

const CSRF_HEADER = "x-csrf-token";

function firstHeader(value: string | string[] | undefined): string | undefined {
  return Array.isArray(value) ? value[0] : value;
}

function safeFilename(value: string): string {
  if (!/^[A-Za-z0-9_.-]{1,160}$/.test(value)) {
    throw new AccountDataRightsError(503, "account_export_unavailable");
  }
  return value;
}

function throwHttp(error: unknown): never {
  if (error instanceof AccountDataRightsError) {
    throw new HttpException(
      {statusCode: error.statusCode, code: error.code, error: error.code},
      error.statusCode
    );
  }
  throw error;
}

@Controller("api/v1/account")
export class AccountDataRightsController {
  constructor(
    @Inject(AccountDataRightsService)
    private readonly dataRights: AccountDataRightsService,
    @Inject(AUTH_PROVIDER) private readonly authProvider: AuthProviderPort,
    @Inject(SessionCookieService) private readonly sessionCookies: SessionCookieService,
    @Inject(AppConfigService) private readonly config: AppConfigService
  ) {}

  @Get("export")
  async export(
    @CurrentPrincipal() principal: AuthenticatedPrincipal,
    @Req() request: FastifyRequest,
    @Res() reply: FastifyReply
  ): Promise<void> {
    const abortController = new AbortController();
    const abort = () => abortController.abort();
    const close = () => {
      if (!reply.raw.writableEnded) abort();
    };
    request.raw.once("aborted", abort);
    reply.raw.once("close", close);
    try {
      const archive = await this.dataRights.export(principal, abortController.signal);
      reply.header("cache-control", "no-store, max-age=0");
      reply.header("content-type", "application/zip");
      reply.header(
        "content-disposition",
        `attachment; filename="${safeFilename(archive.filename)}"`
      );
      reply.header("content-length", String(archive.byteLength));
      reply.header("x-manifest-sha256", archive.manifestSha256);
      reply.header("x-content-type-options", "nosniff");
      reply.hijack();
      await pipeline(archive.stream, reply.raw);
    } catch (error) {
      if (reply.sent || reply.raw.headersSent) {
        reply.raw.destroy();
        return;
      }
      throwHttp(error);
    } finally {
      request.raw.removeListener("aborted", abort);
      reply.raw.removeListener("close", close);
    }
  }

  @Post("deletion/prepare")
  @HttpCode(201)
  async prepareDeletion(
    @CurrentPrincipal() principal: AuthenticatedPrincipal,
    @Headers(CSRF_HEADER) csrfToken: string | undefined,
    @Body() _input: PrepareAccountDeletionDto,
    @Res({passthrough: true}) reply: FastifyReply
  ) {
    try {
      const prepared = await this.dataRights.prepareDeletion(
        principal,
        csrfToken ?? ""
      );
      reply.header("cache-control", "no-store, max-age=0");
      reply.header("set-cookie", prepared.statusCookie);
      return prepared.response;
    } catch (error) {
      throwHttp(error);
    }
  }

  @Post("deletion/confirm")
  @AllowAccountDeletionScope()
  @HttpCode(200)
  async confirmDeletion(
    @CurrentPrincipal() principal: AuthenticatedPrincipal,
    @Headers(CSRF_HEADER) csrfToken: string | undefined,
    @Headers("cookie") rawCookieHeader: string | undefined,
    @Body() input: ConfirmAccountDeletionDto,
    @Res({passthrough: true}) reply: FastifyReply
  ) {
    try {
      const receipt = await this.dataRights.confirmDeletion(
        principal,
        {
          challengeId: input.challenge_id,
          confirmationToken: input.confirmation_token,
          confirmationPhrase: input.confirmation_phrase,
          expectedRevision: input.expected_revision,
          idempotencyKey: input.idempotency_key
        },
        csrfToken ?? "",
        rawCookieHeader
      );
      // The repository has already committed Events -> Tasks -> Sessions ->
      // all auth sessions and the tombstone. Only now may the browser discard
      // its current signed session and CSRF cookies.
      reply.header("cache-control", "no-store, max-age=0");
      reply.header("set-cookie", [
        this.sessionCookies.serializeSessionCookie("", 0),
        this.sessionCookies.serializeCsrfCookie("", 0)
      ]);
      return receipt;
    } catch (error) {
      throwHttp(error);
    }
  }

  @Post("deletion/resume")
  @PublicRoute()
  @HttpCode(200)
  async resumeDeletion(
    @Headers("origin") origin: string | undefined,
    @Headers("sec-fetch-site") fetchSite: string | undefined,
    @Headers("cookie") rawCookieHeader: string | undefined,
    @Body() _input: PrepareAccountDeletionDto,
    @Res({passthrough: true}) reply: FastifyReply
  ) {
    // Authentication may disappear in the final transaction, so the status
    // capability is the authority. Same-origin metadata still protects this
    // ambient HttpOnly credential from mutation CSRF.
    if (
      !new Set(["same-origin", "same-site"]).has(fetchSite ?? "")
      || !this.config.isTrustedOrigin(origin)
    ) throw new ForbiddenException("Account deletion resume origin is invalid");
    try {
      const status = await this.dataRights.resume(rawCookieHeader);
      reply.header("cache-control", "no-store, max-age=0");
      if (status.status === "permanently_deleted") {
        reply.header("set-cookie", [
          this.sessionCookies.serializeSessionCookie("", 0),
          this.sessionCookies.serializeCsrfCookie("", 0)
        ]);
      }
      return status;
    } catch (error) {
      throwHttp(error);
    }
  }

  @Get("deletion/status")
  @PublicRoute()
  async deletionStatus(
    @Headers() headers: Record<string, string | string[] | undefined>
  ) {
    try {
      // Authentication is optional because a successful final transaction
      // deletes every auth session. The separate HttpOnly status capability is
      // the only post-delete authority and yields only a content-free receipt.
      const authenticated = await this.authProvider.authenticate({
        headers,
        method: "GET"
      }).catch(() => null);
      return await this.dataRights.status(
        firstHeader(headers.cookie),
        authenticated?.principal
      );
    } catch (error) {
      throwHttp(error);
    }
  }
}
