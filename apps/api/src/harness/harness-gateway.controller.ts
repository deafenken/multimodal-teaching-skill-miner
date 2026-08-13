import {
  All,
  Controller,
  Inject,
  Param,
  Req,
  Res,
  Optional
} from "@nestjs/common";
import type {FastifyReply, FastifyRequest} from "fastify";
import type {IncomingMessage} from "node:http";
import {Transform} from "node:stream";

import type {AuthenticatedPrincipal} from "../auth/auth-provider.port";
import type {TeacherEntitlementAuthorizationResult} from "../auth/teacher-entitlement.port";
import {CurrentPrincipal} from "../auth/current-principal.decorator";
import {
  TeacherEntitlementAuthorizationService,
  teacherEntitlementReceiptSha256
} from "../auth/teacher-entitlement-authorization";
import {AppConfigService} from "../config/app-config.service";
import {AccountBrowserCacheScopeIssuer} from "../account/account-browser-cache-scope";
import {accessScopeFor} from "../tenancy/access-scope";
import type {AccessScope} from "../tenancy/access-scope";
import {
  HarnessGatewayError,
  HarnessWorkerPoolService
} from "./harness-worker-pool.service";
import {
  harnessRequestBodyAllowed
} from "./harness-request-limits";
import {
  TEACHER_AUTHORITY_PATHS,
  teacherAuthorityRequestContainsOverride
} from "./teacher-authority";
import {resolveSafeguardingRouteLocator} from "./safeguarding-route-locator";

export {
  HARNESS_REQUEST_LIMITS,
  harnessRequestBodyAllowed,
  harnessRequestBodyLimit
} from "./harness-request-limits";

const SYLLABUS_PATH = /^api\/syllabi(?:\/(?:generate|import|[^/]+|[^/]+\/(?:download|versions|curriculum-blueprint|revisions|publish|rollback)|[^/]+\/lessons\/[^/]+\/start-payload))?$/;
const PROJECT_PATH = /^api\/projects(?:\/(?:trash|bootstrap)|\/project_[0-9a-f]{24}(?:\/(?:update|chat-thread|browse|reference|remove-reference|note|trash|restore|export|purge))?)?$/;
const LEARNING_REVIEW_PATH = /^api\/learning-reviews\/(?:due|claim|release)$/;
const BACKGROUND_TASK_PATH = /^api\/tasks\/(?:list|status|cancel|resume)$/;
const METACOGNITION_PATH = /^api\/metacognition\/(?:list|predict|pair)$/;
const ADJUDICATION_PATH = /^api\/adjudication\/(?:candidates|list|enqueue|claim|decide)$/;
const CONSENT_PATH = /^api\/consent\/(?:grant|list|revoke)$/;
const SAFEGUARDING_STATUS_PATH = /^api\/safeguarding\/status$/;
const SAFEGUARDING_STAFF_PATH = /^api\/safeguarding\/(?:list|dispatch|case\/(?:acknowledge|close)|escalation\/(?:overdue|acknowledge))$/;
const PROJECT_EXPORT_PATH = /^api\/projects\/project_[0-9a-f]{24}\/export$/;
export const MAX_PROJECT_EXPORT_RESPONSE_BYTES = 256 * 1024 * 1024;
const EXACT_POST_PATHS = new Set([
  "api/chat",
  "api/start",
  "api/session",
  "api/step",
  "api/command",
  "api/cancel",
  "api/stream",
  "api/attachment",
  "api/resource",
  "api/resource/review",
  "api/curriculum/review",
  "api/curriculum/seal",
  "api/curriculum/revoke"
]);
const SERVER_REMOTE_POLICY_FIELDS = new Set([
  "remote_provider_policy",
  "remote_subject_policy",
  "remote_processing_policy",
  "remote_processing_eligible",
  "likely_minor",
  "guardian_or_school_policy",
  "provider_policy_sha256",
  "subject_policy_sha256"
]);

function requestContainsRemotePolicyOverride(value: unknown): boolean {
  if (!value || typeof value !== "object") return false;
  if (Array.isArray(value)) return value.some(requestContainsRemotePolicyOverride);
  return Object.entries(value as Record<string, unknown>).some(
    ([key, item]) => SERVER_REMOTE_POLICY_FIELDS.has(key)
      || requestContainsRemotePolicyOverride(item)
  );
}

function routeValue(value: string | string[] | undefined): string | undefined {
  const route = Array.isArray(value) ? value.join("/") : value;
  if (
    !route ||
    route.length > 512 ||
    route.startsWith("/") ||
    route.endsWith("/") ||
    route.includes("\\") ||
    route.split("/").some((segment) => !segment || segment === "." || segment === "..")
  ) {
    return undefined;
  }
  return route;
}

function routeAllowed(route: string, method: string): boolean {
  if (method === "GET" || method === "HEAD") {
    return (
      route === "api/bootstrap" ||
      SAFEGUARDING_STATUS_PATH.test(route) ||
      SYLLABUS_PATH.test(route) ||
      PROJECT_PATH.test(route)
    );
  }
  if (method !== "POST") return false;
  return (
    EXACT_POST_PATHS.has(route) ||
    SYLLABUS_PATH.test(route) ||
    PROJECT_PATH.test(route) ||
    LEARNING_REVIEW_PATH.test(route) ||
    BACKGROUND_TASK_PATH.test(route) ||
    METACOGNITION_PATH.test(route) ||
    ADJUDICATION_PATH.test(route) ||
    CONSENT_PATH.test(route) ||
    SAFEGUARDING_STAFF_PATH.test(route)
  );
}

async function boundedBody(stream: IncomingMessage, maximumBytes: number): Promise<Buffer> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const raw of stream) {
    const chunk = Buffer.isBuffer(raw) ? raw : Buffer.from(raw);
    size += chunk.byteLength;
    if (size > maximumBytes) {
      stream.destroy();
      throw new HarnessGatewayError(502, "harness_upstream_unavailable");
    }
    chunks.push(chunk);
  }
  return Buffer.concat(chunks, size);
}

function singleUpstreamHeader(value: string | string[] | undefined): string | undefined {
  return typeof value === "string" ? value : undefined;
}

export function projectExportResponseLength(
  headers: IncomingMessage["headers"]
): number | undefined {
  const rawLength = singleUpstreamHeader(headers["content-length"]);
  const contentType = singleUpstreamHeader(headers["content-type"]);
  const disposition = singleUpstreamHeader(headers["content-disposition"]);
  const manifest = singleUpstreamHeader(headers["x-manifest-sha256"]);
  if (
    !rawLength || !/^[1-9][0-9]{0,9}$/.test(rawLength) ||
    contentType?.split(";", 1)[0]?.trim().toLowerCase() !== "application/zip" ||
    !disposition || disposition.length > 512 || /[\r\n]/.test(disposition) ||
    !manifest || !/^[0-9a-f]{64}$/.test(manifest)
  ) {
    return undefined;
  }
  const length = Number(rawLength);
  return Number.isSafeInteger(length) && length <= MAX_PROJECT_EXPORT_RESPONSE_BYTES
    ? length
    : undefined;
}

export function boundedProjectExportStream(
  upstream: IncomingMessage,
  declaredBytes: number
): Transform {
  let receivedBytes = 0;
  const output = new Transform({
    transform(chunk: Buffer | string, encoding, callback) {
      const bytes = Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk, encoding);
      receivedBytes += bytes.byteLength;
      if (receivedBytes > declaredBytes || receivedBytes > MAX_PROJECT_EXPORT_RESPONSE_BYTES) {
        callback(new Error("project_export_response_limit_exceeded"));
        return;
      }
      callback(null, bytes);
    },
    flush(callback) {
      if (receivedBytes !== declaredBytes) {
        callback(new Error("project_export_response_length_mismatch"));
        return;
      }
      callback();
    }
  });
  upstream.once("error", () => output.destroy(new Error("project_export_upstream_failed")));
  upstream.pipe(output);
  return output;
}

function safeError(reply: FastifyReply, error: unknown): FastifyReply {
  const gatewayError =
    error instanceof HarnessGatewayError
      ? error
      : new HarnessGatewayError(503, "harness_upstream_unavailable");
  return reply.status(gatewayError.statusCode).send({
    statusCode: gatewayError.statusCode,
    code: gatewayError.code,
    error: "Teaching Harness request could not be completed"
  });
}

function authenticatedTeacherAuthorityProjection(
  bootstrap: Record<string, unknown>,
  roleAuthorized: boolean
): Record<string, unknown> | undefined {
  const value = bootstrap.teacher_authority;
  if (!value || typeof value !== "object" || Array.isArray(value)) return undefined;
  const authority = value as Record<string, unknown>;
  if (
    authority.mode !== "authenticated_apps_api" ||
    authority.role_authorized !== false ||
    authority.correct_mastery_updates_enabled !== false ||
    authority.assurance !==
      "deployment_service_role_authorization_not_personal_signature" ||
    authority.raw_identity_exposed !== false ||
    authority.nonce_replay_policy !==
      "append_only_permanent_tombstone_bounded_fail_closed" ||
    !Number.isSafeInteger(authority.replay_store_max_bytes) ||
    Number(authority.replay_store_max_bytes) < 1024 ||
    authority.expired_nonce_tombstones_retained !== true
  ) {
    return undefined;
  }
  return {
    mode: "authenticated_apps_api",
    role_authorized: roleAuthorized,
    correct_mastery_updates_enabled: roleAuthorized,
    assurance: "deployment_service_role_authorization_not_personal_signature",
    raw_identity_exposed: false,
    nonce_replay_policy: authority.nonce_replay_policy,
    replay_store_max_bytes: authority.replay_store_max_bytes,
    expired_nonce_tombstones_retained: true
  };
}

@Controller()
export class HarnessGatewayController {
  constructor(
    @Inject(HarnessWorkerPoolService)
    private readonly workers: HarnessWorkerPoolService,
    @Inject(AppConfigService) private readonly config: AppConfigService,
    @Optional() @Inject(AccountBrowserCacheScopeIssuer)
    private readonly cacheScopes?: AccountBrowserCacheScopeIssuer,
    @Optional() @Inject(TeacherEntitlementAuthorizationService)
    private readonly teacherEntitlements?: TeacherEntitlementAuthorizationService
  ) {}

  @All("api/v1/harness/*")
  async proxy(
    @CurrentPrincipal() principal: AuthenticatedPrincipal,
    @Param("*") rawPath: string | string[] | undefined,
    @Req() request: FastifyRequest,
    @Res() reply: FastifyReply
  ): Promise<FastifyReply | void> {
    const method = request.method.toUpperCase();
    const path = routeValue(rawPath);
    if (!path || !routeAllowed(path, method)) {
      return reply.status(404).send({
        statusCode: 404,
        code: "harness_route_not_found",
        error: "Resource not found"
      });
    }
    const safeguardingStaffPath = SAFEGUARDING_STAFF_PATH.test(path);
    let workerScope: AccessScope;
    try {
      workerScope = accessScopeFor(principal);
    } catch {
      return reply.status(403).send({
        statusCode: 403,
        code: "harness_scope_required",
        error: "A trusted account scope is required"
      });
    }
    let body: Buffer | undefined;
    let parsedBody: Record<string, unknown> | undefined;
    if (method === "POST") {
      try {
        parsedBody = request.body && typeof request.body === "object" && !Array.isArray(request.body)
          ? request.body as Record<string, unknown>
          : {};
        body = Buffer.from(JSON.stringify(parsedBody), "utf8");
      } catch {
        return safeError(reply, new HarnessGatewayError(400, "harness_request_invalid"));
      }
      if (!harnessRequestBodyAllowed(path, body.byteLength)) {
        return reply.status(413).send({
          statusCode: 413,
          code: "harness_request_too_large",
          error: "Teaching Harness request exceeds the route limit"
        });
      }
      if (
        path === "api/chat"
        && (
          typeof parsedBody.request_id !== "string"
          || !/^[A-Za-z0-9][A-Za-z0-9._:-]{7,159}$/.test(parsedBody.request_id)
        )
      ) {
        return reply.status(400).send({
          statusCode: 400,
          code: "chat_request_id_required",
          error: "Chat requires a stable request identity"
        });
      }
      if (requestContainsRemotePolicyOverride(parsedBody)) {
        return reply.status(400).send({
          statusCode: 400,
          code: "remote_processing_policy_input_rejected",
          error: "Remote processing policy is server controlled"
        });
      }
    }
    const teacherAuthorityPath = TEACHER_AUTHORITY_PATHS.has(path);
    const authorityRoles = safeguardingStaffPath
      ? this.config.safeguardingAuthorityRoles
      : this.config.teacherAuthorityRoles;
    const entitlementRequiredCode = safeguardingStaffPath
      ? "safeguarding_entitlement_required"
      : "teacher_entitlement_required";
    const entitlementUnavailableCode = safeguardingStaffPath
      ? "safeguarding_entitlement_unavailable"
      : "teacher_entitlement_unavailable";
    const entitlementRequiredMessage = safeguardingStaffPath
      ? "A current authoritative safeguarding entitlement is required"
      : "A current authoritative teacher entitlement is required";
    const entitlementUnavailableMessage = safeguardingStaffPath
      ? "Safeguarding entitlement could not be verified"
      : "Teacher entitlement could not be verified";
    let teacherEntitlementBinding: string | undefined;
    if (teacherAuthorityPath) {
      if (
        parsedBody === undefined
        || teacherAuthorityRequestContainsOverride(path, parsedBody)
      ) {
        return reply.status(400).send({
          statusCode: 400,
          code: "teacher_authority_input_rejected",
          error: "Teacher authority fields are server controlled"
        });
      }
      if (principal.provider !== "oidc" && this.config.nodeEnv !== "test") {
        return reply.status(403).send({
          statusCode: 403,
          code: entitlementRequiredCode,
          error: entitlementRequiredMessage
        });
      }
      if (this.teacherEntitlements) {
        let decision: TeacherEntitlementAuthorizationResult;
        try {
          decision = await this.teacherEntitlements.authorize(principal, path);
        } catch {
          return reply.status(503).send({
            statusCode: 503,
            code: entitlementUnavailableCode,
            error: entitlementUnavailableMessage
          });
        }
        if (!decision.allowed) {
          const unavailable = new Set([
            "provider_unavailable",
            "snapshot_missing",
            "snapshot_invalid",
            "identity_mismatch",
            "policy_mismatch",
            "snapshot_stale",
            "snapshot_rollback",
            "clock_invalid"
          ]).has(decision.reason);
          return reply.status(unavailable ? 503 : 403).send({
            statusCode: unavailable ? 503 : 403,
            code: unavailable
              ? entitlementUnavailableCode
              : entitlementRequiredCode,
            error: unavailable
              ? entitlementUnavailableMessage
              : entitlementRequiredMessage
          });
        }
        teacherEntitlementBinding = teacherEntitlementReceiptSha256(
          decision.receipt
        );
      } else if (
        safeguardingStaffPath || this.config.nodeEnv !== "test" ||
        !this.config.principalHasTeacherAuthority(principal.roles)
      ) {
        // Directly constructed test controllers may omit the production
        // entitlement provider. Every non-test runtime fails closed.
        return reply.status(503).send({
          statusCode: 503,
          code: entitlementUnavailableCode,
          error: entitlementUnavailableMessage
        });
      }
    }
    if (safeguardingStaffPath) {
      const locator = parsedBody?.safeguarding_route_locator;
      if (typeof locator !== "string") {
        return reply.status(400).send({
          statusCode: 400,
          code: "safeguarding_route_required",
          error: "A trusted safeguarding route locator is required"
        });
      }
      try {
        workerScope = resolveSafeguardingRouteLocator(
          locator,
          this.config.accountScopeKeys?.length
            ? this.config.accountScopeKeys
            : this.config.harnessScopeKeys,
          workerScope.tenantId
        );
      } catch {
        return reply.status(403).send({
          statusCode: 403,
          code: "safeguarding_route_rejected",
          error: "Safeguarding route locator was rejected"
        });
      }
      const workerBody = {...parsedBody};
      delete workerBody.safeguarding_route_locator;
      parsedBody = workerBody;
      body = Buffer.from(JSON.stringify(workerBody), "utf8");
    }
    const abort = new AbortController();
    request.raw.once("aborted", () => abort.abort());
    let upstream: IncomingMessage;
    try {
      upstream = await this.workers.request(workerScope, {
        method: method as "GET" | "HEAD" | "POST",
        path,
        body,
        accept:
          path === "api/stream"
            ? "text/event-stream"
            : request.headers.accept ?? "application/json",
        signal: abort.signal,
        remoteSubjectPolicy: safeguardingStaffPath
          ? undefined
          : principal.remoteSubjectPolicy,
        preserveRemoteSubjectPolicy: safeguardingStaffPath,
        ...(teacherAuthorityPath
          ? {
              // The role decision was made by the fresh server directory
              // above. Pass only deployment-controlled markers and a receipt
              // binding into the scope-signed worker envelope; never reuse
              // the potentially stale roles stored in the browser session.
              teacherRoles: authorityRoles,
              teacherAllowedRoles: authorityRoles,
              teacherPrincipal: {
                issuer: teacherEntitlementBinding
                  ? "teachlab-authoritative-entitlement-v1"
                  : "teachlab-test-session-role-v1",
                subject: teacherEntitlementBinding ?? principal.subject
              }
            }
          : {})
      });
    } catch (error) {
      return safeError(reply, error);
    }

    if ((upstream.statusCode ?? 502) >= 400) {
      // Never pass a Python exception body through the authenticated boundary:
      // even an otherwise benign message could contain a capability or a
      // tenant-private path.  Preserve only the bounded HTTP class.
      upstream.resume();
      const status = new Set([400, 404, 409, 410, 413, 429, 503]).has(
        upstream.statusCode ?? 0
      )
        ? (upstream.statusCode as number)
        : 502;
      return reply.status(status).send({
        statusCode: status,
        code: status === 404 ? "harness_resource_not_found" : "harness_request_rejected",
        error: "Teaching Harness request was rejected"
      });
    }

    if (method === "GET" && PROJECT_EXPORT_PATH.test(path)) {
      const declaredBytes = projectExportResponseLength(upstream.headers);
      if (declaredBytes === undefined) {
        upstream.resume();
        return safeError(reply, new HarnessGatewayError(502, "harness_upstream_unavailable"));
      }
      reply.status(upstream.statusCode ?? 200);
      reply.header("cache-control", "no-store, max-age=0");
      reply.header("content-type", "application/zip");
      reply.header("content-length", String(declaredBytes));
      reply.header("content-disposition", upstream.headers["content-disposition"] as string);
      reply.header("x-manifest-sha256", upstream.headers["x-manifest-sha256"] as string);
      const exportStream = boundedProjectExportStream(upstream, declaredBytes);
      const detach = () => {
        if (!reply.raw.writableEnded) upstream.destroy();
      };
      reply.raw.once("close", detach);
      upstream.once("close", () => reply.raw.removeListener("close", detach));
      exportStream.once("error", () => {
        if (!reply.raw.writableEnded) reply.raw.destroy();
      });
      return reply.send(exportStream);
    }
    const contentType = upstream.headers["content-type"] ?? "application/octet-stream";
    reply.status(upstream.statusCode ?? 200);
    reply.header("cache-control", "no-store, max-age=0");
    reply.header("content-type", contentType);
    for (const name of [
      "content-disposition",
      "x-manifest-sha256",
      "x-harness-run-id",
      "x-harness-turn-id",
      "x-background-task-id",
      "x-background-task-version"
    ]) {
      const value = upstream.headers[name];
      if (typeof value === "string" && !/[\r\n]/.test(value)) reply.header(name, value);
    }
    if (method === "HEAD") {
      upstream.resume();
      return reply.send();
    }
    if (String(contentType).toLowerCase().startsWith("text/event-stream")) {
      reply.header("x-accel-buffering", "no");
      const detach = () => {
        if (!reply.raw.writableEnded) upstream.destroy();
      };
      reply.raw.once("close", detach);
      upstream.once("close", () => reply.raw.removeListener("close", detach));
      upstream.once("error", () => {
        if (!reply.raw.writableEnded) reply.raw.destroy();
      });
      return reply.send(upstream);
    }
    try {
      const upstreamBody = await boundedBody(
        upstream,
        this.config.harnessResponseMaxBytes
      );
      if (method === "GET" && path === "api/bootstrap") {
        let bootstrap: unknown;
        try {
          bootstrap = JSON.parse(upstreamBody.toString("utf8"));
        } catch {
          return safeError(
            reply,
            new HarnessGatewayError(502, "harness_upstream_unavailable")
          );
        }
        if (!bootstrap || typeof bootstrap !== "object" || Array.isArray(bootstrap)) {
          return safeError(
            reply,
            new HarnessGatewayError(502, "harness_upstream_unavailable")
          );
        }
        let roleAuthorized = false;
        if (this.teacherEntitlements) {
          try {
            roleAuthorized = (
              await this.teacherEntitlements.authorize(
                principal,
                "api/resource/review"
              )
            ).allowed;
          } catch {
            roleAuthorized = false;
          }
        } else if (this.config.nodeEnv === "test") {
          roleAuthorized = this.config.principalHasTeacherAuthority(
            principal.roles
          );
        }
        const teacherAuthority = authenticatedTeacherAuthorityProjection(
          bootstrap as Record<string, unknown>,
          roleAuthorized
        );
        if (!teacherAuthority) {
          return safeError(
            reply,
            new HarnessGatewayError(502, "harness_upstream_unavailable")
          );
        }
        return reply.send({
          ...(bootstrap as Record<string, unknown>),
          teacher_authority: teacherAuthority,
          ...(principal.provider === "oidc" && this.cacheScopes
            ? {
                cache_scope: this.cacheScopes.issue(principal),
                account_data_rights: {
                  mode: "authenticated_account_authority",
                  export: "/api/teacher-agent/account/export",
                  deletion_prepare: "/api/teacher-agent/account/deletion/prepare",
                  deletion_confirm: "/api/teacher-agent/account/deletion/confirm",
                  deletion_resume: "/api/teacher-agent/account/deletion/resume",
                  deletion_status: "/api/teacher-agent/account/deletion/status",
                  step_up: "/api/teacher-agent/security/account-step-up",
                  recent_auth_required: true,
                  recent_auth_satisfied: (() => {
                    const authenticatedAt = Date.parse(principal.authenticatedAt ?? "");
                    return principal.identityIssuer === this.config.oidcIssuer
                      && Number.isFinite(authenticatedAt)
                      && authenticatedAt <= Date.now() + 30_000
                      && authenticatedAt >= Date.now()
                        - this.config.oidcAccountStepUpMaxAgeSeconds * 1_000
                      && (principal.assuranceLevel ?? 0) >= 2;
                  })(),
                  minimum_assurance_level: 2,
                  remote_provider_copies_deleted: false
                }
              }
            : {
                account_data_rights: {
                  mode: "local_only_no_account_authority",
                  recent_auth_required: false,
                  remote_provider_copies_deleted: false
                }
              })
        });
      }
      return reply.send(upstreamBody);
    } catch (error) {
      return safeError(reply, error);
    }
  }
}
