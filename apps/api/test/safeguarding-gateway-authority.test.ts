import "reflect-metadata";

import assert from "node:assert/strict";
import {EventEmitter} from "node:events";
import {Readable} from "node:stream";
import test from "node:test";

import type {FastifyReply, FastifyRequest} from "fastify";

import type {AuthenticatedPrincipal} from "../src/auth/auth-provider.port";
import {TeacherEntitlementAuthorizationService} from "../src/auth/teacher-entitlement-authorization";
import type {
  AuthoritativeTeacherEntitlementSnapshot,
  TeacherEntitlementSnapshotProvider
} from "../src/auth/teacher-entitlement.port";
import type {AppConfigService} from "../src/config/app-config.service";
import {HarnessGatewayController} from "../src/harness/harness-gateway.controller";
import type {HarnessWorkerPoolService} from "../src/harness/harness-worker-pool.service";
import {issueSafeguardingRouteLocator} from "../src/harness/safeguarding-route-locator";

const KEY = {version: "k1", secret: "safeguarding-routing-key-with-at-least-32-bytes"};
const CALLER_SCOPE = {tenantId: "ot1_ns1_school-a", ownerId: "os1_ns1_staff-7"};
const LEARNER_SCOPE = {tenantId: CALLER_SCOPE.tenantId, ownerId: "os1_ns1_learner-42"};
const PRINCIPAL: AuthenticatedPrincipal = {
  provider: "oidc",
  identityNamespace: "oidc-issuer-tenant-sub-v1",
  identityIssuer: "https://identity.school.example",
  tenantId: "school-a",
  subject: "staff-7",
  assuranceLevel: 2,
  roles: ["teacher"], // stale cookie role is deliberately irrelevant.
  sessionId: "session-staff-7",
  scopeTenantId: CALLER_SCOPE.tenantId,
  scopeOwnerId: CALLER_SCOPE.ownerId
};

class Directory implements TeacherEntitlementSnapshotProvider {
  constructor(
    private readonly roles: readonly string[],
    private readonly outage = false
  ) {}

  async readAuthoritativeSnapshot(): Promise<AuthoritativeTeacherEntitlementSnapshot> {
    if (this.outage) throw new Error("private directory outage");
    const now = new Date();
    return {
      identity: {
        issuer: PRINCIPAL.identityIssuer!,
        tenantId: PRINCIPAL.tenantId,
        subject: PRINCIPAL.subject
      },
      status: "active",
      roles: this.roles,
      policyVersion: "school-entitlements-v1",
      revision: 11,
      evaluatedAt: now.toISOString(),
      expiresAt: new Date(now.getTime() + 60_000).toISOString()
    };
  }
}

function entitlement(roles: readonly string[], outage = false) {
  return new TeacherEntitlementAuthorizationService(
    new Directory(roles, outage),
    {
      policyId: "school-entitlements",
      version: "school-entitlements-v1",
      requiredRoles: {
        "api/resource/review": ["teacher"],
        "api/curriculum/review": ["teacher"],
        "api/curriculum/seal": ["teacher"],
        "api/curriculum/revoke": ["teacher"],
        "api/adjudication/claim": ["teacher"],
        "api/adjudication/decide": ["teacher"],
        "api/safeguarding/list": ["safeguarding"],
        "api/safeguarding/dispatch": ["safeguarding"],
        "api/safeguarding/case/acknowledge": ["safeguarding"],
        "api/safeguarding/case/close": ["safeguarding"],
        "api/safeguarding/escalation/overdue": ["safeguarding"],
        "api/safeguarding/escalation/acknowledge": ["safeguarding"]
      },
      freshnessTtlMs: 10_000,
      cacheTtlMs: 0,
      providerTimeoutMs: 1_000,
      maxClockSkewMs: 1_000,
      minAssuranceLevel: 2
    },
    Buffer.alloc(32, 0x31),
    Buffer.alloc(32, 0x32)
  );
}

function config(): AppConfigService {
  return {
    nodeEnv: "production",
    harnessResponseMaxBytes: 64 * 1024,
    teacherAuthorityRoles: ["teacher"],
    safeguardingAuthorityRoles: ["safeguarding"],
    accountScopeKeys: [KEY],
    harnessScopeKeys: [KEY]
  } as unknown as AppConfigService;
}

function fakeRequest(body: Record<string, unknown>): FastifyRequest {
  return {
    method: "POST",
    body,
    headers: {accept: "application/json"},
    raw: new EventEmitter()
  } as unknown as FastifyRequest;
}

function fakeReply() {
  const state = {status: 200, payload: undefined as unknown};
  const raw = new EventEmitter() as EventEmitter & {
    writableEnded: boolean;
    destroy: () => void;
  };
  raw.writableEnded = false;
  raw.destroy = () => { raw.writableEnded = true; };
  const reply = {
    raw,
    status(code: number) { state.status = code; return this; },
    header() { return this; },
    send(payload?: unknown) { state.payload = payload; return this; }
  } as unknown as FastifyReply;
  return {reply, state};
}

function workers(calls: Array<{scope: unknown; input: Record<string, unknown>}>) {
  return {
    async request(scope: unknown, input: Record<string, unknown>) {
      calls.push({scope, input});
      const response = Readable.from([Buffer.from("{}")]) as Readable & {
        statusCode: number;
        headers: Record<string, string>;
      };
      response.statusCode = 200;
      response.headers = {"content-type": "application/json"};
      return response;
    }
  } as unknown as HarnessWorkerPoolService;
}

const ROUTES = [
  "api/safeguarding/list",
  "api/safeguarding/dispatch",
  "api/safeguarding/case/acknowledge",
  "api/safeguarding/case/close",
  "api/safeguarding/escalation/overdue",
  "api/safeguarding/escalation/acknowledge"
] as const;

test("all safeguarding routes require the exact fresh role and target only the opaque learner scope", async () => {
  const calls: Array<{scope: unknown; input: Record<string, unknown>}> = [];
  const controller = new HarnessGatewayController(
    workers(calls),
    config(),
    undefined,
    entitlement(["safeguarding"])
  );
  const locator = issueSafeguardingRouteLocator(LEARNER_SCOPE, KEY);
  for (const [index, path] of ROUTES.entries()) {
    const reply = fakeReply();
    const mutation = path === "api/safeguarding/list" ? {} : {
      case_id: `sgc_${"1".repeat(24)}`,
      expected_version: 3
    };
    await controller.proxy(
      {...PRINCIPAL, roles: []},
      path,
      fakeRequest({
        ...mutation,
        safeguarding_idempotency_key: `safeguarding-operation-${index}`,
        safeguarding_route_locator: locator
      }),
      reply.reply
    );
    assert.equal(reply.state.status, 200);
  }
  assert.equal(calls.length, ROUTES.length);
  for (const call of calls) {
    assert.deepEqual(call.scope, LEARNER_SCOPE);
    assert.deepEqual(call.input.teacherRoles, ["safeguarding"]);
    assert.deepEqual(call.input.teacherAllowedRoles, ["safeguarding"]);
    assert.equal(call.input.preserveRemoteSubjectPolicy, true);
    assert.equal(call.input.remoteSubjectPolicy, undefined);
    assert.doesNotMatch(
      (call.input.body as Buffer).toString("utf8"),
      /safeguarding_route_locator|os1_ns1_learner/
    );
  }
});

test("revocation, directory outage, and cross-tenant locators make zero worker calls", async () => {
  for (const [service, expected] of [
    [entitlement(["teacher"]), 403],
    [entitlement(["safeguarding"], true), 503]
  ] as const) {
    const calls: Array<{scope: unknown; input: Record<string, unknown>}> = [];
    const controller = new HarnessGatewayController(workers(calls), config(), undefined, service);
    const reply = fakeReply();
    await controller.proxy(
      PRINCIPAL,
      "api/safeguarding/case/acknowledge",
      fakeRequest({
        case_id: `sgc_${"1".repeat(24)}`,
        expected_version: 1,
        safeguarding_idempotency_key: "safeguarding-denied-001",
        safeguarding_route_locator: issueSafeguardingRouteLocator(LEARNER_SCOPE, KEY)
      }),
      reply.reply
    );
    assert.equal(reply.state.status, expected);
    assert.equal(calls.length, 0);
  }

  const calls: Array<{scope: unknown; input: Record<string, unknown>}> = [];
  const controller = new HarnessGatewayController(
    workers(calls), config(), undefined, entitlement(["safeguarding"])
  );
  const reply = fakeReply();
  const crossTenant = issueSafeguardingRouteLocator(
    {tenantId: "ot1_ns1_other-school", ownerId: LEARNER_SCOPE.ownerId},
    KEY
  );
  await controller.proxy(
    PRINCIPAL,
    "api/safeguarding/list",
    fakeRequest({
      safeguarding_idempotency_key: "safeguarding-cross-tenant-001",
      safeguarding_route_locator: crossTenant
    }),
    reply.reply
  );
  assert.equal(reply.state.status, 403);
  assert.equal(calls.length, 0);
});
