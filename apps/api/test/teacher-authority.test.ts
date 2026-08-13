import "reflect-metadata";

import assert from "node:assert/strict";
import {createHash, createHmac} from "node:crypto";
import {EventEmitter} from "node:events";
import {Readable} from "node:stream";
import test from "node:test";

import type {FastifyReply, FastifyRequest} from "fastify";

import type {AuthenticatedPrincipal} from "../src/auth/auth-provider.port";
import {TeacherEntitlementAuthorizationService} from "../src/auth/teacher-entitlement-authorization";
import type {
  AuthoritativeTeacherEntitlementSnapshot,
  CanonicalEntitlementIdentity,
  TeacherEntitlementSnapshotProvider
} from "../src/auth/teacher-entitlement.port";
import type {AppConfigService} from "../src/config/app-config.service";
import {HarnessGatewayController} from "../src/harness/harness-gateway.controller";
import type {
  HarnessUpstreamRequest,
  HarnessWorkerPoolService
} from "../src/harness/harness-worker-pool.service";
import {
  TEACHER_AUTHORITY_KIND,
  TEACHER_AUTHORITY_OPERATION_SPECS,
  canonicalJson,
  requestContainsAuthorityOverride,
  sha256Canonical,
  teacherAuthorityEnvelope
} from "../src/harness/teacher-authority";

const AUTHORITY_KEY = Buffer.alloc(32, 0x61);
const ENTITLEMENT_NOW = Date.parse("2026-08-12T04:00:00.000Z");
const ENTITLEMENT_IDENTITY: CanonicalEntitlementIdentity = {
  issuer: "https://identity.example.test",
  tenantId: "tenant-a",
  subject: "teacher-a"
};
const BODY = {
  session_id: "teach_session_1",
  item_id: `adj_${"1".repeat(24)}`,
  expected_version: 2,
  adjudication_idempotency_key: "teacher-authority-idempotency-001"
};
const REVIEW_BODY = {
  resource_id: `res_${"2".repeat(20)}`,
  staged_resource_id: `stage_${"3".repeat(24)}`,
  content_sha256: "4".repeat(64),
  expected_review_version: 0,
  resource_review_idempotency_key: "resource-review-idempotency-001",
  original_resource_sha256: "5".repeat(64),
  reviewed_text: "教师已对照原始资料确认的教学上下文。",
  resolved_conflict_ids: [`conflict_${"6".repeat(16)}`],
  excluded_layer_ids: [`layer_${"7".repeat(16)}`],
  attestations: {
    compared_with_original_source: true,
    uncertainties_removed_or_explicit: true,
    not_an_answer_key: true,
    context_only: true
  },
  review_note: "仅作为不受信任的教学上下文。"
};
const CURRICULUM_REVIEW_BODY = {
  syllabus_id: `syl_${"8".repeat(24)}`,
  teacher_spec: {
    source_spans: [{authority: true}],
    rubrics: [{authority: true}],
    item_blueprints: [{authority: true}]
  },
  expected_syllabus_version: 1,
  expected_authority_version: 0,
  curriculum_authority_idempotency_key: "curriculum-review-idempotency-001"
};
const CURRICULUM_SEAL_BODY = {
  syllabus_id: `syl_${"8".repeat(24)}`,
  review_id: `currev_${"9".repeat(24)}`,
  teacher_confirmed_authority: true,
  expected_syllabus_version: 1,
  expected_authority_version: 1,
  curriculum_authority_idempotency_key: "curriculum-seal-idempotency-001"
};
const CURRICULUM_REVOKE_BODY = {
  syllabus_id: `syl_${"8".repeat(24)}`,
  curriculum_id: `cur_${"a".repeat(24)}`,
  reason_code: "teacher_revoked",
  expected_syllabus_version: 1,
  expected_authority_version: 2,
  curriculum_authority_idempotency_key: "curriculum-revoke-idempotency-001"
};

function context(subject = "teacher-a") {
  return {
    scopeId: `scope_${"a".repeat(48)}`,
    scopeKeyVersion: "k2",
    authorityKey: AUTHORITY_KEY,
    principalIssuer: "https://identity.example.test",
    principalSubject: subject,
    roles: ["teacher", "curriculum-editor"],
    allowedRoles: ["teacher"],
    ttlSeconds: 120,
    now: new Date("2026-08-12T04:00:00Z")
  };
}

test("teacher authority envelope binds identity, scope, route, body, role policy and idempotency without raw identity", () => {
  const envelope = teacherAuthorityEnvelope(
    "api/adjudication/decide",
    BODY,
    context()
  );
  assert.equal(envelope.authority_kind, TEACHER_AUTHORITY_KIND);
  assert.equal(envelope.method, "POST");
  assert.equal(envelope.path, "api/adjudication/decide");
  assert.equal(envelope.body_sha256, sha256Canonical(BODY));
  assert.equal(
    envelope.idempotency_key_sha256,
    createHash("sha256")
      .update(BODY.adjudication_idempotency_key, "utf8")
      .digest("hex")
  );
  const serialized = JSON.stringify(envelope);
  assert.doesNotMatch(serialized, /teacher-a|identity\.example|curriculum-editor/);
  assert.match(String(envelope.actor_principal_sha256), /^[0-9a-f]{64}$/);
  assert.notEqual(
    envelope.actor_principal_sha256,
    teacherAuthorityEnvelope(
      "api/adjudication/decide",
      BODY,
      context("teacher-b")
    ).actor_principal_sha256
  );
  const material = {...envelope};
  const signature = material.signature;
  delete material.signature;
  assert.equal(
    signature,
    createHmac("sha256", AUTHORITY_KEY)
      .update(canonicalJson(material), "utf8")
      .digest("base64url")
  );
});

test("teacher authority operation specs bind each exact route to its own idempotency field", () => {
  assert.deepEqual(TEACHER_AUTHORITY_OPERATION_SPECS, {
    "api/adjudication/claim": {
      method: "POST",
      idempotencyField: "adjudication_idempotency_key"
    },
    "api/adjudication/decide": {
      method: "POST",
      idempotencyField: "adjudication_idempotency_key"
    },
    "api/resource/review": {
      method: "POST",
      idempotencyField: "resource_review_idempotency_key"
    },
    "api/curriculum/review": {
      method: "POST",
      idempotencyField: "curriculum_authority_idempotency_key"
    },
    "api/curriculum/seal": {
      method: "POST",
      idempotencyField: "curriculum_authority_idempotency_key"
    },
    "api/curriculum/revoke": {
      method: "POST",
      idempotencyField: "curriculum_authority_idempotency_key"
    },
    "api/safeguarding/list": {
      method: "POST",
      idempotencyField: "safeguarding_idempotency_key"
    },
    "api/safeguarding/dispatch": {
      method: "POST",
      idempotencyField: "safeguarding_idempotency_key"
    },
    "api/safeguarding/case/acknowledge": {
      method: "POST",
      idempotencyField: "safeguarding_idempotency_key"
    },
    "api/safeguarding/case/close": {
      method: "POST",
      idempotencyField: "safeguarding_idempotency_key"
    },
    "api/safeguarding/escalation/overdue": {
      method: "POST",
      idempotencyField: "safeguarding_idempotency_key"
    },
    "api/safeguarding/escalation/acknowledge": {
      method: "POST",
      idempotencyField: "safeguarding_idempotency_key"
    }
  });

  assert.doesNotThrow(() => teacherAuthorityEnvelope(
    "api/curriculum/review",
    CURRICULUM_REVIEW_BODY,
    context()
  ));
  for (const forgedSpec of [
    {...CURRICULUM_REVIEW_BODY.teacher_spec, identity: "browser-forged"},
    {...CURRICULUM_REVIEW_BODY.teacher_spec, nested: {authority_receipt: "forged"}},
    {...CURRICULUM_REVIEW_BODY.teacher_spec, source_spans: [{authority: false}]},
    {...CURRICULUM_REVIEW_BODY.teacher_spec, factual_claims: [{authority: true}]}
  ]) {
    assert.throws(() => teacherAuthorityEnvelope(
      "api/curriculum/review",
      {...CURRICULUM_REVIEW_BODY, teacher_spec: forgedSpec},
      context()
    ), /not eligible/);
  }

  const review = teacherAuthorityEnvelope(
    "api/resource/review",
    REVIEW_BODY,
    context()
  );
  assert.equal(review.method, "POST");
  assert.equal(review.path, "api/resource/review");
  assert.equal(review.body_sha256, sha256Canonical(REVIEW_BODY));
  assert.equal(
    review.idempotency_key_sha256,
    createHash("sha256")
      .update(REVIEW_BODY.resource_review_idempotency_key, "utf8")
      .digest("hex")
  );
  assert.doesNotMatch(
    JSON.stringify(review),
    /teacher-a|identity\.example|resource-review-idempotency/
  );

  assert.throws(
    () => teacherAuthorityEnvelope(
      "api/resource/review",
      {...REVIEW_BODY, resource_review_idempotency_key: undefined},
      context()
    ),
    /idempotency key is missing/
  );
  assert.throws(
    () => teacherAuthorityEnvelope(
      "api/adjudication/claim",
      REVIEW_BODY,
      context()
    ),
    /idempotency key is missing/
  );
});

test("authority overrides are rejected recursively and unsupported routes or roles cannot mint", () => {
  for (const value of [
    {actor: {identity: "forged"}},
    {correction: {authority_receipt: "forged"}},
    {nested: [{roles: ["teacher"]}]},
    {_teacher_authority: {signature: "forged"}},
    {tenant_id: "tenant-other"},
    {tenantId: "tenant-other"},
    {deep: [{ownerId: "owner-other"}]},
    {identity: "forged"},
    {nested: {authenticated_identity: "forged"}},
    {nested: [{principal_subject: "forged"}]},
    {correction: {"authority-envelope": "forged"}},
    {teacherAuthority: {signature: "forged"}}
  ]) {
    assert.equal(requestContainsAuthorityOverride(value), true);
  }
  assert.equal(
    requestContainsAuthorityOverride({correction: {signal: "partial"}}),
    false
  );
  assert.throws(
    () => teacherAuthorityEnvelope("api/start", BODY, context()),
    /not eligible/
  );
  assert.throws(
    () =>
      teacherAuthorityEnvelope("api/adjudication/claim", BODY, {
        ...context(),
        roles: ["learner"]
      }),
    /role is missing/
  );
  for (const idempotency of [
    "1234567",
    "-1234567",
    "_1234567",
    "safe/key0",
    " padded",
    "padded ",
    "x".repeat(161)
  ]) {
    assert.throws(
      () => teacherAuthorityEnvelope(
        "api/adjudication/claim",
        {...BODY, adjudication_idempotency_key: idempotency},
        context()
      ),
      /idempotency key is missing/
    );
  }
});

function incomingJson(value: unknown) {
  const response = Readable.from([Buffer.from(JSON.stringify(value), "utf8")]) as Readable & {
    statusCode: number;
    headers: Record<string, string>;
  };
  response.statusCode = 200;
  response.headers = {"content-type": "application/json"};
  return response;
}

function fakeRequest(body: Record<string, unknown>): FastifyRequest {
  return {
    method: "POST",
    body,
    headers: {accept: "application/json"},
    raw: new EventEmitter()
  } as unknown as FastifyRequest;
}

function fakeGetRequest(): FastifyRequest {
  return {
    method: "GET",
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
    status(code: number) {
      state.status = code;
      return this;
    },
    header() { return this; },
    send(payload?: unknown) {
      state.payload = payload;
      return this;
    }
  } as unknown as FastifyReply;
  return {reply, state};
}

function principal(
  roles: string[],
  provider: "development" | "oidc" = "oidc"
): AuthenticatedPrincipal {
  return {
    tenantId: "tenant-a",
    subject: "teacher-a",
    sessionId: "session-a",
    provider,
    roles,
    ...(provider === "oidc"
      ? {
          identityNamespace: "oidc-issuer-tenant-sub-v1" as const,
          identityIssuer: ENTITLEMENT_IDENTITY.issuer,
          authenticatedAt: "2026-08-12T03:55:00.000Z",
          assuranceLevel: 2,
          scopeTenantId: "tenant-a",
          scopeOwnerId: "teacher-a"
        }
      : {})
  };
}

function entitlementSnapshot(
  overrides: Partial<AuthoritativeTeacherEntitlementSnapshot> = {}
): AuthoritativeTeacherEntitlementSnapshot {
  return {
    identity: {...ENTITLEMENT_IDENTITY},
    status: "active",
    roles: ["teacher"],
    policyVersion: "roles-v7",
    revision: 7,
    evaluatedAt: new Date(ENTITLEMENT_NOW).toISOString(),
    expiresAt: new Date(ENTITLEMENT_NOW + 60_000).toISOString(),
    ...overrides
  };
}

class DirectoryProvider implements TeacherEntitlementSnapshotProvider {
  calls = 0;

  constructor(
    public value: AuthoritativeTeacherEntitlementSnapshot | null = entitlementSnapshot(),
    public failure?: Error
  ) {}

  async readAuthoritativeSnapshot() {
    this.calls += 1;
    if (this.failure) throw this.failure;
    return this.value;
  }
}

function controllerHarness(
  provider = new DirectoryProvider(),
  upstreamValue: unknown = {ok: true}
) {
  const calls: HarnessUpstreamRequest[] = [];
  const workers = {
    async request(_scope: unknown, input: HarnessUpstreamRequest) {
      calls.push(input);
      return incomingJson(upstreamValue);
    }
  } as unknown as HarnessWorkerPoolService;
  const config = {
    nodeEnv: "production",
    oidcIssuer: "https://identity.example.test",
    harnessResponseMaxBytes: 64 * 1024,
    teacherAuthorityRoles: ["teacher"],
    principalHasTeacherAuthority(roles: readonly string[]) {
      return roles.includes("teacher");
    }
  } as unknown as AppConfigService;
  const entitlements = new TeacherEntitlementAuthorizationService(
    provider,
    {
      policyId: "teacher-mutations",
      version: "roles-v7",
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
      providerTimeoutMs: 100,
      maxClockSkewMs: 1_000,
      minAssuranceLevel: 2
    },
    Buffer.alloc(32, 0x62),
    Buffer.alloc(32, 0x72),
    {now: () => new Date(ENTITLEMENT_NOW)}
  );
  return {
    controller: new HarnessGatewayController(
      workers,
      config,
      undefined,
      entitlements
    ),
    calls,
    provider
  };
}

test("non-OIDC and malicious authority inputs fail before directory or worker side effects", async () => {
  const {controller, calls, provider} = controllerHarness();
  for (const [actor, body, status] of [
    [principal(["teacher"], "development"), BODY, 403],
    [principal(["teacher"]), {...BODY, actor: {authenticated: true}}, 400],
    [principal(["teacher"]), {...BODY, correction: {principal: "forged"}}, 400]
  ] as const) {
    const output = fakeReply();
    await controller.proxy(
      actor,
      "api/adjudication/decide",
      fakeRequest(body),
      output.reply
    );
    assert.equal(output.state.status, status);
  }
  assert.equal(calls.length, 0);
  assert.equal(provider.calls, 0);
});

test("bootstrap teacher capability is projected from the directory and never from cookie roles", async () => {
  const bootstrap = {
    teacher_authority: {
      mode: "authenticated_apps_api",
      role_authorized: false,
      correct_mastery_updates_enabled: false,
      assurance: "deployment_service_role_authorization_not_personal_signature",
      raw_identity_exposed: false,
      nonce_replay_policy: "append_only_permanent_tombstone_bounded_fail_closed",
      replay_store_max_bytes: 16 * 1024 * 1024,
      expired_nonce_tombstones_retained: true
    }
  };
  const allowed = controllerHarness(new DirectoryProvider(), bootstrap);
  const allowedReply = fakeReply();
  await allowed.controller.proxy(
    principal(["learner"]),
    "api/bootstrap",
    fakeGetRequest(),
    allowedReply.reply
  );
  assert.equal(
    (allowedReply.state.payload as any).teacher_authority.role_authorized,
    true
  );

  const denied = controllerHarness(
    new DirectoryProvider(entitlementSnapshot({roles: ["learner"]})),
    bootstrap
  );
  const deniedReply = fakeReply();
  await denied.controller.proxy(
    principal(["teacher", "platform-admin"]),
    "api/bootstrap",
    fakeGetRequest(),
    deniedReply.reply
  );
  assert.equal(
    (deniedReply.state.payload as any).teacher_authority.role_authorized,
    false
  );
});

test("fresh directory role authorizes despite a stale role-missing cookie and binds only the receipt hash", async () => {
  const {controller, calls} = controllerHarness();
  const output = fakeReply();
  await controller.proxy(
    principal(["learner"]),
    "api/adjudication/decide",
    fakeRequest(BODY),
    output.reply
  );
  assert.equal(output.state.status, 200);
  assert.equal(calls.length, 1);
  assert.deepEqual(calls[0]?.teacherRoles, ["teacher"]);
  assert.equal(
    calls[0]?.teacherPrincipal?.issuer,
    "teachlab-authoritative-entitlement-v1"
  );
  assert.match(calls[0]?.teacherPrincipal?.subject ?? "", /^[0-9a-f]{64}$/);
  assert.equal(calls[0]?.body?.toString("utf8"), JSON.stringify(BODY));
  assert.doesNotMatch(
    calls[0]?.body?.toString("utf8") ?? "",
    /"(?:teacher-a|tenant-a)"/
  );
});

test("authorized resource review route forwards only private teacher context and rejects browser identity attacks", async () => {
  const {controller, calls} = controllerHarness();
  const output = fakeReply();
  await controller.proxy(
    principal(["teacher"]),
    "api/resource/review",
    fakeRequest(REVIEW_BODY),
    output.reply
  );
  assert.equal(output.state.status, 200);
  assert.equal(calls.length, 1);
  assert.deepEqual(calls[0]?.teacherRoles, ["teacher"]);
  assert.equal(
    calls[0]?.teacherPrincipal?.issuer,
    "teachlab-authoritative-entitlement-v1"
  );
  assert.match(calls[0]?.teacherPrincipal?.subject ?? "", /^[0-9a-f]{64}$/);
  assert.equal(calls[0]?.body?.toString("utf8"), JSON.stringify(REVIEW_BODY));
  assert.doesNotMatch(
    calls[0]?.body?.toString("utf8") ?? "",
    /teacher-a|tenant-a|identity\.example/
  );

  for (const attack of [
    {...REVIEW_BODY, identity: "forged"},
    {...REVIEW_BODY, nested: {authority_receipt: "forged"}},
    {...REVIEW_BODY, attestations: {...REVIEW_BODY.attestations, actor: "forged"}}
  ]) {
    const rejected = fakeReply();
    await controller.proxy(
      principal(["teacher"]),
      "api/resource/review",
      fakeRequest(attack),
      rejected.reply
    );
    assert.equal(rejected.state.status, 400);
  }
  assert.equal(calls.length, 1);
});

test("directory outage, revoked entitlement, missing role, and stale snapshots fail closed on every teacher route", async () => {
  const operations = [
    ["api/resource/review", REVIEW_BODY],
    ["api/curriculum/review", CURRICULUM_REVIEW_BODY],
    ["api/curriculum/seal", CURRICULUM_SEAL_BODY],
    ["api/curriculum/revoke", CURRICULUM_REVOKE_BODY],
    ["api/adjudication/claim", BODY],
    ["api/adjudication/decide", BODY]
  ] as const;
  const failures = [
    [new DirectoryProvider(entitlementSnapshot(), new Error("private outage detail")), 503],
    [new DirectoryProvider(entitlementSnapshot({status: "revoked", roles: []})), 403],
    [new DirectoryProvider(entitlementSnapshot({roles: ["learner"]})), 403],
    [new DirectoryProvider(entitlementSnapshot({
      evaluatedAt: new Date(ENTITLEMENT_NOW - 60_000).toISOString(),
      expiresAt: new Date(ENTITLEMENT_NOW + 60_000).toISOString()
    })), 503]
  ] as const;

  for (const [template, expectedStatus] of failures) {
    for (const [path, body] of operations) {
      const provider = new DirectoryProvider(template.value, template.failure);
      const {controller, calls} = controllerHarness(provider);
      const output = fakeReply();
      await controller.proxy(
        // A forged/stale role-bearing cookie cannot change the result.
        principal(["teacher", "platform-admin"]),
        path,
        fakeRequest(body),
        output.reply
      );
      assert.equal(output.state.status, expectedStatus, path);
      assert.equal(calls.length, 0, `${path} reached a worker`);
      assert.equal(provider.calls, 1, `${path} did not consult the directory`);
      assert.doesNotMatch(JSON.stringify(output.state.payload), /private outage detail/);
    }
  }
});
