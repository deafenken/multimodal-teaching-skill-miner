import {createHash, createHmac, randomBytes} from "node:crypto";

export const TEACHER_AUTHORITY_SCHEMA =
  "teaching_skill_miner.teacher_authority_envelope.v1";
export const TEACHER_AUTHORITY_KIND =
  "authenticated_teacher_server_authorized";

export const TEACHER_AUTHORITY_OPERATION_SPECS = {
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
} as const;

type TeacherAuthorityPath = keyof typeof TEACHER_AUTHORITY_OPERATION_SPECS;

export const TEACHER_AUTHORITY_PATHS: ReadonlySet<string> = new Set(
  Object.keys(TEACHER_AUTHORITY_OPERATION_SPECS)
);

const AUTHORITY_IDEMPOTENCY_KEY = /^[A-Za-z0-9][A-Za-z0-9._:-]{7,159}$/;

const FORBIDDEN_AUTHORITY_FIELDS = new Set([
  "tenant",
  "tenantid",
  "user",
  "userid",
  "owner",
  "ownerid",
  "subject",
  "principal",
  "identity",
  "authenticatedidentity",
  "authenticatedprincipal",
  "principalissuer",
  "principalsubject",
  "role",
  "roles",
  "actor",
  "authority",
  "authorityenvelope",
  "authorityreceipt",
  "teacherauthority"
]);

function authorityFieldName(value: string): string {
  return value.toLowerCase().replaceAll("_", "").replaceAll("-", "");
}

export interface TeacherAuthorityContext {
  scopeId: string;
  scopeKeyVersion: string;
  authorityKey: Buffer;
  principalIssuer: string;
  principalSubject: string;
  roles: readonly string[];
  allowedRoles: readonly string[];
  ttlSeconds: number;
  now?: Date;
}

type CanonicalValue = null | boolean | number | string | CanonicalValue[] | {
  [key: string]: CanonicalValue;
};

export function canonicalJson(value: unknown): string {
  const normalize = (input: unknown): CanonicalValue => {
    if (
      input === null || typeof input === "boolean" || typeof input === "string"
    ) return input;
    if (typeof input === "number" && Number.isFinite(input)) return input;
    if (Array.isArray(input)) return input.map(normalize);
    if (input && typeof input === "object") {
      return Object.fromEntries(
        Object.entries(input as Record<string, unknown>)
          .filter(([, item]) => item !== undefined)
          .sort(([left], [right]) => left.localeCompare(right))
          .map(([key, item]) => [key, normalize(item)])
      );
    }
    throw new Error("teacher authority values must contain canonical JSON");
  };
  return JSON.stringify(normalize(value));
}

export function sha256Canonical(value: unknown): string {
  return createHash("sha256").update(canonicalJson(value), "utf8").digest("hex");
}

export function requestContainsAuthorityOverride(value: unknown): boolean {
  if (Array.isArray(value)) return value.some(requestContainsAuthorityOverride);
  if (!value || typeof value !== "object") return false;
  return Object.entries(value as Record<string, unknown>).some(
    ([key, item]) => FORBIDDEN_AUTHORITY_FIELDS.has(authorityFieldName(key))
      || requestContainsAuthorityOverride(item)
  );
}

function curriculumTeacherSpecContainsAuthorityOverride(
  value: unknown,
  path: readonly string[] = []
): boolean {
  if (Array.isArray(value)) {
    return value.some((item) =>
      curriculumTeacherSpecContainsAuthorityOverride(item, [...path, "*"])
    );
  }
  if (!value || typeof value !== "object") return false;
  return Object.entries(value as Record<string, unknown>).some(([key, item]) => {
    const normalized = authorityFieldName(key);
    if (FORBIDDEN_AUTHORITY_FIELDS.has(normalized)) {
      // These are curriculum content assertions, not caller identity.  Their
      // exact locations and literal value are subsequently checked by the
      // strict Python curriculum schema.  No other authority-shaped key gets
      // exempted from the browser override fence.
      const permittedContentAuthority = normalized === "authority"
        && item === true
        && path.length === 2
        && path[1] === "*"
        && new Set(["source_spans", "rubrics", "item_blueprints"])
          .has(path[0] ?? "");
      if (!permittedContentAuthority) return true;
    }
    return curriculumTeacherSpecContainsAuthorityOverride(item, [...path, key]);
  });
}

export function teacherAuthorityRequestContainsOverride(
  path: string,
  body: Record<string, unknown>
): boolean {
  if (path !== "api/curriculum/review") {
    return requestContainsAuthorityOverride(body);
  }
  return requestContainsAuthorityOverride(
    Object.fromEntries(Object.entries(body).filter(([key]) => key !== "teacher_spec"))
  ) || curriculumTeacherSpecContainsAuthorityOverride(body.teacher_spec);
}

function isoSeconds(value: Date): string {
  return value.toISOString().replace(/\.\d{3}Z$/, "Z");
}

function teacherAuthorityOperation(path: string) {
  return Object.prototype.hasOwnProperty.call(
    TEACHER_AUTHORITY_OPERATION_SPECS,
    path
  )
    ? TEACHER_AUTHORITY_OPERATION_SPECS[path as TeacherAuthorityPath]
    : undefined;
}

export function teacherAuthorityEnvelope(
  path: string,
  body: Record<string, unknown>,
  context: TeacherAuthorityContext
): Record<string, unknown> {
  const operation = teacherAuthorityOperation(path);
  const hasAuthorityOverride = teacherAuthorityRequestContainsOverride(path, body);
  if (!operation || hasAuthorityOverride) {
    throw new Error("teacher authority request is not eligible");
  }
  const roleSet = [...new Set(context.roles)].sort();
  const allowedSet = [...new Set(context.allowedRoles)].sort();
  if (!roleSet.some((role) => allowedSet.includes(role))) {
    throw new Error("teacher authority role is missing");
  }
  const idempotencyKey = body[operation.idempotencyField];
  if (
    typeof idempotencyKey !== "string" ||
    !AUTHORITY_IDEMPOTENCY_KEY.test(idempotencyKey)
  ) {
    throw new Error("teacher authority idempotency key is missing");
  }
  const now = context.now ?? new Date();
  const issuedAt = isoSeconds(now);
  const expiresAt = isoSeconds(new Date(now.getTime() + context.ttlSeconds * 1000));
  const nonce = `tan_${randomBytes(24).toString("hex")}`;
  const actorPrincipalSha256 = createHmac("sha256", context.authorityKey)
    .update(
      `actor-principal-v1\0${context.scopeId}\0${context.principalIssuer.length}:` +
        `${context.principalIssuer}\0${context.principalSubject.length}:` +
        context.principalSubject,
      "utf8"
    )
    .digest("hex");
  const envelope: Record<string, unknown> = {
    schema: TEACHER_AUTHORITY_SCHEMA,
    authority_kind: TEACHER_AUTHORITY_KIND,
    assurance: "deployment_service_role_authorization_not_personal_signature",
    scope_id: context.scopeId,
    scope_key_version: context.scopeKeyVersion,
    actor_principal_sha256: actorPrincipalSha256,
    roles_sha256: sha256Canonical(roleSet),
    role_policy_sha256: sha256Canonical(allowedSet),
    method: operation.method,
    path,
    body_sha256: sha256Canonical(body),
    idempotency_key_sha256: createHash("sha256")
      .update(idempotencyKey, "utf8")
      .digest("hex"),
    issued_at: issuedAt,
    expires_at: expiresAt,
    nonce
  };
  envelope.authority_id = `tauth_${sha256Canonical(envelope).slice(0, 24)}`;
  envelope.signature = createHmac("sha256", context.authorityKey)
    .update(canonicalJson(envelope), "utf8")
    .digest("base64url");
  return envelope;
}
