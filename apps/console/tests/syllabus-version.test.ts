import assert from "node:assert/strict";
import test from "node:test";

import {normalizeSyllabusVersionFamily} from "../lib/syllabus-version.ts";

function family() {
  return {
    schema: "teaching_skill_miner.syllabus_version_projection.v1",
    family_id: "syf_0123456789abcdef01234567",
    version: 2,
    published_revision_id: "syr_0123456789abcdef01234567",
    revisions: [
      {
        revision_id: "syr_0123456789abcdef01234567",
        revision_number: 1,
        syllabus_id: "syl_0123456789abcdef01234567",
        content_sha256: "a".repeat(64),
        parent_revision_id: null,
        created_at_utc: "2026-08-12T01:00:00Z",
        change_summary: "Initial immutable syllabus revision",
        status: "published",
      },
      {
        revision_id: "syr_89abcdef0123456789abcdef",
        revision_number: 2,
        syllabus_id: "syl_89abcdef0123456789abcdef",
        content_sha256: "b".repeat(64),
        parent_revision_id: "syr_0123456789abcdef01234567",
        created_at_utc: "2026-08-12T02:00:00Z",
        change_summary: "Refine prerequisite order",
        status: "draft",
      },
    ],
    actor_boundary: {
      actor_kind: "local_operator_not_authenticated",
      authenticated: false,
      teacher_identity_claimed: false,
    },
  };
}

test("syllabus version projection requires one published immutable revision", () => {
  assert.deepEqual(normalizeSyllabusVersionFamily({version_family: family()}), family());
  const forged = family();
  forged.revisions[1]!.status = "published";
  assert.throws(
    () => normalizeSyllabusVersionFamily(forged),
    /无效的大纲版本账本/,
  );
});

test("syllabus version projection never upgrades a local operator to teacher identity", () => {
  const forged = family();
  forged.actor_boundary.authenticated = true;
  assert.throws(
    () => normalizeSyllabusVersionFamily(forged),
    /无效的大纲版本账本/,
  );
});
