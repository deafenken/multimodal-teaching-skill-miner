import assert from "node:assert/strict";
import test from "node:test";

import {teacherAuthorityUiState} from "../lib/teacher-authority.ts";

test("authenticated teacher UI requires the complete server bootstrap assertion", () => {
  const authorized = teacherAuthorityUiState({
    mode: "authenticated_apps_api",
    role_authorized: true,
    correct_mastery_updates_enabled: true,
    assurance: "deployment_service_role_authorization_not_personal_signature",
    raw_identity_exposed: false,
  });
  assert.deepEqual(authorized, {
    mode: "authenticated_apps_api",
    actionsAllowed: true,
    authenticatedTeacher: true,
    title: "认证教师裁决",
  });

  for (const value of [
    undefined,
    {
      mode: "authenticated_apps_api" as const,
      role_authorized: false,
      correct_mastery_updates_enabled: true,
      assurance: "deployment_service_role_authorization_not_personal_signature" as const,
      raw_identity_exposed: false as const,
    },
    {
      mode: "authenticated_apps_api" as const,
      role_authorized: true,
      correct_mastery_updates_enabled: false,
      assurance: "deployment_service_role_authorization_not_personal_signature" as const,
      raw_identity_exposed: false as const,
    },
  ]) {
    const state = teacherAuthorityUiState(value);
    assert.equal(state.actionsAllowed, false);
    assert.equal(state.authenticatedTeacher, false);
  }
});

test("standalone UI stays explicitly unauthenticated and never enables correct mastery", () => {
  assert.deepEqual(teacherAuthorityUiState({
    mode: "local_python",
    role_authorized: false,
    correct_mastery_updates_enabled: false,
    assurance: "no_authenticated_teacher_identity",
    raw_identity_exposed: false,
  }), {
    mode: "local_python",
    actionsAllowed: true,
    authenticatedTeacher: false,
    title: "本机未认证裁决",
  });
});
