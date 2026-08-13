import type {BootstrapPayload} from "@/lib/types";

export type TeacherAuthorityUiState =
  | {
      mode: "local_python";
      actionsAllowed: true;
      authenticatedTeacher: false;
      title: "本机未认证裁决";
    }
  | {
      mode: "authenticated_apps_api";
      actionsAllowed: boolean;
      authenticatedTeacher: boolean;
      title: "认证教师裁决" | "教师裁决权限不足";
    }
  | {
      mode: "unavailable";
      actionsAllowed: false;
      authenticatedTeacher: false;
      title: "裁决权限不可用";
    };

/**
 * Fail closed unless bootstrap declares one complete, internally consistent
 * deployment boundary. Browser state never upgrades a principal or supplies
 * authority claims; claim/decide authorization remains server-owned.
 */
export function teacherAuthorityUiState(
  authority: BootstrapPayload["teacher_authority"],
): TeacherAuthorityUiState {
  if (
    authority?.mode === "local_python"
    && authority.role_authorized === false
    && authority.correct_mastery_updates_enabled === false
    && authority.assurance === "no_authenticated_teacher_identity"
    && authority.raw_identity_exposed === false
  ) {
    return {
      mode: "local_python",
      actionsAllowed: true,
      authenticatedTeacher: false,
      title: "本机未认证裁决",
    };
  }
  if (
    authority?.mode === "authenticated_apps_api"
    && authority.raw_identity_exposed === false
    && authority.assurance
      === "deployment_service_role_authorization_not_personal_signature"
  ) {
    const authorized = authority.role_authorized === true
      && authority.correct_mastery_updates_enabled === true;
    return {
      mode: "authenticated_apps_api",
      actionsAllowed: authorized,
      authenticatedTeacher: authorized,
      title: authorized ? "认证教师裁决" : "教师裁决权限不足",
    };
  }
  return {
    mode: "unavailable",
    actionsAllowed: false,
    authenticatedTeacher: false,
    title: "裁决权限不可用",
  };
}
