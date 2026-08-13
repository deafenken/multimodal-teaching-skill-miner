export const HARNESS_REQUEST_LIMITS = {
  default: 64 * 1024,
  attachment: 6 * 1024 * 1024,
  resource: 17 * 1024 * 1024,
  resourceReview: 64 * 1024,
  syllabus: 384 * 1024,
  project: 2 * 1024 * 1024,
  adjudication: 64 * 1024,
  consent: 16 * 1024,
  safeguarding: 16 * 1024,
} as const;

export function harnessRequestBodyLimit(route: string): number {
  if (route === "api/attachment") return HARNESS_REQUEST_LIMITS.attachment;
  if (route === "api/resource/review") return HARNESS_REQUEST_LIMITS.resourceReview;
  if (route.startsWith("api/curriculum/")) return HARNESS_REQUEST_LIMITS.syllabus;
  if (route === "api/resource") return HARNESS_REQUEST_LIMITS.resource;
  if (route === "api/syllabi" || route.startsWith("api/syllabi/")) {
    return HARNESS_REQUEST_LIMITS.syllabus;
  }
  if (route === "api/projects" || route.startsWith("api/projects/")) {
    return HARNESS_REQUEST_LIMITS.project;
  }
  if (route.startsWith("api/adjudication/")) {
    return HARNESS_REQUEST_LIMITS.adjudication;
  }
  if (route.startsWith("api/consent/")) return HARNESS_REQUEST_LIMITS.consent;
  if (route.startsWith("api/safeguarding/")) {
    return HARNESS_REQUEST_LIMITS.safeguarding;
  }
  return HARNESS_REQUEST_LIMITS.default;
}

export function harnessRequestBodyAllowed(route: string, byteLength: number): boolean {
  return Number.isSafeInteger(byteLength) && byteLength >= 1 && byteLength <= harnessRequestBodyLimit(route);
}

export function requestBodyLimitForUrl(rawUrl: string): number {
  const pathname = rawUrl.split("?", 1)[0] ?? "";
  const prefix = "/api/v1/harness/";
  return pathname.startsWith(prefix)
    ? harnessRequestBodyLimit(pathname.slice(prefix.length))
    : HARNESS_REQUEST_LIMITS.default;
}
