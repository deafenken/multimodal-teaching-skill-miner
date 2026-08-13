import "reflect-metadata";

import assert from "node:assert/strict";
import {test} from "node:test";

import {ProductionSurfaceGuard, legacySurfacePath} from "../src/operations/production-surface.guard";

test("only pre-v1 compatibility surfaces are exact and modern durable routes remain available", () => {
  for (const path of [
    "/api/bootstrap",
    "/api/step",
    "/api/events",
  ]) assert.equal(legacySurfacePath(path), true, path);
  for (const path of [
    "/health",
    "/ready",
    "/metrics",
    "/api/v1/auth/session",
    "/api/v1/harness/api/bootstrap",
    "/api/v1/account/export",
    "/api/v1/providers/model",
    "/api/v1/sessions",
    "/api/v1/sessions/s1/tasks",
    "/api/v1/tasks/t1",
  ]) assert.equal(legacySurfacePath(path), false, path);
});

test("production guard returns 410 while development retains compatibility", () => {
  const context = (url: string) => ({
    switchToHttp: () => ({getRequest: () => ({url})}),
  }) as never;
  const production = new ProductionSurfaceGuard({nodeEnv: "production"} as never);
  assert.throws(
    () => production.canActivate(context("/api/events/t1?x=1")),
    (error: unknown) => {
      const response = (error as {getResponse?: () => unknown}).getResponse?.() as {code?: string} | undefined;
      return response?.code === "legacy_surface_disabled";
    },
  );
  assert.equal(production.canActivate(context("/api/v1/harness/api/tasks/status")), true);
  const development = new ProductionSurfaceGuard({nodeEnv: "development"} as never);
  assert.equal(development.canActivate(context("/api/v1/tasks/t1")), true);
});
