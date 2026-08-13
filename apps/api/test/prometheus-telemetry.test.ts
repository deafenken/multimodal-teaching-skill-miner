import assert from "node:assert/strict";
import {test} from "node:test";

import {PrometheusTelemetryAdapter} from "../src/observability/prometheus-telemetry.adapter";
import type {HarnessWorkerPoolService} from "../src/harness/harness-worker-pool.service";
import type {ContinuousReadinessService} from "../src/health/continuous-readiness.service";
import type {ResourceGovernor} from "../src/operations/resource-governor";


test("Prometheus telemetry keeps only low-cardinality non-identity labels", () => {
  const telemetry = new PrometheusTelemetryAdapter();
  telemetry.duration("http.server.request", 37.2, {
    requestId: "attacker-controlled-request-id",
    tenantId: "tenant-secret",
    method: "POST",
    route: "/api/v1/sessions/:sessionId/tasks",
    statusCode: 201,
    failed: false
  });
  telemetry.event("task.queued", {
    taskId: "task_secret",
    sessionId: "session_secret",
    provider: "deepseek"
  });

  const output = telemetry.renderPrometheus();
  assert.match(output, /route="\/api\/v1\/sessions\/:sessionId\/tasks"/);
  assert.match(output, /status_class="2xx"/);
  assert.match(output, /event="task.queued",provider="deepseek"/);
  for (const forbidden of [
    "attacker-controlled-request-id",
    "tenant-secret",
    "task_secret",
    "session_secret"
  ]) {
    assert.equal(output.includes(forbidden), false);
  }
  assert.equal(telemetry.safeSnapshot().unsafeAttributesDropped, 2);
  assert.equal(telemetry.safeSnapshot().containsUserIdentifiers, false);
});


test("unsafe route, method, event and provider values collapse to bounded labels", () => {
  const telemetry = new PrometheusTelemetryAdapter();
  telemetry.duration("http.server.request", Number.POSITIVE_INFINITY, {
    method: "FORGED\nMETHOD",
    route: "/api/users/alice@example.test?secret=yes",
    statusCode: 999,
    failed: true
  });
  telemetry.event("user.alice.private", {provider: "bad\nprovider"});
  const output = telemetry.renderPrometheus();
  assert.match(output, /method="OTHER"/);
  assert.match(output, /route="unmatched"/);
  assert.match(output, /status_class="other"/);
  assert.match(output, /event="unknown",provider="none"/);
  assert.equal(output.includes("alice@example.test"), false);
  assert.equal(output.includes("secret=yes"), false);
});


test("histogram buckets and process gauges are valid Prometheus text", () => {
  const telemetry = new PrometheusTelemetryAdapter();
  telemetry.duration("http.server.request", 5, {
    method: "GET",
    route: "/health",
    statusClass: "2xx",
    failed: false
  });
  telemetry.duration("http.server.request", 2_000, {
    method: "GET",
    route: "/health",
    statusClass: "2xx",
    failed: false
  });
  const output = telemetry.renderPrometheus();
  assert.match(output, /le="0.005"[^\n]* 1/);
  assert.match(output, /le="2.5"[^\n]* 2/);
  assert.match(output, /_count[^\n]* 2/);
  assert.match(output, /teachlab_process_resident_memory_bytes \d+/);
  assert.match(output, /teachlab_metrics_scrapes_total 1/);
  assert.match(output, /teachlab_preparse_active_requests 0/);
  assert.match(output, /reason="identity_concurrency"/);
  assert.match(output, /teachlab_harness_workers\{state="ready"\} 0/);
  assert.match(output, /teachlab_readiness_checks_total\{result="failed"\} 0/);
});


test("safeguarding and provider readiness metrics are aggregate-only and low-cardinality", () => {
  const telemetry = new PrometheusTelemetryAdapter();
  telemetry.bindOperationalSources({
    resourceGovernor: {snapshot: () => ({})} as ResourceGovernor,
    readiness: {
      operationalSnapshot: () => ({
        attempts: 1,
        successes: 0,
        failures: 1,
        lastResult: "failed"
      })
    } as ContinuousReadinessService,
    workers: {
      status: () => ({
        safeguardingSupervisorPending: 7,
        safeguardingSupervisorOverdue: 2,
        safeguardingSupervisorOldestPendingAgeSeconds: 901,
        safeguardingSupervisorAttemptedTotal: 11,
        safeguardingSupervisorAcceptedTotal: 8,
        safeguardingSupervisorFailedTotal: 3,
        safeguardingSupervisorLastSuccessAtUtc: "2026-08-12T12:34:56Z",
        safeguardingSupervisorReceiverReadinessAttemptsTotal: 4,
        safeguardingSupervisorReceiverReadinessSuccessesTotal: 3,
        safeguardingSupervisorReceiverReadinessFailuresTotal: 1,
        safeguardingSupervisorLastReceiverSuccessAtUtc: "2026-08-12T12:35:00Z",
        safeguardingSupervisorRetentionEnabled: true,
        safeguardingSupervisorRetentionEligibleCases: 5,
        safeguardingSupervisorRetentionCasesCompactedTotal: 17,
        safeguardingSupervisorRetentionEventsCompactedTotal: 49,
        safeguardingSupervisorRetentionFailures: 0,
        safeguardingSupervisorRetentionBlockedStores: 1,
        safeguardingSupervisorCapacityNearLimitStores: 1,
        safeguardingSupervisorCapacityEvents: 90,
        safeguardingSupervisorCapacityEventLimit: 100,
        safeguardingSupervisorCapacityEventHeadroomMin: 10,
        safeguardingSupervisorCapacityStoreBytes: 900,
        safeguardingSupervisorCapacityStoreByteLimit: 1000,
        safeguardingSupervisorCapacityStoreByteHeadroomMin: 100,
        safeguardingSupervisorCapacityRecentErasureTombstones: 9,
        safeguardingSupervisorCapacityRecentErasureTombstoneLimit: 10,
        safeguardingSupervisorCapacityRecentErasureTombstoneHeadroomMin: 1,
        safeguardingSupervisorErasureFenceInsertedCount: 23,
        safeguardingSupervisorErasureFenceEstimatedFalsePositiveUpperBound:
          0.0000005,
        safeguardingSupervisorErasureFenceFalsePositiveWithinTarget: true,
        providerReadinessAttempts: 5,
        providerReadinessSuccesses: 4,
        providerReadinessFailures: 1,
        providerReadinessLastResult: "ready",
        readyWorkers: 0,
        startingWorkers: 0,
        stoppingWorkers: 0,
        activeRequests: 0
      })
    } as HarnessWorkerPoolService
  });
  const output = telemetry.renderPrometheus();
  assert.match(output, /teachlab_safeguarding_supervisor_pending 7/);
  assert.match(output, /teachlab_safeguarding_supervisor_overdue 2/);
  assert.match(output, /teachlab_safeguarding_supervisor_oldest_pending_age_seconds 901/);
  assert.match(output, /teachlab_safeguarding_supervisor_delivery_attempts_total 11/);
  assert.match(output, /teachlab_safeguarding_supervisor_delivery_accepted_total 8/);
  assert.match(output, /teachlab_safeguarding_supervisor_delivery_failures_total 3/);
  assert.match(output, /teachlab_safeguarding_receiver_readiness_checks_total\{result="failed"\} 1/);
  assert.match(output, /teachlab_safeguarding_retention_enabled 1/);
  assert.match(output, /teachlab_safeguarding_retention_cases_compacted_total 17/);
  assert.match(output, /teachlab_safeguarding_retention_events_compacted_total 49/);
  assert.match(output, /teachlab_safeguarding_retention_blocked_stores 1/);
  assert.match(output, /teachlab_safeguarding_capacity_near_limit_stores 1/);
  assert.match(output, /teachlab_safeguarding_capacity_event_headroom_min 10/);
  assert.match(output, /teachlab_safeguarding_capacity_store_byte_headroom_min 100/);
  assert.match(output, /teachlab_safeguarding_erasure_fence_inserted_count 23/);
  assert.match(
    output,
    /teachlab_safeguarding_erasure_fence_estimated_false_positive_upper_bound 5e-7/
  );
  assert.match(output, /teachlab_harness_provider_readiness_checks_total\{result="attempted"\} 5/);
  assert.match(output, /teachlab_harness_provider_readiness_last_result\{result="ready"\} 1/);
  assert.equal(output.includes("scope_"), false);
  assert.equal(output.includes("tenant"), false);
  assert.equal(output.includes("learner"), false);
});
