import {Injectable, Logger} from "@nestjs/common";

import type {TelemetryAttributes, TelemetryPort} from "./telemetry.port";
import type {ContinuousReadinessService} from "../health/continuous-readiness.service";
import type {HarnessWorkerPoolService} from "../harness/harness-worker-pool.service";
import {preparseOperationalSnapshot} from "../operations/preparse-request-governor";
import type {ResourceGovernor} from "../operations/resource-governor";


const LATENCY_BUCKETS_MS = [5, 10, 25, 50, 100, 250, 500, 1_000, 2_500, 5_000, 10_000];
const METHOD = new Set(["GET", "HEAD", "POST", "PATCH", "DELETE", "OPTIONS"]);
const STATUS_CLASS = /^[1-5]xx$/;
const SAFE_ROUTE = /^\/[A-Za-z0-9_*/:.-]{0,159}$/;
const SAFE_EVENT = /^(?:task\.(?:queued|running|succeeded|failed|cancelled|requires_configuration)|unknown)$/;
const SAFE_PROVIDER = /^[A-Za-z0-9._:-]{1,64}$/;

interface HttpSeries {
  method: string;
  route: string;
  statusClass: string;
  failed: "true" | "false";
  count: number;
  durationCount: number;
  durationSumMs: number;
  buckets: number[];
}

function escapeLabel(value: string): string {
  return value.replace(/\\/g, "\\\\").replace(/"/g, '\\"').replace(/\n/g, "\\n");
}

function labelSet(values: Record<string, string>): string {
  return `{${Object.entries(values)
    .map(([key, value]) => `${key}="${escapeLabel(value)}"`)
    .join(",")}}`;
}

function boundedRoute(value: unknown): string {
  return typeof value === "string" && SAFE_ROUTE.test(value) ? value : "unmatched";
}

function boundedMethod(value: unknown): string {
  return typeof value === "string" && METHOD.has(value.toUpperCase())
    ? value.toUpperCase()
    : "OTHER";
}

function boundedStatusClass(value: unknown): string {
  if (typeof value === "number" && Number.isInteger(value)) {
    const candidate = `${Math.floor(value / 100)}xx`;
    return STATUS_CLASS.test(candidate) ? candidate : "other";
  }
  if (typeof value === "string" && STATUS_CLASS.test(value)) return value;
  return "other";
}

function finiteMilliseconds(value: number): number {
  return Number.isFinite(value) ? Math.max(0, Math.min(value, 86_400_000)) : 0;
}

/**
 * In-process, low-cardinality Prometheus registry.
 *
 * It never stores request IDs, session/task IDs, tenant IDs, user IDs, model
 * text, URLs with query strings, or exception messages.  This is deliberately
 * a single-replica operational signal, not a distributed billing authority.
 */
@Injectable()
export class PrometheusTelemetryAdapter implements TelemetryPort {
  private readonly logger = new Logger("OperationalTelemetry");
  private readonly http = new Map<string, HttpSeries>();
  private readonly events = new Map<string, number>();
  private droppedUnsafeAttributes = 0;
  private scrapeCount = 0;
  private resourceGovernor?: ResourceGovernor;
  private readiness?: ContinuousReadinessService;
  private workers?: HarnessWorkerPoolService;

  bindOperationalSources(sources: {
    resourceGovernor: ResourceGovernor;
    readiness: ContinuousReadinessService;
    workers: HarnessWorkerPoolService;
  }): void {
    this.resourceGovernor = sources.resourceGovernor;
    this.readiness = sources.readiness;
    this.workers = sources.workers;
  }

  event(name: string, attributes: TelemetryAttributes = {}): void {
    const safeName = SAFE_EVENT.test(name) ? name : "unknown";
    const provider =
      typeof attributes.provider === "string" && SAFE_PROVIDER.test(attributes.provider)
        ? attributes.provider
        : "none";
    const key = `${safeName}\0${provider}`;
    this.events.set(key, (this.events.get(key) ?? 0) + 1);
    if (
      Object.keys(attributes).some(
        (key) => !new Set(["provider", "reason"]).has(key)
      )
    ) {
      this.droppedUnsafeAttributes += 1;
    }
    this.logger.log(
      JSON.stringify({kind: "event", name: safeName, provider})
    );
  }

  duration(
    name: string,
    milliseconds: number,
    attributes: TelemetryAttributes = {}
  ): void {
    if (name !== "http.server.request") {
      this.droppedUnsafeAttributes += 1;
      return;
    }
    const method = boundedMethod(attributes.method);
    const route = boundedRoute(attributes.route);
    const statusClass = boundedStatusClass(attributes.statusClass ?? attributes.statusCode);
    const failed = attributes.failed === true ? "true" : "false";
    const key = `${method}\0${route}\0${statusClass}\0${failed}`;
    const series = this.http.get(key) ?? {
      method,
      route,
      statusClass,
      failed,
      count: 0,
      durationCount: 0,
      durationSumMs: 0,
      buckets: LATENCY_BUCKETS_MS.map(() => 0)
    };
    const duration = finiteMilliseconds(milliseconds);
    series.count += 1;
    series.durationCount += 1;
    series.durationSumMs += duration;
    LATENCY_BUCKETS_MS.forEach((bound, index) => {
      if (duration <= bound) series.buckets[index] = (series.buckets[index] ?? 0) + 1;
    });
    this.http.set(key, series);
    if (
      Object.keys(attributes).some(
        (attribute) =>
          !new Set(["method", "route", "statusCode", "statusClass", "failed"]).has(
            attribute
          )
      )
    ) {
      this.droppedUnsafeAttributes += 1;
    }
  }

  renderPrometheus(): string {
    this.scrapeCount += 1;
    const lines = [
      "# HELP teachlab_http_server_requests_total Completed HTTP requests.",
      "# TYPE teachlab_http_server_requests_total counter"
    ];
    for (const series of [...this.http.values()].sort((left, right) =>
      `${left.method}\0${left.route}\0${left.statusClass}\0${left.failed}`.localeCompare(
        `${right.method}\0${right.route}\0${right.statusClass}\0${right.failed}`
      )
    )) {
      const labels = {
        method: series.method,
        route: series.route,
        status_class: series.statusClass,
        failed: series.failed
      };
      lines.push(
        `teachlab_http_server_requests_total${labelSet(labels)} ${series.count}`
      );
    }
    lines.push(
      "# HELP teachlab_http_server_request_duration_seconds HTTP request latency.",
      "# TYPE teachlab_http_server_request_duration_seconds histogram"
    );
    for (const series of [...this.http.values()].sort((left, right) =>
      `${left.method}\0${left.route}\0${left.statusClass}\0${left.failed}`.localeCompare(
        `${right.method}\0${right.route}\0${right.statusClass}\0${right.failed}`
      )
    )) {
      const labels = {
        method: series.method,
        route: series.route,
        status_class: series.statusClass,
        failed: series.failed
      };
      LATENCY_BUCKETS_MS.forEach((bound, index) => {
        lines.push(
          `teachlab_http_server_request_duration_seconds_bucket${labelSet({...labels, le: String(bound / 1_000)})} ${series.buckets[index] ?? 0}`
        );
      });
      lines.push(
        `teachlab_http_server_request_duration_seconds_bucket${labelSet({...labels, le: "+Inf"})} ${series.durationCount}`,
        `teachlab_http_server_request_duration_seconds_sum${labelSet(labels)} ${series.durationSumMs / 1_000}`,
        `teachlab_http_server_request_duration_seconds_count${labelSet(labels)} ${series.durationCount}`
      );
    }
    lines.push(
      "# HELP teachlab_task_events_total Low-cardinality task lifecycle events.",
      "# TYPE teachlab_task_events_total counter"
    );
    for (const [key, count] of [...this.events.entries()].sort()) {
      const [event, provider] = key.split("\0");
      lines.push(
        `teachlab_task_events_total${labelSet({event: event ?? "unknown", provider: provider ?? "none"})} ${count}`
      );
    }
    const memory = process.memoryUsage();
    const preparse = preparseOperationalSnapshot();
    const governor = this.resourceGovernor?.snapshot();
    const readiness = this.readiness?.operationalSnapshot();
    const workers = this.workers?.status();
    lines.push(
      "# HELP teachlab_preparse_active_requests Request bodies currently held before JSON parsing.",
      "# TYPE teachlab_preparse_active_requests gauge",
      `teachlab_preparse_active_requests ${preparse.globalActive}`,
      "# HELP teachlab_preparse_requests_total Bounded pre-parse request outcomes.",
      "# TYPE teachlab_preparse_requests_total counter",
      `teachlab_preparse_requests_total${labelSet({outcome: "accepted"})} ${preparse.accepted}`,
      `teachlab_preparse_requests_total${labelSet({outcome: "capacity_rejected"})} ${preparse.rejectedCapacity}`,
      `teachlab_preparse_requests_total${labelSet({outcome: "body_limit_rejected"})} ${preparse.rejectedBodyLimit}`,
      `teachlab_preparse_requests_total${labelSet({outcome: "completed"})} ${preparse.completed}`,
      `teachlab_preparse_requests_total${labelSet({outcome: "aborted_or_timeout"})} ${preparse.aborted}`,
      "# HELP teachlab_resource_governor_active_requests Post-auth request leases by bounded class.",
      "# TYPE teachlab_resource_governor_active_requests gauge",
      `teachlab_resource_governor_active_requests${labelSet({class: "anonymous"})} ${governor?.activeAnonymous ?? 0}`,
      `teachlab_resource_governor_active_requests${labelSet({class: "authenticated"})} ${governor?.activeAuthenticated ?? 0}`,
      `teachlab_resource_governor_active_requests${labelSet({class: "stream"})} ${governor?.activeStream ?? 0}`,
      "# HELP teachlab_resource_governor_rejections_total Post-auth capacity rejections without identity labels.",
      "# TYPE teachlab_resource_governor_rejections_total counter",
      `teachlab_resource_governor_rejections_total${labelSet({reason: "rate"})} ${governor?.rejectedRate ?? 0}`,
      `teachlab_resource_governor_rejections_total${labelSet({reason: "identity_concurrency"})} ${governor?.rejectedIdentityConcurrency ?? 0}`,
      `teachlab_resource_governor_rejections_total${labelSet({reason: "global_concurrency"})} ${governor?.rejectedGlobalConcurrency ?? 0}`,
      `teachlab_resource_governor_rejections_total${labelSet({reason: "bucket_capacity"})} ${governor?.rejectedCapacity ?? 0}`,
      "# HELP teachlab_harness_workers Harness worker pool states.",
      "# TYPE teachlab_harness_workers gauge",
      `teachlab_harness_workers${labelSet({state: "ready"})} ${workers?.readyWorkers ?? 0}`,
      `teachlab_harness_workers${labelSet({state: "starting"})} ${workers?.startingWorkers ?? 0}`,
      `teachlab_harness_workers${labelSet({state: "stopping"})} ${workers?.stoppingWorkers ?? 0}`,
      `teachlab_harness_workers${labelSet({state: "active_requests"})} ${workers?.activeRequests ?? 0}`,
      "# HELP teachlab_safeguarding_supervisor_pending Pending content-free safeguarding deliveries.",
      "# TYPE teachlab_safeguarding_supervisor_pending gauge",
      `teachlab_safeguarding_supervisor_pending ${workers?.safeguardingSupervisorPending ?? 0}`,
      "# HELP teachlab_safeguarding_supervisor_overdue Overdue content-free safeguarding deliveries.",
      "# TYPE teachlab_safeguarding_supervisor_overdue gauge",
      `teachlab_safeguarding_supervisor_overdue ${workers?.safeguardingSupervisorOverdue ?? 0}`,
      "# HELP teachlab_safeguarding_supervisor_oldest_pending_age_seconds Age of the oldest pending delivery.",
      "# TYPE teachlab_safeguarding_supervisor_oldest_pending_age_seconds gauge",
      `teachlab_safeguarding_supervisor_oldest_pending_age_seconds ${workers?.safeguardingSupervisorOldestPendingAgeSeconds ?? 0}`,
      "# HELP teachlab_safeguarding_supervisor_delivery_attempts_total Content-free delivery attempts.",
      "# TYPE teachlab_safeguarding_supervisor_delivery_attempts_total counter",
      `teachlab_safeguarding_supervisor_delivery_attempts_total ${workers?.safeguardingSupervisorAttemptedTotal ?? 0}`,
      "# HELP teachlab_safeguarding_supervisor_delivery_accepted_total Accepted content-free deliveries.",
      "# TYPE teachlab_safeguarding_supervisor_delivery_accepted_total counter",
      `teachlab_safeguarding_supervisor_delivery_accepted_total ${workers?.safeguardingSupervisorAcceptedTotal ?? 0}`,
      "# HELP teachlab_safeguarding_supervisor_delivery_failures_total Failed content-free delivery attempts.",
      "# TYPE teachlab_safeguarding_supervisor_delivery_failures_total counter",
      `teachlab_safeguarding_supervisor_delivery_failures_total ${workers?.safeguardingSupervisorFailedTotal ?? 0}`,
      "# HELP teachlab_safeguarding_supervisor_last_success_timestamp_seconds Last accepted delivery time, or zero before any success.",
      "# TYPE teachlab_safeguarding_supervisor_last_success_timestamp_seconds gauge",
      `teachlab_safeguarding_supervisor_last_success_timestamp_seconds ${workers?.safeguardingSupervisorLastSuccessAtUtc ? Date.parse(workers.safeguardingSupervisorLastSuccessAtUtc) / 1_000 : 0}`,
      "# HELP teachlab_safeguarding_receiver_readiness_checks_total Content-free receiver readiness checks.",
      "# TYPE teachlab_safeguarding_receiver_readiness_checks_total counter",
      `teachlab_safeguarding_receiver_readiness_checks_total${labelSet({result: "attempted"})} ${workers?.safeguardingSupervisorReceiverReadinessAttemptsTotal ?? 0}`,
      `teachlab_safeguarding_receiver_readiness_checks_total${labelSet({result: "ready"})} ${workers?.safeguardingSupervisorReceiverReadinessSuccessesTotal ?? 0}`,
      `teachlab_safeguarding_receiver_readiness_checks_total${labelSet({result: "failed"})} ${workers?.safeguardingSupervisorReceiverReadinessFailuresTotal ?? 0}`,
      "# HELP teachlab_safeguarding_receiver_last_success_timestamp_seconds Last authenticated receiver readiness success, or zero.",
      "# TYPE teachlab_safeguarding_receiver_last_success_timestamp_seconds gauge",
      `teachlab_safeguarding_receiver_last_success_timestamp_seconds ${workers?.safeguardingSupervisorLastReceiverSuccessAtUtc ? Date.parse(workers.safeguardingSupervisorLastReceiverSuccessAtUtc) / 1_000 : 0}`,
      "# HELP teachlab_safeguarding_retention_enabled Whether closed-case retention is enabled.",
      "# TYPE teachlab_safeguarding_retention_enabled gauge",
      `teachlab_safeguarding_retention_enabled ${workers?.safeguardingSupervisorRetentionEnabled ? 1 : 0}`,
      "# HELP teachlab_safeguarding_retention_eligible_cases Cases eligible in the latest bounded run.",
      "# TYPE teachlab_safeguarding_retention_eligible_cases gauge",
      `teachlab_safeguarding_retention_eligible_cases ${workers?.safeguardingSupervisorRetentionEligibleCases ?? 0}`,
      "# HELP teachlab_safeguarding_retention_cases_compacted_total Closed cases compacted by this supervisor process.",
      "# TYPE teachlab_safeguarding_retention_cases_compacted_total counter",
      `teachlab_safeguarding_retention_cases_compacted_total ${workers?.safeguardingSupervisorRetentionCasesCompactedTotal ?? 0}`,
      "# HELP teachlab_safeguarding_retention_events_compacted_total Case events removed by bounded compaction.",
      "# TYPE teachlab_safeguarding_retention_events_compacted_total counter",
      `teachlab_safeguarding_retention_events_compacted_total ${workers?.safeguardingSupervisorRetentionEventsCompactedTotal ?? 0}`,
      "# HELP teachlab_safeguarding_retention_failures Failed retention operations in the latest run.",
      "# TYPE teachlab_safeguarding_retention_failures gauge",
      `teachlab_safeguarding_retention_failures ${workers?.safeguardingSupervisorRetentionFailures ?? 0}`,
      "# HELP teachlab_safeguarding_retention_blocked_stores Stores blocked from retention in the latest run.",
      "# TYPE teachlab_safeguarding_retention_blocked_stores gauge",
      `teachlab_safeguarding_retention_blocked_stores ${workers?.safeguardingSupervisorRetentionBlockedStores ?? 0}`,
      "# HELP teachlab_safeguarding_capacity_near_limit_stores Stores still near a hard capacity limit.",
      "# TYPE teachlab_safeguarding_capacity_near_limit_stores gauge",
      `teachlab_safeguarding_capacity_near_limit_stores ${workers?.safeguardingSupervisorCapacityNearLimitStores ?? 0}`,
      "# HELP teachlab_safeguarding_capacity_events Aggregate stored safeguarding events.",
      "# TYPE teachlab_safeguarding_capacity_events gauge",
      `teachlab_safeguarding_capacity_events ${workers?.safeguardingSupervisorCapacityEvents ?? 0}`,
      "# HELP teachlab_safeguarding_capacity_event_limit Aggregate event capacity across scanned stores.",
      "# TYPE teachlab_safeguarding_capacity_event_limit gauge",
      `teachlab_safeguarding_capacity_event_limit ${workers?.safeguardingSupervisorCapacityEventLimit ?? 0}`,
      "# HELP teachlab_safeguarding_capacity_event_headroom_min Minimum event headroom across scanned stores, or zero when none are scanned.",
      "# TYPE teachlab_safeguarding_capacity_event_headroom_min gauge",
      `teachlab_safeguarding_capacity_event_headroom_min ${workers?.safeguardingSupervisorCapacityEventHeadroomMin ?? 0}`,
      "# HELP teachlab_safeguarding_capacity_store_bytes Aggregate safeguarding store bytes.",
      "# TYPE teachlab_safeguarding_capacity_store_bytes gauge",
      `teachlab_safeguarding_capacity_store_bytes ${workers?.safeguardingSupervisorCapacityStoreBytes ?? 0}`,
      "# HELP teachlab_safeguarding_capacity_store_byte_limit Aggregate byte capacity across scanned stores.",
      "# TYPE teachlab_safeguarding_capacity_store_byte_limit gauge",
      `teachlab_safeguarding_capacity_store_byte_limit ${workers?.safeguardingSupervisorCapacityStoreByteLimit ?? 0}`,
      "# HELP teachlab_safeguarding_capacity_store_byte_headroom_min Minimum byte headroom across scanned stores, or zero when none are scanned.",
      "# TYPE teachlab_safeguarding_capacity_store_byte_headroom_min gauge",
      `teachlab_safeguarding_capacity_store_byte_headroom_min ${workers?.safeguardingSupervisorCapacityStoreByteHeadroomMin ?? 0}`,
      "# HELP teachlab_safeguarding_capacity_recent_erasure_tombstones Aggregate recent erasure tombstones.",
      "# TYPE teachlab_safeguarding_capacity_recent_erasure_tombstones gauge",
      `teachlab_safeguarding_capacity_recent_erasure_tombstones ${workers?.safeguardingSupervisorCapacityRecentErasureTombstones ?? 0}`,
      "# HELP teachlab_safeguarding_capacity_recent_erasure_tombstone_limit Aggregate recent-erasure tombstone capacity.",
      "# TYPE teachlab_safeguarding_capacity_recent_erasure_tombstone_limit gauge",
      `teachlab_safeguarding_capacity_recent_erasure_tombstone_limit ${workers?.safeguardingSupervisorCapacityRecentErasureTombstoneLimit ?? 0}`,
      "# HELP teachlab_safeguarding_capacity_recent_erasure_tombstone_headroom_min Minimum recent-erasure tombstone headroom, or zero when none are scanned.",
      "# TYPE teachlab_safeguarding_capacity_recent_erasure_tombstone_headroom_min gauge",
      `teachlab_safeguarding_capacity_recent_erasure_tombstone_headroom_min ${workers?.safeguardingSupervisorCapacityRecentErasureTombstoneHeadroomMin ?? 0}`,
      "# HELP teachlab_safeguarding_erasure_fence_inserted_count Aggregate insertions into erasure fences.",
      "# TYPE teachlab_safeguarding_erasure_fence_inserted_count gauge",
      `teachlab_safeguarding_erasure_fence_inserted_count ${workers?.safeguardingSupervisorErasureFenceInsertedCount ?? 0}`,
      "# HELP teachlab_safeguarding_erasure_fence_estimated_false_positive_upper_bound Aggregate estimated false-positive upper bound.",
      "# TYPE teachlab_safeguarding_erasure_fence_estimated_false_positive_upper_bound gauge",
      `teachlab_safeguarding_erasure_fence_estimated_false_positive_upper_bound ${workers?.safeguardingSupervisorErasureFenceEstimatedFalsePositiveUpperBound ?? 0}`,
      "# HELP teachlab_safeguarding_erasure_fence_false_positive_within_target Whether the estimated bound remains within policy.",
      "# TYPE teachlab_safeguarding_erasure_fence_false_positive_within_target gauge",
      `teachlab_safeguarding_erasure_fence_false_positive_within_target ${workers?.safeguardingSupervisorErasureFenceFalsePositiveWithinTarget ? 1 : 0}`,
      "# HELP teachlab_harness_provider_readiness_checks_total Authenticated content-free provider readiness checks.",
      "# TYPE teachlab_harness_provider_readiness_checks_total counter",
      `teachlab_harness_provider_readiness_checks_total${labelSet({result: "attempted"})} ${workers?.providerReadinessAttempts ?? 0}`,
      `teachlab_harness_provider_readiness_checks_total${labelSet({result: "ready"})} ${workers?.providerReadinessSuccesses ?? 0}`,
      `teachlab_harness_provider_readiness_checks_total${labelSet({result: "failed"})} ${workers?.providerReadinessFailures ?? 0}`,
      "# HELP teachlab_harness_provider_readiness_last_result Last bounded provider readiness result.",
      "# TYPE teachlab_harness_provider_readiness_last_result gauge",
      ...(["not_required", "never", "ready", "failed"] as const).map(
        (result) =>
          `teachlab_harness_provider_readiness_last_result${labelSet({result})} ${workers?.providerReadinessLastResult === result ? 1 : 0}`
      ),
      "# HELP teachlab_readiness_checks_total Continuous dependency checks.",
      "# TYPE teachlab_readiness_checks_total counter",
      `teachlab_readiness_checks_total${labelSet({result: "ready"})} ${readiness?.successes ?? 0}`,
      `teachlab_readiness_checks_total${labelSet({result: "failed"})} ${readiness?.failures ?? 0}`,
      "# HELP teachlab_process_resident_memory_bytes Resident memory of the API process.",
      "# TYPE teachlab_process_resident_memory_bytes gauge",
      `teachlab_process_resident_memory_bytes ${memory.rss}`,
      "# HELP teachlab_process_heap_used_bytes V8 heap used by the API process.",
      "# TYPE teachlab_process_heap_used_bytes gauge",
      `teachlab_process_heap_used_bytes ${memory.heapUsed}`,
      "# HELP teachlab_process_uptime_seconds API process uptime.",
      "# TYPE teachlab_process_uptime_seconds gauge",
      `teachlab_process_uptime_seconds ${process.uptime()}`,
      "# HELP teachlab_telemetry_unsafe_attributes_dropped_total Attributes rejected from metrics labels.",
      "# TYPE teachlab_telemetry_unsafe_attributes_dropped_total counter",
      `teachlab_telemetry_unsafe_attributes_dropped_total ${this.droppedUnsafeAttributes}`,
      "# HELP teachlab_metrics_scrapes_total Prometheus scrapes served by this process.",
      "# TYPE teachlab_metrics_scrapes_total counter",
      `teachlab_metrics_scrapes_total ${this.scrapeCount}`,
      ""
    );
    return lines.join("\n");
  }

  safeSnapshot(): {
    httpSeries: number;
    taskSeries: number;
    unsafeAttributesDropped: number;
    scrapeCount: number;
    containsUserIdentifiers: false;
  } {
    return {
      httpSeries: this.http.size,
      taskSeries: this.events.size,
      unsafeAttributesDropped: this.droppedUnsafeAttributes,
      scrapeCount: this.scrapeCount,
      containsUserIdentifiers: false
    };
  }
}
