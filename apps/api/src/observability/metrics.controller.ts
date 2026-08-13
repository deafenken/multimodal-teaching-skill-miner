import {
  Controller,
  Get,
  Headers,
  Inject,
  NotFoundException,
  Res,
  UnauthorizedException
} from "@nestjs/common";
import type {FastifyReply} from "fastify";

import {PublicRoute} from "../auth/public.decorator";
import {MetricsAccessService} from "./metrics-access.service";
import {PrometheusTelemetryAdapter} from "./prometheus-telemetry.adapter";
import {ContinuousReadinessService} from "../health/continuous-readiness.service";
import {HarnessWorkerPoolService} from "../harness/harness-worker-pool.service";
import {RESOURCE_GOVERNOR} from "../operations/resource-governance.interceptor";
import type {ResourceGovernor} from "../operations/resource-governor";


@Controller()
export class MetricsController {
  constructor(
    @Inject(MetricsAccessService) private readonly access: MetricsAccessService,
    @Inject(PrometheusTelemetryAdapter)
    private readonly telemetry: PrometheusTelemetryAdapter,
    @Inject(RESOURCE_GOVERNOR) resourceGovernor: ResourceGovernor,
    @Inject(ContinuousReadinessService) readiness: ContinuousReadinessService,
    @Inject(HarnessWorkerPoolService) workers: HarnessWorkerPoolService,
  ) {
    telemetry.bindOperationalSources({resourceGovernor, readiness, workers});
  }

  @Get("metrics")
  @PublicRoute()
  metrics(
    @Headers("authorization") authorization: string | string[] | undefined,
    @Res() reply: FastifyReply
  ): FastifyReply {
    if (!this.access.enabled) throw new NotFoundException("Resource not found");
    if (!this.access.authorize(authorization)) {
      reply.header("www-authenticate", 'Bearer realm="teachlab-metrics"');
      throw new UnauthorizedException("Metrics authentication is required");
    }
    return reply
      .header("cache-control", "no-store")
      .header("content-type", "text/plain; version=0.0.4; charset=utf-8")
      .send(this.telemetry.renderPrometheus());
  }
}
