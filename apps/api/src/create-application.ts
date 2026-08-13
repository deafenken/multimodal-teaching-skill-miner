import {ValidationPipe} from "@nestjs/common";
import {NestFactory} from "@nestjs/core";
import {
  FastifyAdapter,
  type NestFastifyApplication
} from "@nestjs/platform-fastify";

import {AppModule} from "./app.module";
import {AppConfigService} from "./config/app-config.service";
import {PostgresDatabase} from "./database/postgres-database";
import {HarnessWorkerPoolService} from "./harness/harness-worker-pool.service";
import {requestBodyLimitForUrl} from "./harness/harness-request-limits";
import {
  PreparseCapacityError,
  PreparseRequestGovernor,
  registerPreparseMetricsSource,
  RouteLimitedBodyStream,
  type PreparseLease,
} from "./operations/preparse-request-governor";

export interface CreateApplicationOptions {
  logger?: false | Array<"log" | "error" | "warn" | "debug" | "verbose" | "fatal">;
}

// The Python resource contract accepts a 17 MiB JSON envelope (base64 plus
// metadata). Fastify parses before the authenticated Harness controller, so a
// smaller global default would make that allowlisted route unreachable. The
// Python boundary still applies its narrower per-route limits.
export const APPS_API_BODY_LIMIT_BYTES = 18 * 1024 * 1024;

export function trustedProxyPolicy(environment = process.env.NODE_ENV): false | string[] {
  // The API has no published production port. Only a private edge peer may
  // supply forwarding metadata; Caddy overwrites it before proxying.
  return environment === "production"
    ? ["loopback", "linklocal", "uniquelocal"]
    : false;
}

export async function createApplication(
  options: CreateApplicationOptions = {}
): Promise<NestFastifyApplication> {
  const adapter = new FastifyAdapter({
    bodyLimit: APPS_API_BODY_LIMIT_BYTES,
    trustProxy: trustedProxyPolicy(),
    requestTimeout: 15_000,
    connectionTimeout: 10_000,
    keepAliveTimeout: 5_000,
  });
  // Retain a global hard cap plus a real-client bucket. Development has
  // trustProxy disabled and therefore uses the direct socket identity.
  const preparseGovernor = new PreparseRequestGovernor(32, 4);
  registerPreparseMetricsSource(preparseGovernor);
  const preparseLeases = new WeakMap<object, PreparseLease>();
  const releasePreparseLease = (request: object, outcome: "completed" | "aborted") => {
    preparseLeases.get(request)?.release(outcome);
    preparseLeases.delete(request);
  };
  adapter.getInstance().addHook("onRequest", (request, reply, done) => {
    const method = request.method.toUpperCase();
    if (!new Set(["POST", "PATCH", "DELETE"]).has(method)) {
      done();
      return;
    }
    const rawLength = request.headers["content-length"];
    const maximumBytes = requestBodyLimitForUrl(request.url);
    const byteLength = rawLength === undefined
      ? undefined
      : typeof rawLength === "string" && /^[0-9]+$/.test(rawLength)
        ? Number(rawLength)
        : Number.NaN;
    if (
      (byteLength !== undefined && !Number.isSafeInteger(byteLength))
      || (byteLength !== undefined && byteLength > maximumBytes)
    ) {
      preparseGovernor.recordBodyLimitExceeded();
      reply.status(413).send({
        statusCode: 413,
        code: "request_body_too_large",
        error: "Request body exceeds the route limit"
      });
      return;
    }
    try {
      const lease = preparseGovernor.acquire(request.ip || "unknown-peer");
      preparseLeases.set(request, lease);
      let finished = false;
      reply.raw.once("finish", () => {
        finished = true;
        releasePreparseLease(request, "completed");
      });
      reply.raw.once("close", () => releasePreparseLease(request, finished ? "completed" : "aborted"));
      reply.raw.once("error", () => releasePreparseLease(request, "aborted"));
    } catch (error) {
      if (!(error instanceof PreparseCapacityError)) throw error;
      reply
        .header("retry-after", "1")
        .status(429)
        .send({
          statusCode: 429,
          code: error.code,
          error: "Request body capacity is temporarily unavailable",
        });
      return;
    }
    done();
  });
  adapter.getInstance().addHook("preParsing", (request, _reply, payload, done) => {
    if (!preparseLeases.has(request)) {
      done(null, payload);
      return;
    }
    const limited = new RouteLimitedBodyStream(
      requestBodyLimitForUrl(request.url),
      () => preparseGovernor.recordBodyLimitExceeded(),
    );
    payload.pipe(limited);
    done(null, limited);
  });
  adapter.getInstance().addHook("onError", (request, _reply, _error, done) => {
    releasePreparseLease(request, "aborted");
    done();
  });
  const app = await NestFactory.create<NestFastifyApplication>(AppModule, adapter, {
    logger: options.logger
  });
  let config: AppConfigService;
  try {
    config = app.get(AppConfigService);
    config.assertSafeForStartup();
    await app.get(PostgresDatabase).assertReady();
    await app.get(HarnessWorkerPoolService).assertReady();
  } catch (error) {
    await app.close().catch(() => undefined);
    throw error;
  }

  app.useGlobalPipes(
    new ValidationPipe({
      transform: true,
      whitelist: true,
      forbidNonWhitelisted: true,
      stopAtFirstError: false
    })
  );
  app.enableCors({
    origin: config.corsOrigins,
    credentials: true,
    methods: ["GET", "HEAD", "POST", "PATCH", "DELETE", "OPTIONS"]
  });
  app.enableShutdownHooks();
  return app;
}
