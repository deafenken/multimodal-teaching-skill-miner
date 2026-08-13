import {Injectable, Logger} from "@nestjs/common";

import type {TelemetryAttributes, TelemetryPort} from "./telemetry.port";

@Injectable()
export class LoggingTelemetryAdapter implements TelemetryPort {
  private readonly logger = new Logger("Telemetry");

  event(name: string, attributes: TelemetryAttributes = {}): void {
    this.logger.log(JSON.stringify({kind: "event", name, ...attributes}));
  }

  duration(
    name: string,
    milliseconds: number,
    attributes: TelemetryAttributes = {}
  ): void {
    this.logger.log(
      JSON.stringify({kind: "duration", name, milliseconds: Math.round(milliseconds), ...attributes})
    );
  }
}
