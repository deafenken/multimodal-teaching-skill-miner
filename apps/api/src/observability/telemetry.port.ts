export type TelemetryAttributes = Record<
  string,
  string | number | boolean | undefined
>;

export interface TelemetryPort {
  event(name: string, attributes?: TelemetryAttributes): void;
  duration(name: string, milliseconds: number, attributes?: TelemetryAttributes): void;
}
