import {timingSafeEqual} from "node:crypto";
import {Injectable} from "@nestjs/common";


export class MetricsAccessConfigurationError extends Error {
  constructor() {
    super("metrics access configuration is invalid");
    this.name = "MetricsAccessConfigurationError";
  }
}

function secureEqual(left: string, right: string): boolean {
  const leftBytes = Buffer.from(left, "utf8");
  const rightBytes = Buffer.from(right, "utf8");
  return leftBytes.length === rightBytes.length && timingSafeEqual(leftBytes, rightBytes);
}

@Injectable()
export class MetricsAccessService {
  private configuredToken?: string;

  configureToken(token: string | undefined): void {
    if (!token) {
      this.configuredToken = undefined;
      return;
    }
    if (
      token.length < 32
      || token.length > 512
      || /\s/.test(token)
      || !/^[A-Za-z0-9._~-]+$/.test(token)
    ) {
      throw new MetricsAccessConfigurationError();
    }
    this.configuredToken = token;
  }

  get enabled(): boolean {
    return this.configuredToken !== undefined;
  }

  authorize(header: string | string[] | undefined): boolean {
    if (!this.configuredToken) return false;
    const value = Array.isArray(header) ? header[0] : header;
    if (typeof value !== "string" || value.length > 520) return false;
    const match = /^Bearer ([A-Za-z0-9._~-]{32,512})$/.exec(value);
    return Boolean(match?.[1] && secureEqual(match[1], this.configuredToken));
  }
}
