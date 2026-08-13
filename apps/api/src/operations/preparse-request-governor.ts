import {createHash} from "node:crypto";
import {Transform, type TransformCallback} from "node:stream";

export class PreparseCapacityError extends Error {
  readonly statusCode = 429;
  readonly code = "preparse_capacity_exhausted";

  constructor() {
    super("Request body capacity is temporarily unavailable");
    this.name = "PreparseCapacityError";
  }
}

export class PreparseBodyLimitError extends Error {
  readonly statusCode = 413;
  readonly code = "FST_ERR_CTP_BODY_TOO_LARGE";

  constructor() {
    super("Request body exceeds the route limit");
    this.name = "PreparseBodyLimitError";
  }
}

export interface PreparseLease {
  release(outcome?: "completed" | "aborted"): void;
}

export interface PreparseOperationalSnapshot {
  globalActive: number;
  peerBuckets: number;
  accepted: number;
  rejectedCapacity: number;
  rejectedBodyLimit: number;
  completed: number;
  aborted: number;
}

let activeMetricsSource: PreparseRequestGovernor | undefined;

export function registerPreparseMetricsSource(source: PreparseRequestGovernor): void {
  activeMetricsSource = source;
}

export function preparseOperationalSnapshot(): PreparseOperationalSnapshot {
  return activeMetricsSource?.snapshot() ?? {
    globalActive: 0,
    peerBuckets: 0,
    accepted: 0,
    rejectedCapacity: 0,
    rejectedBodyLimit: 0,
    completed: 0,
    aborted: 0,
  };
}

/**
 * Bounds bodies before authentication and JSON parsing. Peer identities are
 * hashed immediately and are never exported as telemetry labels.
 */
export class PreparseRequestGovernor {
  private globalActive = 0;
  private readonly peers = new Map<string, number>();
  private accepted = 0;
  private rejectedCapacity = 0;
  private rejectedBodyLimit = 0;
  private completed = 0;
  private aborted = 0;

  constructor(
    private readonly maximumGlobal = 32,
    private readonly maximumPerPeer = 4,
  ) {
    if (
      !Number.isInteger(maximumGlobal)
      || !Number.isInteger(maximumPerPeer)
      || maximumGlobal < 1
      || maximumGlobal > 1_024
      || maximumPerPeer < 1
      || maximumPerPeer > maximumGlobal
    ) {
      throw new Error("Pre-parse request capacity is invalid");
    }
  }

  acquire(peer: string): PreparseLease {
    const key = createHash("sha256")
      .update("teachlab-preparse-peer-v1\0", "utf8")
      .update(peer.slice(0, 512), "utf8")
      .digest("hex");
    const active = this.peers.get(key) ?? 0;
    if (this.globalActive >= this.maximumGlobal || active >= this.maximumPerPeer) {
      this.rejectedCapacity += 1;
      throw new PreparseCapacityError();
    }
    this.accepted += 1;
    this.globalActive += 1;
    this.peers.set(key, active + 1);
    let released = false;
    return {
      release: (outcome = "completed") => {
        if (released) return;
        released = true;
        if (outcome === "completed") this.completed += 1;
        else this.aborted += 1;
        this.globalActive = Math.max(0, this.globalActive - 1);
        const remaining = Math.max(0, (this.peers.get(key) ?? 1) - 1);
        if (remaining === 0) this.peers.delete(key);
        else this.peers.set(key, remaining);
      },
    };
  }

  recordBodyLimitExceeded(): void {
    this.rejectedBodyLimit += 1;
  }

  snapshot(): PreparseOperationalSnapshot {
    return {
      globalActive: this.globalActive,
      peerBuckets: this.peers.size,
      accepted: this.accepted,
      rejectedCapacity: this.rejectedCapacity,
      rejectedBodyLimit: this.rejectedBodyLimit,
      completed: this.completed,
      aborted: this.aborted,
    };
  }
}

export class RouteLimitedBodyStream extends Transform {
  receivedEncodedLength = 0;

  constructor(
    private readonly maximumBytes: number,
    private readonly onLimitExceeded: () => void = () => undefined,
  ) {
    super();
    if (!Number.isSafeInteger(maximumBytes) || maximumBytes < 1) {
      throw new Error("Route body limit is invalid");
    }
  }

  override _transform(
    chunk: Buffer | string,
    encoding: BufferEncoding,
    callback: TransformCallback,
  ): void {
    const bytes = Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk, encoding);
    this.receivedEncodedLength += bytes.byteLength;
    if (this.receivedEncodedLength > this.maximumBytes) {
      this.onLimitExceeded();
      callback(new PreparseBodyLimitError());
      return;
    }
    callback(null, bytes);
  }
}
