import {createHash} from "node:crypto";
import {Readable} from "node:stream";

import {AccountDataRightsError} from "./account-data-rights.errors";
import {
  ACCOUNT_EXPORT_MANIFEST_SCHEMA,
  ACCOUNT_EXPORT_SCHEMA,
  type AccountExportArchive,
  type AccountExportEntry
} from "./account-data-rights.types";

export const ACCOUNT_EXPORT_MAX_DATA_BYTES = 384 * 1024 * 1024;
export const ACCOUNT_EXPORT_MAX_ARCHIVE_BYTES = 400 * 1024 * 1024;
export const ACCOUNT_EXPORT_MAX_ENTRIES = 20_000;
export const ACCOUNT_EXPORT_MAX_ENTRY_BYTES = 256 * 1024 * 1024;
const MAX_MANIFEST_BYTES = 16 * 1024 * 1024;
const UTF8_FLAG = 0x0800;
const ZIP_VERSION = 20;
const ZIP_DOS_DATE_1980_01_01 = 0x0021;
const UINT32_MAX = 0xffff_ffff;
const SAFE_ARCHIVE_PART = /^[A-Za-z0-9_.-]{1,200}$/;
const SHA256_PATTERN = /^[0-9a-f]{64}$/;

export interface AccountExportManifestEntry {
  path: string;
  bytes: number;
  sha256: string;
  media_type: string;
  data_class: string;
}

export interface AccountExportManifest {
  schema: typeof ACCOUNT_EXPORT_MANIFEST_SCHEMA;
  export_schema: typeof ACCOUNT_EXPORT_SCHEMA;
  exported_at: string;
  postgres_captured_at: string;
  worker_captured_at: string;
  entry_count: number;
  entries: AccountExportManifestEntry[];
  claim_boundary: {
    authenticated_scope_derived_server_side: true;
    contains_private_content: true;
    contains_raw_tenant_or_subject_claims: false;
    contains_session_or_csrf_secrets: false;
    contains_host_scope_paths: false;
  };
}

interface CentralRecord {
  name: Buffer;
  crc32: number;
  byteLength: number;
  localOffset: number;
}

function canonicalValue(value: unknown): string {
  if (value === null) return "null";
  if (typeof value === "string" || typeof value === "boolean") {
    return JSON.stringify(value);
  }
  if (typeof value === "number") {
    if (!Number.isFinite(value)) throw new Error("Non-finite canonical JSON number");
    return JSON.stringify(value);
  }
  if (Array.isArray(value)) return `[${value.map(canonicalValue).join(",")}]`;
  if (typeof value === "object") {
    const row = value as Record<string, unknown>;
    return `{${Object.keys(row)
      .sort()
      .map((key) => `${JSON.stringify(key)}:${canonicalValue(row[key])}`)
      .join(",")}}`;
  }
  throw new Error("Unsupported canonical JSON value");
}

export function canonicalAccountJson(value: unknown): Buffer {
  return Buffer.from(`${canonicalValue(value)}\n`, "utf8");
}

const CRC32_TABLE = (() => {
  const table = new Uint32Array(256);
  for (let value = 0; value < table.length; value += 1) {
    let current = value;
    for (let bit = 0; bit < 8; bit += 1) {
      current = (current & 1) !== 0
        ? 0xedb8_8320 ^ (current >>> 1)
        : current >>> 1;
    }
    table[value] = current >>> 0;
  }
  return table;
})();

export function crc32(content: Uint8Array, initial = 0): number {
  let current = (initial ^ 0xffff_ffff) >>> 0;
  for (const byte of content) {
    current = (CRC32_TABLE[(current ^ byte) & 0xff]! ^ (current >>> 8)) >>> 0;
  }
  return (current ^ 0xffff_ffff) >>> 0;
}

function safeArchivePath(path: string): string {
  if (
    !path
    || path.length > 1_024
    || path.startsWith("/")
    || path.endsWith("/")
    || path.includes("\\")
    || path.includes("\u0000")
  ) {
    throw new AccountDataRightsError(503, "account_export_unavailable");
  }
  const parts = path.split("/");
  if (
    !parts.length
    || parts.some((part) =>
      part === "." || part === ".." || !SAFE_ARCHIVE_PART.test(part)
    )
  ) {
    throw new AccountDataRightsError(503, "account_export_unavailable");
  }
  return path;
}

function assertEntry(entry: AccountExportEntry): void {
  safeArchivePath(entry.archivePath);
  if (
    !Number.isSafeInteger(entry.byteLength)
    || entry.byteLength < 0
    || entry.byteLength > ACCOUNT_EXPORT_MAX_ENTRY_BYTES
    || !SHA256_PATTERN.test(entry.sha256)
    || !Number.isInteger(entry.crc32)
    || entry.crc32 < 0
    || entry.crc32 > UINT32_MAX
    || !entry.mediaType
    || entry.mediaType.length > 200
    || !entry.dataClass
    || entry.dataClass.length > 200
  ) {
    throw new AccountDataRightsError(503, "account_export_unavailable");
  }
}

function localHeader(record: CentralRecord): Buffer {
  const header = Buffer.alloc(30);
  header.writeUInt32LE(0x0403_4b50, 0);
  header.writeUInt16LE(ZIP_VERSION, 4);
  header.writeUInt16LE(UTF8_FLAG, 6);
  header.writeUInt16LE(0, 8);
  header.writeUInt16LE(0, 10);
  header.writeUInt16LE(ZIP_DOS_DATE_1980_01_01, 12);
  header.writeUInt32LE(record.crc32, 14);
  header.writeUInt32LE(record.byteLength, 18);
  header.writeUInt32LE(record.byteLength, 22);
  header.writeUInt16LE(record.name.byteLength, 26);
  header.writeUInt16LE(0, 28);
  return Buffer.concat([header, record.name]);
}

function centralHeader(record: CentralRecord): Buffer {
  const header = Buffer.alloc(46);
  header.writeUInt32LE(0x0201_4b50, 0);
  header.writeUInt16LE(0x0314, 4);
  header.writeUInt16LE(ZIP_VERSION, 6);
  header.writeUInt16LE(UTF8_FLAG, 8);
  header.writeUInt16LE(0, 10);
  header.writeUInt16LE(0, 12);
  header.writeUInt16LE(ZIP_DOS_DATE_1980_01_01, 14);
  header.writeUInt32LE(record.crc32, 16);
  header.writeUInt32LE(record.byteLength, 20);
  header.writeUInt32LE(record.byteLength, 24);
  header.writeUInt16LE(record.name.byteLength, 28);
  header.writeUInt16LE(0, 30);
  header.writeUInt16LE(0, 32);
  header.writeUInt16LE(0, 34);
  header.writeUInt16LE(0, 36);
  header.writeUInt32LE(0o100600 * 0x1_0000, 38);
  header.writeUInt32LE(record.localOffset, 42);
  return Buffer.concat([header, record.name]);
}

function endOfCentralDirectory(
  entryCount: number,
  centralBytes: number,
  centralOffset: number
): Buffer {
  const end = Buffer.alloc(22);
  end.writeUInt32LE(0x0605_4b50, 0);
  end.writeUInt16LE(0, 4);
  end.writeUInt16LE(0, 6);
  end.writeUInt16LE(entryCount, 8);
  end.writeUInt16LE(entryCount, 10);
  end.writeUInt32LE(centralBytes, 12);
  end.writeUInt32LE(centralOffset, 16);
  end.writeUInt16LE(0, 20);
  return end;
}

export function bufferAccountExportEntry(input: {
  archivePath: string;
  content: Buffer | string;
  mediaType?: string;
  dataClass: string;
}): AccountExportEntry {
  const content = Buffer.isBuffer(input.content)
    ? Buffer.from(input.content)
    : Buffer.from(input.content, "utf8");
  const archivePath = safeArchivePath(input.archivePath);
  if (content.byteLength > ACCOUNT_EXPORT_MAX_ENTRY_BYTES) {
    throw new AccountDataRightsError(413, "account_export_limit_exceeded");
  }
  return {
    archivePath,
    byteLength: content.byteLength,
    sha256: createHash("sha256").update(content).digest("hex"),
    crc32: crc32(content),
    mediaType: input.mediaType ?? "application/json",
    dataClass: input.dataClass,
    open: () => Readable.from([content])
  };
}

function totalArchiveBytes(entries: readonly AccountExportEntry[]): number {
  let local = 0;
  let central = 0;
  for (const entry of entries) {
    const nameBytes = Buffer.byteLength(entry.archivePath, "utf8");
    local += 30 + nameBytes + entry.byteLength;
    central += 46 + nameBytes;
  }
  return local + central + 22;
}

export function buildAccountExportArchive(input: {
  entries: readonly AccountExportEntry[];
  exportedAt: string;
  postgresCapturedAt: string;
  workerCapturedAt: string;
  signal?: AbortSignal;
  onFinally?: () => Promise<void>;
}): AccountExportArchive {
  if (!Number.isFinite(Date.parse(input.exportedAt))) {
    throw new AccountDataRightsError(503, "account_export_unavailable");
  }
  const sorted = [...input.entries].sort((left, right) =>
    left.archivePath.localeCompare(right.archivePath)
  );
  if (sorted.length > ACCOUNT_EXPORT_MAX_ENTRIES) {
    throw new AccountDataRightsError(413, "account_export_limit_exceeded");
  }
  for (const entry of sorted) assertEntry(entry);
  if (new Set(sorted.map((entry) => entry.archivePath)).size !== sorted.length) {
    throw new AccountDataRightsError(503, "account_export_unavailable");
  }
  const dataBytes = sorted.reduce((sum, entry) => sum + entry.byteLength, 0);
  if (!Number.isSafeInteger(dataBytes) || dataBytes > ACCOUNT_EXPORT_MAX_DATA_BYTES) {
    throw new AccountDataRightsError(413, "account_export_limit_exceeded");
  }

  const manifest: AccountExportManifest = {
    schema: ACCOUNT_EXPORT_MANIFEST_SCHEMA,
    export_schema: ACCOUNT_EXPORT_SCHEMA,
    exported_at: input.exportedAt,
    postgres_captured_at: input.postgresCapturedAt,
    worker_captured_at: input.workerCapturedAt,
    entry_count: sorted.length,
    entries: sorted.map((entry) => ({
      path: entry.archivePath,
      bytes: entry.byteLength,
      sha256: entry.sha256,
      media_type: entry.mediaType,
      data_class: entry.dataClass
    })),
    claim_boundary: {
      authenticated_scope_derived_server_side: true,
      contains_private_content: true,
      contains_raw_tenant_or_subject_claims: false,
      contains_session_or_csrf_secrets: false,
      contains_host_scope_paths: false
    }
  };
  const manifestBytes = canonicalAccountJson(manifest);
  if (manifestBytes.byteLength > MAX_MANIFEST_BYTES) {
    throw new AccountDataRightsError(413, "account_export_limit_exceeded");
  }
  const manifestSha256 = createHash("sha256").update(manifestBytes).digest("hex");
  const archiveEntries = [
    ...sorted,
    bufferAccountExportEntry({
      archivePath: "manifest.json",
      content: manifestBytes,
      dataClass: "canonical_export_manifest"
    }),
    bufferAccountExportEntry({
      archivePath: "manifest.sha256",
      content: `${manifestSha256}  manifest.json\n`,
      mediaType: "text/plain",
      dataClass: "manifest_integrity_digest"
    })
  ];
  const byteLength = totalArchiveBytes(archiveEntries);
  if (byteLength > ACCOUNT_EXPORT_MAX_ARCHIVE_BYTES || byteLength > UINT32_MAX) {
    throw new AccountDataRightsError(413, "account_export_limit_exceeded");
  }

  const generate = async function* (): AsyncGenerator<Buffer> {
    const central: CentralRecord[] = [];
    let offset = 0;
    try {
      for (const entry of archiveEntries) {
        if (input.signal?.aborted) throw new Error("account export aborted");
        const record: CentralRecord = {
          name: Buffer.from(entry.archivePath, "utf8"),
          crc32: entry.crc32,
          byteLength: entry.byteLength,
          localOffset: offset
        };
        const header = localHeader(record);
        central.push(record);
        offset += header.byteLength;
        yield header;

        const digest = createHash("sha256");
        let observedCrc = 0;
        let observedBytes = 0;
        const source = entry.open();
        try {
          for await (const rawChunk of source) {
            if (input.signal?.aborted) throw new Error("account export aborted");
            const chunk = Buffer.isBuffer(rawChunk)
              ? rawChunk
              : Buffer.from(rawChunk as Uint8Array);
            observedBytes += chunk.byteLength;
            if (observedBytes > entry.byteLength) {
              throw new AccountDataRightsError(
                503,
                "account_export_changed_during_stream"
              );
            }
            digest.update(chunk);
            observedCrc = crc32(chunk, observedCrc);
            offset += chunk.byteLength;
            yield chunk;
          }
        } finally {
          source.destroy();
        }
        if (
          observedBytes !== entry.byteLength
          || observedCrc !== entry.crc32
          || digest.digest("hex") !== entry.sha256
        ) {
          throw new AccountDataRightsError(
            503,
            "account_export_changed_during_stream"
          );
        }
      }

      const centralOffset = offset;
      for (const record of central) {
        const header = centralHeader(record);
        offset += header.byteLength;
        yield header;
      }
      const centralBytes = offset - centralOffset;
      yield endOfCentralDirectory(central.length, centralBytes, centralOffset);
    } finally {
      await input.onFinally?.();
    }
  };

  return {
    filename: "teachlab-account-private-export.zip",
    stream: Readable.from(generate()),
    byteLength,
    manifestSha256,
    entryCount: sorted.length
  };
}

interface ParsedCentralEntry {
  path: string;
  crc32: number;
  byteLength: number;
  localOffset: number;
}

/** Strict validator used by tests, restore drills, and offline integrity checks. */
export function validateAccountExportArchive(payload: Buffer): AccountExportManifest {
  if (!payload.byteLength || payload.byteLength > ACCOUNT_EXPORT_MAX_ARCHIVE_BYTES) {
    throw new AccountDataRightsError(413, "account_export_limit_exceeded");
  }
  if (payload.byteLength < 22 || payload.readUInt32LE(payload.byteLength - 22) !== 0x0605_4b50) {
    throw new AccountDataRightsError(503, "account_export_unavailable");
  }
  const end = payload.byteLength - 22;
  const count = payload.readUInt16LE(end + 10);
  const centralBytes = payload.readUInt32LE(end + 12);
  const centralOffset = payload.readUInt32LE(end + 16);
  if (
    count > ACCOUNT_EXPORT_MAX_ENTRIES + 2
    || centralOffset + centralBytes !== end
    || payload.readUInt16LE(end + 20) !== 0
  ) {
    throw new AccountDataRightsError(503, "account_export_unavailable");
  }

  const central: ParsedCentralEntry[] = [];
  let cursor = centralOffset;
  for (let index = 0; index < count; index += 1) {
    if (cursor + 46 > end || payload.readUInt32LE(cursor) !== 0x0201_4b50) {
      throw new AccountDataRightsError(503, "account_export_unavailable");
    }
    const flags = payload.readUInt16LE(cursor + 8);
    const method = payload.readUInt16LE(cursor + 10);
    const compressed = payload.readUInt32LE(cursor + 20);
    const uncompressed = payload.readUInt32LE(cursor + 24);
    const nameLength = payload.readUInt16LE(cursor + 28);
    const extraLength = payload.readUInt16LE(cursor + 30);
    const commentLength = payload.readUInt16LE(cursor + 32);
    const localOffset = payload.readUInt32LE(cursor + 42);
    const next = cursor + 46 + nameLength + extraLength + commentLength;
    if (
      flags !== UTF8_FLAG
      || method !== 0
      || compressed !== uncompressed
      || uncompressed > ACCOUNT_EXPORT_MAX_ENTRY_BYTES
      || extraLength !== 0
      || commentLength !== 0
      || next > end
    ) {
      throw new AccountDataRightsError(503, "account_export_unavailable");
    }
    const path = payload.subarray(cursor + 46, cursor + 46 + nameLength).toString("utf8");
    safeArchivePath(path);
    central.push({
      path,
      crc32: payload.readUInt32LE(cursor + 16),
      byteLength: uncompressed,
      localOffset
    });
    cursor = next;
  }
  if (cursor !== end || new Set(central.map((entry) => entry.path)).size !== count) {
    throw new AccountDataRightsError(503, "account_export_unavailable");
  }

  const content = new Map<string, Buffer>();
  let totalDataBytes = 0;
  let expectedOffset = 0;
  for (const entry of [...central].sort((left, right) => left.localOffset - right.localOffset)) {
    const offset = entry.localOffset;
    if (
      offset !== expectedOffset
      || offset + 30 > centralOffset
      || payload.readUInt32LE(offset) !== 0x0403_4b50
      || payload.readUInt16LE(offset + 6) !== UTF8_FLAG
      || payload.readUInt16LE(offset + 8) !== 0
    ) {
      throw new AccountDataRightsError(503, "account_export_unavailable");
    }
    const nameLength = payload.readUInt16LE(offset + 26);
    const extraLength = payload.readUInt16LE(offset + 28);
    const dataStart = offset + 30 + nameLength + extraLength;
    const dataEnd = dataStart + entry.byteLength;
    const localName = payload.subarray(offset + 30, offset + 30 + nameLength).toString("utf8");
    if (
      extraLength !== 0
      || localName !== entry.path
      || payload.readUInt32LE(offset + 14) !== entry.crc32
      || payload.readUInt32LE(offset + 18) !== entry.byteLength
      || payload.readUInt32LE(offset + 22) !== entry.byteLength
      || dataEnd > centralOffset
    ) {
      throw new AccountDataRightsError(503, "account_export_unavailable");
    }
    const bytes = payload.subarray(dataStart, dataEnd);
    if (crc32(bytes) !== entry.crc32) {
      throw new AccountDataRightsError(503, "account_export_unavailable");
    }
    totalDataBytes += bytes.byteLength;
    if (totalDataBytes > ACCOUNT_EXPORT_MAX_DATA_BYTES + MAX_MANIFEST_BYTES + 256) {
      throw new AccountDataRightsError(413, "account_export_limit_exceeded");
    }
    content.set(entry.path, bytes);
    expectedOffset = dataEnd;
  }
  if (expectedOffset !== centralOffset) {
    throw new AccountDataRightsError(503, "account_export_unavailable");
  }

  const rawManifest = content.get("manifest.json");
  const manifestDigest = content.get("manifest.sha256")?.toString("ascii");
  if (!rawManifest || rawManifest.byteLength > MAX_MANIFEST_BYTES || !manifestDigest) {
    throw new AccountDataRightsError(503, "account_export_unavailable");
  }
  let manifest: unknown;
  try {
    manifest = JSON.parse(rawManifest.toString("utf8"));
  } catch {
    throw new AccountDataRightsError(503, "account_export_unavailable");
  }
  if (
    !manifest
    || typeof manifest !== "object"
    || Array.isArray(manifest)
    || canonicalAccountJson(manifest).compare(rawManifest) !== 0
  ) {
    throw new AccountDataRightsError(503, "account_export_unavailable");
  }
  const row = manifest as unknown as AccountExportManifest;
  const digest = createHash("sha256").update(rawManifest).digest("hex");
  if (
    row.schema !== ACCOUNT_EXPORT_MANIFEST_SCHEMA
    || row.export_schema !== ACCOUNT_EXPORT_SCHEMA
    || manifestDigest !== `${digest}  manifest.json\n`
    || !Array.isArray(row.entries)
    || row.entry_count !== row.entries.length
    || row.entries.length !== count - 2
  ) {
    throw new AccountDataRightsError(503, "account_export_unavailable");
  }
  const declared = new Set<string>();
  for (const declaredEntry of row.entries) {
    if (
      !declaredEntry
      || typeof declaredEntry !== "object"
      || typeof declaredEntry.path !== "string"
      || declared.has(declaredEntry.path)
      || !Number.isSafeInteger(declaredEntry.bytes)
      || !SHA256_PATTERN.test(declaredEntry.sha256)
    ) {
      throw new AccountDataRightsError(503, "account_export_unavailable");
    }
    declared.add(declaredEntry.path);
    const bytes = content.get(declaredEntry.path);
    if (
      !bytes
      || bytes.byteLength !== declaredEntry.bytes
      || createHash("sha256").update(bytes).digest("hex") !== declaredEntry.sha256
    ) {
      throw new AccountDataRightsError(503, "account_export_unavailable");
    }
  }
  if (
    [...content.keys()].some((path) =>
      path !== "manifest.json" && path !== "manifest.sha256" && !declared.has(path)
    )
  ) {
    throw new AccountDataRightsError(503, "account_export_unavailable");
  }
  return row;
}
