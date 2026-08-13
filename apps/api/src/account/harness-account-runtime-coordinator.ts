import {createHash, randomBytes} from "node:crypto";
import {createReadStream} from "node:fs";
import {lstat, mkdir, open, readFile, readdir, rename, rm} from "node:fs/promises";
import {basename, dirname, join, relative, resolve, sep} from "node:path";

import {Inject, Injectable, Optional} from "@nestjs/common";

import {AppConfigService} from "../config/app-config.service";
import {
  ACCOUNT_EXPORT_MAX_DATA_BYTES,
  ACCOUNT_EXPORT_MAX_ENTRIES,
  ACCOUNT_EXPORT_MAX_ENTRY_BYTES,
  crc32
} from "./account-export-archive";
import {AccountDataRightsError} from "./account-data-rights.errors";
import type {AccountScopeRuntimeCoordinatorPort} from "./account-data-rights.repository.port";
import type {
  AccountExportEntry,
  AccountRuntimeDrainResult,
  AccountScopeHashes,
  AccountWorkerExportLease,
  AccountWorkerPurgeResult
} from "./account-data-rights.types";
import {HarnessWorkerPoolService} from "../harness/harness-worker-pool.service";
import {scopeMigrationSecurityMetadataNames} from "../harness/harness-scope-key-migration";
import type {AccessScope} from "../tenancy/access-scope";
import {TASK_ORCHESTRATOR} from "../platform/tokens";
import type {TaskOrchestratorPort} from "../tasks/task-orchestrator.port";

const QUARANTINE_DIRECTORY = ".account-deletion-quarantine-v1";
const MAX_EXPORT_FILES = 50_000;
const ACCOUNT_EXPORT_SECURITY_METADATA_PATHS = new Set([
  "syllabi/.curriculum_signing_keyring.json",
  "syllabi/..curriculum_signing_keyring.json.lock"
]);
const CURRICULUM_KEYRING_TEMP_PATH =
  /^syllabi\/\.\.curriculum_signing_keyring\.json\.[A-Za-z0-9_.-]{1,160}\.tmp$/;
const PURGE_RECEIPT_SCHEMA = "teachlab.account_worker_purge_receipt.v1";

function inside(parent: string, child: string): boolean {
  const path = relative(resolve(parent), resolve(child));
  return Boolean(path) && path !== ".." && !path.startsWith(`..${sep}`) && !path.startsWith(sep);
}

function safePart(value: string): string {
  if (!/^[A-Za-z0-9_.-]{1,200}$/.test(value) || value === "." || value === "..") {
    throw new AccountDataRightsError(503, "account_export_unavailable");
  }
  return value;
}

async function syncDirectory(path: string): Promise<void> {
  const handle = await open(path, "r");
  try {
    await handle.sync();
  } finally {
    await handle.close();
  }
}

async function walkFiles(
  root: string,
  options: {excludeScopeSecurityMetadata?: boolean} = {}
): Promise<readonly string[]> {
  const output: string[] = [];
  const visit = async (directory: string): Promise<void> => {
    const entries = await readdir(directory, {withFileTypes: true});
    entries.sort((left, right) => left.name.localeCompare(right.name));
    for (const entry of entries) {
      safePart(entry.name);
      if (
        directory === root
        && options.excludeScopeSecurityMetadata
        && scopeMigrationSecurityMetadataNames.has(entry.name)
      ) continue;
      const path = join(directory, entry.name);
      const relativePath = relative(root, path).split(sep).join("/");
      if (
        options.excludeScopeSecurityMetadata
        && (
          ACCOUNT_EXPORT_SECURITY_METADATA_PATHS.has(relativePath)
          || CURRICULUM_KEYRING_TEMP_PATH.test(relativePath)
        )
      ) continue;
      const metadata = await lstat(path);
      if (metadata.isSymbolicLink()) {
        throw new AccountDataRightsError(503, "account_export_unavailable");
      }
      if (metadata.isDirectory()) await visit(path);
      else if (metadata.isFile()) output.push(path);
      else throw new AccountDataRightsError(503, "account_export_unavailable");
      if (output.length > MAX_EXPORT_FILES) {
        throw new AccountDataRightsError(413, "account_export_limit_exceeded");
      }
    }
  };
  await visit(root);
  return output;
}

/**
 * Count an already fenced and atomically quarantined tree without following
 * links. Export limits must not become a reason that a large account can never
 * complete permanent deletion. The exact quarantine path is service-created,
 * private, and removed only after this traversal succeeds.
 */
async function countQuarantinedTree(
  root: string
): Promise<{files: number; bytes: number}> {
  let files = 0;
  let bytes = 0;
  const pending = [root];
  while (pending.length) {
    const directory = pending.pop()!;
    const entries = await readdir(directory, {withFileTypes: true});
    for (const entry of entries) {
      safePart(entry.name);
      const path = join(directory, entry.name);
      const metadata = await lstat(path);
      if (metadata.isDirectory() && !metadata.isSymbolicLink()) {
        pending.push(path);
        continue;
      }
      files += 1;
      bytes += metadata.size;
      if (!Number.isSafeInteger(files) || !Number.isSafeInteger(bytes)) {
        throw new Error("Account worker purge count exceeded safe integer range");
      }
    }
  }
  return {files, bytes};
}

function purgeReceipt(value: unknown, operationDigest: string): AccountWorkerPurgeResult {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    throw new Error("Invalid account worker purge receipt");
  }
  const row = value as Record<string, unknown>;
  const exactKeys = [
    "schema", "operation_id_sha256", "worker_files", "worker_bytes", "worker_roots"
  ];
  const count = (candidate: unknown) =>
    Number.isSafeInteger(candidate) && Number(candidate) >= 0;
  if (
    Object.keys(row).sort().join("\0") !== [...exactKeys].sort().join("\0")
    || row.schema !== PURGE_RECEIPT_SCHEMA
    || row.operation_id_sha256 !== operationDigest
    || !count(row.worker_files)
    || !count(row.worker_bytes)
    || !count(row.worker_roots)
  ) throw new Error("Invalid account worker purge receipt");
  return {
    workerFiles: Number(row.worker_files),
    workerBytes: Number(row.worker_bytes),
    workerRoots: Number(row.worker_roots)
  };
}

async function readPurgeReceipt(
  path: string,
  operationDigest: string
): Promise<AccountWorkerPurgeResult | undefined> {
  try {
    const metadata = await lstat(path);
    if (!metadata.isFile() || metadata.isSymbolicLink() || metadata.size > 2_048) {
      throw new Error("Unsafe account worker purge receipt");
    }
    return purgeReceipt(JSON.parse(await readFile(path, "utf8")), operationDigest);
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code === "ENOENT") return undefined;
    throw error;
  }
}

async function persistPurgeReceipt(
  path: string,
  operationDigest: string,
  counts: AccountWorkerPurgeResult
): Promise<void> {
  const directory = dirname(path);
  const temporary = join(
    directory,
    `.${basename(path)}.${process.pid}.${randomBytes(8).toString("hex")}.tmp`
  );
  const content = `${JSON.stringify({
    schema: PURGE_RECEIPT_SCHEMA,
    operation_id_sha256: operationDigest,
    worker_files: counts.workerFiles,
    worker_bytes: counts.workerBytes,
    worker_roots: counts.workerRoots
  })}\n`;
  const handle = await open(temporary, "wx", 0o600);
  try {
    await handle.writeFile(content, "utf8");
    await handle.sync();
  } finally {
    await handle.close();
  }
  try {
    await rename(temporary, path);
    await syncDirectory(directory);
  } catch (error) {
    await rm(temporary, {force: true}).catch(() => undefined);
    throw error;
  }
}

async function fileEntry(
  root: string,
  path: string,
  rootIndex: number
): Promise<AccountExportEntry> {
  const archiveRelative = relative(root, path).split(sep).map(safePart).join("/");
  const metadata = await lstat(path);
  if (
    !metadata.isFile()
    || metadata.isSymbolicLink()
    || !Number.isSafeInteger(metadata.size)
    || metadata.size > ACCOUNT_EXPORT_MAX_ENTRY_BYTES
  ) throw new AccountDataRightsError(413, "account_export_limit_exceeded");
  const hasher = createHash("sha256");
  let checksum = 0;
  let bytes = 0;
  for await (const chunk of createReadStream(path, {flags: "r"})) {
    const value = Buffer.from(chunk as Buffer);
    bytes += value.byteLength;
    if (bytes > ACCOUNT_EXPORT_MAX_ENTRY_BYTES) {
      throw new AccountDataRightsError(413, "account_export_limit_exceeded");
    }
    hasher.update(value);
    checksum = crc32(value, checksum);
  }
  if (bytes !== metadata.size) {
    throw new AccountDataRightsError(503, "account_export_changed_during_stream");
  }
  return {
    archivePath: `scope-root-${String(rootIndex + 1).padStart(2, "0")}/${archiveRelative}`,
    byteLength: bytes,
    sha256: hasher.digest("hex"),
    crc32: checksum,
    mediaType: path.endsWith(".json") ? "application/json" : "application/octet-stream",
    dataClass: "harness_scope_private",
    open: () => createReadStream(path, {flags: "r"})
  };
}

@Injectable()
export class HarnessAccountRuntimeCoordinator
  implements AccountScopeRuntimeCoordinatorPort {
  constructor(
    @Inject(HarnessWorkerPoolService) private readonly workers: HarnessWorkerPoolService,
    @Inject(AppConfigService) private readonly config: AppConfigService,
    @Optional() @Inject(TASK_ORCHESTRATOR) private readonly tasks?: TaskOrchestratorPort
  ) {}

  async exportWorkerData(
    scope: AccessScope,
    _hashes: AccountScopeHashes,
    signal?: AbortSignal
  ): Promise<AccountWorkerExportLease> {
    const lease = await this.workers.beginAccountExport(scope);
    try {
      const entries: AccountExportEntry[] = [];
      let total = 0;
      for (const [rootIndex, value] of lease.roots.entries()) {
        if (signal?.aborted) throw new AccountDataRightsError(503, "account_export_unavailable");
        const files = await walkFiles(value.privateRoot, {
          excludeScopeSecurityMetadata: true
        });
        for (const path of files) {
          const entry = await fileEntry(value.privateRoot, path, rootIndex);
          total += entry.byteLength;
          if (entries.length >= ACCOUNT_EXPORT_MAX_ENTRIES || total > ACCOUNT_EXPORT_MAX_DATA_BYTES) {
            throw new AccountDataRightsError(413, "account_export_limit_exceeded");
          }
          entries.push(entry);
        }
      }
      return {capturedAt: new Date().toISOString(), entries, release: lease.release};
    } catch (error) {
      await lease.release();
      throw error;
    }
  }

  async fence(
    scope: AccessScope,
    _hashes: AccountScopeHashes,
    operationId: string
  ): Promise<void> {
    await this.workers.fenceAccountScope(scope, operationId);
  }

  async drain(
    scope: AccessScope,
    _hashes: AccountScopeHashes,
    operationId: string,
    timeoutMs: number
  ): Promise<AccountRuntimeDrainResult> {
    const result = await this.workers.drainAccountScope(scope, operationId, timeoutMs);
    return {
      apiTasksCancelled: 0,
      apiTasksHandedOff: 0,
      workerTasksCancelled: result.activeRequestsCancelled,
      workerTasksHandedOff: 0
    };
  }

  async quarantine(
    scope: AccessScope,
    _hashes: AccountScopeHashes,
    operationId: string
  ): Promise<void> {
    const root = this.config.harnessWorkerRoot;
    if (!root || !/^adel_[0-9a-f]{32}$/.test(operationId)) {
      throw new Error("Account worker quarantine is unavailable");
    }
    const destinationRoot = this.quarantineRoot(root, operationId);
    await mkdir(destinationRoot, {recursive: true, mode: 0o700});
    const roots = await this.workers.accountRoots(scope);
    for (const value of roots) {
      if (!inside(root, value.privateRoot)) throw new Error("Unsafe worker root");
      const destination = join(
        destinationRoot,
        `${safePart(value.keyVersion)}-${safePart(value.scopeId)}`
      );
      try {
        const destinationMetadata = await lstat(destination);
        if (destinationMetadata.isSymbolicLink() || !destinationMetadata.isDirectory()) {
          throw new Error("Unsafe worker quarantine");
        }
        continue;
      } catch (error) {
        if ((error as NodeJS.ErrnoException).code !== "ENOENT") throw error;
      }
      await rename(value.privateRoot, destination);
      await syncDirectory(dirname(value.privateRoot));
      await syncDirectory(destinationRoot);
    }
  }

  async purgeQuarantine(
    _scope: AccessScope,
    _hashes: AccountScopeHashes,
    operationId: string
  ): Promise<AccountWorkerPurgeResult> {
    const root = this.config.harnessWorkerRoot;
    if (!root || !/^adel_[0-9a-f]{32}$/.test(operationId)) {
      throw new Error("Account worker purge is unavailable");
    }
    const quarantine = this.quarantineRoot(root, operationId);
    const operationDigest = basename(quarantine);
    const receiptPath = join(
      dirname(quarantine),
      `${operationDigest}.purge-receipt.json`
    );
    const priorReceipt = await readPurgeReceipt(receiptPath, operationDigest);
    if (priorReceipt) {
      await rm(quarantine, {recursive: true, force: true, maxRetries: 0});
      await syncDirectory(dirname(quarantine));
      return priorReceipt;
    }
    let metadata;
    try {
      metadata = await lstat(quarantine);
    } catch (error) {
      if ((error as NodeJS.ErrnoException).code === "ENOENT") {
        throw new Error("Account worker purge receipt is unavailable");
      }
      throw error;
    }
    if (metadata.isSymbolicLink() || !metadata.isDirectory() || !inside(root, quarantine)) {
      throw new Error("Unsafe worker quarantine");
    }
    const rootEntries = (await readdir(quarantine, {withFileTypes: true}))
      .filter((entry) => entry.isDirectory());
    const counted = await countQuarantinedTree(quarantine);
    const result = {
      workerFiles: counted.files,
      workerBytes: counted.bytes,
      workerRoots: rootEntries.length
    };
    // Persist content-free counts before the irreversible recursive removal.
    // A crash in either following window can replay this receipt exactly.
    await persistPurgeReceipt(receiptPath, operationDigest, result);
    await rm(quarantine, {recursive: true, force: false, maxRetries: 0});
    await syncDirectory(dirname(quarantine));
    return result;
  }

  async fenceApiTasks(scope: AccessScope): Promise<void> {
    await this.tasks?.fenceScope(scope);
  }

  async drainApiTasks(
    scope: AccessScope,
    timeoutMs: number
  ): Promise<Pick<AccountRuntimeDrainResult, "apiTasksCancelled" | "apiTasksHandedOff">> {
    const result = await this.tasks?.drainScope(scope, timeoutMs);
    return {
      apiTasksCancelled: result?.cancelled ?? 0,
      apiTasksHandedOff: result?.handedOff ?? 0
    };
  }

  private quarantineRoot(root: string, operationId: string): string {
    const operationDigest = createHash("sha256").update(operationId, "utf8").digest("hex");
    const path = join(root, QUARANTINE_DIRECTORY, operationDigest);
    if (!inside(root, path) || basename(path) !== operationDigest) {
      throw new Error("Unsafe account quarantine target");
    }
    return path;
  }
}
