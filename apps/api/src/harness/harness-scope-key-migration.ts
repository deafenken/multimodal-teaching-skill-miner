import {
  createCipheriv,
  createDecipheriv,
  createHash,
  createHmac,
  randomBytes,
  timingSafeEqual
} from "node:crypto";
import {createReadStream} from "node:fs";
import {
  chmod,
  copyFile,
  lstat,
  mkdir,
  open,
  readFile,
  readdir,
  rename,
  rm,
  unlink,
  writeFile
} from "node:fs/promises";
import {basename, dirname, join, relative, resolve, sep} from "node:path";

const ACTIVE_MARKER = ".teachlab-scope-key-migration-active-v1.json";
const RETIRING_MARKER = ".teachlab-scope-key-migration-retiring-v1.json";
const TOMBSTONE_MARKER = ".teachlab-scope-key-migration-tombstone-v1.json";
const DATA_KEY_ENVELOPE = ".teachlab-scope-data-key-envelope-v1.json";
const LOCK_DIRECTORY = ".scope-key-migration-locks-v1";
const MARKER_SCHEMA = "teachlab.harness_scope_key_migration.v1";
const ENVELOPE_SCHEMA = "teachlab.harness_scope_data_key_envelope.v1";
const MATERIAL_SCHEMA = "teachlab.harness_scope_stable_material.v1";
const MAX_MIGRATION_ENTRIES = 50_000;
const MAX_MIGRATION_BYTES = 512 * 1024 * 1024;
const LOCK_WAIT_MS = 30_000;
const localMigrationGates = new Map<string, Promise<void>>();

const RESERVED_MARKERS = new Set([
  ACTIVE_MARKER,
  RETIRING_MARKER,
  TOMBSTONE_MARKER,
  DATA_KEY_ENVELOPE
]);

export interface HarnessAuthorityScopeBinding {
  scopeId: string;
  keyVersion: string;
}

export interface HarnessScopeRootIdentity {
  key: string;
  scopeId: string;
  keyVersion: string;
  privateRoot: string;
  /** Version-derived wrapping key on input; stable unwrapped data key on output. */
  dataKey: Buffer;
  learnerScopeId?: string;
  authorityScopeBindings?: readonly HarnessAuthorityScopeBinding[];
}

export type ScopeMigrationCheckpoint =
  | "copy_verified"
  | "active_committed"
  | "previous_retirement_started"
  | "previous_tombstoned";

export interface ScopeMigrationOptions {
  checkpoint?: (
    value: ScopeMigrationCheckpoint
  ) => void | Promise<void>;
  now?: () => number;
}

interface Manifest {
  entries: number;
  bytes: number;
  sha256: string;
}

interface MarkerPayload {
  schema: typeof MARKER_SCHEMA;
  state: "active_committed" | "retiring_previous" | "previous_tombstone";
  source_key_version: string;
  target_key_version: string;
  source_manifest_sha256: string;
  source_entry_count: number;
  source_byte_count: number;
  target_scope_sha256: string;
  migration_id: string;
}

interface SignedMarker extends MarkerPayload {
  signature_sha256: string;
}

interface StableScopeMaterial {
  dataKey: Buffer;
  learnerScopeId: string;
  authorityScopeBindings: readonly HarnessAuthorityScopeBinding[];
}

interface DataKeyEnvelope {
  schema: typeof ENVELOPE_SCHEMA;
  algorithm: "aes-256-gcm";
  wrapping_key_version: string;
  target_scope_sha256: string;
  source_key_version: string;
  nonce: string;
  ciphertext: string;
  auth_tag: string;
}

function inside(parent: string, child: string): boolean {
  const value = relative(resolve(parent), resolve(child));
  return (
    Boolean(value) &&
    value !== ".." &&
    !value.startsWith(`..${sep}`) &&
    !value.startsWith(sep)
  );
}

function safeName(value: string): string {
  if (
    !/^[A-Za-z0-9_.-]{1,240}$/.test(value) ||
    value === "." ||
    value === ".."
  ) {
    throw new Error("unsafe scope migration entry");
  }
  return value;
}

function canonicalMarker(payload: MarkerPayload): string {
  return JSON.stringify({
    migration_id: payload.migration_id,
    schema: payload.schema,
    source_byte_count: payload.source_byte_count,
    source_entry_count: payload.source_entry_count,
    source_key_version: payload.source_key_version,
    source_manifest_sha256: payload.source_manifest_sha256,
    state: payload.state,
    target_key_version: payload.target_key_version,
    target_scope_sha256: payload.target_scope_sha256
  });
}

function signMarker(payload: MarkerPayload, key: Buffer): SignedMarker {
  return {
    ...payload,
    signature_sha256: createHmac("sha256", key)
      .update(`scope-key-migration-marker-v1\0${canonicalMarker(payload)}`, "utf8")
      .digest("hex")
  };
}

function validVersion(value: unknown): value is string {
  return typeof value === "string" && /^k[1-9][0-9]{0,8}$/.test(value);
}

function parseSignedMarker(value: unknown): SignedMarker | undefined {
  if (!value || typeof value !== "object" || Array.isArray(value)) return undefined;
  const row = value as Record<string, unknown>;
  const keys = Object.keys(row).sort();
  const expected = [
    "migration_id",
    "schema",
    "signature_sha256",
    "source_byte_count",
    "source_entry_count",
    "source_key_version",
    "source_manifest_sha256",
    "state",
    "target_key_version",
    "target_scope_sha256"
  ].sort();
  if (keys.length !== expected.length || keys.some((key, index) => key !== expected[index])) {
    return undefined;
  }
  if (
    row.schema !== MARKER_SCHEMA ||
    ![
      "active_committed",
      "retiring_previous",
      "previous_tombstone"
    ].includes(String(row.state)) ||
    !validVersion(row.source_key_version) ||
    !validVersion(row.target_key_version) ||
    typeof row.source_manifest_sha256 !== "string" ||
    !/^[0-9a-f]{64}$/.test(row.source_manifest_sha256) ||
    !Number.isSafeInteger(row.source_entry_count) ||
    Number(row.source_entry_count) < 0 ||
    !Number.isSafeInteger(row.source_byte_count) ||
    Number(row.source_byte_count) < 0 ||
    typeof row.target_scope_sha256 !== "string" ||
    !/^[0-9a-f]{64}$/.test(row.target_scope_sha256) ||
    typeof row.migration_id !== "string" ||
    !/^smig_[0-9a-f]{48}$/.test(row.migration_id) ||
    typeof row.signature_sha256 !== "string" ||
    !/^[0-9a-f]{64}$/.test(row.signature_sha256)
  ) {
    return undefined;
  }
  return row as unknown as SignedMarker;
}

function markerPayload(marker: SignedMarker): MarkerPayload {
  const {signature_sha256: _signature, ...payload} = marker;
  return payload;
}

function markerSignatureValid(marker: SignedMarker, key: Buffer): boolean {
  const expected = Buffer.from(signMarker(markerPayload(marker), key).signature_sha256, "hex");
  const received = Buffer.from(marker.signature_sha256, "hex");
  return expected.byteLength === received.byteLength && timingSafeEqual(expected, received);
}

function validScopeId(value: unknown): value is string {
  return typeof value === "string" && /^scope_[0-9a-f]{48}$/.test(value);
}

function targetScopeSha256(identity: HarnessScopeRootIdentity): string {
  return createHmac("sha256", identity.dataKey)
    .update(`scope-key-migration-target-v1\0${identity.scopeId}`, "utf8")
    .digest("hex");
}

function canonicalEnvelopeAad(
  envelope: Pick<
    DataKeyEnvelope,
    | "schema"
    | "algorithm"
    | "wrapping_key_version"
    | "target_scope_sha256"
    | "source_key_version"
  >
): Buffer {
  return Buffer.from(JSON.stringify({
    algorithm: envelope.algorithm,
    schema: envelope.schema,
    source_key_version: envelope.source_key_version,
    target_scope_sha256: envelope.target_scope_sha256,
    wrapping_key_version: envelope.wrapping_key_version
  }), "utf8");
}

function normalizedBindings(
  bindings: readonly HarnessAuthorityScopeBinding[]
): HarnessAuthorityScopeBinding[] {
  if (bindings.length < 1 || bindings.length > 16) {
    throw new Error("scope material authority bindings are invalid");
  }
  const unique = new Map<string, HarnessAuthorityScopeBinding>();
  for (const binding of bindings) {
    if (!validScopeId(binding.scopeId) || !validVersion(binding.keyVersion)) {
      throw new Error("scope material authority binding is invalid");
    }
    unique.set(`${binding.keyVersion}:${binding.scopeId}`, {
      scopeId: binding.scopeId,
      keyVersion: binding.keyVersion
    });
  }
  return [...unique.values()].sort(
    (left, right) => left.keyVersion.localeCompare(right.keyVersion)
      || left.scopeId.localeCompare(right.scopeId)
  );
}

function canonicalMaterial(material: StableScopeMaterial): Buffer {
  if (
    !Buffer.isBuffer(material.dataKey) ||
    material.dataKey.byteLength !== 32 ||
    !validScopeId(material.learnerScopeId)
  ) {
    throw new Error("scope stable material is invalid");
  }
  return Buffer.from(JSON.stringify({
    authority_scope_bindings: normalizedBindings(material.authorityScopeBindings)
      .map((binding) => ({
        key_version: binding.keyVersion,
        scope_id: binding.scopeId
      })),
    data_key: material.dataKey.toString("base64url"),
    learner_scope_id: material.learnerScopeId,
    schema: MATERIAL_SCHEMA
  }), "utf8");
}

function parseMaterial(value: Buffer): StableScopeMaterial {
  if (value.byteLength < 1 || value.byteLength > 4096) {
    throw new Error("scope stable material is invalid");
  }
  let parsed: unknown;
  try {
    parsed = JSON.parse(value.toString("utf8"));
  } catch {
    throw new Error("scope stable material is invalid");
  }
  if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) {
    throw new Error("scope stable material is invalid");
  }
  const row = parsed as Record<string, unknown>;
  if (
    Object.keys(row).sort().join("\0") !== [
      "authority_scope_bindings",
      "data_key",
      "learner_scope_id",
      "schema"
    ].sort().join("\0") ||
    row.schema !== MATERIAL_SCHEMA ||
    typeof row.data_key !== "string" ||
    !/^[A-Za-z0-9_-]{43}$/.test(row.data_key) ||
    !validScopeId(row.learner_scope_id) ||
    !Array.isArray(row.authority_scope_bindings)
  ) {
    throw new Error("scope stable material is invalid");
  }
  const dataKey = Buffer.from(row.data_key, "base64url");
  const bindings = row.authority_scope_bindings.map((binding) => {
    if (!binding || typeof binding !== "object" || Array.isArray(binding)) {
      throw new Error("scope stable material authority binding is invalid");
    }
    const item = binding as Record<string, unknown>;
    if (
      Object.keys(item).sort().join("\0") !== "key_version\0scope_id" ||
      !validVersion(item.key_version) ||
      !validScopeId(item.scope_id)
    ) {
      throw new Error("scope stable material authority binding is invalid");
    }
    return {keyVersion: item.key_version, scopeId: item.scope_id};
  });
  const material = {
    dataKey,
    learnerScopeId: row.learner_scope_id,
    authorityScopeBindings: normalizedBindings(bindings)
  };
  // Reject non-canonical plaintext before accepting the decrypted key.
  const canonical = canonicalMaterial(material);
  if (value.byteLength !== canonical.byteLength || !timingSafeEqual(value, canonical)) {
    throw new Error("scope stable material is not canonical");
  }
  return material;
}

function parseEnvelope(value: unknown): DataKeyEnvelope {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    throw new Error("scope data key envelope is invalid");
  }
  const row = value as Record<string, unknown>;
  if (
    Object.keys(row).sort().join("\0") !== [
      "algorithm",
      "auth_tag",
      "ciphertext",
      "nonce",
      "schema",
      "source_key_version",
      "target_scope_sha256",
      "wrapping_key_version"
    ].sort().join("\0") ||
    row.schema !== ENVELOPE_SCHEMA ||
    row.algorithm !== "aes-256-gcm" ||
    !validVersion(row.wrapping_key_version) ||
    !validVersion(row.source_key_version) ||
    typeof row.target_scope_sha256 !== "string" ||
    !/^[0-9a-f]{64}$/.test(row.target_scope_sha256) ||
    typeof row.nonce !== "string" ||
    !/^[A-Za-z0-9_-]{16}$/.test(row.nonce) ||
    typeof row.ciphertext !== "string" ||
    !/^[A-Za-z0-9_-]{1,5462}$/.test(row.ciphertext) ||
    typeof row.auth_tag !== "string" ||
    !/^[A-Za-z0-9_-]{22}$/.test(row.auth_tag)
  ) {
    throw new Error("scope data key envelope is invalid");
  }
  return row as unknown as DataKeyEnvelope;
}

function sealMaterial(
  material: StableScopeMaterial,
  target: HarnessScopeRootIdentity,
  sourceKeyVersion: string
): DataKeyEnvelope {
  if (!validVersion(sourceKeyVersion) || target.dataKey.byteLength !== 32) {
    throw new Error("scope data key wrapping material is invalid");
  }
  const header: Pick<
    DataKeyEnvelope,
    | "schema"
    | "algorithm"
    | "wrapping_key_version"
    | "target_scope_sha256"
    | "source_key_version"
  > = {
    schema: ENVELOPE_SCHEMA,
    algorithm: "aes-256-gcm" as const,
    wrapping_key_version: target.keyVersion,
    target_scope_sha256: targetScopeSha256(target),
    source_key_version: sourceKeyVersion
  };
  const nonce = randomBytes(12);
  const cipher = createCipheriv("aes-256-gcm", target.dataKey, nonce, {
    authTagLength: 16
  });
  cipher.setAAD(canonicalEnvelopeAad(header));
  const ciphertext = Buffer.concat([
    cipher.update(canonicalMaterial(material)),
    cipher.final()
  ]);
  return {
    ...header,
    nonce: nonce.toString("base64url"),
    ciphertext: ciphertext.toString("base64url"),
    auth_tag: cipher.getAuthTag().toString("base64url")
  };
}

function unsealMaterial(
  envelope: DataKeyEnvelope,
  identity: HarnessScopeRootIdentity
): StableScopeMaterial {
  if (
    envelope.wrapping_key_version !== identity.keyVersion ||
    envelope.target_scope_sha256 !== targetScopeSha256(identity)
  ) {
    throw new Error("scope data key envelope binding is invalid");
  }
  const decipher = createDecipheriv(
    "aes-256-gcm",
    identity.dataKey,
    Buffer.from(envelope.nonce, "base64url"),
    {authTagLength: 16}
  );
  decipher.setAAD(canonicalEnvelopeAad(envelope));
  decipher.setAuthTag(Buffer.from(envelope.auth_tag, "base64url"));
  try {
    return parseMaterial(Buffer.concat([
      decipher.update(Buffer.from(envelope.ciphertext, "base64url")),
      decipher.final()
    ]));
  } catch {
    throw new Error("scope data key envelope authentication failed");
  }
}

async function readSealedMaterial(
  root: string,
  identity: HarnessScopeRootIdentity
): Promise<StableScopeMaterial | undefined> {
  try {
    const content = await readFile(join(root, DATA_KEY_ENVELOPE), "utf8");
    if (Buffer.byteLength(content, "utf8") > 8192) {
      throw new Error("scope data key envelope is invalid");
    }
    return unsealMaterial(parseEnvelope(JSON.parse(content)), identity);
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code === "ENOENT") return undefined;
    if (error instanceof SyntaxError) throw new Error("scope data key envelope is invalid");
    throw error;
  }
}

async function writeSealedMaterial(
  root: string,
  identity: HarnessScopeRootIdentity,
  material: StableScopeMaterial,
  sourceKeyVersion: string
): Promise<void> {
  const envelope = sealMaterial(material, identity, sourceKeyVersion);
  await writeDurableJson(join(root, DATA_KEY_ENVELOPE), envelope);
  const verified = await readSealedMaterial(root, identity);
  if (!verified || !timingSafeEqual(verified.dataKey, material.dataKey)) {
    throw new Error("scope data key envelope verification failed");
  }
}

function legacyMaterial(identity: HarnessScopeRootIdentity): StableScopeMaterial {
  return {
    dataKey: Buffer.from(identity.dataKey),
    learnerScopeId: identity.scopeId,
    authorityScopeBindings: [{
      scopeId: identity.scopeId,
      keyVersion: identity.keyVersion
    }]
  };
}

function extendMaterialForTarget(
  material: StableScopeMaterial,
  source: HarnessScopeRootIdentity
): StableScopeMaterial {
  return {
    dataKey: Buffer.from(material.dataKey),
    learnerScopeId: material.learnerScopeId,
    authorityScopeBindings: normalizedBindings([
      ...material.authorityScopeBindings,
      {scopeId: source.scopeId, keyVersion: source.keyVersion}
    ])
  };
}

function identityWithMaterial<T extends HarnessScopeRootIdentity>(
  identity: T,
  material: StableScopeMaterial
): T {
  return {
    ...identity,
    dataKey: Buffer.from(material.dataKey),
    learnerScopeId: material.learnerScopeId,
    authorityScopeBindings: material.authorityScopeBindings.map((binding) => ({...binding}))
  };
}

async function syncDirectory(path: string): Promise<void> {
  const handle = await open(path, "r");
  try {
    await handle.sync();
  } finally {
    await handle.close();
  }
}

async function privateDirectory(path: string): Promise<boolean> {
  try {
    const metadata = await lstat(path);
    if (
      metadata.isSymbolicLink() ||
      !metadata.isDirectory() ||
      (metadata.mode & 0o077) !== 0
    ) {
      throw new Error("unsafe scope migration directory");
    }
    return true;
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code === "ENOENT") return false;
    throw error;
  }
}

async function manifest(root: string): Promise<Manifest> {
  const digest = createHash("sha256");
  let entries = 0;
  let bytes = 0;
  const visit = async (directory: string): Promise<void> => {
    const children = await readdir(directory, {withFileTypes: true});
    children.sort((left, right) => left.name.localeCompare(right.name));
    for (const child of children) {
      safeName(child.name);
      if (directory === root && RESERVED_MARKERS.has(child.name)) continue;
      const path = join(directory, child.name);
      if (!inside(root, path)) throw new Error("unsafe scope migration path");
      const metadata = await lstat(path);
      if (metadata.isSymbolicLink()) throw new Error("scope migration rejects symlinks");
      const archivePath = relative(root, path).split(sep).join("/");
      entries += 1;
      if (entries > MAX_MIGRATION_ENTRIES) throw new Error("scope migration entry limit");
      if (metadata.isDirectory()) {
        digest.update(`d\0${archivePath}\0`, "utf8");
        await visit(path);
      } else if (metadata.isFile()) {
        if (!Number.isSafeInteger(metadata.size)) throw new Error("scope migration size invalid");
        bytes += metadata.size;
        if (bytes > MAX_MIGRATION_BYTES) throw new Error("scope migration byte limit");
        const fileDigest = createHash("sha256");
        let observed = 0;
        for await (const chunk of createReadStream(path, {flags: "r"})) {
          const buffer = Buffer.from(chunk as Buffer);
          observed += buffer.byteLength;
          if (observed > metadata.size) throw new Error("scope changed during migration");
          fileDigest.update(buffer);
        }
        if (observed !== metadata.size) throw new Error("scope changed during migration");
        digest.update(
          `f\0${archivePath}\0${metadata.size}\0${fileDigest.digest("hex")}\0`,
          "utf8"
        );
      } else {
        throw new Error("unsupported scope migration entry");
      }
    }
  };
  await visit(root);
  return {entries, bytes, sha256: digest.digest("hex")};
}

async function copyTree(
  source: string,
  destination: string,
  treeRoot: string = source
): Promise<void> {
  const children = await readdir(source, {withFileTypes: true});
  children.sort((left, right) => left.name.localeCompare(right.name));
  for (const child of children) {
    safeName(child.name);
    if (source === treeRoot && RESERVED_MARKERS.has(child.name)) continue;
    const sourcePath = join(source, child.name);
    const destinationPath = join(destination, child.name);
    const metadata = await lstat(sourcePath);
    if (metadata.isSymbolicLink()) throw new Error("scope migration rejects symlinks");
    if (metadata.isDirectory()) {
      await mkdir(destinationPath, {mode: 0o700});
      await copyTree(sourcePath, destinationPath, treeRoot);
      await chmod(destinationPath, 0o700);
      await syncDirectory(destinationPath);
    } else if (metadata.isFile()) {
      await copyFile(sourcePath, destinationPath);
      await chmod(destinationPath, 0o600);
      const handle = await open(destinationPath, "r");
      try {
        await handle.sync();
      } finally {
        await handle.close();
      }
    } else {
      throw new Error("unsupported scope migration entry");
    }
  }
  await syncDirectory(destination);
}

async function readMarker(path: string): Promise<SignedMarker | undefined> {
  try {
    const content = await readFile(path, "utf8");
    if (Buffer.byteLength(content, "utf8") > 4096) return undefined;
    return parseSignedMarker(JSON.parse(content));
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code === "ENOENT") return undefined;
    if (error instanceof SyntaxError) return undefined;
    throw error;
  }
}

async function writeDurableJson(path: string, value: unknown): Promise<void> {
  await writeFile(path, `${JSON.stringify(value)}\n`, {encoding: "utf8", mode: 0o600, flag: "wx"});
  const handle = await open(path, "r");
  try {
    await handle.sync();
  } finally {
    await handle.close();
  }
  await syncDirectory(dirname(path));
}

async function acquireMigrationLock(
  root: string,
  active: HarnessScopeRootIdentity,
  now: () => number
): Promise<() => Promise<void>> {
  const lockRoot = join(root, LOCK_DIRECTORY);
  await mkdir(lockRoot, {recursive: true, mode: 0o700});
  await chmod(lockRoot, 0o700);
  const lockName = createHmac("sha256", active.dataKey)
    .update("scope-key-migration-lock-v1", "utf8")
    .digest("hex");
  const lockPath = join(lockRoot, `${lockName}.lock`);
  if (!inside(root, lockPath) || basename(lockPath) !== `${lockName}.lock`) {
    throw new Error("unsafe scope migration lock");
  }
  const token = randomBytes(24).toString("hex");
  const deadline = now() + LOCK_WAIT_MS;
  while (true) {
    try {
      const handle = await open(lockPath, "wx", 0o600);
      try {
        await handle.writeFile(`${JSON.stringify({pid: process.pid, token})}\n`, "utf8");
        await handle.sync();
      } finally {
        await handle.close();
      }
      await syncDirectory(lockRoot);
      return async () => {
        try {
          const value = JSON.parse(await readFile(lockPath, "utf8")) as Record<string, unknown>;
          if (value.token === token) {
            await unlink(lockPath);
            await syncDirectory(lockRoot);
          }
        } catch (error) {
          if ((error as NodeJS.ErrnoException).code !== "ENOENT") throw error;
        }
      };
    } catch (error) {
      if ((error as NodeJS.ErrnoException).code !== "EEXIST") throw error;
      let ownerAlive = true;
      try {
        const value = JSON.parse(await readFile(lockPath, "utf8")) as Record<string, unknown>;
        if (!Number.isInteger(value.pid) || Number(value.pid) < 1) throw new Error("invalid lock");
        try {
          process.kill(Number(value.pid), 0);
        } catch (killError) {
          if ((killError as NodeJS.ErrnoException).code === "ESRCH") ownerAlive = false;
          else throw killError;
        }
      } catch (lockError) {
        if ((lockError as NodeJS.ErrnoException).code === "ENOENT") continue;
        if (lockError instanceof SyntaxError || (lockError as Error).message === "invalid lock") {
          // A peer may have created the lock but not completed its first
          // durable write. Only reap an invalid lock after the full bounded
          // wait and a second age check; never steal a freshly-created lock.
          const metadata = await lstat(lockPath).catch(() => undefined);
          ownerAlive = !metadata || now() - metadata.mtimeMs < LOCK_WAIT_MS;
        } else {
          throw lockError;
        }
      }
      if (!ownerAlive) {
        await unlink(lockPath).catch((unlinkError: NodeJS.ErrnoException) => {
          if (unlinkError.code !== "ENOENT") throw unlinkError;
        });
        continue;
      }
      if (now() >= deadline) throw new Error("scope migration lock timeout");
      await new Promise<void>((resolvePromise) => {
        const timer = setTimeout(resolvePromise, 25);
        timer.unref();
      });
    }
  }
}

async function acquireLocalMigrationGate(key: string): Promise<() => void> {
  const previous = localMigrationGates.get(key) ?? Promise.resolve();
  let release!: () => void;
  const current = new Promise<void>((resolvePromise) => {
    release = resolvePromise;
  });
  localMigrationGates.set(key, current);
  await previous;
  return () => {
    if (localMigrationGates.get(key) === current) localMigrationGates.delete(key);
    release();
  };
}

function payloadFor(
  state: MarkerPayload["state"],
  source: HarnessScopeRootIdentity,
  target: HarnessScopeRootIdentity,
  sourceManifest: Manifest
): MarkerPayload {
  const targetScopeSha256 = createHmac("sha256", target.dataKey)
    .update(`scope-key-migration-target-v1\0${target.scopeId}`, "utf8")
    .digest("hex");
  const migrationId = `smig_${createHmac("sha256", target.dataKey)
    .update(
      `scope-key-migration-id-v1\0${source.keyVersion}\0${sourceManifest.sha256}`,
      "utf8"
    )
    .digest("hex")
    .slice(0, 48)}`;
  return {
    schema: MARKER_SCHEMA,
    state,
    source_key_version: source.keyVersion,
    target_key_version: target.keyVersion,
    source_manifest_sha256: sourceManifest.sha256,
    source_entry_count: sourceManifest.entries,
    source_byte_count: sourceManifest.bytes,
    target_scope_sha256: targetScopeSha256,
    migration_id: migrationId
  };
}

function markerMatches(
  marker: SignedMarker,
  state: MarkerPayload["state"],
  source: HarnessScopeRootIdentity,
  target: HarnessScopeRootIdentity,
  activeMarker?: SignedMarker
): boolean {
  const expectedTargetScopeSha256 = createHmac("sha256", target.dataKey)
    .update(`scope-key-migration-target-v1\0${target.scopeId}`, "utf8")
    .digest("hex");
  const expectedMigrationId = `smig_${createHmac("sha256", target.dataKey)
    .update(
      `scope-key-migration-id-v1\0${source.keyVersion}\0${marker.source_manifest_sha256}`,
      "utf8"
    )
    .digest("hex")
    .slice(0, 48)}`;
  return (
    marker.state === state &&
    marker.source_key_version === source.keyVersion &&
    marker.target_key_version === target.keyVersion &&
    marker.target_scope_sha256 === expectedTargetScopeSha256 &&
    marker.migration_id === expectedMigrationId &&
    markerSignatureValid(marker, target.dataKey) &&
    (!activeMarker ||
      (marker.migration_id === activeMarker.migration_id &&
        marker.source_manifest_sha256 === activeMarker.source_manifest_sha256 &&
        marker.source_entry_count === activeMarker.source_entry_count &&
        marker.source_byte_count === activeMarker.source_byte_count &&
        marker.target_scope_sha256 === activeMarker.target_scope_sha256))
  );
}

async function finalTombstone(
  source: HarnessScopeRootIdentity,
  identities: readonly HarnessScopeRootIdentity[]
): Promise<boolean> {
  const children = await readdir(source.privateRoot);
  if (children.length !== 1 || children[0] !== TOMBSTONE_MARKER) return false;
  const marker = await readMarker(join(source.privateRoot, TOMBSTONE_MARKER));
  if (!marker || marker.state !== "previous_tombstone") return false;
  const target = identities.find((identity) => identity.keyVersion === marker.target_key_version);
  return Boolean(target && markerMatches(marker, "previous_tombstone", source, target));
}

async function retirePrevious(
  source: HarnessScopeRootIdentity,
  target: HarnessScopeRootIdentity,
  activeMarker: SignedMarker,
  checkpoint?: ScopeMigrationOptions["checkpoint"]
): Promise<void> {
  if (await finalTombstone(source, [target])) return;
  const retiringPath = join(source.privateRoot, RETIRING_MARKER);
  let retiring = await readMarker(retiringPath);
  if (!retiring) {
    const observed = await manifest(source.privateRoot);
    if (
      observed.sha256 !== activeMarker.source_manifest_sha256 ||
      observed.entries !== activeMarker.source_entry_count ||
      observed.bytes !== activeMarker.source_byte_count
    ) {
      throw new Error("previous scope changed after migration");
    }
    retiring = signMarker(
      {...markerPayload(activeMarker), state: "retiring_previous"},
      target.dataKey
    );
    await writeDurableJson(retiringPath, retiring);
    await checkpoint?.("previous_retirement_started");
  } else if (!markerMatches(retiring, "retiring_previous", source, target, activeMarker)) {
    throw new Error("invalid scope retirement marker");
  }

  const children = await readdir(source.privateRoot, {withFileTypes: true});
  for (const child of children) {
    safeName(child.name);
    if (child.name === RETIRING_MARKER) continue;
    const path = join(source.privateRoot, child.name);
    if (!inside(source.privateRoot, path)) throw new Error("unsafe previous scope path");
    await rm(path, {recursive: child.isDirectory(), force: false, maxRetries: 0});
  }
  const tombstone = signMarker(
    {...markerPayload(activeMarker), state: "previous_tombstone"},
    target.dataKey
  );
  const temporary = join(source.privateRoot, `${TOMBSTONE_MARKER}.new`);
  await writeDurableJson(temporary, tombstone);
  await rename(temporary, join(source.privateRoot, TOMBSTONE_MARKER));
  await unlink(retiringPath);
  await syncDirectory(source.privateRoot);
  await checkpoint?.("previous_tombstoned");
}

async function migratePrevious<T extends HarnessScopeRootIdentity>(
  root: string,
  source: T,
  target: T,
  checkpoint?: ScopeMigrationOptions["checkpoint"]
): Promise<T> {
  const targetVersionRoot = dirname(target.privateRoot);
  await mkdir(targetVersionRoot, {recursive: true, mode: 0o700});
  await chmod(targetVersionRoot, 0o700);
  const staging = join(targetVersionRoot, `.${target.scopeId}.migrating-v1`);
  if (!inside(root, staging)) throw new Error("unsafe scope migration staging root");
  if (await privateDirectory(staging)) await rm(staging, {recursive: true, force: false});
  await mkdir(staging, {mode: 0o700});
  try {
    const sourceMaterial = extendMaterialForTarget(
      (await readSealedMaterial(source.privateRoot, source)) ?? legacyMaterial(source),
      source
    );
    const before = await manifest(source.privateRoot);
    await copyTree(source.privateRoot, staging);
    const copied = await manifest(staging);
    const after = await manifest(source.privateRoot);
    if (
      copied.sha256 !== before.sha256 ||
      copied.entries !== before.entries ||
      copied.bytes !== before.bytes ||
      after.sha256 !== before.sha256 ||
      after.entries !== before.entries ||
      after.bytes !== before.bytes
    ) {
      throw new Error("scope changed during migration");
    }
    const activeMarker = signMarker(
      payloadFor("active_committed", source, target, before),
      target.dataKey
    );
    await writeSealedMaterial(
      staging,
      target,
      sourceMaterial,
      source.keyVersion
    );
    await writeDurableJson(join(staging, ACTIVE_MARKER), activeMarker);
    await syncDirectory(staging);
    await checkpoint?.("copy_verified");
    await rename(staging, target.privateRoot);
    await syncDirectory(targetVersionRoot);
    await checkpoint?.("active_committed");
    await retirePrevious(source, target, activeMarker, checkpoint);
    return identityWithMaterial(target, sourceMaterial);
  } catch (error) {
    if (!(await privateDirectory(target.privateRoot))) {
      await rm(staging, {recursive: true, force: true}).catch(() => undefined);
    }
    throw error;
  }
}

async function activeMarker(
  active: HarnessScopeRootIdentity
): Promise<SignedMarker | undefined> {
  const marker = await readMarker(join(active.privateRoot, ACTIVE_MARKER));
  if (!marker || !markerMatches(marker, "active_committed", { ...active, keyVersion: marker.source_key_version }, active)) {
    return undefined;
  }
  return marker;
}

/**
 * Selects the active key root. A retained previous-key root is copied into a
 * verified staging directory, atomically renamed, then reduced in place to a
 * signed hash-only tombstone. No raw tenant/subject value is written.
 */
export async function ensureActiveScopeRoot<T extends HarnessScopeRootIdentity>(
  root: string,
  identities: readonly T[],
  options: ScopeMigrationOptions = {}
): Promise<T> {
  const active = identities[0];
  if (!active || !inside(root, active.privateRoot)) throw new Error("scope migration unavailable");
  const now = options.now ?? (() => Date.now());
  const localKey = createHmac("sha256", active.dataKey)
    .update("scope-key-migration-local-gate-v1", "utf8")
    .digest("hex");
  const releaseLocal = await acquireLocalMigrationGate(localKey);
  let release: (() => Promise<void>) | undefined;
  try {
    release = await acquireMigrationLock(root, active, now);
    const existing: T[] = [];
    for (const identity of identities) {
      if (!inside(root, identity.privateRoot)) throw new Error("unsafe scope root");
      if (await privateDirectory(identity.privateRoot)) existing.push(identity);
    }
    const activeExists = existing.includes(active);
    const previousData: T[] = [];
    for (const identity of existing.slice(activeExists ? 1 : 0)) {
      if (!(await finalTombstone(identity, identities))) previousData.push(identity);
    }

    if (activeExists) {
      if (previousData.length === 0) {
        let material = await readSealedMaterial(active.privateRoot, active);
        if (!material) {
          const marker = await activeMarker(active);
          if (marker && marker.source_key_version !== active.keyVersion) {
            // A migrated root without its sealed stable key cannot safely fall
            // back to a newly-derived key after its source was tombstoned.
            throw new Error("migrated scope data key envelope is missing");
          }
          material = legacyMaterial(active);
          await writeSealedMaterial(
            active.privateRoot,
            active,
            material,
            active.keyVersion
          );
        }
        return identityWithMaterial(active, material);
      }
      const marker = await activeMarker(active);
      if (!marker) {
        const activeContents = await manifest(active.privateRoot);
        if (activeContents.entries !== 0 || previousData.length !== 1) {
          throw new Error("ambiguous active and previous scope roots");
        }
        await rm(active.privateRoot, {recursive: true, force: false});
        await syncDirectory(dirname(active.privateRoot));
        return await migratePrevious(root, previousData[0]!, active, options.checkpoint);
      }
      const source = previousData.find(
        (identity) => identity.keyVersion === marker.source_key_version
      );
      if (!source || previousData.length !== 1) {
        throw new Error("ambiguous previous scope roots");
      }
      let material = await readSealedMaterial(active.privateRoot, active);
      if (!material) {
        material = extendMaterialForTarget(
          (await readSealedMaterial(source.privateRoot, source)) ?? legacyMaterial(source),
          source
        );
        await writeSealedMaterial(
          active.privateRoot,
          active,
          material,
          source.keyVersion
        );
      }
      await retirePrevious(source, active, marker, options.checkpoint);
      return identityWithMaterial(active, material);
    }

    if (previousData.length > 1) throw new Error("ambiguous previous scope roots");
    if (previousData.length === 1) {
      return await migratePrevious(root, previousData[0]!, active, options.checkpoint);
    }
    if (existing.length > 0) {
      throw new Error("active scope root missing after prior migration");
    }
    const targetVersionRoot = dirname(active.privateRoot);
    await mkdir(targetVersionRoot, {recursive: true, mode: 0o700});
    await chmod(targetVersionRoot, 0o700);
    await mkdir(active.privateRoot, {mode: 0o700});
    const material = legacyMaterial(active);
    await writeSealedMaterial(
      active.privateRoot,
      active,
      material,
      active.keyVersion
    );
    await syncDirectory(targetVersionRoot);
    return identityWithMaterial(active, material);
  } finally {
    try {
      await release?.();
    } finally {
      releaseLocal();
    }
  }
}

export const scopeMigrationMarkerNames = Object.freeze({
  active: ACTIVE_MARKER,
  retiring: RETIRING_MARKER,
  tombstone: TOMBSTONE_MARKER,
  dataKeyEnvelope: DATA_KEY_ENVELOPE
});

export const scopeMigrationSecurityMetadataNames: ReadonlySet<string> =
  Object.freeze(new Set(RESERVED_MARKERS));
