#!/usr/bin/env node

import assert from "node:assert/strict";
import {spawn, spawnSync} from "node:child_process";
import {createHash, randomBytes} from "node:crypto";
import fs from "node:fs";
import net from "node:net";
import os from "node:os";
import path from "node:path";
import process from "node:process";
import {fileURLToPath} from "node:url";

const launcherScript = fileURLToPath(import.meta.url);
const root = path.resolve(path.dirname(launcherScript), "..");
const lifecycleSelfTest = process.argv[2] === "--lifecycle-self-test";
const runtimeSelfTest = process.argv[2] === "--runtime-self-test";
const consoleSupervisorMode = process.argv[2] === "--console-supervisor";
const daemonMode = process.argv[2] === "--daemon";
const stopMode = process.argv[2] === "--stop";

async function runConsoleSupervisor(runtimeDirectory, port) {
  if (
    process.env.TEACHLAB_LAUNCH_ROLE !== "console"
    || !/^[0-9a-f]{32}$/.test(String(process.env.TEACHLAB_LAUNCH_ID))
    || !path.isAbsolute(String(runtimeDirectory))
    || !/^\d{1,5}$/.test(String(port))
  ) {
    console.error("Console supervisor 启动身份或参数无效。");
    return 2;
  }
  const childEnvironment = {...process.env};
  delete childEnvironment.TEACHLAB_LAUNCH_ID;
  delete childEnvironment.TEACHLAB_LAUNCH_ROLE;
  const serverEntry = path.join(runtimeDirectory, "server.js");
  let serverStat = null;
  try { serverStat = fs.lstatSync(serverEntry); } catch {}
  if (!serverStat?.isFile() || serverStat.isSymbolicLink()) {
    console.error("Console production runtime 缺少 server.js。");
    return 2;
  }
  const serverChild = spawn(
    process.execPath,
    [serverEntry],
    {
      cwd: runtimeDirectory,
      env: {...childEnvironment, HOSTNAME: "127.0.0.1", PORT: String(port), NODE_ENV: "production"},
      stdio: "inherit",
      detached: false,
    },
  );
  let requestedExitCode = null;
  let forceTimer = null;
  const forwardSignal = (signal, exitCode) => {
    if (requestedExitCode !== null) return;
    requestedExitCode = exitCode;
    try { serverChild.kill(signal); } catch {}
    forceTimer = setTimeout(() => {
      try { serverChild.kill("SIGKILL"); } catch {}
    }, 1_200);
  };
  process.on("SIGINT", () => forwardSignal("SIGINT", 130));
  process.on("SIGTERM", () => forwardSignal("SIGTERM", 143));
  return await new Promise((resolve) => {
    serverChild.once("error", () => resolve(1));
    serverChild.once("exit", (code, signal) => {
      if (forceTimer) clearTimeout(forceTimer);
      if (requestedExitCode !== null) resolve(requestedExitCode);
      else if (Number.isInteger(code)) resolve(code);
      else resolve(signal === "SIGTERM" ? 143 : signal === "SIGINT" ? 130 : 1);
    });
  });
}

if (consoleSupervisorMode) {
  process.exit(await runConsoleSupervisor(process.argv[3], process.argv[4]));
}

const argumentOffset = daemonMode ? 3 : 2;
const python = process.argv[argumentOffset];
const apiKeyFile = process.argv[argumentOffset + 1];
const npm = process.argv[argumentOffset + 2] || "npm";
if (!lifecycleSelfTest && !runtimeSelfTest && !stopMode && (!python || !apiKeyFile)) {
  console.error("usage: start_teacher_agent_console.mjs <python> <api-key-file> [npm]");
  process.exit(2);
}

const children = [];
let shuttingDown = false;
let shutdownTimer = null;
let shutdownFinalizer = null;
let launcherLockReleasePermitted = true;
const sessionStore = process.env.TEACHLAB_SESSION_STORE || `${root}/.private/teacher_agent_console_sessions_lessonflow_v2.jsonl`;
const syllabusStore = process.env.TEACHLAB_SYLLABUS_STORE || `${root}/.private/teacher_agent_console_syllabi`;
const projectStore = process.env.TEACHLAB_PROJECT_STORE || `${root}/.private/teacher_agent_console_projects`;
const resourceIndexStore = process.env.TEACHLAB_RESOURCE_INDEX_STORE || `${root}/.private/teacher_agent_console_resource_index`;
const resourceReviewStore = process.env.TEACHLAB_RESOURCE_REVIEW_STORE || `${root}/.private/teacher_agent_console_resource_reviews`;
const learningRecordStore = process.env.TEACHLAB_LEARNING_RECORD_STORE || `${root}/.private/teacher_agent_console_learning_records.jsonl`;
const metacognitionStore = process.env.TEACHLAB_METACOGNITION_STORE || `${root}/.private/teacher_agent_console_metacognition.jsonl`;
const adjudicationStore = process.env.TEACHLAB_ADJUDICATION_STORE || `${root}/.private/teacher_agent_console_adjudications.jsonl`;
const learnerKeySecretFile = process.env.TEACHLAB_LEARNER_KEY_SECRET_FILE || `${root}/.private/teacher_agent_console_learner_key.secret`;
const consentStore = process.env.TEACHLAB_CONSENT_STORE || `${root}/.private/teacher_agent_console_remote_consent.json`;
const consentSigningSecretFile = process.env.TEACHLAB_CONSENT_SIGNING_SECRET_FILE || `${root}/.private/teacher_agent_console_remote_consent.secret`;
const runtimeActivityFile = process.env.TEACHLAB_RUNTIME_ACTIVITY_FILE || `${root}/.private/teacher_agent_console_activity`;
const idleShutdownMinutes = Number(process.env.TEACHLAB_IDLE_SHUTDOWN_MINUTES || 0);
const learnerTenantId = process.env.TEACHLAB_LEARNER_TENANT_ID || "teachlab-local-v1";
const launcherLock = process.env.TEACHLAB_LAUNCHER_LOCK || `${root}/.private/teacher_agent_console_launcher.lock`;
const daemonHandoffToken = /^[0-9a-f]{32}$/.test(String(process.env.TEACHLAB_DAEMON_HANDOFF_TOKEN || ""))
  ? String(process.env.TEACHLAB_DAEMON_HANDOFF_TOKEN)
  : randomBytes(16).toString("hex");
const materializedSecrets = [];
let staleMaterializedSecretsCleaned = false;
const LOCK_SCHEMA = "teachlab.console_launcher.v5";
const LEGACY_LOCK_SCHEMA = "teachlab.console_launcher.v4";
const OLDER_LOCK_SCHEMA = "teachlab.console_launcher.v3";
const ACQUISITION_LOCK_SCHEMA = "teachlab.console_launcher.acquire.v1";
const STABLE_RECOVERY_PHASES = new Set(["runtime_preflight", "backend_started", "children_started", "ready", "stopping"]);
const TRANSIENT_LAUNCH_PHASES = new Set(["spawning_backend", "spawning_console"]);
const ALL_LAUNCH_PHASES = new Set([...STABLE_RECOVERY_PHASES, ...TRANSIENT_LAUNCH_PHASES]);
let launcherRecord = null;

function validateRuntimeConfiguration() {
  if (!Number.isSafeInteger(idleShutdownMinutes) || idleShutdownMinutes < 0 || idleShutdownMinutes > 1440) {
    throw new Error("TEACHLAB_IDLE_SHUTDOWN_MINUTES 必须是 0–1440 的整数。");
  }
  const preferredPort = Number(process.env.TEACHLAB_CONSOLE_PORT || 3000);
  if (!Number.isSafeInteger(preferredPort) || preferredPort < 1024 || preferredPort > 65516) {
    throw new Error("TEACHLAB_CONSOLE_PORT 必须是 1024–65516 的整数，以保留 20 个候选端口。");
  }
}

function ensurePrivateSecret(secretFile, label) {
  const directory = path.dirname(secretFile);
  fs.mkdirSync(directory, {recursive: true, mode: 0o700});
  const directoryStat = fs.lstatSync(directory);
  if (!directoryStat.isDirectory() || directoryStat.isSymbolicLink()) {
    throw new Error(`${label}目录必须是本机私有目录。`);
  }
  fs.chmodSync(directory, 0o700);
  try {
    const existing = fs.lstatSync(secretFile);
    if (
      !existing.isFile()
      || existing.isSymbolicLink()
      || (existing.mode & 0o077) !== 0
      || existing.size < 32
      || existing.size > 4096
    ) {
      throw new Error(`${label}必须是 32–4096 字节的 mode-0600 普通文件。`);
    }
    return;
  } catch (error) {
    if (!(error instanceof Error) || !Reflect.has(error, "code") || error.code !== "ENOENT") throw error;
  }
  const descriptor = fs.openSync(secretFile, "wx", 0o600);
  try {
    fs.writeFileSync(descriptor, randomBytes(32));
    fs.fsyncSync(descriptor);
  } finally {
    fs.closeSync(descriptor);
  }
  const directoryDescriptor = fs.openSync(directory, "r");
  try { fs.fsyncSync(directoryDescriptor); } finally { fs.closeSync(directoryDescriptor); }
}

function strictFallbackSecret(secretFile, label, {allowShort = false} = {}) {
  const parent = fs.lstatSync(path.dirname(secretFile));
  if (!parent.isDirectory() || parent.isSymbolicLink() || (parent.mode & 0o077) !== 0) {
    throw new Error(`${label} fallback 的父目录必须是非符号链接的 mode-0700 私有目录。`);
  }
  const stat = fs.lstatSync(secretFile);
  if (!stat.isFile() || stat.isSymbolicLink() || (stat.mode & 0o077) !== 0
    || stat.size < (allowShort ? 1 : 32) || stat.size > 4096) {
    throw new Error(`${label} fallback 必须是非符号链接的 mode-0600 普通文件。`);
  }
  return secretFile;
}

function inspectSymlinkSecretTarget(secretFile, label, {allowShort = false} = {}) {
  const linkParent = fs.lstatSync(path.dirname(secretFile));
  if (
    !linkParent.isDirectory()
    || linkParent.isSymbolicLink()
    || (linkParent.mode & 0o077) !== 0
  ) {
    throw new Error(`${label} fallback 链接所在目录必须是非符号链接的 mode-0700 私有目录。`);
  }
  const link = fs.lstatSync(secretFile);
  if (!link.isSymbolicLink()) {
    throw new Error(`${label} fallback 不是可解析的私有密钥链接。`);
  }
  let resolved;
  try {
    resolved = fs.realpathSync.native(secretFile);
  } catch (error) {
    throw new Error(`${label} fallback 链接目标无法解析。`, {cause: error});
  }
  if (!path.isAbsolute(resolved) || resolved === path.resolve(secretFile)) {
    throw new Error(`${label} fallback 链接目标无效。`);
  }
  const targetParent = fs.lstatSync(path.dirname(resolved));
  if (
    !targetParent.isDirectory()
    || targetParent.isSymbolicLink()
    || (targetParent.mode & 0o077) !== 0
  ) {
    throw new Error(`${label} fallback 链接目标的父目录必须是 mode-0700 私有目录。`);
  }
  const target = fs.lstatSync(resolved);
  if (
    !target.isFile()
    || target.isSymbolicLink()
    || (target.mode & 0o077) !== 0
    || target.size < (allowShort ? 1 : 32)
    || target.size > 4096
  ) {
    throw new Error(`${label} fallback 链接目标必须是 1–4096 字节的 owner-only 普通文件。`);
  }
  return {resolved, target};
}

function materializeSymlinkSecret(
  secretFile,
  label,
  runtimeSecretDirectory,
  {allowShort = false} = {},
) {
  const inspected = inspectSymlinkSecretTarget(secretFile, label, {allowShort});
  const noFollow = fs.constants.O_NOFOLLOW ?? 0;
  let sourceDescriptor;
  try {
    // Resolve the link first, then refuse a final symlink and verify the inode
    // again after opening. The child process receives only the private copy.
    sourceDescriptor = fs.openSync(inspected.resolved, fs.constants.O_RDONLY | noFollow);
    const opened = fs.fstatSync(sourceDescriptor);
    if (
      !opened.isFile()
      || (opened.mode & 0o077) !== 0
      || opened.dev !== inspected.target.dev
      || opened.ino !== inspected.target.ino
      || opened.size < (allowShort ? 1 : 32)
      || opened.size > 4096
    ) {
      throw new Error(`${label} fallback 链接目标在读取期间发生变化。`);
    }
    const contents = fs.readFileSync(sourceDescriptor);
    if (contents.length !== opened.size) {
      throw new Error(`${label} fallback 链接目标大小在读取期间发生变化。`);
    }
    const target = path.join(
      runtimeSecretDirectory,
      `${launcherRecord.launch_id}-${materializedSecrets.length}.secret`,
    );
    let descriptor;
    try {
      descriptor = fs.openSync(target, "wx", 0o600);
      fs.writeFileSync(descriptor, contents);
      fs.fsyncSync(descriptor);
    } finally {
      if (descriptor !== undefined) fs.closeSync(descriptor);
    }
    fsyncDirectory(runtimeSecretDirectory);
    materializedSecrets.push(target);
    return target;
  } catch (error) {
    throw error instanceof Error ? error : new Error(`${label} fallback 链接读取失败。`);
  } finally {
    if (sourceDescriptor !== undefined) fs.closeSync(sourceDescriptor);
  }
}

function keychainValue(service, account) {
  if (process.platform !== "darwin" || process.env.TEACHLAB_DISABLE_KEYCHAIN === "1") return null;
  const result = spawnSync("/usr/bin/security", [
    "find-generic-password", "-s", service, "-a", account, "-w",
  ], {encoding: "utf8", maxBuffer: 8192});
  if (result.status !== 0 || result.error) return null;
  const value = result.stdout.replace(/[\r\n]+$/, "");
  return value && Buffer.byteLength(value) <= 4096 ? value : null;
}

function resolveSecretFile({
  fallback,
  label,
  service,
  allowShort = false,
  generateFallback = false,
  allowSymlinkFallback = false,
}) {
  const runtimeSecretDirectory = process.env.TEACHLAB_RUNTIME_SECRET_DIR
    || path.join(root, ".private", "runtime-secrets");
  fs.mkdirSync(runtimeSecretDirectory, {recursive: true, mode: 0o700});
  const runtimeSecretDirectoryStat = fs.lstatSync(runtimeSecretDirectory);
  if (!runtimeSecretDirectoryStat.isDirectory() || runtimeSecretDirectoryStat.isSymbolicLink()) {
    throw new Error("运行时密钥目录必须是本机普通目录，不能是符号链接。");
  }
  fs.chmodSync(runtimeSecretDirectory, 0o700);
  if (!staleMaterializedSecretsCleaned) {
    for (const name of fs.readdirSync(runtimeSecretDirectory)) {
      if (!/^[0-9a-f]{32}-\d+\.secret$/.test(name)) continue;
      const stale = path.join(runtimeSecretDirectory, name);
      const stat = fs.lstatSync(stale);
      if (stat.isFile() && !stat.isSymbolicLink() && (stat.mode & 0o077) === 0) fs.unlinkSync(stale);
    }
    staleMaterializedSecretsCleaned = true;
  }
  const account = process.env.TEACHLAB_KEYCHAIN_ACCOUNT || os.userInfo().username;
  const value = keychainValue(service, account);
  if (value !== null) {
    if (!allowShort && Buffer.byteLength(value) < 32) throw new Error(`${label} 的 Keychain 项长度不足 32 字节。`);
    const target = path.join(runtimeSecretDirectory, `${launcherRecord.launch_id}-${materializedSecrets.length}.secret`);
    const descriptor = fs.openSync(target, "wx", 0o600);
    try { fs.writeFileSync(descriptor, value); fs.fsyncSync(descriptor); } finally { fs.closeSync(descriptor); }
    materializedSecrets.push(target);
    return target;
  }
  if (generateFallback && !fs.existsSync(fallback)) ensurePrivateSecret(fallback, label);
  try {
    if (allowSymlinkFallback && fs.lstatSync(fallback).isSymbolicLink()) {
      return materializeSymlinkSecret(
        fallback,
        label,
        runtimeSecretDirectory,
        {allowShort},
      );
    }
    return strictFallbackSecret(fallback, label, {allowShort});
  } catch (error) {
    throw new Error(
      `${label}未在 macOS Keychain（service=${service}, account=${account}）中找到，且 0600 fallback 不可用：`
      + `${error instanceof Error ? error.message : error}`,
    );
  }
}

function cleanupMaterializedSecrets() {
  for (const secret of materializedSecrets.splice(0)) {
    try { fs.unlinkSync(secret); } catch {}
  }
}

function processIsAlive(pid) {
  if (!Number.isSafeInteger(pid) || pid < 2) return false;
  try {
    process.kill(pid, 0);
    return true;
  } catch (error) {
    if (error instanceof Error && Reflect.has(error, "code") && error.code === "ESRCH") return false;
    // EPERM still means that a process owns the PID.  Failing closed avoids
    // launching a second Console when ownership cannot be inspected.
    return true;
  }
}

function readLauncherRecord() {
  const state = inspectLauncherLockFile();
  return state.record;
}

// The launcher lock is a security boundary, not just a JSON hint.  Never
// follow a symlink (or accept a group/world-readable lock) while deciding
// whether another instance owns the Console.  Returning an explicit state lets
// callers distinguish an absent lock from a malformed/stale one and fail
// closed without spawning a duplicate daemon.
function inspectLauncherLockFile() {
  try {
    const stat = fs.lstatSync(launcherLock);
    if (!stat.isFile() || stat.isSymbolicLink() || (stat.mode & 0o077) !== 0) {
      return {exists: true, valid: false, record: null, stat};
    }
    let parsed = null;
    try {
      parsed = JSON.parse(fs.readFileSync(launcherLock, "utf8"));
    } catch {
      // A zero-byte lock can be the short O_EXCL→write handoff window.  Any
      // non-empty malformed lock remains present and is handled fail-closed.
      parsed = null;
    }
    return {
      exists: true,
      valid: true,
      record: parsed && typeof parsed === "object" && !Array.isArray(parsed) ? parsed : null,
      stat,
    };
  } catch (error) {
    if (error instanceof Error && Reflect.has(error, "code") && error.code === "ENOENT") {
      return {exists: false, valid: true, record: null, stat: null};
    }
    return {exists: true, valid: false, record: null, stat: null, error};
  }
}

function launcherLockIsInitializing() {
  const state = inspectLauncherLockFile();
  if (!state.exists || !state.valid || state.record !== null || !state.stat) return false;
  const age = Date.now() - state.stat.mtimeMs;
  return state.stat.size === 0 && age >= 0 && age < 5_000;
}

function canonicalDigest(value) {
  return createHash("sha256").update(
    Buffer.isBuffer(value) || value instanceof Uint8Array ? value : String(value),
  ).digest("hex");
}

async function verifyProductionRuntime(runtimeDirectory) {
  const manifestPath = path.join(runtimeDirectory, "runtime-manifest.json");
  const manifest = JSON.parse(fs.readFileSync(manifestPath, "utf8"));
  if (
    manifest?.schema !== "teachlab.console.runtime.v1"
    || !/^[0-9A-Za-z._-]+-[0-9a-f]{16}$/.test(String(manifest.release_id))
    || !Array.isArray(manifest.files)
    || manifest.files.length < 1
    || manifest.files.length > 20_000
    || manifest.node_min_major !== 22
    || Number(process.versions.node.split(".")[0]) < manifest.node_min_major
  ) throw new Error("Console production runtime manifest 无效或 Node.js 不兼容。");
  const lockDigest = canonicalDigest(fs.readFileSync(path.join(root, "apps", "console", "package-lock.json")));
  if (manifest.package_lock_sha256 !== lockDigest) {
    throw new Error("Console production runtime 与当前 package-lock.json 不一致，必须重新构建。");
  }
  const expectedPaths = new Set();
  for (const entry of manifest.files) {
    if (!entry || typeof entry.path !== "string" || path.isAbsolute(entry.path)
      || entry.path.split(path.sep).some((part) => part === "" || part === "." || part === "..")) {
      throw new Error("Console production runtime file manifest 无效。");
    }
    if (expectedPaths.has(entry.path)) throw new Error("Console production runtime manifest 存在重复路径。");
    expectedPaths.add(entry.path);
  }
  for (let offset = 0; offset < manifest.files.length; offset += 64) {
    await Promise.all(manifest.files.slice(offset, offset + 64).map(async (entry) => {
      const target = path.join(runtimeDirectory, entry.path);
      const [stat, bytes] = await Promise.all([fs.promises.lstat(target), fs.promises.readFile(target)]);
      if (!stat.isFile() || stat.isSymbolicLink() || stat.size !== entry.bytes
        || canonicalDigest(bytes) !== entry.sha256) {
        throw new Error(`Console production runtime 完整性校验失败：${entry.path}`);
      }
    }));
  }
  const visit = (relative) => {
    const absolute = path.join(runtimeDirectory, relative);
    const stat = fs.lstatSync(absolute);
    if (stat.isSymbolicLink()) throw new Error(`Console runtime 含符号链接：${relative}`);
    if (stat.isDirectory()) {
      for (const name of fs.readdirSync(absolute)) visit(path.join(relative, name));
    } else if (stat.isFile() && relative !== "runtime-manifest.json" && !expectedPaths.delete(relative)) {
      throw new Error(`Console runtime 含未登记文件：${relative}`);
    } else if (!stat.isFile()) throw new Error(`Console runtime 含不支持的文件系统项：${relative}`);
  };
  for (const name of fs.readdirSync(runtimeDirectory)) visit(name);
  if (expectedPaths.size) throw new Error("Console runtime manifest 引用了缺失文件。");
  if (!fs.existsSync(path.join(runtimeDirectory, "server.js"))) throw new Error("Console runtime 缺少 server.js。");
  return manifest;
}

async function ensureProductionRuntime() {
  const stateRoot = process.env.TEACHLAB_CONSOLE_RUNTIME_ROOT || path.join(root, ".private", "console-runtime");
  const channelPath = path.join(stateRoot, "channel.json");
  const configuredExpectedReleaseId = process.env.TEACHLAB_EXPECTED_CONSOLE_RELEASE_ID?.trim();
  const computeExpectedReleaseId = () => {
    if (configuredExpectedReleaseId) return configuredExpectedReleaseId;
    const expected = spawnSync(process.execPath, [
      path.join(root, "scripts", "build_teacher_agent_console_runtime.mjs"), "--dry-run",
    ], {cwd: root, env: {...process.env, TEACHLAB_NPM: npm}, encoding: "utf8"});
    if (expected.error || expected.status !== 0) {
      throw new Error("Console 依赖未通过 package-lock 校验，无法选择 production runtime。");
    }
    try { return JSON.parse(expected.stdout.trim()).release_id; } catch { return null; }
  };
  const expectedReleaseId = computeExpectedReleaseId();
  if (!/^[0-9A-Za-z._-]+-[0-9a-f]{16}$/.test(String(expectedReleaseId))) {
    throw new Error("无法计算当前 Console 源码与锁文件的 release id。");
  }
  const resolveCurrent = async () => {
    const channel = JSON.parse(fs.readFileSync(channelPath, "utf8"));
    if (channel?.schema !== "teachlab.console.channel.v1" || !/^[0-9A-Za-z._-]+-[0-9a-f]{16}$/.test(String(channel.current))) {
      throw new Error("Console release channel manifest 无效。");
    }
    if (channel.current !== expectedReleaseId) throw new Error("Console production runtime 与当前源码或锁文件不一致。");
    const directory = path.join(stateRoot, "releases", channel.current);
    const manifestPath = path.join(directory, "runtime-manifest.json");
    if (!/^[0-9a-f]{64}$/.test(String(channel.current_manifest_sha256))
      || canonicalDigest(fs.readFileSync(manifestPath)) !== channel.current_manifest_sha256) {
      throw new Error("Console release channel 未绑定有效的 runtime manifest 哈希。");
    }
    const manifest = await verifyProductionRuntime(directory);
    if (manifest.release_id !== channel.current) throw new Error("Console runtime manifest 的 release id 与 channel 不一致。");
    if (computeExpectedReleaseId() !== expectedReleaseId) {
      throw new Error("Console 源码或锁文件在 runtime 验证期间发生变化；拒绝启动混合版本。");
    }
    return {directory, manifest};
  };
  try {
    return await resolveCurrent();
  } catch (initialError) {
    if (process.env.TEACHLAB_ALLOW_CONSOLE_BUILD === "0") throw initialError;
    console.log("未找到匹配当前锁文件的 production runtime，正在受控构建…");
    const result = spawnSync(process.execPath, [path.join(root, "scripts", "build_teacher_agent_console_runtime.mjs")], {
      cwd: root,
      env: {...process.env, TEACHLAB_NPM: npm},
      stdio: "inherit",
    });
    if (result.error || result.status !== 0) throw new Error("Console production runtime 构建失败。");
    if (computeExpectedReleaseId() !== expectedReleaseId) {
      throw new Error("Console 源码或锁文件在 production build 期间发生变化；请在变更稳定后重试。");
    }
    return await resolveCurrent();
  }
}

function parseProcessIdentityLine(line) {
  const match = String(line).trimEnd().match(/^\s*(\d+)\s+(\d+)\s+(\d+)\s+(.{24})\s+(.+)$/);
  if (!match) return null;
  const pid = Number(match[1]);
  const ppid = Number(match[2]);
  const pgid = Number(match[3]);
  const startedAtOs = match[4].trim();
  const command = match[5].trim();
  if (
    !Number.isSafeInteger(pid)
    || pid < 2
    || !Number.isSafeInteger(ppid)
    || ppid < 0
    || !Number.isSafeInteger(pgid)
    || pgid < 2
    || !command
  ) return null;
  return {pid, ppid, pgid, started_at_os: startedAtOs, command, command_sha256: canonicalDigest(command)};
}

function inspectProcess(pid) {
  if (!Number.isSafeInteger(pid) || pid < 2) return null;
  const result = spawnSync("ps", [
    "-ww", "-p", String(pid), "-o", "pid=", "-o", "ppid=", "-o", "pgid=", "-o", "lstart=", "-o", "command=",
  ], {encoding: "utf8"});
  if (result.error || result.status !== 0) return null;
  return parseProcessIdentityLine(result.stdout);
}

function launcherProcessIdentity() {
  const identity = inspectProcess(process.pid);
  if (!identity || identity.pid !== process.pid || identity.ppid < 0 || identity.pgid < 2) {
    throw new Error("无法验证当前 Console 启动器的进程身份；本次拒绝写入单实例锁。");
  }
  return {
    launcher_started_at_os: identity.started_at_os,
    launcher_command_sha256: identity.command_sha256,
  };
}

function launcherRecordOwnerMatchesProcess(record, pid = record?.launcher_pid) {
  if (!record || !Number.isSafeInteger(pid) || pid < 2 || !processIsAlive(pid)) return false;
  const identity = inspectProcess(pid);
  if (!identity || identity.pid !== pid) return false;
  if (record.schema === LOCK_SCHEMA) {
    return identity.started_at_os === record.launcher_started_at_os
      && identity.command_sha256 === record.launcher_command_sha256
      && leaderCommandMatchesLauncher(identity.command);
  }
  // v4/v3 records predate the launcher identity fields.  They may still block
  // a duplicate, but only when the live command is recognisably this launcher;
  // an unrelated process reusing the old PID must not be reported as running.
  return leaderCommandMatchesLauncher(identity.command);
}

function leaderCommandMatchesLauncher(command) {
  const normalized = String(command).replace(/\s+/g, " ").trim();
  if (/(?:^|\s)--(?:console-supervisor|daemon|stop|lifecycle-self-test|runtime-self-test)(?:\s|$)/.test(normalized)) {
    return false;
  }
  return /(?:^|\/)(?:node|nodejs)(?:\s|$)/i.test(normalized)
    && /(?:^|\/)start_teacher_agent_console\.mjs(?:\s|$)/.test(normalized)
    && !/(?:^|\s)--console-supervisor(?:\s|$)/.test(normalized);
}

function listProcessGroup(pgid) {
  if (!Number.isSafeInteger(pgid) || pgid < 2) return [];
  const result = spawnSync("ps", [
    "-ww", "-axo", "pid=", "-o", "ppid=", "-o", "pgid=", "-o", "lstart=", "-o", "command=",
  ], {encoding: "utf8"});
  if (result.error || result.status !== 0) {
    throw new Error("无法只读检查遗留 Console 进程组；为避免误杀，本次拒绝启动。");
  }
  return result.stdout
    .split(/\r?\n/)
    .map(parseProcessIdentityLine)
    .filter((item) => item && item.pgid === pgid);
}

function processHasLaunchIdentity(pid, launchId, role) {
  const result = spawnSync("ps", ["eww", "-p", String(pid), "-o", "command="], {encoding: "utf8"});
  if (result.error || result.status !== 0) return false;
  return result.stdout.includes(`TEACHLAB_LAUNCH_ID=${launchId}`)
    && result.stdout.includes(`TEACHLAB_LAUNCH_ROLE=${role}`);
}

function blockingSleep(milliseconds) {
  Atomics.wait(new Int32Array(new SharedArrayBuffer(4)), 0, 0, milliseconds);
}

function leaderCommandMatchesRole(role, command) {
  const normalized = String(command).replace(/\s+/g, " ").trim();
  if (role === "backend") {
    return /(?:^| )-m teaching_skill_miner teacher-agent-dashboard(?: |$)/.test(normalized);
  }
  if (role === "console") {
    return /(?:^|\/)start_teacher_agent_console\.mjs --console-supervisor(?: |$)/.test(normalized);
  }
  return false;
}

function consolePostcssWorkerMatches(command) {
  const normalized = String(command).replace(/\s+/g, " ").trim();
  const nodeCommand = normalized.match(/^(?:\S*\/)?node\s+(.+)$/);
  if (!nodeCommand) return false;
  const expectedPrefix = `${path.join(root, "apps", "console", ".next", "postcss.js")} `;
  if (!nodeCommand[1].startsWith(expectedPrefix)) return false;
  const token = nodeCommand[1].slice(expectedPrefix.length);
  return /^[1-9]\d{0,9}$/.test(token) && Number(token) <= 2_147_483_647;
}

function groupMemberMatchesRole(role, command) {
  const normalized = String(command).replace(/\s+/g, " ").trim();
  const lower = normalized.toLowerCase();
  if (role === "backend") {
    return lower.includes("teaching_skill_miner") || lower.includes("teacher-agent-dashboard");
  }
  if (role === "console") {
    return /(?:^|[\s/])npm(?:-cli\.js)?\s+run\s+dev(?:\s|$)/i.test(normalized)
      || /(?:^|\/)next\/dist\/bin\/next\s+dev(?:\s|$)/i.test(normalized)
      || /(?:^|\/)node_modules\/(?:\.bin\/next|next\/dist\/bin\/next)\s+dev(?:\s|$)/i.test(normalized)
      || /(?:^|\/)node\s+[^;&|]*(?:console-runtime\/releases\/[^/]+|teachlab-console-runtime-[^/]+)\/server\.js(?:\s|$)/i.test(normalized)
      || /(?:^|[\s/])next-server(?:\s|$)/i.test(normalized)
      || /(?:^|\/)sh\s+-c\s+[^;&|]*\bnext\s+dev(?:\s|$)/i.test(normalized)
      || consolePostcssWorkerMatches(normalized);
  }
  return false;
}

function groupHasVerifiedParentChain(leaderPid, members) {
  const byPid = new Map(members.map((member) => [member.pid, member]));
  if (!byPid.has(leaderPid)) return false;
  for (const member of members) {
    if (member.pid === leaderPid) continue;
    const seen = new Set();
    let current = member;
    while (current.pid !== leaderPid) {
      if (seen.has(current.pid)) return false;
      seen.add(current.pid);
      current = byPid.get(current.ppid);
      if (!current) return false;
    }
  }
  return true;
}

function childIdentityRecord(role, identity) {
  if (!identity || identity.pid !== identity.pgid || !leaderCommandMatchesRole(role, identity.command)) {
    throw new Error(`无法验证 ${role} 子进程的独立进程组身份。`);
  }
  return {
    role,
    pid: identity.pid,
    pgid: identity.pgid,
    started_at_os: identity.started_at_os,
    command_sha256: identity.command_sha256,
  };
}

const realRegistrationAdapter = {
  inspectProcess,
  processHasLaunchIdentity,
  processIsAlive,
  sleep: blockingSleep,
  now: Date.now,
};

function waitForChildIdentity(child, role, launchId, adapter = realRegistrationAdapter) {
  if (!child || !Number.isSafeInteger(child.pid) || child.pid < 2) {
    throw new Error(`无法取得 ${role} 子进程 PID。`);
  }
  const now = adapter.now ?? Date.now;
  const timeoutMs = 5_000;
  const deadline = now() + timeoutMs;
  let lastObservation = "process_not_visible";
  while (now() < deadline) {
    if (child.exitCode !== null || child.signalCode !== null || !adapter.processIsAlive(child.pid)) {
      throw new Error(`${role} 子进程在完成身份登记前已经退出。`);
    }
    const identity = adapter.inspectProcess(child.pid);
    if (!identity) {
      lastObservation = "process_not_visible";
      adapter.sleep(20);
      continue;
    }
    if (identity.pid !== child.pid || identity.pgid !== child.pid) {
      lastObservation = "process_group_not_ready";
      adapter.sleep(20);
      continue;
    }
    if (!leaderCommandMatchesRole(role, identity.command)) {
      lastObservation = "exec_not_ready";
      adapter.sleep(20);
      continue;
    }
    if (!adapter.processHasLaunchIdentity(child.pid, launchId, role)) {
      lastObservation = "launch_identity_not_visible";
      adapter.sleep(20);
      continue;
    }
    return identity;
  }
  throw new Error(`等待 ${role} 子进程完成 exec 和启动身份登记超时（${lastObservation}）。`);
}

function validateLockRecord(record, {allowTransient = false} = {}) {
  if (
    !record
    || ![LOCK_SCHEMA, LEGACY_LOCK_SCHEMA, OLDER_LOCK_SCHEMA].includes(record.schema)
    || record.project_root !== root
    || !/^[0-9a-f]{32}$/.test(String(record.launch_id))
    || !Number.isSafeInteger(record.launcher_pid)
    || record.launcher_pid < 2
    || typeof record.phase !== "string"
    || (record.schema !== OLDER_LOCK_SCHEMA && !/^[0-9a-f]{32}$/.test(String(record.control_token)))
    || (record.schema === LOCK_SCHEMA && (
      typeof record.launcher_started_at_os !== "string"
      || !/^[0-9a-f]{64}$/.test(String(record.launcher_command_sha256))
      || !/^[0-9a-f]{32}$/.test(String(record.handoff_token))
    ))
    || !Array.isArray(record.children)
    || record.children.length > 2
  ) {
    throw new Error("发现无法验证的旧 Console 锁；为避免误杀或重复启动，请先人工检查遗留进程。");
  }
  const roles = new Set();
  const groups = new Set();
  for (const child of record.children) {
    if (
      !child
      || !["backend", "console"].includes(child.role)
      || roles.has(child.role)
      || !Number.isSafeInteger(child.pid)
      || child.pid < 2
      || child.pgid !== child.pid
      || groups.has(child.pgid)
      || typeof child.started_at_os !== "string"
      || !/^[0-9a-f]{64}$/.test(String(child.command_sha256))
    ) {
      throw new Error("Console 锁中的子进程身份无效；本次不会发送任何信号。");
    }
    roles.add(child.role);
    groups.add(child.pgid);
  }
  if (!ALL_LAUNCH_PHASES.has(record.phase)) {
    throw new Error("Console 锁中的生命周期阶段无效；本次不会发送任何信号。");
  }
  if (!allowTransient && !STABLE_RECOVERY_PHASES.has(record.phase)) {
    throw new Error("上次 Console 在子进程登记窗口中异常退出；无法安全确定完整进程组，本次拒绝自动清理。");
  }
  if (record.phase === "backend_started" && (record.children.length !== 1 || !roles.has("backend"))) {
    throw new Error("Console 后端阶段记录不完整；本次不会发送任何信号。");
  }
  if (record.phase === "runtime_preflight" && record.children.length !== 0) {
    throw new Error("Console runtime 预检阶段不应存在子进程；本次不会发送任何信号。");
  }
  if (["children_started", "ready"].includes(record.phase) && (record.children.length !== 2 || !roles.has("backend") || !roles.has("console"))) {
    throw new Error("Console 就绪阶段记录不完整；本次不会发送任何信号。");
  }
  return record;
}

const realRecoveryAdapter = {
  inspectProcess,
  listProcessGroup,
  processHasLaunchIdentity,
  signalGroup(pgid, signal) {
    if (process.platform === "win32") throw new Error("Windows 上不支持安全的遗留 POSIX 进程组回收；本次拒绝启动。");
    try {
      process.kill(-pgid, signal);
    } catch (error) {
      if (!(error instanceof Error) || !Reflect.has(error, "code") || error.code !== "ESRCH") throw error;
    }
  },
  sleep: blockingSleep,
};

function inspectRecordedChild(record, child, adapter) {
  const leader = adapter.inspectProcess(child.pid);
  const members = adapter.listProcessGroup(child.pgid);
  if (!leader) {
    if (members.length) {
      throw new Error(`遗留 ${child.role} 组长已消失但进程组仍有成员；为避免 PID/PGID 复用误杀，本次拒绝清理。`);
    }
    return null;
  }
  if (
    leader.pid !== child.pid
    || leader.pgid !== child.pgid
    || leader.started_at_os !== child.started_at_os
    || leader.command_sha256 !== child.command_sha256
    || !leaderCommandMatchesRole(child.role, leader.command)
    || !adapter.processHasLaunchIdentity(child.pid, record.launch_id, child.role)
  ) {
    throw new Error(`遗留 ${child.role} PID 身份已变化；疑似 PID 复用，本次不会发送信号。`);
  }
  if (
    !members.some((member) => member.pid === leader.pid)
    || !groupHasVerifiedParentChain(leader.pid, members)
    || members.some((member) => member.pid !== leader.pid && (
      member.pgid !== child.pgid
      || !groupMemberMatchesRole(child.role, member.command)
    ))
  ) {
    throw new Error(`遗留 ${child.role} 进程组含有无法验证的成员；本次不会发送信号。`);
  }
  return {child, members};
}

function sameProcessGroupSnapshot(left, right) {
  if (left.length !== right.length) return false;
  const expected = new Map(left.map((member) => [member.pid, member]));
  return right.every((member) => {
    const original = expected.get(member.pid);
    return original
      && original.pgid === member.pgid
      && original.ppid === member.ppid
      && original.started_at_os === member.started_at_os
      && original.command_sha256 === member.command_sha256;
  });
}

function recoverStaleChildren(record, adapter = realRecoveryAdapter) {
  validateLockRecord(record);
  const validated = [];
  for (const child of record.children) {
    const inspected = inspectRecordedChild(record, child, adapter);
    if (inspected) validated.push(inspected);
  }
  // Validate every group a second time immediately before the first signal.
  // This narrows the unavoidable inspect/kill race and, importantly, prevents
  // signaling one group before discovering that another identity changed.
  const readyToSignal = [];
  for (const item of validated) {
    const current = inspectRecordedChild(record, item.child, adapter);
    if (!current) continue;
    if (!sameProcessGroupSnapshot(item.members, current.members)) {
      throw new Error("遗留进程组在清理前发生身份变化；本次不会发送任何信号。");
    }
    readyToSignal.push(current);
  }
  for (const item of readyToSignal) adapter.signalGroup(item.child.pgid, "SIGTERM");
  const deadline = Date.now() + 2_000;
  while (Date.now() < deadline) {
    let remaining = 0;
    for (const item of readyToSignal) {
      const current = adapter.listProcessGroup(item.child.pgid);
      if (!current.length) continue;
      remaining += current.length;
      const originals = new Map(item.members.map((member) => [member.pid, member]));
      if (current.some((member) => {
        const original = originals.get(member.pid);
        return !original || original.started_at_os !== member.started_at_os || original.command_sha256 !== member.command_sha256;
      })) {
        throw new Error("遗留进程组在关闭期间发生身份变化；本次拒绝继续操作。");
      }
    }
    if (!remaining) return;
    adapter.sleep(50);
  }
  throw new Error("遗留 Console 进程组未在 SIGTERM 后退出；为避免误杀，本次拒绝启动，请人工检查记录的 PID/PGID。");
}

function writeLockRecord(record, {initialDescriptor = null} = {}) {
  const encoded = `${JSON.stringify(record)}\n`;
  if (initialDescriptor !== null) {
    fs.writeFileSync(initialDescriptor, encoded);
    fs.fsyncSync(initialDescriptor);
    return;
  }
  const currentRecord = readLauncherRecord();
  if (currentRecord?.launcher_pid !== process.pid || currentRecord?.launch_id !== record.launch_id) {
    throw new Error("Console 单实例锁所有者已变化。");
  }
  const temporary = `${launcherLock}.${process.pid}.${Date.now()}.tmp`;
  const descriptor = fs.openSync(temporary, "wx", 0o600);
  try {
    fs.writeFileSync(descriptor, encoded);
    fs.fsyncSync(descriptor);
  } finally {
    fs.closeSync(descriptor);
  }
  try {
    fs.renameSync(temporary, launcherLock);
    const directory = fs.openSync(path.dirname(launcherLock), "r");
    try {
      fs.fsyncSync(directory);
    } finally {
      fs.closeSync(directory);
    }
  } catch (error) {
    try { fs.unlinkSync(temporary); } catch {}
    throw error;
  }
}

function fsyncDirectory(directoryPath) {
  const descriptor = fs.openSync(directoryPath, "r");
  try {
    fs.fsyncSync(descriptor);
  } finally {
    fs.closeSync(descriptor);
  }
}

function readJsonRecord(recordPath) {
  try {
    const parsed = JSON.parse(fs.readFileSync(recordPath, "utf8"));
    return parsed && typeof parsed === "object" && !Array.isArray(parsed) ? parsed : null;
  } catch {
    return null;
  }
}

function acquireLauncherAcquisitionGuard(lockPath = launcherLock, ownerAlive = processIsAlive) {
  const guardPath = `${lockPath}.acquire`;
  const token = randomBytes(16).toString("hex");
  let descriptor;
  try {
    descriptor = fs.openSync(guardPath, "wx", 0o600);
  } catch (error) {
    if (!(error instanceof Error) || !Reflect.has(error, "code") || error.code !== "EEXIST") throw error;
    const existing = readJsonRecord(guardPath);
    const owner = Number(existing?.owner_pid ?? 0);
    if (
      existing?.schema === ACQUISITION_LOCK_SCHEMA
      && existing?.project_root === root
      && Number.isSafeInteger(owner)
      && owner >= 2
      && ownerAlive(owner)
    ) {
      throw new Error(`另一个 TeachLab Console 启动器正在仲裁单实例锁（PID ${owner}）。`);
    }
    throw new Error("发现无法安全接管的 Console 启动仲裁锁；上次启动可能在锁回收中崩溃，请人工检查后再试。");
  }
  try {
    fs.writeFileSync(descriptor, `${JSON.stringify({
      schema: ACQUISITION_LOCK_SCHEMA,
      project_root: root,
      owner_pid: process.pid,
      token,
      started_at: new Date().toISOString(),
    })}\n`);
    fs.fsyncSync(descriptor);
  } catch (error) {
    fs.closeSync(descriptor);
    try { fs.unlinkSync(guardPath); } catch {}
    throw error;
  }
  fs.closeSync(descriptor);
  fsyncDirectory(path.dirname(guardPath));
  return () => {
    const current = readJsonRecord(guardPath);
    if (current?.owner_pid !== process.pid || current?.token !== token) {
      throw new Error("Console 启动仲裁锁所有者已变化；本次不会删除该锁。");
    }
    fs.unlinkSync(guardPath);
    fsyncDirectory(path.dirname(guardPath));
  };
}

function assertLockRecordUnchanged(expected, current) {
  if (
    !current
    || current.launch_id !== expected.launch_id
    || canonicalDigest(JSON.stringify(current)) !== canonicalDigest(JSON.stringify(expected))
  ) {
    throw new Error("Console 锁在遗留进程回收期间发生变化；本次不会删除该锁。");
  }
}

function updateLauncherRecord(phase, additions = {}) {
  launcherRecord = {...launcherRecord, ...additions, phase};
  writeLockRecord(launcherRecord);
}

function acquireLauncherLock() {
  fs.mkdirSync(path.dirname(launcherLock), {recursive: true, mode: 0o700});
  const lockDirectory = fs.lstatSync(path.dirname(launcherLock));
  if (!lockDirectory.isDirectory() || lockDirectory.isSymbolicLink()) {
    throw new Error("Console launcher lock 目录必须是本机普通目录，不能是符号链接。");
  }
  fs.chmodSync(path.dirname(launcherLock), 0o700);
  const releaseAcquisitionGuard = acquireLauncherAcquisitionGuard();
  try {
    for (let attempt = 0; attempt < 3; attempt += 1) {
      try {
        // Resolve our own identity before creating O_EXCL.  If `ps` is
        // temporarily unavailable, do not leave an empty lock that the next
        // click would have to treat as an unverifiable stale record.
        const currentLauncherIdentity = launcherProcessIdentity();
        const descriptor = fs.openSync(launcherLock, "wx", 0o600);
        try {
          launcherRecord = {
            schema: LOCK_SCHEMA,
            project_root: root,
            launch_id: randomBytes(16).toString("hex"),
            control_token: randomBytes(16).toString("hex"),
            launcher_pid: process.pid,
            ...currentLauncherIdentity,
            handoff_token: daemonHandoffToken,
            started_at: new Date().toISOString(),
            preferred_port: Number(process.env.TEACHLAB_CONSOLE_PORT || 3000),
            // This phase is recoverable because no external child effect can
            // occur before it is durably written by O_EXCL + fsync.
            phase: "runtime_preflight",
            children: [],
          };
          writeLockRecord(launcherRecord, {initialDescriptor: descriptor});
        } finally {
          fs.closeSync(descriptor);
        }
        return;
      } catch (error) {
        if (!(error instanceof Error) || !Reflect.has(error, "code") || error.code !== "EEXIST") throw error;
        const existingState = inspectLauncherLockFile();
        if (!existingState.valid) {
          throw new Error("发现无法验证的旧 Console 锁；本次不会删除锁或启动新进程组。");
        }
        const existing = existingState.record;
        const owner = Number(existing?.launcher_pid ?? existing?.pid ?? 0);
        if (existing && launcherRecordOwnerMatchesProcess(existing)) {
          throw new Error(`TeachLab Console 已在运行（launcher PID ${owner}）；本次不会重复启动。`);
        }
        if (existing && processIsAlive(owner)) {
          throw new Error("Console 单实例锁记录的 owner PID 仍存活但身份不匹配；本次拒绝启动。");
        }
        // Another launcher may have won O_EXCL but not yet written its owner
        // record.  Never unlink a fresh, temporarily empty lock: that race would
        // let two complete backend/Next process groups start at once.
        if (!owner && launcherLockIsInitializing()) {
          throw new Error("TeachLab Console 正在启动；本次不会重复创建进程组。");
        }
        if (!existingState.exists || !existing) {
          throw new Error("发现无法验证的旧 Console 锁；本次不会删除锁或启动新进程组。");
        }
        validateLockRecord(existing);
        recoverStaleChildren(existing);
        assertLockRecordUnchanged(existing, readLauncherRecord());
        try {
          fs.unlinkSync(launcherLock);
          fsyncDirectory(path.dirname(launcherLock));
        } catch (unlinkError) {
          if (!(unlinkError instanceof Error) || !Reflect.has(unlinkError, "code") || unlinkError.code !== "ENOENT") throw unlinkError;
        }
      }
    }
    throw new Error("无法取得 TeachLab Console 单实例锁；请检查 .private 目录权限。");
  } finally {
    releaseAcquisitionGuard();
  }
}

function releaseLauncherLock() {
  if (!launcherLockReleasePermitted) return;
  const current = readLauncherRecord();
  if (current?.launcher_pid !== process.pid || current?.launch_id !== launcherRecord?.launch_id) return;
  try {
    fs.unlinkSync(launcherLock);
  } catch (error) {
    if (!(error instanceof Error) || !Reflect.has(error, "code") || error.code !== "ENOENT") {
      console.error(`无法清理 Console 单实例锁：${error instanceof Error ? error.message : error}`);
    }
  }
}

function stopRequestPath(record) {
  return `${launcherLock}.stop.${record.control_token}`;
}

async function requestRunningLauncherStop() {
  const record = validateLockRecord(readLauncherRecord());
  if (!/^[0-9a-f]{32}$/.test(String(record.control_token))) {
    throw new Error("旧版 Console 不支持控制请求；请关闭其原终端窗口，再用新版入口启动。");
  }
  if (!processIsAlive(record.launcher_pid)) throw new Error("Console 启动器已不存在；请重新启动以执行安全遗留回收。");
  const requestPath = stopRequestPath(record);
  try {
    const descriptor = fs.openSync(requestPath, "wx", 0o600);
    try {
      fs.writeFileSync(descriptor, `${JSON.stringify({launch_id: record.launch_id, requested_at: new Date().toISOString()})}\n`);
      fs.fsyncSync(descriptor);
    } finally {
      fs.closeSync(descriptor);
    }
    fsyncDirectory(path.dirname(requestPath));
  } catch (error) {
    if (error?.code !== "EEXIST") throw error;
  }
  const deadline = Date.now() + 10_000;
  while (Date.now() < deadline) {
    if (!fs.existsSync(launcherLock)) return;
    await new Promise((resolve) => setTimeout(resolve, 100));
  }
  throw new Error("Console 未在 10 秒内完成关闭；锁已保留，后续启动会继续验证进程身份。");
}

async function runDaemonLauncher() {
  const logDirectory = process.env.TEACHLAB_LOG_DIR || path.join(root, ".private", "logs");
  fs.mkdirSync(logDirectory, {recursive: true, mode: 0o700});
  const logDirectoryStat = fs.lstatSync(logDirectory);
  if (!logDirectoryStat.isDirectory() || logDirectoryStat.isSymbolicLink()) {
    throw new Error("Console 日志目录必须是本机普通目录，不能是符号链接。");
  }
  fs.chmodSync(logDirectory, 0o700);
  const logPath = path.join(logDirectory, "teacher-agent-console.log");
  const configuredLogLimit = Number(process.env.TEACHLAB_LOG_MAX_BYTES || 10 * 1024 * 1024);
  if (!Number.isSafeInteger(configuredLogLimit) || configuredLogLimit < 1024 * 1024 || configuredLogLimit > 100 * 1024 * 1024) {
    throw new Error("TEACHLAB_LOG_MAX_BYTES 必须是 1–100 MiB 的整数。");
  }
  try {
    const stat = fs.lstatSync(logPath);
    if (!stat.isFile() || stat.isSymbolicLink() || (stat.mode & 0o077) !== 0) {
      throw new Error("Console 日志必须是 mode-0600 普通文件。");
    }
    if (stat.size >= configuredLogLimit) {
      const rotated = `${logPath}.1`;
      fs.rmSync(rotated, {force: true});
      fs.renameSync(logPath, rotated);
    }
  } catch (error) {
    if (error?.code !== "ENOENT") throw error;
  }
  const descriptor = fs.openSync(logPath, "a", 0o600);
  const existingState = inspectLauncherLockFile();
  if (!existingState.valid) {
    fs.closeSync(descriptor);
    throw new Error("发现无法验证的 Console 单实例锁；为避免误认旧端口或重复启动，请人工检查锁文件。");
  }
  const existing = existingState.record;
  const existingOwner = Number(existing?.launcher_pid ?? 0);
  if (existing) {
    // Validate structure and lifecycle phase before trusting the owner PID.
    // Otherwise a hand-written/malformed record whose PID happens to be this
    // process could make a stale port look like a healthy Console.
    try {
      validateLockRecord(existing, {allowTransient: true});
    } catch (error) {
      fs.closeSync(descriptor);
      throw error;
    }
  }
  if (existing && launcherRecordOwnerMatchesProcess(existing)) {
    if (existing.phase === "ready" && Number.isSafeInteger(existing.console_port)) {
      console.log(`TeachLab Console 已在运行（launcher PID ${existingOwner}）。`);
    } else {
      console.log(`TeachLab Console 正在启动（launcher PID ${existingOwner}）。`);
    }
    fs.closeSync(descriptor);
    return;
  }
  if (existing && processIsAlive(existingOwner)) {
    fs.closeSync(descriptor);
    throw new Error("Console 单实例锁记录的 owner PID 仍存活但身份不匹配；本次拒绝重复启动。");
  }
  if (existingState.exists && !existing) {
    fs.closeSync(descriptor);
    if (launcherLockIsInitializing()) {
      throw new Error("TeachLab Console 正在启动；本次不会重复创建进程组。");
    }
    throw new Error("发现无法验证的 Console 单实例锁；为避免误认旧端口或重复启动，请人工检查锁文件。");
  }
  const child = spawn(process.execPath, [launcherScript, python, apiKeyFile, npm], {
    cwd: root,
    detached: true,
    stdio: ["ignore", descriptor, descriptor],
    env: {...process.env, TEACHLAB_DAEMON_CHILD: "1", TEACHLAB_DAEMON_HANDOFF_TOKEN: daemonHandoffToken},
  });
  fs.closeSync(descriptor);
  let childExit = null;
  let childError = null;
  let childFailureObservedAt = null;
  child.once("exit", (code, signal) => {
    childExit = {code, signal};
    childFailureObservedAt ??= Date.now();
  });
  child.once("error", (error) => {
    childError = error;
    childFailureObservedAt ??= Date.now();
  });
  child.unref();
  const startupTimeoutMs = Number(process.env.TEACHLAB_DAEMON_STARTUP_TIMEOUT_MS || 10 * 60_000);
  if (!Number.isSafeInteger(startupTimeoutMs) || startupTimeoutMs < 60_000 || startupTimeoutMs > 30 * 60_000) {
    throw new Error("TEACHLAB_DAEMON_STARTUP_TIMEOUT_MS 必须是 60000–1800000 的整数。");
  }
  const deadline = Date.now() + startupTimeoutMs;
  while (Date.now() < deadline) {
    const record = readLauncherRecord();
    if (
      record?.launcher_pid === child.pid
      && record.handoff_token === daemonHandoffToken
      && record.phase === "ready"
      && Number.isSafeInteger(record.console_port)
      && record.console_port >= 1024
      && record.console_port <= 65535
    ) {
      console.log(`TeachLab Console 已在后台启动：http://127.0.0.1:${record.console_port}`);
      console.log(`日志：${logPath}`);
      console.log("关闭此终端窗口不会停止服务；运行 ./打开题目二教学Agent.command --stop 可停止。 ");
      return;
    }
    // Detached child process visibility can race with exec on macOS. Treat
    // the launcher's durable lock/readiness record as authoritative and keep
    // a bounded grace period before declaring a missing child failed.
    if (childFailureObservedAt && Date.now() - childFailureObservedAt > 15_000) {
      const owner = Number(record?.launcher_pid ?? 0);
      if (!record || owner !== child.pid || record.handoff_token !== daemonHandoffToken || !processIsAlive(owner)) {
        if (childError) {
          throw new Error(`后台 Console 启动失败，请检查日志：${logPath}`, {cause: childError});
        }
        throw new Error(
          `后台 Console 启动失败（${childExit?.code ?? childExit?.signal ?? "unknown"}），请检查日志：${logPath}`,
        );
      }
    }
    await new Promise((resolve) => setTimeout(resolve, 250));
  }
  try {
    if (process.platform !== "win32") process.kill(-child.pid, "SIGTERM");
    else child.kill("SIGTERM");
  } catch {}
  const stopDeadline = Date.now() + 3_000;
  while (processIsAlive(child.pid) && Date.now() < stopDeadline) {
    await new Promise((resolve) => setTimeout(resolve, 50));
  }
  if (processIsAlive(child.pid)) {
    try {
      if (process.platform !== "win32") process.kill(-child.pid, "SIGKILL");
      else child.kill("SIGKILL");
    } catch {}
  }
  throw new Error(`后台 Console 启动超时，已停止本次受控进程组；请检查日志：${logPath}`);
}

const CHILD_TREE_GONE = "gone";
const CHILD_TREE_RUNNING = "running";
const CHILD_TREE_UNKNOWN = "unknown";

const realShutdownAdapter = {
  platform: process.platform,
  killGroup(pgid, signal) {
    process.kill(-pgid, signal);
  },
};

function childIsRunning(child) {
  return child.exitCode === null && child.signalCode === null;
}

function childTreeState(child, adapter = realShutdownAdapter) {
  if (adapter.platform === "win32" || !child.pid) {
    return childIsRunning(child) ? CHILD_TREE_RUNNING : CHILD_TREE_GONE;
  }
  try {
    adapter.killGroup(child.pid, 0);
    return CHILD_TREE_RUNNING;
  } catch (error) {
    const code = error instanceof Error && Reflect.has(error, "code") ? error.code : null;
    if (code === "ESRCH") return CHILD_TREE_GONE;
    // EPERM proves neither absence nor ownership. Other probe failures are
    // equally non-authoritative, so shutdown must retain the lock.
    return CHILD_TREE_UNKNOWN;
  }
}

function signalChildTree(child, signal, adapter = realShutdownAdapter) {
  try {
    if (adapter.platform !== "win32" && child.pid) {
      adapter.killGroup(child.pid, signal);
      return CHILD_TREE_RUNNING;
    }
    if (childIsRunning(child)) {
      child.kill(signal);
      return CHILD_TREE_RUNNING;
    }
    return CHILD_TREE_GONE;
  } catch (error) {
    const code = error instanceof Error && Reflect.has(error, "code") ? error.code : null;
    if (code === "ESRCH") return CHILD_TREE_GONE;
    return CHILD_TREE_UNKNOWN;
  }
}

function shutdownDisposition(states, {force = false} = {}) {
  if (states.every((state) => state === CHILD_TREE_GONE)) return "release";
  return force ? "retain" : "wait";
}

async function runLifecycleSelfTest() {
  const backendIdentity = {
    pid: 4101,
    ppid: 4000,
    pgid: 4101,
    started_at_os: "Wed Aug 12 01:00:00 2026",
    command: "/opt/venv/bin/python -m teaching_skill_miner teacher-agent-dashboard --port 0",
  };
  backendIdentity.command_sha256 = canonicalDigest(backendIdentity.command);
  const consoleIdentity = {
    pid: 4102,
    ppid: 4000,
    pgid: 4102,
    started_at_os: "Wed Aug 12 01:00:01 2026",
    command: `/usr/local/bin/node ${root}/scripts/start_teacher_agent_console.mjs --console-supervisor npm 3030`,
  };
  consoleIdentity.command_sha256 = canonicalDigest(consoleIdentity.command);
  const record = {
    schema: LOCK_SCHEMA,
    project_root: root,
    launch_id: "0123456789abcdef0123456789abcdef",
    control_token: "fedcba9876543210fedcba9876543210",
    launcher_pid: 4000,
    launcher_started_at_os: "Wed Aug 12 00:59:59 2026",
    launcher_command_sha256: canonicalDigest("/usr/local/bin/node scripts/start_teacher_agent_console.mjs --daemon"),
    handoff_token: "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
    phase: "ready",
    children: [
      childIdentityRecord("backend", backendIdentity),
      childIdentityRecord("console", consoleIdentity),
    ],
  };

  const adapterFor = (identities, groupMembers) => {
    const signals = [];
    const groups = new Map(groupMembers);
    return {
      signals,
      inspectProcess(pid) {
        return identities.get(pid) ?? null;
      },
      listProcessGroup(pgid) {
        return groups.get(pgid) ?? [];
      },
      processHasLaunchIdentity(pid, launchId, role) {
        return launchId === record.launch_id
          && ((pid === 4101 && role === "backend") || (pid === 4102 && role === "console"));
      },
      signalGroup(pgid, signal) {
        signals.push({pgid, signal});
        groups.delete(pgid);
      },
      sleep() {},
    };
  };

  const consoleNpm = {
    pid: 4110,
    ppid: 4102,
    pgid: 4102,
    started_at_os: "Wed Aug 12 01:00:02 2026",
    command: "/opt/homebrew/bin/node /opt/homebrew/lib/node_modules/npm/bin/npm-cli.js run dev -- --port 3030",
  };
  consoleNpm.command_sha256 = canonicalDigest(consoleNpm.command);
  const consoleNext = {
    pid: 4111,
    ppid: 4110,
    pgid: 4102,
    started_at_os: "Wed Aug 12 01:00:03 2026",
    command: "/opt/homebrew/bin/node /workspace/node_modules/next/dist/bin/next dev --port 3030",
  };
  consoleNext.command_sha256 = canonicalDigest(consoleNext.command);
  const consoleServer = {
    pid: 4112,
    ppid: 4111,
    pgid: 4102,
    started_at_os: "Wed Aug 12 01:00:04 2026",
    // Keep the lifecycle self-test hermetic.  Python-only CI jobs intentionally
    // do not install Console dependencies, and this synthetic process identity
    // does not need the locally installed Next.js version to exercise recovery.
    command: "next-server (v0.0.0-lifecycle-self-test)",
  };
  consoleServer.command_sha256 = canonicalDigest(consoleServer.command);

  const validAdapter = adapterFor(
    new Map([[4101, backendIdentity], [4102, consoleIdentity]]),
    new Map([
      [4101, [backendIdentity]],
      [4102, [consoleIdentity, consoleNpm, consoleNext, consoleServer]],
    ]),
  );
  recoverStaleChildren(record, validAdapter);
  assert.deepEqual(validAdapter.signals, [
    {pgid: 4101, signal: "SIGTERM"},
    {pgid: 4102, signal: "SIGTERM"},
  ]);

  const reusedBackend = {...backendIdentity, started_at_os: "Wed Aug 12 02:00:00 2026"};
  const reusedAdapter = adapterFor(
    new Map([[4101, reusedBackend], [4102, consoleIdentity]]),
    new Map([[4101, [reusedBackend]], [4102, [consoleIdentity]]]),
  );
  assert.throws(() => recoverStaleChildren(record, reusedAdapter), /PID 身份已变化/);
  assert.deepEqual(reusedAdapter.signals, []);

  const missingNonceAdapter = adapterFor(
    new Map([[4101, backendIdentity], [4102, consoleIdentity]]),
    new Map([[4101, [backendIdentity]], [4102, [consoleIdentity]]]),
  );
  missingNonceAdapter.processHasLaunchIdentity = () => false;
  assert.throws(() => recoverStaleChildren(record, missingNonceAdapter), /PID 身份已变化/);
  assert.deepEqual(missingNonceAdapter.signals, []);

  const foreignMember = {
    pid: 4199,
    ppid: 4101,
    pgid: 4101,
    started_at_os: "Wed Aug 12 01:00:02 2026",
    command: "/usr/bin/sleep 999",
  };
  foreignMember.command_sha256 = canonicalDigest(foreignMember.command);
  const foreignAdapter = adapterFor(
    new Map([[4101, backendIdentity], [4102, consoleIdentity]]),
    new Map([[4101, [backendIdentity, foreignMember]], [4102, [consoleIdentity]]]),
  );
  assert.throws(() => recoverStaleChildren(record, foreignAdapter), /无法验证的成员/);
  assert.deepEqual(foreignAdapter.signals, []);

  const detachedNext = {...consoleNext, ppid: 4999};
  const brokenChainAdapter = adapterFor(
    new Map([[4101, backendIdentity], [4102, consoleIdentity]]),
    new Map([
      [4101, [backendIdentity]],
      [4102, [consoleIdentity, consoleNpm, detachedNext, consoleServer]],
    ]),
  );
  assert.throws(() => recoverStaleChildren(record, brokenChainAdapter), /无法验证的成员/);
  assert.deepEqual(brokenChainAdapter.signals, []);

  assert.throws(
    () => validateLockRecord({...record, phase: "spawning_console"}),
    /登记窗口中异常退出/,
  );
  assert.doesNotThrow(() => validateLockRecord({...record, phase: "runtime_preflight", children: []}));
  assert.throws(
    () => validateLockRecord({...record, phase: "runtime_preflight"}),
    /预检阶段不应存在子进程/,
  );
  assert.throws(
    () => validateLockRecord({...record, project_root: "/tmp/not-this-project"}),
    /无法验证的旧 Console 锁/,
  );
  assert.equal(
    launcherRecordCoversChildren(record, [{pid: 4101}, {pid: 4102}]),
    true,
  );
  assert.equal(
    launcherRecordCoversChildren({...record, phase: "spawning_console"}, [{pid: 4101}, {pid: 4102}]),
    false,
  );
  assert.equal(
    launcherRecordCoversChildren(
      {...record, phase: "stopping", children: [record.children[0]]},
      [{pid: 4101}, {pid: 4102}],
    ),
    false,
  );
  const parsed = parseProcessIdentityLine(
    " 4101 4000 4101 Wed Aug 12 01:00:00 2026     /opt/python -m teaching_skill_miner teacher-agent-dashboard",
  );
  assert.equal(parsed?.pid, 4101);
  assert.equal(parsed?.ppid, 4000);
  assert.equal(parsed?.pgid, 4101);

  let registrationInspections = 0;
  const registered = waitForChildIdentity(
    {pid: 4101, exitCode: null, signalCode: null},
    "backend",
    record.launch_id,
    {
      processIsAlive: () => true,
      inspectProcess() {
        registrationInspections += 1;
        if (registrationInspections === 1) return null;
        if (registrationInspections === 2) {
          const preExec = {...backendIdentity, command: process.execPath};
          return {...preExec, command_sha256: canonicalDigest(preExec.command)};
        }
        return backendIdentity;
      },
      processHasLaunchIdentity: () => true,
      sleep() {},
    },
  );
  assert.equal(registered.command_sha256, backendIdentity.command_sha256);
  assert.equal(registrationInspections, 3);

  let simulatedNow = 0;
  assert.throws(
    () => waitForChildIdentity(
      {pid: 4101, exitCode: null, signalCode: null},
      "backend",
      record.launch_id,
      {
        processIsAlive: () => true,
        inspectProcess: () => null,
        processHasLaunchIdentity: () => false,
        now: () => simulatedNow,
        sleep(milliseconds) { simulatedNow += milliseconds; },
      },
    ),
    /process_not_visible/,
  );

  const permissionDenied = Object.assign(new Error("permission denied"), {code: "EPERM"});
  const epermShutdownAdapter = {
    platform: "darwin",
    killGroup() { throw permissionDenied; },
  };
  const simulatedChild = {pid: 4101, exitCode: null, signalCode: null};
  assert.equal(childTreeState(simulatedChild, epermShutdownAdapter), CHILD_TREE_UNKNOWN);
  assert.equal(signalChildTree(simulatedChild, "SIGTERM", epermShutdownAdapter), CHILD_TREE_UNKNOWN);
  assert.equal(shutdownDisposition([CHILD_TREE_UNKNOWN]), "wait");
  assert.equal(shutdownDisposition([CHILD_TREE_UNKNOWN], {force: true}), "retain");
  assert.equal(shutdownDisposition([CHILD_TREE_GONE], {force: true}), "release");

  const testDirectory = fs.mkdtempSync(path.join(os.tmpdir(), "teachlab-launcher-guard-"));
  const testLock = path.join(testDirectory, "launcher.lock");
  let releaseFirstGuard = null;
  let releaseSecondGuard = null;
  try {
    releaseFirstGuard = acquireLauncherAcquisitionGuard(testLock);
    assert.throws(
      () => acquireLauncherAcquisitionGuard(testLock),
      /正在仲裁单实例锁/,
    );
    releaseFirstGuard();
    releaseFirstGuard = null;
    releaseSecondGuard = acquireLauncherAcquisitionGuard(testLock);
    assert.throws(
      () => assertLockRecordUnchanged(record, {...record, launch_id: "fedcba9876543210fedcba9876543210"}),
      /回收期间发生变化/,
    );
  } finally {
    if (releaseFirstGuard) releaseFirstGuard();
    if (releaseSecondGuard) releaseSecondGuard();
    fs.rmdirSync(testDirectory);
  }

  if (process.platform !== "win32") {
    const probeLaunchId = randomBytes(16).toString("hex");
    const probe = spawn(
      process.execPath,
      ["-e", "setInterval(() => undefined, 1000)"],
      {
        detached: true,
        stdio: "ignore",
        env: {
          ...process.env,
          TEACHLAB_LAUNCH_ID: probeLaunchId,
          TEACHLAB_LAUNCH_ROLE: "lifecycle_probe",
        },
      },
    );
    probe.unref();
    const probeExit = new Promise((resolve, reject) => {
      probe.once("exit", resolve);
      probe.once("error", reject);
    });
    try {
      let observed = null;
      const deadline = Date.now() + 1_000;
      while (!observed && Date.now() < deadline) {
        observed = inspectProcess(probe.pid);
        if (!observed) Atomics.wait(new Int32Array(new SharedArrayBuffer(4)), 0, 0, 20);
      }
      assert.equal(observed?.pid, probe.pid);
      assert.equal(observed?.pgid, probe.pid);
      assert.equal(processHasLaunchIdentity(probe.pid, probeLaunchId, "lifecycle_probe"), true);
      assert.equal(processHasLaunchIdentity(probe.pid, probeLaunchId, "console"), false);
      assert.equal(childTreeState(probe), CHILD_TREE_RUNNING);
      assert.equal(signalChildTree(probe, "SIGTERM"), CHILD_TREE_RUNNING);
      await Promise.race([
        probeExit,
        new Promise((_, reject) => setTimeout(
          () => reject(new Error("lifecycle probe did not exit after SIGTERM")),
          1_000,
        )),
      ]);
      assert.equal(childTreeState(probe), CHILD_TREE_GONE);
    } finally {
      if (childTreeState(probe) !== CHILD_TREE_GONE) {
        try { process.kill(-probe.pid, "SIGTERM"); } catch {}
      }
    }

    const supervisorDirectory = fs.mkdtempSync(path.join(os.tmpdir(), "teachlab-console-runtime-"));
    const fakeServer = path.join(supervisorDirectory, "server.js");
    fs.writeFileSync(
      fakeServer,
      "process.on('SIGTERM', () => process.exit(0));\n"
      + "process.on('SIGINT', () => process.exit(0));\n"
      + "setInterval(() => undefined, 1000);\n",
      {mode: 0o600},
    );
    const supervisorLaunchId = randomBytes(16).toString("hex");
    const supervisor = spawn(
      process.execPath,
      [launcherScript, "--console-supervisor", supervisorDirectory, "39001"],
      {
        cwd: root,
        detached: true,
        stdio: "ignore",
        env: {
          ...process.env,
          TEACHLAB_LAUNCH_ID: supervisorLaunchId,
          TEACHLAB_LAUNCH_ROLE: "console",
        },
      },
    );
    supervisor.unref();
    const supervisorExit = new Promise((resolve, reject) => {
      supervisor.once("exit", resolve);
      supervisor.once("error", reject);
    });
    try {
      let supervisorIdentity = null;
      let supervisorMembers = [];
      const supervisorDeadline = Date.now() + 2_000;
      while (Date.now() < supervisorDeadline) {
        supervisorIdentity = inspectProcess(supervisor.pid);
        supervisorMembers = listProcessGroup(supervisor.pid);
        if (
          supervisorIdentity
          && supervisorMembers.length >= 2
        ) break;
        await new Promise((resolve) => setTimeout(resolve, 20));
      }
      const supervisorChildRecord = childIdentityRecord("console", supervisorIdentity);
      const inspectedSupervisor = inspectRecordedChild(
        {launch_id: supervisorLaunchId},
        supervisorChildRecord,
        realRecoveryAdapter,
      );
      assert.ok(inspectedSupervisor);
      const serverMember = supervisorMembers.find((member) => member.pid !== supervisor.pid);
      assert.ok(serverMember);
      assert.equal(serverMember.ppid, supervisor.pid);
      assert.equal(processHasLaunchIdentity(serverMember.pid, supervisorLaunchId, "console"), false);
      assert.equal(
        consolePostcssWorkerMatches(`node /tmp/other-project/.next/postcss.js 53152`),
        false,
      );
      assert.equal(
        consolePostcssWorkerMatches(`node ${path.join(root, "apps", "console", ".next", "postcss.js")} ../../bad`),
        false,
      );
      assert.equal(signalChildTree(supervisor, "SIGTERM"), CHILD_TREE_RUNNING);
      await Promise.race([
        supervisorExit,
        new Promise((_, reject) => setTimeout(
          () => reject(new Error("console supervisor did not exit after SIGTERM")),
          2_000,
        )),
      ]);
      const groupExitDeadline = Date.now() + 1_000;
      while (childTreeState(supervisor) !== CHILD_TREE_GONE && Date.now() < groupExitDeadline) {
        await new Promise((resolve) => setTimeout(resolve, 20));
      }
      assert.equal(childTreeState(supervisor), CHILD_TREE_GONE);
    } finally {
      if (childTreeState(supervisor) !== CHILD_TREE_GONE) {
        try { process.kill(-supervisor.pid, "SIGKILL"); } catch {}
      }
      try { fs.unlinkSync(fakeServer); } catch {}
      try { fs.rmdirSync(supervisorDirectory); } catch {}
    }
  }
}

if (lifecycleSelfTest) {
  await runLifecycleSelfTest();
  console.log("TeachLab Console lifecycle self-test passed.");
  process.exit(0);
}

if (runtimeSelfTest) {
  try {
    const runtime = await ensureProductionRuntime();
    console.log(JSON.stringify({status: "verified", release_id: runtime.manifest.release_id}));
    process.exit(0);
  } catch (error) {
    console.error(error instanceof Error ? error.message : error);
    process.exit(1);
  }
}

if (stopMode) {
  try { await requestRunningLauncherStop(); console.log("TeachLab Console 已停止。"); process.exit(0); }
  catch (error) { console.error(error instanceof Error ? error.message : error); process.exit(1); }
}

if (daemonMode) {
  try { await runDaemonLauncher(); process.exit(0); }
  catch (error) { console.error(error instanceof Error ? error.message : error); process.exit(1); }
}

try {
  acquireLauncherLock();
} catch (error) {
  console.error(error instanceof Error ? error.message : error);
  process.exit(1);
}
process.on("exit", releaseLauncherLock);
process.on("exit", cleanupMaterializedSecrets);

function launcherRecordCoversChildren(record, childProcesses) {
  if (!record || !STABLE_RECOVERY_PHASES.has(record.phase) || !Array.isArray(record.children)) return false;
  const recordedPids = new Set(record.children.map((child) => child.pid));
  return record.children.length === childProcesses.length
    && childProcesses.every((child) => Number.isSafeInteger(child.pid) && recordedPids.has(child.pid));
}

function clearShutdownTimers() {
  if (shutdownTimer) clearTimeout(shutdownTimer);
  if (shutdownFinalizer) clearTimeout(shutdownFinalizer);
  shutdownTimer = null;
  shutdownFinalizer = null;
}

function maybeFinishShutdown(exitCode, {force = false} = {}) {
  const states = children.map((child) => childTreeState(child));
  const disposition = shutdownDisposition(states, {force});
  if (disposition === "release") {
    launcherLockReleasePermitted = true;
    clearShutdownTimers();
    process.exit(exitCode);
  }
  if (disposition === "wait") return;
  launcherLockReleasePermitted = false;
  clearShutdownTimers();
  const unknown = states.filter((state) => state === CHILD_TREE_UNKNOWN).length;
  const running = states.filter((state) => state === CHILD_TREE_RUNNING).length;
  console.error(
    `无法证明全部 Console 子进程组已回收（running=${running}, unknown=${unknown}）；`
    + "单实例锁将保留，后续启动会先验证遗留进程。",
  );
  process.exit(exitCode);
}

function terminateChildren(exitCode = 0) {
  if (shuttingDown) return;
  shuttingDown = true;
  process.exitCode = exitCode;
  if (children.length) launcherLockReleasePermitted = false;
  // Never convert an unstable spawning_* record into a recoverable "stopping"
  // record. A crash between spawn and identity registration could otherwise
  // hide an unrecorded orphan and let the next launcher start a duplicate.
  if (launcherRecordCoversChildren(launcherRecord, children)) {
    try {
      updateLauncherRecord("stopping");
    } catch (error) {
      console.error(`无法记录 Console 停止阶段：${error instanceof Error ? error.message : error}`);
    }
  }
  for (const child of children) signalChildTree(child, "SIGTERM");
  for (const child of children) child.once("exit", () => maybeFinishShutdown(exitCode));
  shutdownTimer = setTimeout(() => {
    for (const child of children) {
      if (childTreeState(child) !== CHILD_TREE_GONE) signalChildTree(child, "SIGKILL");
    }
    shutdownFinalizer = setTimeout(
      () => maybeFinishShutdown(exitCode, {force: true}),
      250,
    );
    maybeFinishShutdown(exitCode);
  }, 1500);
  maybeFinishShutdown(exitCode);
}

process.on("SIGINT", () => terminateChildren(130));
process.on("SIGTERM", () => terminateChildren(143));
const stopRequestTimer = setInterval(() => {
  if (!launcherRecord?.control_token || shuttingDown) return;
  const requestPath = stopRequestPath(launcherRecord);
  if (!fs.existsSync(requestPath)) return;
  try { fs.unlinkSync(requestPath); } catch {}
  terminateChildren(0);
}, 500);
stopRequestTimer.unref();
if (Number.isFinite(idleShutdownMinutes) && idleShutdownMinutes > 0) {
  const idleTimer = setInterval(() => {
    if (shuttingDown) return;
    try {
      if (Date.now() - fs.statSync(runtimeActivityFile).mtimeMs >= idleShutdownMinutes * 60_000) {
        console.log(`Console 已空闲 ${idleShutdownMinutes} 分钟，按显式 idle policy 安全停止。`);
        terminateChildren(0);
      }
    } catch {}
  }, Math.min(60_000, Math.max(1_000, idleShutdownMinutes * 10_000)));
  idleTimer.unref();
}

function freePort(port) {
  return new Promise((resolve) => {
    const server = net.createServer();
    server.once("error", () => resolve(false));
    server.listen(port, "127.0.0.1", () => server.close(() => resolve(true)));
  });
}

async function chooseConsolePort() {
  const preferred = Number(process.env.TEACHLAB_CONSOLE_PORT || 3000);
  for (let port = preferred; port < preferred + 20; port += 1) {
    if (await freePort(port)) return port;
  }
  throw new Error(
    `没有可用的 Console 端口（已检查 ${preferred}-${preferred + 19}）`,
  );
}

function waitForCapability(child) {
  return new Promise((resolve, reject) => {
    let output = "";
    let settled = false;
    const cleanup = () => {
      clearTimeout(timeout);
      child.stdout?.off("data", onData);
      child.off("error", onError);
      child.off("exit", onExit);
    };
    const finish = (callback, value) => {
      if (settled) return;
      settled = true;
      cleanup();
      callback(value);
    };
    const onData = (chunk) => {
      output += chunk.toString();
      if (Buffer.byteLength(output, "utf8") > 64 * 1024) {
        finish(reject, new Error("Teaching Agent 后端在 capability 就绪前输出超过 64 KiB。"));
        return;
      }
      const match = output.match(new RegExp('"dashboard_url"\\s*:\\s*"(http://127\\.0\\.0\\.1:[^"]+)"'));
      if (!match) return;
      finish(resolve, match[1].replaceAll("\\/", "/"));
    };
    const onError = (error) => finish(reject, error);
    const onExit = (code, signal) => finish(
      reject,
      new Error(`Teaching Agent 后端在就绪前退出（${code ?? signal ?? "unknown"}）`),
    );
    const timeout = setTimeout(
      () => finish(reject, new Error("Teaching Agent 后端启动超时")),
      60000,
    );
    child.stdout?.on("data", onData);
    child.once("error", onError);
    child.once("exit", onExit);
  });
}

async function waitForConsole(url) {
  const deadline = Date.now() + 60000;
  while (Date.now() < deadline) {
    try {
      const health = await fetch(`${url}/health`, {
        cache: "no-store",
        redirect: "error",
        signal: AbortSignal.timeout(2_000),
      });
      if (!health.ok) throw new Error(`health returned ${health.status}`);
      const ready = await fetch(`${url}/ready`, {
        cache: "no-store",
        redirect: "error",
        signal: AbortSignal.timeout(3_000),
      });
      if (ready.status === 200) {
        const payload = await ready.json();
        if (payload?.status === "ready" && payload?.backend === "ready") return;
      }
    } catch {
      // Next is still compiling.
    }
    await new Promise((resolve) => setTimeout(resolve, 300));
  }
  throw new Error("Console /ready 未在超时前返回严格 200 ready；503 不会被当作就绪");
}

try {
  validateRuntimeConfiguration();
  const productionRuntime = await ensureProductionRuntime();
  fs.mkdirSync(path.dirname(runtimeActivityFile), {recursive: true, mode: 0o700});
  try {
    const activityStat = fs.lstatSync(runtimeActivityFile);
    if (!activityStat.isFile() || activityStat.isSymbolicLink() || (activityStat.mode & 0o077) !== 0) {
      throw new Error("Console activity 记录必须是 mode-0600 普通文件。");
    }
  } catch (error) {
    if (error?.code !== "ENOENT") throw error;
    fs.closeSync(fs.openSync(runtimeActivityFile, "wx", 0o600));
  }
  const now = new Date();
  fs.utimesSync(runtimeActivityFile, now, now);
  const resolvedApiKeyFile = resolveSecretFile({
    fallback: apiKeyFile,
    label: "DeepSeek API 密钥",
    service: process.env.TEACHLAB_API_KEYCHAIN_SERVICE || "TeachLab DeepSeek API Key",
    allowShort: true,
    allowSymlinkFallback: true,
  });
  const resolvedLearnerKeyFile = resolveSecretFile({
    fallback: learnerKeySecretFile,
    label: "学习记录密钥",
    service: process.env.TEACHLAB_LEARNER_KEYCHAIN_SERVICE || "TeachLab Learner Key",
    generateFallback: true,
  });
  const resolvedConsentKeyFile = resolveSecretFile({
    fallback: consentSigningSecretFile,
    label: "远程同意签名密钥",
    service: process.env.TEACHLAB_CONSENT_KEYCHAIN_SERVICE || "TeachLab Consent Signing Key",
    generateFallback: true,
  });
  updateLauncherRecord("spawning_backend");
  const backend = spawn(python, [
    "-m", "teaching_skill_miner", "teacher-agent-dashboard",
    "--port", "0", "--no-browser",
    "--agent-backend", "deepseek",
    "--model", process.env.TEACHLAB_DEEPSEEK_MODEL || "deepseek-v4-flash",
    "--api-key-file", resolvedApiKeyFile,
    "--allow-remote-student-data",
    "--session-store", sessionStore,
    "--syllabus-store", syllabusStore,
    "--project-store", projectStore,
    "--resource-index-store", resourceIndexStore,
    "--resource-review-store", resourceReviewStore,
    "--learning-record-store", learningRecordStore,
    "--metacognition-store", metacognitionStore,
    "--adjudication-store", adjudicationStore,
    "--consent-store", consentStore,
    "--consent-signing-secret-file", resolvedConsentKeyFile,
    "--remote-processing-region", process.env.TEACHLAB_REMOTE_PROCESSING_REGION || "provider_managed",
    "--remote-provider-retention-days", process.env.TEACHLAB_REMOTE_PROVIDER_RETENTION_DAYS || "30",
    "--learner-key-secret-file", resolvedLearnerKeyFile,
    "--learner-tenant-id", learnerTenantId,
  ], {
    cwd: root,
    env: {
      ...process.env,
      TEACHLAB_LAUNCH_ID: launcherRecord.launch_id,
      TEACHLAB_LAUNCH_ROLE: "backend",
    },
    stdio: ["ignore", "pipe", "pipe"],
    detached: process.platform !== "win32",
  });
  children.push(backend);
  launcherLockReleasePermitted = false;
  const backendIdentity = waitForChildIdentity(backend, "backend", launcherRecord.launch_id);
  updateLauncherRecord("backend_started", {
    children: [childIdentityRecord("backend", backendIdentity)],
  });
  backend.stderr?.on("data", (chunk) => process.stderr.write(`[Teaching Agent] ${chunk}`));
  backend.once("exit", (code) => {
    if (!shuttingDown) terminateChildren(code ?? 1);
  });

  const capabilityUrl = await waitForCapability(backend);
  // The readiness listener is removed once the capability URL is parsed.
  // Keep draining stdout afterwards so a verbose backend can never fill its
  // pipe and stall while the Console is running for a long time.
  backend.stdout?.resume();
  const port = await chooseConsolePort();
  const consoleUrl = `http://127.0.0.1:${port}`;
  updateLauncherRecord("spawning_console", {console_port: port});
  const consoleProcess = spawn(process.execPath, [launcherScript, "--console-supervisor", productionRuntime.directory, String(port)], {
    cwd: root,
    env: {
      ...process.env,
      TEACHLAB_HARNESS_MODE: "local_python",
      TEACHER_AGENT_CAPABILITY_URL: capabilityUrl,
      // Share one ephemeral signing key across all Next route bundles. It is
      // never exposed through NEXT_PUBLIC_* and dies with this Console launch.
      TEACHLAB_LOCAL_SECURITY_SECRET: randomBytes(32).toString("hex"),
      NEXT_TELEMETRY_DISABLED: "1",
      TEACHLAB_RELEASE_VERSION: productionRuntime.manifest.version,
      TEACHLAB_RELEASE_ID: productionRuntime.manifest.release_id,
      TEACHLAB_RUNTIME_ACTIVITY_FILE: runtimeActivityFile,
      TEACHLAB_RUNTIME_STOP_REQUEST: stopRequestPath(launcherRecord),
      TEACHLAB_LAUNCH_ID: launcherRecord.launch_id,
      TEACHLAB_LAUNCH_ROLE: "console",
    },
    stdio: "inherit",
    detached: process.platform !== "win32",
  });
  children.push(consoleProcess);
  launcherLockReleasePermitted = false;
  const consoleIdentity = waitForChildIdentity(consoleProcess, "console", launcherRecord.launch_id);
  updateLauncherRecord("children_started", {
    children: [
      ...launcherRecord.children,
      childIdentityRecord("console", consoleIdentity),
    ],
  });
  consoleProcess.once("exit", (code) => {
    if (!shuttingDown) terminateChildren(code ?? 1);
  });
  await waitForConsole(consoleUrl);
  updateLauncherRecord("ready");
  console.log(`\nTeachLab Console 已连接真实 Teaching Agent：${consoleUrl}`);
  console.log("后端 capability URL 仅由 Next 服务端持有；浏览器不会看到访问令牌。\n");
  if (process.platform === "darwin" && process.env.TEACHLAB_OPEN_BROWSER === "1") {
    spawn("open", [consoleUrl], {stdio: "ignore", detached: true}).unref();
  }
  await new Promise(() => undefined);
} catch (error) {
  console.error(error instanceof Error ? error.message : error);
  terminateChildren(1);
}
