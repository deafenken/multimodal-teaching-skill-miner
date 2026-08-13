#!/usr/bin/env node

import {spawnSync} from "node:child_process";
import {createHash, randomBytes} from "node:crypto";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import process from "node:process";
import {fileURLToPath} from "node:url";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const app = path.join(root, "apps", "console");
const stateRoot = process.env.TEACHLAB_CONSOLE_RUNTIME_ROOT
  || path.join(root, ".private", "console-runtime");
const releases = path.join(stateRoot, "releases");
const channelPath = path.join(stateRoot, "channel.json");
const lockPath = path.join(stateRoot, "build.lock");
const dryRun = process.argv.includes("--dry-run");
const selfTest = process.argv.includes("--self-test");

function sha256(bytes) {
  return createHash("sha256").update(bytes).digest("hex");
}

function atomicJson(target, value) {
  fs.mkdirSync(path.dirname(target), {recursive: true, mode: 0o700});
  const temporary = `${target}.${process.pid}.tmp`;
  const descriptor = fs.openSync(temporary, "wx", 0o600);
  try {
    fs.writeFileSync(descriptor, `${JSON.stringify(value, null, 2)}\n`);
    fs.fsyncSync(descriptor);
  } finally {
    fs.closeSync(descriptor);
  }
  fs.renameSync(temporary, target);
  const directory = fs.openSync(path.dirname(target), "r");
  try { fs.fsyncSync(directory); } finally { fs.closeSync(directory); }
}

function sourceFiles() {
  const roots = ["app", "components", "lib", "public"];
  const result = [
    "instrumentation.ts",
    "middleware.ts",
    "next-env.d.ts",
    "next.config.ts",
    "package.json",
    "package-lock.json",
    "postcss.config.mjs",
    "tsconfig.json",
  ];
  const visit = (relative) => {
    const absolute = path.join(app, relative);
    if (!fs.existsSync(absolute)) return;
    const stat = fs.lstatSync(absolute);
    if (stat.isSymbolicLink()) throw new Error(`Console build input cannot be a symlink: ${relative}`);
    if (stat.isDirectory()) {
      for (const name of fs.readdirSync(absolute).sort()) visit(path.join(relative, name));
    } else if (stat.isFile()) result.push(relative);
  };
  for (const directory of roots) visit(directory);
  return [...new Set(result)].filter((relative) => fs.existsSync(path.join(app, relative))).sort();
}

function sourceFingerprint(files, baseDirectory = app) {
  const hash = createHash("sha256");
  for (const relative of files) {
    const bytes = fs.readFileSync(path.join(baseDirectory, relative));
    hash.update(`${relative}\0${bytes.length}\0`);
    hash.update(bytes);
  }
  return hash.digest("hex");
}

function acquireBuildLock() {
  fs.mkdirSync(stateRoot, {recursive: true, mode: 0o700});
  const stateStat = fs.lstatSync(stateRoot);
  if (!stateStat.isDirectory() || stateStat.isSymbolicLink()) {
    throw new Error("Console runtime state root must be a regular local directory");
  }
  fs.chmodSync(stateRoot, 0o700);
  const token = randomBytes(16).toString("hex");
  let descriptor = null;
  for (let attempt = 0; attempt < 2; attempt += 1) {
    try {
      descriptor = fs.openSync(lockPath, "wx", 0o600);
      break;
    } catch (error) {
      if (error?.code !== "EEXIST") throw error;
      let existing = null;
      try { existing = JSON.parse(fs.readFileSync(lockPath, "utf8")); } catch {}
      const pid = Number(existing?.pid);
      let alive = false;
      if (Number.isSafeInteger(pid) && pid >= 2) {
        try { process.kill(pid, 0); alive = true; } catch (probe) { alive = probe?.code !== "ESRCH"; }
      }
      if (alive) throw new Error("another Console production build is already running");
      const stat = fs.lstatSync(lockPath);
      if (!stat.isFile() || stat.isSymbolicLink() || (stat.mode & 0o077) !== 0) {
        throw new Error("stale Console build lock cannot be validated; refusing to remove it");
      }
      fs.unlinkSync(lockPath);
    }
  }
  if (descriptor === null) throw new Error("could not acquire Console production build lock");
  fs.writeFileSync(descriptor, `${JSON.stringify({pid: process.pid, token, started_at: new Date().toISOString()})}\n`);
  fs.closeSync(descriptor);
  return () => {
    try {
      const current = JSON.parse(fs.readFileSync(lockPath, "utf8"));
      if (current.pid === process.pid && current.token === token) fs.unlinkSync(lockPath);
    } catch {}
  };
}

function verifyInstalledDependencies(npm) {
  if (!fs.existsSync(path.join(app, "package-lock.json"))) throw new Error("Console package-lock.json is required");
  const result = spawnSync(npm, ["ls", "--all", "--json"], {cwd: app, encoding: "utf8"});
  if (result.error || result.status !== 0) {
    throw new Error("Console dependencies do not match package-lock.json; run npm ci before building");
  }
}

function copyTree(source, target) {
  if (!fs.existsSync(source)) return;
  fs.cpSync(source, target, {recursive: true, dereference: true, errorOnExist: true, force: false});
}

function copyBuildInput(source, target) {
  fs.copyFileSync(source, target, fs.constants.COPYFILE_EXCL);
  // Release source snapshots intentionally use read-only files. The isolated
  // build workspace is private scratch state and Next may update generated
  // config declarations such as tsconfig.json while compiling. Normalize the
  // copied input instead of weakening the immutable source snapshot.
  fs.chmodSync(target, 0o600);
}

function artifactFiles(directory) {
  const entries = [];
  const visit = (relative) => {
    const absolute = path.join(directory, relative);
    const stat = fs.lstatSync(absolute);
    if (stat.isSymbolicLink()) throw new Error(`runtime artifact cannot contain symlink: ${relative}`);
    if (stat.isDirectory()) {
      for (const name of fs.readdirSync(absolute).sort()) visit(path.join(relative, name));
    } else if (stat.isFile()) {
      entries.push({path: relative, bytes: stat.size, sha256: sha256(fs.readFileSync(absolute))});
    } else throw new Error(`runtime artifact contains unsupported filesystem entry: ${relative}`);
  };
  for (const name of fs.readdirSync(directory).sort()) if (name !== "runtime-manifest.json") visit(name);
  return entries;
}

async function validExistingRelease(directory, expectedSource, expectedLock) {
  try {
    const manifest = JSON.parse(fs.readFileSync(path.join(directory, "runtime-manifest.json"), "utf8"));
    if (manifest?.schema !== "teachlab.console.runtime.v1"
      || manifest.release_id !== path.basename(directory)
      || manifest.source_sha256 !== expectedSource
      || manifest.package_lock_sha256 !== expectedLock
      || !Array.isArray(manifest.files)
      || manifest.files.length < 1
      || manifest.files.length > 20_000) return false;
    for (let offset = 0; offset < manifest.files.length; offset += 64) {
      const verified = await Promise.all(manifest.files.slice(offset, offset + 64).map(async (entry) => {
        if (!entry || typeof entry.path !== "string" || path.isAbsolute(entry.path)
          || entry.path.split(path.sep).some((part) => part === "" || part === "." || part === "..")) return false;
        const target = path.join(directory, entry.path);
        const [stat, bytes] = await Promise.all([fs.promises.lstat(target), fs.promises.readFile(target)]);
        return stat.isFile() && !stat.isSymbolicLink() && stat.size === entry.bytes
          && sha256(bytes) === entry.sha256;
      }));
      if (verified.includes(false)) return false;
    }
    return true;
  } catch {
    return false;
  }
}

function switchChannel(releaseId) {
  const manifestSha256 = sha256(fs.readFileSync(path.join(releases, releaseId, "runtime-manifest.json")));
  const prior = (() => {
    try { return JSON.parse(fs.readFileSync(channelPath, "utf8")); } catch { return {}; }
  })();
  const previous = prior.current ?? null;
  atomicJson(channelPath, {
    schema: "teachlab.console.channel.v1",
    current: releaseId,
    current_manifest_sha256: manifestSha256,
    previous: previous === releaseId ? prior.previous ?? null : previous,
    previous_manifest_sha256: previous === releaseId
      ? prior.previous_manifest_sha256 ?? null
      : prior.current_manifest_sha256 ?? null,
    upgraded_at: new Date().toISOString(),
    rollback: "set current to previous only after runtime-manifest verification",
  });
}

async function runSelfTest() {
  const parent = fs.mkdtempSync(path.join(os.tmpdir(), "teachlab-runtime-build-test-"));
  const releaseId = "0.1.0-bbbbbbbbbbbbbbbb";
  const directory = path.join(parent, releaseId);
  fs.mkdirSync(directory);
  try {
    const readOnlySource = path.join(parent, "read-only-source.json");
    const writableCopy = path.join(directory, "writable-copy.json");
    fs.writeFileSync(readOnlySource, "{}\n", {mode: 0o400});
    copyBuildInput(readOnlySource, writableCopy);
    const writableMode = fs.lstatSync(writableCopy).mode & 0o777;
    if (writableMode !== 0o600) throw new Error("read-only snapshot input was not normalized in build scratch");
    fs.appendFileSync(writableCopy, "\n");
    const relative = path.join(".next-production-test", "server", "app", "api", "[...path]", "route.js");
    const target = path.join(directory, relative);
    fs.mkdirSync(path.dirname(target), {recursive: true});
    fs.writeFileSync(target, "verified runtime bytes\n");
    const lockDigest = "a".repeat(64);
    const sourceDigest = "b".repeat(64);
    fs.writeFileSync(path.join(directory, "runtime-manifest.json"), `${JSON.stringify({
      schema: "teachlab.console.runtime.v1",
      release_id: releaseId,
      source_sha256: sourceDigest,
      package_lock_sha256: lockDigest,
      files: [{path: relative, bytes: fs.statSync(target).size, sha256: sha256(fs.readFileSync(target))}],
    })}\n`);
    if (!await validExistingRelease(directory, sourceDigest, lockDigest)) throw new Error("valid runtime was rejected");
    fs.appendFileSync(target, "tamper");
    if (await validExistingRelease(directory, sourceDigest, lockDigest)) throw new Error("tampered runtime was accepted");
    console.log("TeachLab Console isolated build self-test passed.");
  } finally {
    fs.rmSync(parent, {recursive: true, force: true});
  }
}

if (selfTest) {
  await runSelfTest();
  process.exit(0);
}

const npm = process.env.TEACHLAB_NPM || "npm";
verifyInstalledDependencies(npm);
const inputs = sourceFiles();
const fingerprint = sourceFingerprint(inputs);
const pkg = JSON.parse(fs.readFileSync(path.join(app, "package.json"), "utf8"));
const releaseId = `${pkg.version}-${fingerprint.slice(0, 16)}`;
const release = path.join(releases, releaseId);
const packageLockDigest = sha256(fs.readFileSync(path.join(app, "package-lock.json")));
const releaseManifestPath = path.join(release, "runtime-manifest.json");
if (dryRun) {
  const valid = fs.existsSync(releaseManifestPath)
    && await validExistingRelease(release, fingerprint, packageLockDigest);
  console.log(JSON.stringify({status: valid ? "present" : "build_required", release_id: releaseId}));
  process.exit(0);
}

if (fs.existsSync(releaseManifestPath) && await validExistingRelease(release, fingerprint, packageLockDigest)) {
  const reuseLock = acquireBuildLock();
  try {
    if (sourceFingerprint(inputs) !== fingerprint
      || !await validExistingRelease(release, fingerprint, packageLockDigest)) {
      throw new Error("Console source or runtime changed during channel arbitration");
    }
    switchChannel(releaseId);
  } finally {
    reuseLock();
  }
  console.log(JSON.stringify({status: "reused", release_id: releaseId, runtime_dir: release}));
  process.exit(0);
}

const releaseLock = acquireBuildLock();
const distName = `.next-production-${fingerprint.slice(0, 16)}`;
const buildWorkspace = path.join(stateRoot, `.build-${releaseId}-${process.pid}`);
const dist = path.join(buildWorkspace, distName);
const staging = path.join(releases, `.${releaseId}.${process.pid}.staging`);
try {
  for (const name of fs.readdirSync(stateRoot)) {
    if (!/^\.build-[0-9A-Za-z._-]+-\d+$/.test(name)) continue;
    const abandoned = path.join(stateRoot, name);
    const stat = fs.lstatSync(abandoned);
    if (stat.isDirectory() && !stat.isSymbolicLink()) fs.rmSync(abandoned, {recursive: true});
  }
  if (fs.existsSync(releases)) {
    for (const name of fs.readdirSync(releases)) {
      if (!/^\.[0-9A-Za-z._-]+\.\d+\.staging$/.test(name)) continue;
      const abandoned = path.join(releases, name);
      const stat = fs.lstatSync(abandoned);
      if (stat.isDirectory() && !stat.isSymbolicLink()) fs.rmSync(abandoned, {recursive: true});
    }
  }
  if (fs.existsSync(release)) throw new Error("existing Console release is incomplete or corrupted; refusing to overwrite it");
  if (fs.existsSync(dist)) fs.rmSync(dist, {recursive: true});
  fs.mkdirSync(releases, {recursive: true, mode: 0o700});
  fs.mkdirSync(staging, {recursive: false, mode: 0o700});
  fs.mkdirSync(buildWorkspace, {recursive: false, mode: 0o700});
  for (const relative of inputs) {
    const target = path.join(buildWorkspace, relative);
    fs.mkdirSync(path.dirname(target), {recursive: true});
    copyBuildInput(path.join(app, relative), target);
  }
  if (sourceFingerprint(inputs, buildWorkspace) !== fingerprint) {
    throw new Error("Console sources changed while the isolated build snapshot was being copied; rerun the build");
  }
  // `npm ci` creates an exact private dependency tree from the lock file.
  // Sharing or symlinking the checkout's node_modules can produce a standalone
  // artifact that still points back into a mutable development installation.
  const install = spawnSync(npm, ["ci", "--ignore-scripts", "--no-audit", "--no-fund"], {
    cwd: buildWorkspace,
    env: {...process.env, NEXT_TELEMETRY_DISABLED: "1"},
    stdio: "inherit",
  });
  if (install.error || install.status !== 0) throw new Error(`Console locked dependency install failed (${install.status ?? "spawn"})`);
  const build = spawnSync(npm, ["run", "build"], {
    cwd: buildWorkspace,
    encoding: "utf8",
    env: {...process.env, NEXT_TELEMETRY_DISABLED: "1", TEACHLAB_NEXT_DIST_DIR: distName},
    stdio: "inherit",
  });
  if (build.error || build.status !== 0) throw new Error(`Console production build failed (${build.status ?? "spawn"})`);
  const standalone = path.join(dist, "standalone");
  if (!fs.existsSync(path.join(standalone, "server.js"))) throw new Error("Next standalone server.js was not produced");
  copyTree(standalone, staging);
  copyTree(path.join(dist, "static"), path.join(staging, distName, "static"));
  copyTree(path.join(buildWorkspace, "public"), path.join(staging, "public"));
  const manifest = {
    schema: "teachlab.console.runtime.v1",
    release_id: releaseId,
    version: pkg.version,
    source_sha256: fingerprint,
    package_lock_sha256: packageLockDigest,
    node_min_major: 22,
    built_with_node: process.versions.node,
    next_dist_dir: distName,
    built_at: new Date().toISOString(),
    distributable: false,
    distribution_status: "local_production_runtime_not_codesigned",
    files: artifactFiles(staging),
  };
  if (sourceFingerprint(inputs) !== fingerprint) {
    throw new Error("Console sources changed while the isolated production build was running; refusing to publish a stale channel");
  }
  atomicJson(path.join(staging, "runtime-manifest.json"), manifest);
  fs.renameSync(staging, release);
  switchChannel(releaseId);
  console.log(JSON.stringify({status: "built", release_id: releaseId, runtime_dir: release}));
} finally {
  try { if (fs.existsSync(staging)) fs.rmSync(staging, {recursive: true}); } catch {}
  try { if (fs.existsSync(buildWorkspace)) fs.rmSync(buildWorkspace, {recursive: true}); } catch {}
  releaseLock();
}
