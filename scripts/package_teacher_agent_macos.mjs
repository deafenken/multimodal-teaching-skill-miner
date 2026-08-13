#!/usr/bin/env node

import {spawnSync} from "node:child_process";
import {createHash} from "node:crypto";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import process from "node:process";
import {fileURLToPath} from "node:url";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const output = path.resolve(process.env.TEACHLAB_PACKAGE_OUTPUT || path.join(root, "artifacts", "macos"));
const runtimeRoot = process.env.TEACHLAB_CONSOLE_RUNTIME_ROOT || path.join(root, ".private", "console-runtime");
const identity = process.env.TEACHLAB_CODESIGN_IDENTITY?.trim() || "";
const notaryProfile = process.env.TEACHLAB_NOTARY_PROFILE?.trim() || "";
const releaseRequested = process.env.TEACHLAB_DISTRIBUTABLE === "1";
const embeddedNodeRoot = process.env.TEACHLAB_EMBEDDED_NODE?.trim() || "";
const embeddedPythonRoot = process.env.TEACHLAB_EMBEDDED_PYTHON?.trim() || "";
const embeddedNodeManifestSha256 = process.env.TEACHLAB_EMBEDDED_NODE_MANIFEST_SHA256?.trim() || "";
const embeddedPythonManifestSha256 = process.env.TEACHLAB_EMBEDDED_PYTHON_MANIFEST_SHA256?.trim() || "";
const embeddedRuntimeSelfTest = process.argv[2] === "--embedded-runtime-self-test";
const packageVersion = JSON.parse(fs.readFileSync(path.join(root, "apps", "console", "package.json"), "utf8")).version;

function run(command, args, options = {}) {
  const result = spawnSync(command, args, {
    encoding: "utf8",
    stdio: options.capture ? "pipe" : "inherit",
    cwd: options.cwd,
    env: options.env,
  });
  if (result.error || result.status !== 0) throw new Error(`${command} failed (${result.status ?? "spawn"})${result.stderr ? `: ${result.stderr.trim()}` : ""}`);
  return result.stdout?.trim() ?? "";
}

function sha256(file) {
  return createHash("sha256").update(fs.readFileSync(file)).digest("hex");
}

function canonicalJson(value) {
  if (Array.isArray(value)) return `[${value.map(canonicalJson).join(",")}]`;
  if (value && typeof value === "object") {
    return `{${Object.keys(value).sort().map((key) => `${JSON.stringify(key)}:${canonicalJson(value[key])}`).join(",")}}`;
  }
  return JSON.stringify(value);
}

function inside(rootDirectory, candidate) {
  const relative = path.relative(rootDirectory, candidate);
  return relative === "" || (!relative.startsWith(`..${path.sep}`) && relative !== ".." && !path.isAbsolute(relative));
}

function runtimeTreeRows(runtimeDirectory, manifestName = "embedded-runtime.json") {
  const rows = [];
  const visit = (directory, prefix = "") => {
    for (const name of fs.readdirSync(directory).sort()) {
      const absolute = path.join(directory, name);
      const relative = prefix ? `${prefix}/${name}` : name;
      if (relative === manifestName) continue;
      const stat = fs.lstatSync(absolute);
      const mode = stat.mode & 0o777;
      if (stat.isDirectory()) {
        rows.push({path: relative, type: "directory", mode});
        visit(absolute, relative);
      } else if (stat.isFile()) {
        rows.push({path: relative, type: "file", mode, size: stat.size, sha256: sha256(absolute)});
      } else if (stat.isSymbolicLink()) {
        const target = fs.readlinkSync(absolute);
        const resolved = fs.realpathSync(absolute);
        if (!inside(runtimeDirectory, resolved)) throw new Error(`embedded runtime symlink escapes its root: ${relative}`);
        rows.push({path: relative, type: "symlink", mode, target});
      } else {
        throw new Error(`embedded runtime contains an unsupported filesystem entry: ${relative}`);
      }
    }
  };
  visit(runtimeDirectory);
  return rows;
}

function runtimeTreeSha256(runtimeDirectory) {
  return createHash("sha256").update(canonicalJson(runtimeTreeRows(runtimeDirectory))).digest("hex");
}

function copyPublishedRuntimeResources(projectDestination) {
  const pyprojectSource = fs.readFileSync(path.join(root, "pyproject.toml"), "utf8");
  const dataFilesSection = pyprojectSource.split("[tool.setuptools.data-files]", 2)[1]?.split("[tool.pytest.ini_options]", 1)[0] || "";
  const publicPaths = [...dataFilesSection.matchAll(/"((?:data|schema|configs)\/[A-Za-z0-9._/+ -]+)"/g)]
    .map((match) => match[1]);
  if (!publicPaths.includes("data/teacher_agent_skill_library_v2.json")
      || !publicPaths.includes("schema/teacher_agent_live_session.schema.json")) {
    throw new Error("pyproject public runtime resource contract is incomplete");
  }
  for (const relative of [...new Set(publicPaths)].sort()) {
    const source = path.join(root, relative);
    const stat = fs.lstatSync(source);
    if (!stat.isFile() || stat.isSymbolicLink() || !inside(root, fs.realpathSync(source))) {
      throw new Error(`public runtime resource is unsafe: ${relative}`);
    }
    const destination = path.join(projectDestination, relative);
    fs.mkdirSync(path.dirname(destination), {recursive: true});
    fs.copyFileSync(source, destination);
  }
}

function checkedEmbeddedRuntime(kind, rawDirectory, expectedManifestSha256) {
  if (!path.isAbsolute(rawDirectory)) throw new Error(`embedded ${kind} runtime path must be absolute`);
  const directory = fs.realpathSync(rawDirectory);
  const directoryStat = fs.lstatSync(rawDirectory);
  if (!directoryStat.isDirectory() || directoryStat.isSymbolicLink()) {
    throw new Error(`embedded ${kind} runtime must be a real directory`);
  }
  if (!/^[0-9a-f]{64}$/.test(expectedManifestSha256)) {
    throw new Error(`embedded ${kind} runtime requires an out-of-band manifest SHA-256`);
  }
  const manifestPath = path.join(directory, "embedded-runtime.json");
  const manifestStat = fs.lstatSync(manifestPath);
  if (!manifestStat.isFile() || manifestStat.isSymbolicLink() || sha256(manifestPath) !== expectedManifestSha256) {
    throw new Error(`embedded ${kind} runtime manifest is missing or does not match its pinned SHA-256`);
  }
  const manifest = JSON.parse(fs.readFileSync(manifestPath, "utf8"));
  const allowedKeys = new Set([
    "schema", "runtime_id", "kind", "version", "platform", "architectures", "executable",
    "tree_sha256", "self_contained", "license_files", "provenance",
  ]);
  if (!manifest || typeof manifest !== "object" || Array.isArray(manifest)
      || Object.keys(manifest).some((key) => !allowedKeys.has(key))
      || manifest.schema !== "teachlab.embedded-runtime.v1"
      || manifest.kind !== kind
      || typeof manifest.runtime_id !== "string" || !/^[A-Za-z0-9._-]{1,96}$/.test(manifest.runtime_id)
      || typeof manifest.version !== "string" || !/^\d+\.\d+\.\d+$/.test(manifest.version)
      || manifest.platform !== "darwin"
      || !Array.isArray(manifest.architectures) || !manifest.architectures.includes(process.arch)
      || typeof manifest.executable !== "string" || !/^(?!\/)(?!.*(?:^|\/)\.\.(?:\/|$))[A-Za-z0-9._/+ -]{1,240}$/.test(manifest.executable)
      || !/^[0-9a-f]{64}$/.test(String(manifest.tree_sha256))
      || manifest.self_contained !== true
      || !Array.isArray(manifest.license_files) || manifest.license_files.length < 1
      || !manifest.license_files.every((item) => typeof item === "string" && /^(?!\/)(?!.*(?:^|\/)\.\.(?:\/|$))[A-Za-z0-9._/+ -]{1,240}$/.test(item))
      || !manifest.provenance || typeof manifest.provenance !== "object" || Array.isArray(manifest.provenance)
      || typeof manifest.provenance.source_uri !== "string" || !/^https:\/\//.test(manifest.provenance.source_uri)
      || !/^[0-9a-f]{64}$/.test(String(manifest.provenance.source_sha256))
      || typeof manifest.provenance.builder !== "string" || manifest.provenance.builder.length < 1 || manifest.provenance.builder.length > 160) {
    throw new Error(`embedded ${kind} runtime manifest contract is invalid`);
  }
  if (runtimeTreeSha256(directory) !== manifest.tree_sha256) {
    throw new Error(`embedded ${kind} runtime tree does not match its manifest`);
  }
  const executable = path.join(directory, manifest.executable);
  const executableReal = fs.realpathSync(executable);
  const executableStat = fs.lstatSync(executable);
  if (!inside(directory, executableReal) || !executableStat.isFile() || executableStat.isSymbolicLink() || !(executableStat.mode & 0o111)) {
    throw new Error(`embedded ${kind} executable is unsafe or not executable`);
  }
  for (const license of manifest.license_files) {
    const licensePath = path.join(directory, license);
    const licenseStat = fs.lstatSync(licensePath);
    if (!inside(directory, fs.realpathSync(licensePath)) || !licenseStat.isFile() || licenseStat.isSymbolicLink()) {
      throw new Error(`embedded ${kind} license file is unsafe`);
    }
  }
  const major = Number(manifest.version.split(".")[0]);
  if ((kind === "node" && major < 22) || (kind === "python" && (major < 3 || major === 3 && Number(manifest.version.split(".")[1]) < 10))) {
    throw new Error(`embedded ${kind} runtime version is below the supported minimum`);
  }
  const probe = kind === "node"
    ? run(executable, ["-e", "const v=process.versions.node; if (!v) process.exit(2); process.stdout.write(v)"], {capture: true})
    : run(executable, ["-I", "-c", "import cryptography,sys; print('.'.join(map(str,sys.version_info[:3])))"], {capture: true});
  if (probe.trim() !== manifest.version) {
    throw new Error(`embedded ${kind} executable version does not match its manifest`);
  }
  return {directory, manifest, manifestSha256: expectedManifestSha256};
}

function runEmbeddedRuntimeSelfTest() {
  const temporary = fs.mkdtempSync(path.join(os.tmpdir(), "teachlab-embedded-runtime-self-test-"));
  try {
    for (const [kind, version] of [["node", "22.1.2"], ["python", "3.11.9"]]) {
      const directory = path.join(temporary, kind);
      fs.mkdirSync(path.join(directory, "bin"), {recursive: true});
      fs.writeFileSync(path.join(directory, "LICENSE"), "self-test only\n");
      fs.writeFileSync(path.join(directory, "bin", kind), `#!/bin/sh\nprintf '%s' '${version}'\n`, {mode: 0o755});
      const manifest = {
        schema: "teachlab.embedded-runtime.v1",
        runtime_id: `self-test-${kind}`,
        kind,
        version,
        platform: "darwin",
        architectures: [process.arch],
        executable: `bin/${kind}`,
        tree_sha256: runtimeTreeSha256(directory),
        self_contained: true,
        license_files: ["LICENSE"],
        provenance: {
          source_uri: `https://example.invalid/${kind}.tar.gz`,
          source_sha256: "a".repeat(64),
          builder: "TeachLab deterministic packaging self-test",
        },
      };
      const manifestPath = path.join(directory, "embedded-runtime.json");
      fs.writeFileSync(manifestPath, `${JSON.stringify(manifest)}\n`);
      checkedEmbeddedRuntime(kind, directory, sha256(manifestPath));
      fs.appendFileSync(path.join(directory, "LICENSE"), "tampered\n");
      let rejected = false;
      try { checkedEmbeddedRuntime(kind, directory, sha256(manifestPath)); }
      catch { rejected = true; }
      if (!rejected) throw new Error(`embedded ${kind} tree tamper was accepted`);
    }
    console.log("TeachLab embedded runtime packaging self-test passed.");
  } finally {
    fs.rmSync(temporary, {recursive: true, force: true});
  }
}

if (embeddedRuntimeSelfTest) {
  runEmbeddedRuntimeSelfTest();
  process.exit(0);
}

if (releaseRequested && (!identity || !notaryProfile || !embeddedNodeRoot || !embeddedPythonRoot)) {
  throw new Error("distributable release requires codesign identity, notary profile, and audited embedded Node/Python runtimes");
}
if (process.platform !== "darwin") throw new Error("macOS .app/DMG packaging must run on macOS");
fs.mkdirSync(output, {recursive: true, mode: 0o700});
const outputStat = fs.lstatSync(output);
if (!outputStat.isDirectory() || outputStat.isSymbolicLink()) {
  throw new Error("macOS package output must be a regular directory, not a symlink");
}
let embeddedNode = null;
let embeddedPython = null;
if (releaseRequested) {
  embeddedNode = checkedEmbeddedRuntime("node", embeddedNodeRoot, embeddedNodeManifestSha256);
  embeddedPython = checkedEmbeddedRuntime("python", embeddedPythonRoot, embeddedPythonManifestSha256);
}

run(process.execPath, [path.join(root, "scripts", "build_teacher_agent_console_runtime.mjs")]);
run(process.execPath, [path.join(root, "scripts", "start_teacher_agent_console.mjs"), "--runtime-self-test"]);
if (releaseRequested) {
  const nodeExecutable = path.join(embeddedNode.directory, embeddedNode.manifest.executable);
  const pythonExecutable = path.join(embeddedPython.directory, embeddedPython.manifest.executable);
  run(nodeExecutable, [path.join(root, "scripts", "start_teacher_agent_console.mjs"), "--runtime-self-test"], {
    env: {...process.env, TEACHLAB_CONSOLE_RUNTIME_ROOT: runtimeRoot},
  });
  run(pythonExecutable, ["-c", "import cryptography,teaching_skill_miner; print('embedded-python-app-smoke-ok')"], {
    cwd: root,
    env: {...process.env, PYTHONPATH: root},
  });
}
const channel = JSON.parse(fs.readFileSync(path.join(runtimeRoot, "channel.json"), "utf8"));
const runtime = path.join(runtimeRoot, "releases", channel.current);
const runtimeManifestSha256 = sha256(path.join(runtime, "runtime-manifest.json"));
const staging = fs.mkdtempSync(path.join(os.tmpdir(), "teachlab-macos-package-"));
process.once("exit", () => {
  try { fs.rmSync(staging, {recursive: true, force: true}); } catch {}
});
const app = path.join(staging, releaseRequested
  ? `TeachLab Console ${packageVersion}.app`
  : `TeachLab Console ${packageVersion} (Developer).app`);
const contents = path.join(app, "Contents");
const resources = path.join(contents, "Resources");
const macos = path.join(contents, "MacOS");
fs.mkdirSync(resources, {recursive: true});
fs.mkdirSync(macos, {recursive: true});
const consoleSbom = run(process.env.TEACHLAB_NPM || "npm", [
  "--prefix", path.join(root, "apps", "console"), "sbom", "--sbom-format", "cyclonedx",
], {capture: true});
JSON.parse(consoleSbom);
fs.writeFileSync(path.join(resources, "console-sbom.cdx.json"), `${consoleSbom}\n`);
const consoleSbomSha256 = sha256(path.join(resources, "console-sbom.cdx.json"));
fs.copyFileSync(path.join(root, "LICENSE"), path.join(resources, "LICENSE"));
fs.copyFileSync(path.join(root, "pyproject.toml"), path.join(resources, "pyproject.toml"));
fs.copyFileSync(
  path.join(root, "docs", "teacher_agent_console_production_runtime.md"),
  path.join(resources, "PRODUCTION_RUNTIME.md"),
);

if (releaseRequested) {
  for (const [kind, verified] of [["node", embeddedNode], ["python", embeddedPython]]) {
    const destination = path.join(resources, "embedded", kind);
    fs.cpSync(verified.directory, destination, {
      recursive: true,
      dereference: false,
      preserveTimestamps: false,
      verbatimSymlinks: true,
    });
    if (runtimeTreeSha256(destination) !== verified.manifest.tree_sha256
        || sha256(path.join(destination, "embedded-runtime.json")) !== verified.manifestSha256) {
      throw new Error(`copied embedded ${kind} runtime failed its immutable manifest check`);
    }
  }
}

fs.cpSync(runtime, path.join(resources, "console-runtime"), {recursive: true});
fs.cpSync(path.join(root, "teaching_skill_miner"), path.join(resources, "project", "teaching_skill_miner"), {
  recursive: true,
  dereference: false,
  filter(source) {
    const name = path.basename(source);
    if (name === "__pycache__" || name.endsWith(".pyc") || name === ".DS_Store") return false;
    if (fs.lstatSync(source).isSymbolicLink()) throw new Error(`Python package input cannot be a symlink: ${source}`);
    return true;
  },
});
copyPublishedRuntimeResources(path.join(resources, "project"));
fs.mkdirSync(path.join(resources, "project", "apps", "console"), {recursive: true});
fs.copyFileSync(path.join(root, "apps", "console", "package-lock.json"), path.join(resources, "project", "apps", "console", "package-lock.json"));
fs.mkdirSync(path.join(resources, "project", "scripts"), {recursive: true});
fs.copyFileSync(path.join(root, "scripts", "start_teacher_agent_console.mjs"), path.join(resources, "project", "scripts", "start_teacher_agent_console.mjs"));

const executable = path.join(macos, "TeachLab Console");
const runtimeSelection = releaseRequested
  ? `node_bin="$resources/embedded/node/${embeddedNode.manifest.executable}"
python_bin="$resources/embedded/python/${embeddedPython.manifest.executable}"
npm_bin=/usr/bin/false`
  : `node_bin=\${TEACHLAB_NODE:-$(command -v node || true)}
python_bin=\${TEACHLAB_PYTHON:-$(command -v python3 || true)}
npm_bin=\${TEACHLAB_NPM:-npm}`;
fs.writeFileSync(executable, `#!/bin/zsh
set -eu
resources=\${0:A:h:h}/Resources
support=\${HOME}/Library/Application Support/TeachLab
mkdir -p \"$support\"
chmod 700 \"$support\"
${runtimeSelection}
if [[ -z \"$node_bin\" || -z \"$python_bin\" ]]; then
  print -u2 \"Developer build requires Node.js 22+ and Python 3.10+; this unsigned artifact is not a distributable release.\"
  exit 1
fi
if ! \"$node_bin\" -e 'process.exit(Number(process.versions.node.split(".")[0]) < 22 ? 1 : 0)' \
  || ! \"$python_bin\" -c 'import sys; raise SystemExit(sys.version_info < (3, 10))'; then
  print -u2 \"Developer build requires Node.js 22+ and Python 3.10+.\"
  exit 1
fi
export PYTHONPATH=\"$resources/project\"
export TEACHLAB_CONSOLE_RUNTIME_ROOT=\"$support/console-runtime\"
export TEACHLAB_ALLOW_CONSOLE_BUILD=0
export TEACHLAB_EXPECTED_CONSOLE_RELEASE_ID=${channel.current}
export TEACHLAB_LAUNCHER_LOCK=\"$support/launcher.lock\"
export TEACHLAB_SESSION_STORE=\"$support/sessions.jsonl\"
export TEACHLAB_SYLLABUS_STORE=\"$support/syllabi\"
export TEACHLAB_PROJECT_STORE=\"$support/projects\"
export TEACHLAB_RESOURCE_INDEX_STORE=\"$support/resource-index\"
export TEACHLAB_LEARNING_RECORD_STORE=\"$support/learning-records.jsonl\"
export TEACHLAB_METACOGNITION_STORE=\"$support/metacognition.jsonl\"
export TEACHLAB_ADJUDICATION_STORE=\"$support/adjudications.jsonl\"
export TEACHLAB_CONSENT_STORE=\"$support/remote-consent.json\"
export TEACHLAB_CONSENT_SIGNING_SECRET_FILE=\"$support/consent-signing.secret\"
export TEACHLAB_LEARNER_KEY_SECRET_FILE=\"$support/learner-key.secret\"
export TEACHLAB_RUNTIME_ACTIVITY_FILE=\"$support/activity\"
export TEACHLAB_RUNTIME_SECRET_DIR=\"$support/runtime-secrets\"
export TEACHLAB_LOG_DIR=\"$support/logs\"
if ! \"$python_bin\" -c 'import cryptography, teaching_skill_miner'; then
  print -u2 \"Developer build requires the Python dependencies declared in bundled pyproject.toml.\"
  exit 1
fi
if [[ ! -d \"$TEACHLAB_CONSOLE_RUNTIME_ROOT/releases/${channel.current}\" ]]; then
  mkdir -p \"$TEACHLAB_CONSOLE_RUNTIME_ROOT/releases\"
  install_lock=\"$TEACHLAB_CONSOLE_RUNTIME_ROOT/install.lock\"
  set -C
  if ! print \"$$\" > \"$install_lock\" 2>/dev/null; then
    owner=\$(cat \"$install_lock\" 2>/dev/null || true)
    if [[ -f \"$install_lock\" && ! -L \"$install_lock\" && \"$owner\" == <-> ]] \
      && ! kill -0 \"$owner\" 2>/dev/null; then
      rm -f \"$install_lock\"
      if ! print \"$$\" > \"$install_lock\" 2>/dev/null; then
        print -u2 \"Another TeachLab runtime installation won the recovery race.\"
        exit 1
      fi
    else
      print -u2 \"Another TeachLab runtime installation is in progress or its lock is unverifiable.\"
      exit 1
    fi
  fi
  set +C
  trap 'rm -f \"$install_lock\"' EXIT
  install_tmp=\"$TEACHLAB_CONSOLE_RUNTIME_ROOT/releases/.${channel.current}.$$\"
  channel_tmp=\"$TEACHLAB_CONSOLE_RUNTIME_ROOT/channel.json.$$\"
  rm -rf \"$install_tmp\"
  cp -R \"$resources/console-runtime\" \"$install_tmp\"
  mv \"$install_tmp\" \"$TEACHLAB_CONSOLE_RUNTIME_ROOT/releases/${channel.current}\"
  print '{"schema":"teachlab.console.channel.v1","current":"${channel.current}","current_manifest_sha256":"${runtimeManifestSha256}","previous":null,"previous_manifest_sha256":null}' > \"$channel_tmp\"
  chmod 600 \"$channel_tmp\"
  mv \"$channel_tmp\" \"$TEACHLAB_CONSOLE_RUNTIME_ROOT/channel.json\"
  rm -f \"$install_lock\"
  trap - EXIT
fi
exec \"$node_bin\" \"$resources/project/scripts/start_teacher_agent_console.mjs\" --daemon \"$python_bin\" \"$support/deepseek_api.txt\" \"$npm_bin\"
`, {mode: 0o755});

fs.writeFileSync(path.join(contents, "Info.plist"), `<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
<key>CFBundleExecutable</key><string>TeachLab Console</string>
<key>CFBundleIdentifier</key><string>${releaseRequested ? "org.teachlab.console" : "org.teachlab.console.developer"}</string>
<key>CFBundleName</key><string>${releaseRequested ? "TeachLab Console" : "TeachLab Console (Developer)"}</string>
<key>CFBundleShortVersionString</key><string>${packageVersion}</string>
<key>CFBundleVersion</key><string>${packageVersion.replaceAll(".", "")}</string>
<key>LSMinimumSystemVersion</key><string>13.0</string>
</dict></plist>\n`);

let signed = false;
if (identity) {
  run("/usr/bin/codesign", ["--force", "--deep", "--options", "runtime", "--timestamp", "--sign", identity, app]);
  run("/usr/bin/codesign", ["--verify", "--deep", "--strict", app]);
  signed = true;
}
fs.mkdirSync(output, {recursive: true});
const finalApp = path.join(output, path.basename(app));
fs.rmSync(finalApp, {recursive: true, force: true});
try {
  fs.renameSync(app, finalApp);
} catch (error) {
  if (error?.code !== "EXDEV") throw error;
  fs.cpSync(app, finalApp, {recursive: true, dereference: false});
  fs.rmSync(app, {recursive: true, force: true});
}
if (signed) run("/usr/bin/codesign", ["--verify", "--deep", "--strict", finalApp]);
const artifactFlavor = releaseRequested ? "signed-notarized" : `${identity ? "signed" : "unsigned"}-developer`;
const dmg = path.join(output, `TeachLab-Console-${packageVersion}-${artifactFlavor}.dmg`);
fs.rmSync(dmg, {force: true});
run("/usr/bin/hdiutil", ["create", "-quiet", "-volname", releaseRequested ? "TeachLab Console" : "TeachLab Console Developer", "-srcfolder", finalApp, "-ov", "-format", "UDZO", dmg]);
let notarized = false;
if (notaryProfile && signed) {
  run("/usr/bin/xcrun", ["notarytool", "submit", dmg, "--keychain-profile", notaryProfile, "--wait"]);
  run("/usr/bin/xcrun", ["stapler", "staple", dmg]);
  run("/usr/bin/xcrun", ["stapler", "validate", dmg]);
  notarized = true;
}
const manifest = {
  schema: "teachlab.console.macos-package.v1",
  version: packageVersion,
  console_release_id: channel.current,
  runtime_manifest_sha256: runtimeManifestSha256,
  console_sbom_sha256: consoleSbomSha256,
  previous_release_id: channel.previous ?? null,
  upgrade_policy: "install side-by-side; verify runtime manifest before switching current",
  rollback_policy: "restore previous_release_id only after verifying its immutable runtime manifest",
  artifact: path.basename(dmg),
  artifact_sha256: sha256(dmg),
  app_codesigned: signed,
  dmg_notarized: notarized,
  distributable: Boolean(releaseRequested && signed && notarized && embeddedNode && embeddedPython),
  distribution_status: releaseRequested
    ? "signed_notarized_self_contained_release"
    : signed && notarized ? "signed_notarized_developer_artifact" : "unsigned_developer_artifact",
  required_external_runtime: releaseRequested ? null : {node: ">=22", python: ">=3.10"},
  embedded_runtimes: releaseRequested ? {
    node: {
      runtime_id: embeddedNode.manifest.runtime_id,
      version: embeddedNode.manifest.version,
      manifest_sha256: embeddedNode.manifestSha256,
      tree_sha256: embeddedNode.manifest.tree_sha256,
    },
    python: {
      runtime_id: embeddedPython.manifest.runtime_id,
      version: embeddedPython.manifest.version,
      manifest_sha256: embeddedPython.manifestSha256,
      tree_sha256: embeddedPython.manifest.tree_sha256,
    },
  } : null,
  secret_contract: {
    preferred: "macOS Keychain generic password",
    api_service: "TeachLab DeepSeek API Key",
    fallback: "~/Library/Application Support/TeachLab/deepseek_api.txt mode 0600",
  },
};
fs.writeFileSync(path.join(output, "release-manifest.json"), `${JSON.stringify(manifest, null, 2)}\n`, {mode: 0o600});
fs.rmSync(staging, {recursive: true, force: true});
console.log(JSON.stringify({status: "packaged", ...manifest}));
