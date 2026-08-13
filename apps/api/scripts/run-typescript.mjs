import {readdir, rm} from "node:fs/promises";
import {resolve} from "node:path";
import {spawnSync} from "node:child_process";

const mode = process.argv[2];
if (mode !== "build" && mode !== "test") {
  process.stderr.write("Usage: node scripts/run-typescript.mjs <build|test>\n");
  process.exit(2);
}

const appRoot = process.cwd();
const outputName = mode === "build" ? "dist" : ".test-dist";
const outputPath = resolve(appRoot, outputName);
await rm(outputPath, {recursive: true, force: true});

const tscPath = resolve(appRoot, "node_modules/typescript/bin/tsc");
const config = mode === "build" ? "tsconfig.build.json" : "tsconfig.test.json";
const compile = spawnSync(process.execPath, [tscPath, "-p", config], {
  cwd: appRoot,
  stdio: "inherit"
});
if (compile.error) throw compile.error;
if (compile.status !== 0) process.exit(compile.status ?? 1);

if (mode === "test") {
  const testDirectory = resolve(outputPath, "test");
  const testFiles = (await readdir(testDirectory))
    .filter((name) => name.endsWith(".test.js"))
    .sort()
    .map((name) => resolve(testDirectory, name));
  const tests = spawnSync(process.execPath, ["--test", ...testFiles], {
    cwd: appRoot,
    stdio: "inherit"
  });
  if (tests.error) throw tests.error;
  if (tests.status !== 0) process.exit(tests.status ?? 1);
}
