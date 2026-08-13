import assert from "node:assert/strict";
import {chmodSync, mkdirSync, mkdtempSync, realpathSync, rmSync, symlinkSync, writeFileSync} from "node:fs";
import {tmpdir} from "node:os";
import {join} from "node:path";
import test, {type TestContext} from "node:test";

import {
  PrivateSecretConfigurationError,
  privateSecret
} from "../src/config/private-secret";

function fixture(context: TestContext): {directory: string; path: string} {
  const root = realpathSync(mkdtempSync(join(tmpdir(), "teachlab-secret-")));
  context.after(() => rmSync(root, {recursive: true, force: true}));
  const directory = join(root, "private");
  mkdirSync(directory, {mode: 0o700});
  chmodSync(directory, 0o700);
  const path = join(directory, "secret");
  writeFileSync(path, `${"a".repeat(43)}\n`, {mode: 0o600});
  chmodSync(path, 0o600);
  return {directory, path};
}

test("production secrets require an owned private regular file", (context) => {
  const {path} = fixture(context);
  assert.equal(
    privateSecret({
      name: "SESSION_SECRET",
      file: path,
      inline: undefined,
      production: true,
      required: true,
      pattern: /^[A-Za-z0-9_-]+$/
    }),
    "a".repeat(43)
  );
  assert.throws(
    () => privateSecret({
      name: "SESSION_SECRET",
      file: undefined,
      inline: "b".repeat(43),
      production: true,
      required: true
    }),
    (error) => error instanceof PrivateSecretConfigurationError && error.code === "inline_forbidden"
  );
});

test("ambiguous, permissive, linked, malformed, and non-canonical sources fail closed", (context) => {
  const {directory, path} = fixture(context);
  const linked = join(directory, "linked");
  symlinkSync(path, linked);
  for (const candidate of [
    () => privateSecret({name: "X", file: path, inline: "x".repeat(32), production: false, required: true}),
    () => privateSecret({name: "X", file: linked, inline: undefined, production: true, required: true}),
    () => privateSecret({name: "X", file: `${directory}/./secret`, inline: undefined, production: true, required: true})
  ]) assert.throws(candidate, PrivateSecretConfigurationError);

  chmodSync(path, 0o640);
  assert.throws(
    () => privateSecret({name: "X", file: path, inline: undefined, production: true, required: true}),
    (error) => error instanceof PrivateSecretConfigurationError && error.code === "unsafe_file"
  );
});

test("development inline values remain explicit and bounded", () => {
  assert.equal(
    privateSecret({
      name: "TEST_SECRET",
      file: undefined,
      inline: "fixture-secret-value",
      production: false,
      required: true,
      minimumBytes: 8,
      maximumBytes: 64,
      pattern: /^[a-z-]+$/
    }),
    "fixture-secret-value"
  );
  assert.equal(
    privateSecret({name: "OPTIONAL", file: undefined, inline: undefined, production: false, required: false}),
    undefined
  );
  assert.throws(
    () => privateSecret({name: "TEST_SECRET", file: undefined, inline: " short ", production: false, required: true, minimumBytes: 1}),
    PrivateSecretConfigurationError
  );
});
