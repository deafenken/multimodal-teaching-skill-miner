import assert from "node:assert/strict";
import {createHash} from "node:crypto";
import {
  chmod,
  mkdir,
  mkdtemp,
  readFile,
  readdir,
  rm,
  writeFile
} from "node:fs/promises";
import {tmpdir} from "node:os";
import {join} from "node:path";
import test from "node:test";

import {
  ensureActiveScopeRoot,
  scopeMigrationMarkerNames,
  type HarnessScopeRootIdentity,
  type ScopeMigrationCheckpoint
} from "../src/harness/harness-scope-key-migration";

function identity(
  root: string,
  keyVersion: "k1" | "k2",
  suffix: string
): HarnessScopeRootIdentity {
  const scopeId = `scope_${createHash("sha256")
    .update(`${keyVersion}:${suffix}`, "utf8")
    .digest("hex")
    .slice(0, 48)}`;
  return {
    key: `${keyVersion}:${scopeId}`,
    scopeId,
    keyVersion,
    privateRoot: join(root, keyVersion, scopeId),
    dataKey: createHash("sha256").update(`key:${keyVersion}:${suffix}`, "utf8").digest()
  };
}

async function fixture() {
  const root = await mkdtemp(join(tmpdir(), "teachlab-scope-migration-"));
  await chmod(root, 0o700);
  const active = identity(root, "k2", "same-principal");
  const previous = identity(root, "k1", "same-principal");
  await mkdir(join(previous.privateRoot, "sessions"), {recursive: true, mode: 0o700});
  await writeFile(
    join(previous.privateRoot, "sessions", "durable.json"),
    JSON.stringify({session_id: "opaque-session", rounds: 3}),
    {encoding: "utf8", mode: 0o600}
  );
  await writeFile(join(previous.privateRoot, "teacher_authority_replay.jsonl"), "sealed\n", {
    encoding: "utf8",
    mode: 0o600
  });
  return {root, active, previous};
}

test("previous-key durable data is verified into the active root and the old root becomes hash-only", async () => {
  const value = await fixture();
  try {
    const selected = await ensureActiveScopeRoot(value.root, [value.active, value.previous]);
    assert.equal(selected.keyVersion, "k2");
    assert.equal(selected.dataKey.equals(value.previous.dataKey), true);
    assert.equal(selected.dataKey.equals(value.active.dataKey), false);
    assert.equal(selected.learnerScopeId, value.previous.scopeId);
    assert.deepEqual(selected.authorityScopeBindings, [{
      keyVersion: "k1",
      scopeId: value.previous.scopeId
    }]);
    assert.deepEqual(
      JSON.parse(await readFile(join(value.active.privateRoot, "sessions", "durable.json"), "utf8")),
      {session_id: "opaque-session", rounds: 3}
    );
    assert.equal(
      await readFile(join(value.active.privateRoot, "teacher_authority_replay.jsonl"), "utf8"),
      "sealed\n"
    );
    assert.deepEqual(await readdir(value.previous.privateRoot), [
      scopeMigrationMarkerNames.tombstone
    ]);
    const tombstone = await readFile(
      join(value.previous.privateRoot, scopeMigrationMarkerNames.tombstone),
      "utf8"
    );
    assert.doesNotMatch(tombstone, /opaque-session|durable\.json|teacher_authority/);
    const envelope = await readFile(
      join(value.active.privateRoot, scopeMigrationMarkerNames.dataKeyEnvelope),
      "utf8"
    );
    assert.doesNotMatch(envelope, new RegExp(value.previous.dataKey.toString("base64url")));
    assert.doesNotMatch(envelope, new RegExp(value.previous.scopeId));

    // Removing the retained previous key after rollout cannot hide or fork the
    // migrated data. New writes continue only in the active root.
    await writeFile(join(value.active.privateRoot, "after-rotation.json"), "active-only", {
      mode: 0o600
    });
    const activeOnly = await ensureActiveScopeRoot(value.root, [value.active]);
    assert.equal(activeOnly.privateRoot, value.active.privateRoot);
    assert.equal(
      await readFile(join(activeOnly.privateRoot, "after-rotation.json"), "utf8"),
      "active-only"
    );
    assert.equal(activeOnly.dataKey.equals(value.previous.dataKey), true);
    assert.equal(activeOnly.learnerScopeId, value.previous.scopeId);
  } finally {
    await rm(value.root, {recursive: true, force: true});
  }
});

for (const failurePoint of ["copy_verified", "active_committed"] as const) {
  test(`migration is restart-safe after injected ${failurePoint} failure`, async () => {
    const value = await fixture();
    try {
      let injected = false;
      await assert.rejects(
        ensureActiveScopeRoot(value.root, [value.active, value.previous], {
          checkpoint: (checkpoint: ScopeMigrationCheckpoint) => {
            if (!injected && checkpoint === failurePoint) {
              injected = true;
              throw new Error("injected migration crash");
            }
          }
        }),
        /injected migration crash/
      );
      assert.equal(injected, true);
      assert.equal(
        JSON.parse(
          await readFile(join(value.previous.privateRoot, "sessions", "durable.json"), "utf8")
        ).session_id,
        "opaque-session"
      );

      await ensureActiveScopeRoot(value.root, [value.active, value.previous]);
      assert.equal(
        JSON.parse(
          await readFile(join(value.active.privateRoot, "sessions", "durable.json"), "utf8")
        ).session_id,
        "opaque-session"
      );
      assert.deepEqual(await readdir(value.previous.privateRoot), [
        scopeMigrationMarkerNames.tombstone
      ]);
    } finally {
      await rm(value.root, {recursive: true, force: true});
    }
  });
}

test("concurrent migration attempts serialize and converge on one active root", async () => {
  const value = await fixture();
  try {
    const results = await Promise.all([
      ensureActiveScopeRoot(value.root, [value.active, value.previous]),
      ensureActiveScopeRoot(value.root, [value.active, value.previous])
    ]);
    assert.deepEqual(results.map((result) => result.key), [value.active.key, value.active.key]);
    assert.deepEqual(await readdir(value.previous.privateRoot), [
      scopeMigrationMarkerNames.tombstone
    ]);
    assert.equal(
      JSON.parse(
        await readFile(join(value.active.privateRoot, "sessions", "durable.json"), "utf8")
      ).rounds,
      3
    );
  } finally {
    await rm(value.root, {recursive: true, force: true});
  }
});

test("ambiguous non-empty active and previous roots fail closed without deleting either", async () => {
  const value = await fixture();
  try {
    await mkdir(value.active.privateRoot, {recursive: true, mode: 0o700});
    await writeFile(join(value.active.privateRoot, "split-brain.json"), "new", {mode: 0o600});
    await assert.rejects(
      ensureActiveScopeRoot(value.root, [value.active, value.previous]),
      /ambiguous active and previous scope roots/
    );
    assert.equal(await readFile(join(value.active.privateRoot, "split-brain.json"), "utf8"), "new");
    assert.equal(
      JSON.parse(
        await readFile(join(value.previous.privateRoot, "sessions", "durable.json"), "utf8")
      ).session_id,
      "opaque-session"
    );
  } finally {
    await rm(value.root, {recursive: true, force: true});
  }
});
