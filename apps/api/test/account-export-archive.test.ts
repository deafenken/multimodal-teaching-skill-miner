import assert from "node:assert/strict";
import {createHash} from "node:crypto";
import {Readable} from "node:stream";
import {test} from "node:test";

import {AccountDataRightsError} from "../src/account/account-data-rights.errors";
import {
  ACCOUNT_EXPORT_MAX_ENTRY_BYTES,
  bufferAccountExportEntry,
  buildAccountExportArchive,
  crc32,
  validateAccountExportArchive
} from "../src/account/account-export-archive";
import type {AccountExportEntry} from "../src/account/account-data-rights.types";

async function bytes(stream: Readable): Promise<Buffer> {
  const chunks: Buffer[] = [];
  for await (const raw of stream) {
    chunks.push(Buffer.isBuffer(raw) ? raw : Buffer.from(raw as Uint8Array));
  }
  return Buffer.concat(chunks);
}

test("account export streams a deterministic STORE-only ZIP with canonical hashes", async () => {
  let released = 0;
  const entries = [
    bufferAccountExportEntry({
      archivePath: "postgres/sessions.json",
      content: "[{\"id\":\"session-a\"}]\n",
      dataClass: "teaching_sessions_private"
    }),
    bufferAccountExportEntry({
      archivePath: "worker/projects/project-a.json",
      content: "{\"private\":true}\n",
      dataClass: "learning_project_private"
    })
  ];
  const archive = buildAccountExportArchive({
    entries: [...entries].reverse(),
    exportedAt: "2026-08-12T00:00:00.000Z",
    postgresCapturedAt: "2026-08-12T00:00:00.000Z",
    workerCapturedAt: "2026-08-12T00:00:00.000Z",
    onFinally: async () => { released += 1; }
  });
  const payload = await bytes(archive.stream);
  assert.equal(payload.byteLength, archive.byteLength);
  assert.equal(released, 1);
  const manifest = validateAccountExportArchive(payload);
  assert.equal(manifest.entry_count, 2);
  assert.deepEqual(manifest.entries.map((entry) => entry.path), [
    "postgres/sessions.json",
    "worker/projects/project-a.json"
  ]);
  assert.equal(manifest.claim_boundary.contains_raw_tenant_or_subject_claims, false);
  assert.equal(manifest.claim_boundary.contains_session_or_csrf_secrets, false);
  assert.equal(manifest.claim_boundary.contains_host_scope_paths, false);
  assert.equal(archive.manifestSha256.length, 64);
  // The first local file header explicitly declares method=STORE (0), so an
  // attacker cannot smuggle a high-expansion compressed entry into this writer.
  assert.equal(payload.readUInt32LE(0), 0x0403_4b50);
  assert.equal(payload.readUInt16LE(8), 0);
});

test("account export validator detects payload tampering and compression tricks", async () => {
  const archive = buildAccountExportArchive({
    entries: [bufferAccountExportEntry({
      archivePath: "postgres/events.json",
      content: "[{\"secret\":\"learner-text\"}]\n",
      dataClass: "task_events_private"
    })],
    exportedAt: "2026-08-12T00:00:00.000Z",
    postgresCapturedAt: "2026-08-12T00:00:00.000Z",
    workerCapturedAt: "2026-08-12T00:00:00.000Z"
  });
  const payload = await bytes(archive.stream);
  const tampered = Buffer.from(payload);
  const nameLength = tampered.readUInt16LE(26);
  const tamperOffset = 30 + nameLength + 3;
  tampered.writeUInt8(tampered.readUInt8(tamperOffset) ^ 0x01, tamperOffset);
  assert.throws(
    () => validateAccountExportArchive(tampered),
    (error) => error instanceof AccountDataRightsError
  );

  const compressed = Buffer.from(payload);
  compressed.writeUInt16LE(8, 8);
  assert.throws(
    () => validateAccountExportArchive(compressed),
    /account_export_unavailable/
  );
});

test("account export rejects traversal, duplicate names, and declared oversize before streaming", () => {
  assert.throws(
    () => bufferAccountExportEntry({
      archivePath: "worker/../scope-secret.json",
      content: "{}",
      dataClass: "unsafe"
    }),
    /account_export_unavailable/
  );
  const entry = bufferAccountExportEntry({
    archivePath: "postgres/tasks.json",
    content: "[]\n",
    dataClass: "tasks"
  });
  assert.throws(
    () => buildAccountExportArchive({
      entries: [entry, entry],
      exportedAt: "2026-08-12T00:00:00.000Z",
      postgresCapturedAt: "2026-08-12T00:00:00.000Z",
      workerCapturedAt: "2026-08-12T00:00:00.000Z"
    }),
    /account_export_unavailable/
  );
  const oversized: AccountExportEntry = {
    archivePath: "worker/huge.bin",
    byteLength: ACCOUNT_EXPORT_MAX_ENTRY_BYTES + 1,
    sha256: "a".repeat(64),
    crc32: 0,
    mediaType: "application/octet-stream",
    dataClass: "private",
    open: () => Readable.from([])
  };
  assert.throws(
    () => buildAccountExportArchive({
      entries: [oversized],
      exportedAt: "2026-08-12T00:00:00.000Z",
      postgresCapturedAt: "2026-08-12T00:00:00.000Z",
      workerCapturedAt: "2026-08-12T00:00:00.000Z"
    }),
    /account_export_unavailable/
  );
});

test("account export aborts changed source streams and always releases the snapshot lease", async () => {
  const declared = Buffer.from("original", "utf8");
  const changed = Buffer.from("mutated!", "utf8");
  let released = 0;
  const entry: AccountExportEntry = {
    archivePath: "worker/sessions.jsonl",
    byteLength: declared.byteLength,
    sha256: createHash("sha256").update(declared).digest("hex"),
    crc32: crc32(declared),
    mediaType: "application/x-ndjson",
    dataClass: "teaching_session_private",
    open: () => Readable.from([changed])
  };
  const archive = buildAccountExportArchive({
    entries: [entry],
    exportedAt: "2026-08-12T00:00:00.000Z",
    postgresCapturedAt: "2026-08-12T00:00:00.000Z",
    workerCapturedAt: "2026-08-12T00:00:00.000Z",
    onFinally: async () => { released += 1; }
  });
  await assert.rejects(bytes(archive.stream), /account_export_changed_during_stream/);
  assert.equal(released, 1);
});
