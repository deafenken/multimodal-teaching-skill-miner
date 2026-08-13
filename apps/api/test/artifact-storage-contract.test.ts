import assert from "node:assert/strict";
import {createHash} from "node:crypto";
import {test} from "node:test";

import type {
  DatabaseQueryResult,
  TenantDatabasePort,
  TenantTransaction
} from "../src/database/tenant-database.port";
import {PostgresArtifactStorageAdapter} from "../src/platform/postgres-artifact-storage.adapter";
import type {AccessScope} from "../src/tenancy/access-scope";

const ALICE: AccessScope = {tenantId: "school-a", ownerId: "alice"};
const BOB: AccessScope = {tenantId: "school-b", ownerId: "bob"};
const BYTES = new TextEncoder().encode("durable-artifact");
const HASH = createHash("sha256").update(BYTES).digest("hex");

interface ScriptedResult {
  rows: unknown[];
  rowCount?: number;
}

class RecordingTenantDatabase implements TenantDatabasePort {
  readonly calls: Array<{
    scope: AccessScope;
    sql: string;
    parameters: unknown[];
  }> = [];

  constructor(private readonly scripted: ScriptedResult[]) {}

  async withTenant<T>(
    scope: AccessScope,
    operation: (transaction: TenantTransaction) => Promise<T>
  ): Promise<T> {
    const transaction: TenantTransaction = {
      query: async <Row>(sql: string, parameters: unknown[] = []) => {
        this.calls.push({scope: {...scope}, sql, parameters});
        const result = this.scripted.shift();
        assert.ok(result, `Unexpected SQL: ${sql}`);
        return {
          rows: result.rows as Row[],
          rowCount: result.rowCount ?? result.rows.length
        } satisfies DatabaseQueryResult<Row>;
      }
    };
    return operation(transaction);
  }
}

function artifactRow(overrides: Record<string, unknown> = {}) {
  return {
    artifact_key: "exports/lesson.json",
    content_type: "application/json",
    byte_length: BYTES.byteLength,
    sha256: HASH,
    metadata: {kind: "lesson"},
    content: Buffer.from(BYTES),
    created_at: "2026-08-13T00:00:00.000Z",
    updated_at: "2026-08-13T00:00:01.000Z",
    version: 1,
    ...overrides
  };
}

test("Postgres artifacts bind tenant scope, hash, size, metadata and version", async () => {
  const database = new RecordingTenantDatabase([
    {rows: [artifactRow()]},
    {rows: [artifactRow({version: 2, updated_at: "2026-08-13T00:00:02.000Z"})]},
    {rows: [artifactRow()]}
  ]);
  const storage = new PostgresArtifactStorageAdapter(database);

  const first = await storage.put({
    ...ALICE,
    key: "exports/lesson.json",
    contentType: "application/json",
    bytes: BYTES,
    metadata: {kind: "lesson"}
  });
  assert.equal(first.sha256, HASH);
  assert.equal(first.size, BYTES.byteLength);
  assert.equal(first.version, 1);

  const second = await storage.put({
    ...ALICE,
    key: "exports/lesson.json",
    contentType: "application/json",
    bytes: BYTES,
    metadata: {kind: "lesson"}
  });
  assert.equal(second.version, 2);
  assert.equal(second.createdAt, first.createdAt);
  assert.match(database.calls[0]?.sql ?? "", /ON CONFLICT \(tenant_id, owner_id, artifact_key\)/);
  assert.match(database.calls[0]?.sql ?? "", /version = public\.teachlab_artifacts\.version \+ 1/);
  assert.deepEqual(database.calls[0]?.scope, ALICE);
  assert.deepEqual(database.calls[0]?.parameters.slice(0, 4), [
    ALICE.tenantId,
    ALICE.ownerId,
    "exports/lesson.json",
    "application/json"
  ]);

  const read = await storage.get(ALICE, "exports/lesson.json");
  assert.equal(read?.artifact.sha256, HASH);
  assert.deepEqual([...read!.bytes], [...BYTES]);
  assert.deepEqual(database.calls[2]?.scope, ALICE);
  assert.deepEqual(database.calls[2]?.parameters, [
    ALICE.tenantId,
    ALICE.ownerId,
    "exports/lesson.json"
  ]);
});

test("Postgres artifact reads fail closed when stored bytes do not match the digest", async () => {
  const database = new RecordingTenantDatabase([
    {rows: [artifactRow({content: Buffer.from("tampered")})]}
  ]);
  const storage = new PostgresArtifactStorageAdapter(database);
  await assert.rejects(
    storage.get(ALICE, "exports/lesson.json"),
    /artifact integrity check failed/
  );
});

test("Postgres artifact writes fail closed when the database returns corrupted bytes", async () => {
  const database = new RecordingTenantDatabase([
    {rows: [artifactRow({content: Buffer.from("tampered")})]}
  ]);
  const storage = new PostgresArtifactStorageAdapter(database);
  await assert.rejects(
    storage.put({
      ...ALICE,
      key: "exports/lesson.json",
      contentType: "application/json",
      bytes: BYTES,
      metadata: {kind: "lesson"}
    }),
    /artifact integrity check failed/
  );
});

test("Postgres artifact rows reject malformed metadata and timestamps", async () => {
  for (const overrides of [
    {metadata: "not-json"},
    {created_at: "not-a-timestamp"},
    {updated_at: "not-a-timestamp"}
  ]) {
    const database = new RecordingTenantDatabase([
      {rows: [artifactRow(overrides)]}
    ]);
    const storage = new PostgresArtifactStorageAdapter(database);
    await assert.rejects(
      storage.get(ALICE, "exports/lesson.json"),
      /Invalid PostgreSQL artifact (metadata|timestamp)/
    );
  }
});

test("artifact lookup remains tenant isolated and deleteScope returns exact row count and bytes", async () => {
  const database = new RecordingTenantDatabase([
    {rows: []},
    {rows: [{byte_length: String(BYTES.byteLength)}, {byte_length: "3"}], rowCount: 2}
  ]);
  const storage = new PostgresArtifactStorageAdapter(database);
  assert.equal(await storage.get(BOB, "exports/lesson.json"), undefined);
  const deleted = await storage.deleteScope(ALICE);
  assert.deepEqual(deleted, {count: 2, bytes: BYTES.byteLength + 3});
  assert.deepEqual(database.calls[0]?.parameters, [
    BOB.tenantId,
    BOB.ownerId,
    "exports/lesson.json"
  ]);
  assert.deepEqual(database.calls[1]?.parameters, [ALICE.tenantId, ALICE.ownerId]);
  assert.match(database.calls[1]?.sql ?? "", /RETURNING byte_length/);
});

test("artifact input validation rejects traversal, invalid metadata, and oversized bytes before SQL", async () => {
  const database = new RecordingTenantDatabase([]);
  const storage = new PostgresArtifactStorageAdapter(database);
  for (const input of [
    {...ALICE, key: "../secret", contentType: "text/plain", bytes: BYTES},
    {...ALICE, key: "safe.txt", contentType: "text/plain", bytes: BYTES, metadata: {"bad key": "x"}},
    {...ALICE, key: "safe.txt", contentType: "text/plain", bytes: new Uint8Array(16 * 1024 * 1024 + 1)}
  ]) {
    await assert.rejects(Promise.resolve().then(() => storage.put(input)), /Invalid artifact/);
  }
  assert.equal(database.calls.length, 0);
});

test("artifact delete is scoped and distinguishes an absent key", async () => {
  const database = new RecordingTenantDatabase([
    {rows: [], rowCount: 0},
    {rows: [], rowCount: 1}
  ]);
  const storage = new PostgresArtifactStorageAdapter(database);
  assert.equal(await storage.delete(ALICE, "exports/missing.json"), false);
  assert.equal(await storage.delete(ALICE, "exports/lesson.json"), true);
  assert.deepEqual(database.calls.map((call) => call.parameters), [
    [ALICE.tenantId, ALICE.ownerId, "exports/missing.json"],
    [ALICE.tenantId, ALICE.ownerId, "exports/lesson.json"]
  ]);
});

test("artifact deletion rejects malformed database byte totals", async () => {
  const database = new RecordingTenantDatabase([
    {rows: [{byte_length: "not-a-number"}], rowCount: 1}
  ]);
  const storage = new PostgresArtifactStorageAdapter(database);
  await assert.rejects(storage.deleteScope(ALICE), /Invalid PostgreSQL artifact length/);
});
