import {createHash} from "node:crypto";

import {Injectable} from "@nestjs/common";

import type {
  ArtifactStoragePort,
  StoreArtifactInput,
  StoredArtifact
} from "./artifact-storage.port";
import type {AccessScope} from "../tenancy/access-scope";

const MAX_ARTIFACT_BYTES = 16 * 1024 * 1024;
const KEY_PATTERN = /^[A-Za-z0-9][A-Za-z0-9._\/-]{0,511}$/;

function scopedKey(scope: AccessScope, key: string): string {
  return `${scope.tenantId}\u0000${scope.ownerId}\u0000${key}`;
}

function validate(input: StoreArtifactInput): void {
  if (
    !input
    || typeof input.key !== "string"
    || typeof input.contentType !== "string"
    || !(input.bytes instanceof Uint8Array)
  ) throw new Error("Invalid artifact content");
  if (
    !KEY_PATTERN.test(input.key)
    || input.key.includes("//")
    || input.key.split("/").some((part) => part === "." || part === "..")
  ) throw new Error("Invalid artifact key");
  if (
    !/^[\x21-\x7e]{1,255}$/.test(input.contentType)
    || input.bytes.byteLength > MAX_ARTIFACT_BYTES
  ) throw new Error("Invalid artifact content");
  const metadata = input.metadata ?? {};
  if (
    !metadata || typeof metadata !== "object" || Array.isArray(metadata)
    || Object.entries(metadata).some(([, value]) => typeof value !== "string")
    || Object.keys(metadata).length > 64
    || Object.entries(metadata).some(([key, value]) =>
      !/^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$/.test(key)
      || Buffer.byteLength(value, "utf8") > 1_024
      || /[\u0000\r\n]/.test(value)
    )
  ) throw new Error("Invalid artifact metadata");
}

@Injectable()
export class InMemoryArtifactStorageAdapter implements ArtifactStoragePort {
  private readonly objects = new Map<
    string,
    {artifact: StoredArtifact; bytes: Uint8Array}
  >();

  async put(input: StoreArtifactInput): Promise<StoredArtifact> {
    validate(input);
    const current = this.objects.get(scopedKey(input, input.key));
    const now = new Date().toISOString();
    const artifact: StoredArtifact = {
      key: input.key,
      contentType: input.contentType,
      size: input.bytes.byteLength,
      sha256: createHash("sha256").update(input.bytes).digest("hex"),
      metadata: {...input.metadata},
      createdAt: current?.artifact.createdAt ?? now,
      updatedAt: now,
      version: (current?.artifact.version ?? 0) + 1
    };
    this.objects.set(scopedKey(input, input.key), {
      artifact,
      bytes: input.bytes.slice()
    });
    return structuredClone(artifact);
  }

  async get(
    scope: AccessScope,
    key: string
  ): Promise<{artifact: StoredArtifact; bytes: Uint8Array} | undefined> {
    const stored = this.objects.get(scopedKey(scope, key));
    if (!stored) return undefined;
    return {artifact: structuredClone(stored.artifact), bytes: stored.bytes.slice()};
  }

  async list(scope: AccessScope): Promise<StoredArtifact[]> {
    const prefix = `${scope.tenantId}\u0000${scope.ownerId}\u0000`;
    return [...this.objects.entries()]
      .filter(([key]) => key.startsWith(prefix))
      .map(([, stored]) => structuredClone(stored.artifact))
      .sort((left, right) => left.key.localeCompare(right.key));
  }

  async delete(scope: AccessScope, key: string): Promise<boolean> {
    return this.objects.delete(scopedKey(scope, key));
  }

  async deleteScope(scope: AccessScope): Promise<{count: number; bytes: number}> {
    const prefix = `${scope.tenantId}\u0000${scope.ownerId}\u0000`;
    let count = 0;
    let bytes = 0;
    for (const [key, value] of this.objects) {
      if (!key.startsWith(prefix)) continue;
      count += 1;
      bytes += value.bytes.byteLength;
      this.objects.delete(key);
    }
    return {count, bytes};
  }

  async probe(): Promise<"memory_local_only"> {
    return "memory_local_only";
  }
}
