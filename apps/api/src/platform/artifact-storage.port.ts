import type {AccessScope} from "../tenancy/access-scope";

export interface StoreArtifactInput extends AccessScope {
  key: string;
  contentType: string;
  bytes: Uint8Array;
  metadata?: Record<string, string>;
}

export interface StoredArtifact {
  key: string;
  contentType: string;
  size: number;
  sha256: string;
  metadata: Record<string, string>;
  createdAt: string;
  updatedAt: string;
  version: number;
}

export interface ArtifactStoragePort {
  put(input: StoreArtifactInput): Promise<StoredArtifact>;
  get(
    scope: AccessScope,
    key: string
  ): Promise<{artifact: StoredArtifact; bytes: Uint8Array} | undefined>;
  list(scope: AccessScope): Promise<StoredArtifact[]>;
  delete(scope: AccessScope, key: string): Promise<boolean>;
  deleteScope(scope: AccessScope): Promise<{count: number; bytes: number}>;
  probe(): Promise<"postgres_durable_force_rls" | "memory_local_only">;
}
