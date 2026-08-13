import type {
  CreateSessionRecord,
  TeachingSession,
  UpdateSessionRecord,
  UpdateSessionResult
} from "./session.types";
import type {AccessScope} from "../tenancy/access-scope";

export interface SessionRepositoryPort {
  create(input: CreateSessionRecord): Promise<TeachingSession>;
  listByOwner(scope: AccessScope): Promise<TeachingSession[]>;
  findOwnedById(scope: AccessScope, id: string): Promise<TeachingSession | undefined>;
  updateOwned(
    scope: AccessScope,
    id: string,
    patch: UpdateSessionRecord,
    expectedVersion: number
  ): Promise<UpdateSessionResult>;
}
