import {SetMetadata} from "@nestjs/common";

export const ALLOW_REVOKED_SESSION = "teachlab:allow-revoked-session";

/** Permit only a still-valid, signed revoked session to repeat its own logout. */
export const AllowRevokedSession = () => SetMetadata(ALLOW_REVOKED_SESSION, true);
