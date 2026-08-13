import {SetMetadata} from "@nestjs/common";

export const ALLOW_ACCOUNT_DELETION_SCOPE = "allow-account-deletion-scope";

/** Continue only an already-authorized, server-minted deletion workflow. */
export const AllowAccountDeletionScope = () =>
  SetMetadata(ALLOW_ACCOUNT_DELETION_SCOPE, true);
