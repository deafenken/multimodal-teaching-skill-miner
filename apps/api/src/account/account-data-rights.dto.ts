import {IsInt, IsString, Matches, Min} from "class-validator";

export class PrepareAccountDeletionDto {}

export class ConfirmAccountDeletionDto {
  @IsString()
  @Matches(/^adelc_[0-9a-f]{32}$/)
  challenge_id!: string;

  @IsString()
  @Matches(/^[A-Za-z0-9_-]{43}$/)
  confirmation_token!: string;

  @IsString()
  @Matches(/^PERMANENTLY DELETE MY TEACHLAB ACCOUNT$/)
  confirmation_phrase!: string;

  @IsInt()
  @Min(1)
  expected_revision!: number;

  @IsString()
  @Matches(/^[A-Za-z0-9][A-Za-z0-9._:-]{7,159}$/)
  idempotency_key!: string;
}
