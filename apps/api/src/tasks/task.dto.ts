import {IsOptional, IsString, Length, MaxLength} from "class-validator";

export class CreateTaskDto {
  @IsString()
  @Length(1, 20_000)
  message!: string;

  @IsOptional()
  @IsString()
  @MaxLength(128)
  clientRequestId?: string;
}

export class CompatibilityStepDto {
  @IsOptional()
  @IsString()
  sessionId?: string;

  @IsOptional()
  @IsString()
  session_id?: string;

  @IsOptional()
  @IsString()
  message?: string;

  @IsOptional()
  @IsString()
  learner_response?: string;

  @IsOptional()
  @IsString()
  clientRequestId?: string;

  @IsOptional()
  @IsString()
  idempotency_key?: string;
}
