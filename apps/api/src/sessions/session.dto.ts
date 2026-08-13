import {IsEnum, IsOptional, IsString, Length} from "class-validator";

import type {SessionStatus} from "./session.types";

export class CreateSessionDto {
  @IsString()
  @Length(1, 120)
  title!: string;

  @IsString()
  @Length(1, 80)
  learner!: string;
}

export class UpdateSessionDto {
  @IsOptional()
  @IsString()
  @Length(1, 120)
  title?: string;

  @IsOptional()
  @IsString()
  @Length(1, 80)
  learner?: string;

  @IsOptional()
  @IsEnum(["active", "succeeded", "terminated_unable"])
  status?: SessionStatus;
}
