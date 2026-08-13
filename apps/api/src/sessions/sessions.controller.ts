import {
  BadRequestException,
  Body,
  Controller,
  Get,
  Headers,
  HttpException,
  HttpStatus,
  Inject,
  Param,
  Patch,
  Post,
  Res
} from "@nestjs/common";
import type {FastifyReply} from "fastify";

import {CurrentPrincipal} from "../auth/current-principal.decorator";
import type {AuthenticatedPrincipal} from "../auth/auth-provider.port";
import {CreateSessionDto, UpdateSessionDto} from "./session.dto";
import {SessionsService} from "./sessions.service";
import {accessScopeFor} from "../tenancy/access-scope";

function parseIfMatch(value: string | undefined): number {
  if (!value) {
    throw new HttpException(
      {statusCode: 428, code: "if_match_required", message: "If-Match is required"},
      HttpStatus.PRECONDITION_REQUIRED
    );
  }
  const match = /^\"?(\d+)\"?$/.exec(value.trim());
  const version = match?.[1] ? Number.parseInt(match[1], 10) : Number.NaN;
  if (!Number.isSafeInteger(version) || version < 1) {
    throw new BadRequestException("If-Match must contain a positive session version");
  }
  return version;
}

@Controller("api/v1/sessions")
export class SessionsController {
  constructor(@Inject(SessionsService) private readonly sessions: SessionsService) {}

  @Get()
  list(@CurrentPrincipal() principal: AuthenticatedPrincipal) {
    return this.sessions.list(accessScopeFor(principal));
  }

  @Post()
  async create(
    @CurrentPrincipal() principal: AuthenticatedPrincipal,
    @Body() input: CreateSessionDto,
    @Res({passthrough: true}) reply: FastifyReply
  ) {
    const session = await this.sessions.create(
      accessScopeFor(principal),
      input.title,
      input.learner
    );
    reply.header("etag", `\"${session.version}\"`);
    return session;
  }

  @Get(":sessionId")
  async get(
    @CurrentPrincipal() principal: AuthenticatedPrincipal,
    @Param("sessionId") sessionId: string,
    @Res({passthrough: true}) reply: FastifyReply
  ) {
    const session = await this.sessions.getOwned(accessScopeFor(principal), sessionId);
    reply.header("etag", `\"${session.version}\"`);
    return session;
  }

  @Patch(":sessionId")
  async update(
    @CurrentPrincipal() principal: AuthenticatedPrincipal,
    @Param("sessionId") sessionId: string,
    @Headers("if-match") ifMatch: string | undefined,
    @Body() input: UpdateSessionDto,
    @Res({passthrough: true}) reply: FastifyReply
  ) {
    if (Object.values(input).every((value) => value === undefined)) {
      throw new BadRequestException("At least one session field is required");
    }
    const session = await this.sessions.updateOwned(
      accessScopeFor(principal),
      sessionId,
      input,
      parseIfMatch(ifMatch)
    );
    reply.header("etag", `\"${session.version}\"`);
    return session;
  }
}
