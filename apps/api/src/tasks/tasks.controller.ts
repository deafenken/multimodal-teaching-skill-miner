import {
  BadRequestException,
  Body,
  Controller,
  Get,
  HttpCode,
  Inject,
  Param,
  Post
} from "@nestjs/common";

import type {AuthenticatedPrincipal} from "../auth/auth-provider.port";
import {CurrentPrincipal} from "../auth/current-principal.decorator";
import {CompatibilityStepDto, CreateTaskDto} from "./task.dto";
import {TasksService} from "./tasks.service";
import {accessScopeFor} from "../tenancy/access-scope";

@Controller()
export class TasksController {
  constructor(@Inject(TasksService) private readonly tasks: TasksService) {}

  @Post("api/v1/sessions/:sessionId/tasks")
  @HttpCode(202)
  enqueue(
    @CurrentPrincipal() principal: AuthenticatedPrincipal,
    @Param("sessionId") sessionId: string,
    @Body() input: CreateTaskDto
  ) {
    return this.tasks.enqueue(
      accessScopeFor(principal),
      sessionId,
      input.message,
      input.clientRequestId
    );
  }

  @Get("api/v1/tasks/:taskId")
  get(
    @CurrentPrincipal() principal: AuthenticatedPrincipal,
    @Param("taskId") taskId: string
  ) {
    return this.tasks.getOwned(accessScopeFor(principal), taskId);
  }

  @Post("api/v1/tasks/:taskId/cancel")
  @HttpCode(202)
  cancel(
    @CurrentPrincipal() principal: AuthenticatedPrincipal,
    @Param("taskId") taskId: string
  ) {
    return this.tasks.cancelOwned(accessScopeFor(principal), taskId);
  }

  @Post("api/step")
  @HttpCode(202)
  compatibilityStep(
    @CurrentPrincipal() principal: AuthenticatedPrincipal,
    @Body() input: CompatibilityStepDto
  ) {
    const sessionId = input.sessionId ?? input.session_id;
    const message = input.message ?? input.learner_response;
    if (!sessionId || !message?.trim()) {
      throw new BadRequestException(
        "sessionId/session_id and message/learner_response are required"
      );
    }
    return this.tasks.enqueue(
      accessScopeFor(principal),
      sessionId,
      message.trim(),
      input.clientRequestId ?? input.idempotency_key
    );
  }
}
