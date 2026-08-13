import {createParamDecorator, ExecutionContext} from "@nestjs/common";

import type {AuthenticatedFastifyRequest} from "./authentication.guard";

export const CurrentPrincipal = createParamDecorator(
  (_data: unknown, context: ExecutionContext) => {
    const request = context.switchToHttp().getRequest<AuthenticatedFastifyRequest>();
    return request.principal;
  }
);
