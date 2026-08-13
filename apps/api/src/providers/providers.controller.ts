import {Controller, Get, Inject} from "@nestjs/common";

import {MODEL_PROVIDER} from "../platform/tokens";
import type {ModelProviderPort} from "./model-provider.port";

@Controller("api/v1/providers")
export class ProvidersController {
  constructor(@Inject(MODEL_PROVIDER) private readonly modelProvider: ModelProviderPort) {}

  @Get("model")
  status() {
    return this.modelProvider.status();
  }
}
