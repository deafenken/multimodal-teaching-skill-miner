import {AppConfigService} from "../config/app-config.service";
import type {
  ModelProviderPort,
  ModelProviderStatus,
  ModelStreamEvent,
  ModelTurnCommand
} from "./model-provider.port";

export class UnconfiguredAnthropicProvider implements ModelProviderPort {
  constructor(private readonly config: AppConfigService) {}

  async status(): Promise<ModelProviderStatus> {
    const supported = this.config.modelProvider === "anthropic";
    return {
      provider: this.config.modelProvider,
      model: this.config.modelName,
      configured: false,
      streaming: true,
      toolUse: true,
      reason: !supported
        ? "unsupported_provider"
        : this.config.anthropicCredentialPresent
          ? "adapter_not_enabled"
          : "credential_missing"
    };
  }

  async *streamTurn(_command: ModelTurnCommand): AsyncIterable<ModelStreamEvent> {
    throw new Error(
      "Anthropic runtime adapter is not enabled in the MVP BFF; run it in an isolated worker"
    );
  }
}
