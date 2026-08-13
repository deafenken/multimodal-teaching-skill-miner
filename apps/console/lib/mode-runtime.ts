export type WorkbenchMode = "chat" | "teach";

export interface ModeRequestIdentity {
  projectId: string | null;
  surfaceId: string | null;
}

export interface ModeRequestToken extends ModeRequestIdentity {
  mode: WorkbenchMode;
  generation: number;
}

interface ModeSlot {
  generation: number;
  controller: AbortController | null;
}

/**
 * Owns the two independent request slots used by the workbench.  Keeping the
 * generations beside their transport AbortControllers makes it impossible for
 * stopping Chat to invalidate Teach (or vice versa). Aborting a slot only
 * detaches its HTTP subscription; the server task continues unless the user
 * sends the explicit durable task-cancel command.
 */
export class ModeRequestSlots {
  readonly #slots: Record<WorkbenchMode, ModeSlot> = {
    chat: {generation: 0, controller: null},
    teach: {generation: 0, controller: null},
  };

  begin(mode: WorkbenchMode, identity: ModeRequestIdentity, controller = new AbortController()): ModeRequestToken {
    const slot = this.#slots[mode];
    slot.controller?.abort();
    slot.generation += 1;
    slot.controller = controller;
    return {mode, generation: slot.generation, ...identity};
  }

  isCurrent(token: ModeRequestToken, identity: ModeRequestIdentity): boolean {
    return this.#slots[token.mode].generation === token.generation
      && token.projectId === identity.projectId
      && token.surfaceId === identity.surfaceId;
  }

  finish(token: ModeRequestToken, controller: AbortController): boolean {
    const slot = this.#slots[token.mode];
    if (slot.generation !== token.generation || slot.controller !== controller) return false;
    slot.controller = null;
    return true;
  }

  cancel(mode: WorkbenchMode): number {
    const slot = this.#slots[mode];
    slot.generation += 1;
    slot.controller?.abort();
    slot.controller = null;
    return slot.generation;
  }

  cancelAll(): Record<WorkbenchMode, number> {
    return {chat: this.cancel("chat"), teach: this.cancel("teach")};
  }

  generation(mode: WorkbenchMode): number {
    return this.#slots[mode].generation;
  }
}

export function nextChatFollowUp<T extends {threadId: string}>(
  prompts: readonly T[],
  chatRunning: boolean,
  activeThreadId: string,
): T | undefined {
  if (chatRunning) return undefined;
  return prompts.find((prompt) => prompt.threadId === activeThreadId);
}
