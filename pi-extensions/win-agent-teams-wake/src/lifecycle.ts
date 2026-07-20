/**
 * Extension-owned lifecycle (plan §3.1, review-4 §4).
 *
 * `session_shutdown` carries no AbortSignal and `ctx.signal` is a per-turn
 * signal, so the extension owns its lifetime `AbortController`: create it on
 * `session_start`, start exactly one loop, and on `session_shutdown` abort it
 * and await loop completion. Reset is idempotent across reload/new/resume/fork.
 */
import { isAbortError } from "./util";

export interface LifecycleDeps {
  createController: () => AbortController;
  startLoop: (signal: AbortSignal) => Promise<void>;
}

export interface Lifecycle {
  start(): Promise<void>;
  shutdown(): Promise<void>;
  isRunning(): boolean;
}

export function createLifecycle(deps: LifecycleDeps): Lifecycle {
  let controller: AbortController | null = null;
  let loop: Promise<void> | null = null;

  async function teardown(): Promise<void> {
    if (controller) {
      controller.abort();
    }
    const pending = loop;
    controller = null;
    loop = null;
    if (pending) {
      try {
        await pending;
      } catch (err) {
        if (!isAbortError(err)) {
          throw err;
        }
      }
    }
  }

  return {
    async start(): Promise<void> {
      // Idempotent: abort any pre-existing loop before starting a new one so at
      // most one child is ever live across reload/new/resume/fork.
      await teardown();
      const c = deps.createController();
      controller = c;
      loop = deps.startLoop(c.signal);
    },
    async shutdown(): Promise<void> {
      await teardown();
    },
    isRunning(): boolean {
      return loop !== null;
    },
  };
}
