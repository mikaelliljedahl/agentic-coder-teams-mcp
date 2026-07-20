import type { Sleep } from "./types";

/**
 * Error raised on abort. Matches (by `name`) the `AbortError` DOMException that
 * Node's AbortController/`pi.exec` raise at runtime, so `isAbortError` covers
 * both our own aborts and real child-termination aborts.
 */
export class AbortError extends Error {
  constructor(message = "aborted") {
    super(message);
    this.name = "AbortError";
  }
}

/** Thrown/observed when the owned AbortController fires; treated as a clean loop exit. */
export function isAbortError(err: unknown): boolean {
  return (
    typeof err === "object" && err !== null && (err as { name?: string }).name === "AbortError"
  );
}

/** Abortable delay. Rejects with an AbortError if the signal is (or becomes) aborted. */
export const sleep: Sleep = (ms, signal) =>
  new Promise<void>((resolve, reject) => {
    if (signal?.aborted) {
      reject(new AbortError());
      return;
    }
    const timer = setTimeout(() => {
      cleanup();
      resolve();
    }, ms);
    const onAbort = (): void => {
      cleanup();
      reject(new AbortError());
    };
    const cleanup = (): void => {
      clearTimeout(timer);
      signal?.removeEventListener("abort", onAbort);
    };
    signal?.addEventListener("abort", onAbort, { once: true });
  });

/** Capped exponential backoff used for discovery, watch errors, and ACK polling. */
export class Backoff {
  private current: number;
  constructor(
    private readonly baseMs: number,
    private readonly capMs: number,
    private readonly factor = 2,
  ) {
    this.current = baseMs;
  }

  next(): number {
    const value = this.current;
    this.current = Math.min(this.current * this.factor, this.capMs);
    return value;
  }

  reset(): void {
    this.current = this.baseMs;
  }
}
