/**
 * Minimal Pi surface used by this extension, expressed as plain types so the
 * state machine can be unit-tested with fakes and never depends on Pi at
 * runtime. The real adapter (`pi-adapter.ts`) binds these to `ExtensionAPI`.
 *
 * These mirror the pinned upstream signatures (plan §10):
 *   - `ExecOptions` has `{ signal?, timeout?, cwd? }` and NO `env`.
 *   - `sendMessage` returns `void` (no turn handle).
 */

export interface ExecOptions {
  signal?: AbortSignal;
  timeout?: number;
  cwd?: string;
}

export interface ExecResult {
  stdout: string;
  stderr: string;
  code: number;
  killed?: boolean;
}

export type PiExec = (
  command: string,
  args: string[],
  options?: ExecOptions,
) => Promise<ExecResult>;

export interface CustomMessageInput {
  customType: string;
  content: string;
  display: boolean;
  details?: unknown;
}

export interface SendMessageOptions {
  triggerTurn?: boolean;
  deliverAs?: "steer" | "followUp" | "nextTurn";
}

export type PiSendMessage = (message: CustomMessageInput, options?: SendMessageOptions) => void;

export type Sleep = (ms: number, signal?: AbortSignal) => Promise<void>;
