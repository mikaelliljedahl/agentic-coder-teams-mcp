/**
 * Deterministic fakes for driving WakeMachine in unit tests.
 *
 * The machine is strictly single-flight (one awaited child at a time), so the
 * harness scripts one response array per subcommand (consumed in order, last
 * item is sticky) and stops the loop by aborting once a bound/predicate is hit.
 */
import type { SenderStatus } from "../src/cli";
import type { CustomMessageInput, ExecOptions, ExecResult, SendMessageOptions } from "../src/types";
import { AbortError } from "../src/util";

export function okSessionDir(sessionId: string, sessionDir: string): ExecResult {
  return { stdout: `${sessionId}\t${sessionDir}\tteam-lead\n`, stderr: "", code: 0, killed: false };
}

export function noSession(): ExecResult {
  return { stdout: "", stderr: "", code: 3, killed: false };
}

export function status(senders: Record<string, SenderStatus>): ExecResult {
  const payload = { schema: "inbox-status/1", reader: "team-lead", senders };
  return { stdout: JSON.stringify(payload), stderr: "", code: 0, killed: false };
}

export function sender(total: number, cursor: number): SenderStatus {
  return { total, cursor, unread: total - Math.min(cursor, total) };
}

export function watchMessage(from: string[], path: string): ExecResult {
  return {
    stdout: JSON.stringify({ reason: "message", from, path }),
    stderr: "",
    code: 0,
    killed: false,
  };
}

export function watchTimeout(): ExecResult {
  return { stdout: "", stderr: "", code: 2, killed: false };
}

export interface RecordedCall {
  sub: string;
  args: string[];
  sawAbortedSignal: boolean;
}

export interface RecordedSend {
  message: CustomMessageInput;
  options: SendMessageOptions | undefined;
}

class Script {
  private idx = 0;
  constructor(private readonly items: ExecResult[]) {}
  next(): ExecResult {
    if (this.items.length === 0) {
      throw new Error("harness: no scripted response for this subcommand");
    }
    const i = Math.min(this.idx, this.items.length - 1);
    this.idx += 1;
    return this.items[i];
  }
}

export interface HarnessOptions {
  sessionDir?: ExecResult[];
  watch?: ExecResult[];
  inboxStatus?: ExecResult[];
  /** Hard cap on completed exec calls before the loop is aborted (safety net). */
  maxCalls?: number;
  /** Abort as soon as this predicate returns true (evaluated after each exec call). */
  abortWhen?: (h: { calls: RecordedCall[]; sends: RecordedSend[] }) => boolean;
}

export class Harness {
  readonly controller = new AbortController();
  readonly calls: RecordedCall[] = [];
  readonly sends: RecordedSend[] = [];
  private callCount = 0;
  private readonly scripts: Record<string, Script>;
  private readonly maxCalls: number;
  private readonly abortWhen: HarnessOptions["abortWhen"];

  constructor(opts: HarnessOptions = {}) {
    this.scripts = {
      "session-dir": new Script(opts.sessionDir ?? [okSessionDir("sid-1", "/base/sid-1")]),
      "inbox-status": new Script(opts.inboxStatus ?? [status({})]),
      watch: new Script(opts.watch ?? [watchTimeout()]),
    };
    this.maxCalls = opts.maxCalls ?? 100;
    this.abortWhen = opts.abortWhen;
  }

  readonly exec = async (
    command: string,
    args: string[],
    options?: ExecOptions,
  ): Promise<ExecResult> => {
    if (options?.signal?.aborted) {
      throw new AbortError();
    }
    const sub = args[0];
    const script = this.scripts[sub];
    if (!script) {
      throw new Error(`harness: unexpected subcommand ${sub} (${command})`);
    }
    this.calls.push({ sub, args, sawAbortedSignal: false });
    const result = script.next();
    this.callCount += 1;
    const stop =
      this.callCount >= this.maxCalls ||
      (this.abortWhen ? this.abortWhen({ calls: this.calls, sends: this.sends }) : false);
    if (stop) {
      this.controller.abort();
    }
    return result;
  };

  readonly sendMessage = (message: CustomMessageInput, options?: SendMessageOptions): void => {
    this.sends.push({ message, options });
  };

  readonly sleep = async (_ms: number, signal?: AbortSignal): Promise<void> => {
    if (signal?.aborted) {
      throw new AbortError();
    }
  };

  count(sub: string): number {
    return this.calls.filter((c) => c.sub === sub).length;
  }
}
