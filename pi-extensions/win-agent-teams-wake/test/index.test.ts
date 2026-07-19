import { describe, expect, it } from "vitest";
import activate from "../index";
import type { ExecOptions, ExecResult } from "../src/types";
import { AbortError } from "../src/util";

/**
 * T9 committed loadability evidence (blocker 4a).
 *
 * Drives the package's default-export factory with a FAKE ExtensionAPI to prove
 * it loads and wires as documented: it registers `session_start` /
 * `session_shutdown` handlers, `session_start` starts exactly one owned loop
 * (one live `watch` child), and `session_shutdown` aborts it. This does NOT
 * exercise a live Pi/model run (see README.md for the manual smoke test).
 */
describe("activate — package wiring (T9 loadability)", () => {
  it("registers lifecycle handlers, runs one owned loop, and shuts it down", async () => {
    const handlers: Record<string, () => Promise<void> | void> = {};
    let liveWatch = 0;
    let maxLiveWatch = 0;
    let sends = 0;

    const fakePi = {
      on: (event: string, handler: () => Promise<void> | void): void => {
        handlers[event] = handler;
      },
      exec: async (
        _command: string,
        args: string[],
        options?: ExecOptions,
      ): Promise<ExecResult> => {
        if (options?.signal?.aborted) {
          throw new AbortError();
        }
        const sub = args[0];
        if (sub === "session-dir") {
          return { stdout: "sid-1\t/base/sid-1\tteam-lead\n", stderr: "", code: 0, killed: false };
        }
        if (sub === "watch") {
          // A live child that blocks until the owned controller aborts.
          liveWatch += 1;
          maxLiveWatch = Math.max(maxLiveWatch, liveWatch);
          return await new Promise<ExecResult>((_resolve, reject) => {
            const signal = options?.signal;
            const finish = (): void => {
              liveWatch -= 1;
              reject(new AbortError());
            };
            if (signal?.aborted) {
              finish();
              return;
            }
            signal?.addEventListener("abort", finish);
          });
        }
        return {
          stdout: JSON.stringify({ schema: "inbox-status/1", reader: "team-lead", senders: {} }),
          stderr: "",
          code: 0,
          killed: false,
        };
      },
      sendMessage: (): void => {
        sends += 1;
      },
    };

    // The factory must accept a plain ExtensionAPI-shaped object without throwing.
    activate(fakePi as unknown as Parameters<typeof activate>[0]);

    expect(typeof handlers["session_start"]).toBe("function");
    expect(typeof handlers["session_shutdown"]).toBe("function");

    await handlers["session_start"]();
    // Give the loop a tick to reach the blocking watch child.
    await new Promise((r) => setTimeout(r, 10));
    expect(maxLiveWatch).toBe(1); // exactly one owned loop / live child

    await handlers["session_shutdown"](); // abort the owned controller + await teardown
    expect(liveWatch).toBe(0); // the child was aborted
    expect(sends).toBe(0); // nothing injected in this idle scenario
  });
});
