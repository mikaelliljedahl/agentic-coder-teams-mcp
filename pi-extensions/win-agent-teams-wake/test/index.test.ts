import { afterEach, beforeEach, describe, expect, it } from "vitest";
import activate from "../index";
import type { ExecOptions, ExecResult } from "../src/types";
import { AbortError } from "../src/util";

/** A minimal fake ExtensionAPI recording the handlers activate() registers. */
function fakeHandlerRecorder(): {
  handlers: Record<string, () => Promise<void> | void>;
  pi: Parameters<typeof activate>[0];
} {
  const handlers: Record<string, () => Promise<void> | void> = {};
  const pi = {
    on: (event: string, handler: () => Promise<void> | void): void => {
      handlers[event] = handler;
    },
    exec: async (): Promise<ExecResult> => ({
      stdout: "",
      stderr: "",
      code: 2,
      killed: false,
    }),
    sendMessage: (): void => {},
  };
  return { handlers, pi: pi as unknown as Parameters<typeof activate>[0] };
}

describe("activate — activation guard (§2)", () => {
  const saved = {
    session: process.env.WIN_AGENT_TEAMS_SESSION_DIR,
    lead: process.env.WIN_AGENT_TEAMS_LEAD,
  };
  beforeEach(() => {
    delete process.env.WIN_AGENT_TEAMS_SESSION_DIR;
    delete process.env.WIN_AGENT_TEAMS_LEAD;
  });
  afterEach(() => {
    if (saved.session === undefined) delete process.env.WIN_AGENT_TEAMS_SESSION_DIR;
    else process.env.WIN_AGENT_TEAMS_SESSION_DIR = saved.session;
    if (saved.lead === undefined) delete process.env.WIN_AGENT_TEAMS_LEAD;
    else process.env.WIN_AGENT_TEAMS_LEAD = saved.lead;
  });

  it("is a true no-op when neither session dir nor lead opt-in is set", () => {
    const { handlers, pi } = fakeHandlerRecorder();
    activate(pi);
    expect(handlers["session_start"]).toBeUndefined();
    expect(handlers["session_shutdown"]).toBeUndefined();
  });

  it("activates for a spawned agent (WIN_AGENT_TEAMS_SESSION_DIR present, no flag)", () => {
    process.env.WIN_AGENT_TEAMS_SESSION_DIR = "/base/sid-1";
    const { handlers, pi } = fakeHandlerRecorder();
    activate(pi);
    expect(typeof handlers["session_start"]).toBe("function");
    expect(typeof handlers["session_shutdown"]).toBe("function");
  });

  it("activates for the flagged root lead (WIN_AGENT_TEAMS_LEAD=1, no session dir)", () => {
    process.env.WIN_AGENT_TEAMS_LEAD = "1";
    const { handlers, pi } = fakeHandlerRecorder();
    activate(pi);
    expect(typeof handlers["session_start"]).toBe("function");
  });
});

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
  const savedSession = process.env.WIN_AGENT_TEAMS_SESSION_DIR;
  beforeEach(() => {
    // The T9 loop only starts when the activation guard passes; a spawned agent
    // has WIN_AGENT_TEAMS_SESSION_DIR injected by the backend.
    process.env.WIN_AGENT_TEAMS_SESSION_DIR = "/base/sid-1";
  });
  afterEach(() => {
    if (savedSession === undefined) delete process.env.WIN_AGENT_TEAMS_SESSION_DIR;
    else process.env.WIN_AGENT_TEAMS_SESSION_DIR = savedSession;
  });

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
