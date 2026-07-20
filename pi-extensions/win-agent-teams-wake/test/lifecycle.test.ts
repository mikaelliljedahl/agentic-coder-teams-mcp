import { describe, expect, it } from "vitest";
import { createLifecycle } from "../src/lifecycle";
import { WakeMachine } from "../src/state-machine";
import type { ExecOptions, ExecResult } from "../src/types";
import { AbortError, sleep } from "../src/util";
import { okSessionDir } from "./harness";

describe("createLifecycle — single-flight & idempotent reset (T11)", () => {
  it("never runs two loops at once across start/reload/resume and resets idempotently", async () => {
    let live = 0;
    let maxLive = 0;
    let starts = 0;
    const abortedSignals: boolean[] = [];

    const lifecycle = createLifecycle({
      createController: () => new AbortController(),
      startLoop: (signal) => {
        starts += 1;
        live += 1;
        maxLive = Math.max(maxLive, live);
        return new Promise<void>((resolve) => {
          signal.addEventListener("abort", () => {
            abortedSignals.push(true);
            live -= 1;
            resolve();
          });
        });
      },
    });

    await lifecycle.start(); // session_start
    expect(lifecycle.isRunning()).toBe(true);
    await lifecycle.start(); // reload without shutdown: must abort the first
    await lifecycle.start(); // resume: still only one live
    expect(maxLive).toBe(1);

    await lifecycle.shutdown();
    expect(lifecycle.isRunning()).toBe(false);
    // idempotent shutdown
    await lifecycle.shutdown();

    await lifecycle.start(); // start again after shutdown
    expect(lifecycle.isRunning()).toBe(true);
    await lifecycle.shutdown();

    expect(maxLive).toBe(1);
    expect(starts).toBe(4);
    expect(abortedSignals.length).toBe(4); // every started loop was aborted exactly once
  });
});

describe("createLifecycle + WakeMachine — one live production child (T11 strengthened)", () => {
  it("keeps at most one live pi.exec watch/inbox-status child across start/reload/resume", async () => {
    // Counts REAL production children (watch / inbox-status) issued by the
    // actual WakeMachine, not synthetic loop promises. session-dir is the
    // discovery probe and is intentionally not counted.
    let liveChildren = 0;
    let maxLive = 0;

    const exec = async (
      _command: string,
      args: string[],
      options?: ExecOptions,
    ): Promise<ExecResult> => {
      if (options?.signal?.aborted) {
        throw new AbortError();
      }
      const sub = args[0];
      if (sub === "session-dir") {
        return okSessionDir("sid-1", "/base/sid-1");
      }
      // watch (and, in other scenarios, inbox-status) are the live children.
      liveChildren += 1;
      maxLive = Math.max(maxLive, liveChildren);
      try {
        return await new Promise<ExecResult>((_resolve, reject) => {
          const signal = options?.signal;
          const finish = (): void => reject(new AbortError());
          if (signal?.aborted) {
            finish();
            return;
          }
          signal?.addEventListener("abort", finish);
        });
      } finally {
        liveChildren -= 1;
      }
    };

    const startLoop = (signal: AbortSignal): Promise<void> =>
      new WakeMachine({
        exec,
        sendMessage: () => {},
        sleep,
        reader: "team-lead",
        watchTimeoutSec: 5,
        ackBudget: 3,
      }).run(signal);

    const lifecycle = createLifecycle({
      createController: () => new AbortController(),
      startLoop,
    });

    const tick = (): Promise<void> => new Promise((r) => setTimeout(r, 10));

    await lifecycle.start(); // session_start
    await tick();
    await lifecycle.start(); // reload without shutdown -> must abort the first child
    await tick();
    await lifecycle.start(); // resume -> still only one live child
    await tick();

    expect(maxLive).toBe(1);

    await lifecycle.shutdown();
    expect(liveChildren).toBe(0);
    expect(lifecycle.isRunning()).toBe(false);
  });
});

describe("session_shutdown aborts the owned controller (T10)", () => {
  it("terminates the live watch child and never injects after shutdown", async () => {
    const sends: unknown[] = [];
    let watchSawAbort = false;

    // A fake exec whose `watch` call blocks until the owned signal aborts,
    // modelling a live child that Pi terminates on shutdown.
    const exec = async (
      _command: string,
      args: string[],
      options?: ExecOptions,
    ): Promise<ExecResult> => {
      const sub = args[0];
      if (sub === "session-dir") {
        return okSessionDir("sid-1", "/base/sid-1");
      }
      if (sub === "watch") {
        return await new Promise<ExecResult>((_resolve, reject) => {
          const signal = options?.signal;
          if (signal?.aborted) {
            watchSawAbort = true;
            return reject(new AbortError());
          }
          signal?.addEventListener("abort", () => {
            watchSawAbort = true;
            reject(new AbortError());
          });
        });
      }
      // inbox-status should not be reached before shutdown in this scenario
      return {
        stdout: JSON.stringify({ schema: "inbox-status/1", reader: "team-lead", senders: {} }),
        stderr: "",
        code: 0,
        killed: false,
      };
    };

    const lifecycle = createLifecycle({
      createController: () => new AbortController(),
      startLoop: (signal) =>
        new WakeMachine({
          exec,
          sendMessage: (m) => {
            sends.push(m);
          },
          sleep: async (_ms, signal2) => {
            if (signal2?.aborted) throw new AbortError();
          },
          reader: "team-lead",
          watchTimeoutSec: 5,
          ackBudget: 3,
        }).run(signal),
    });

    await lifecycle.start();
    // give the loop a tick to reach the blocking watch
    await new Promise((r) => setTimeout(r, 10));

    await lifecycle.shutdown(); // must abort the owned controller and await loop completion

    expect(watchSawAbort).toBe(true);
    expect(sends).toHaveLength(0);
    expect(lifecycle.isRunning()).toBe(false);
  });
});
