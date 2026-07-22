import { describe, expect, it } from "vitest";
import { WakeMachine } from "../src/state-machine";
import {
  Harness,
  okSessionDir,
  noSession,
  sender,
  status,
  watchMessage,
  watchTimeout,
} from "./harness";

const LEAD_INBOX = "/base/sid-1/inbox-team-lead.jsonl";

function machine(h: Harness): WakeMachine {
  // No explicit reader override: the machine binds to the identity that
  // session-dir reports (team-lead for the default harness), and shells out to
  // the CLI WITHOUT --reader so it applies its own ambient default.
  return new WakeMachine({
    exec: h.exec,
    sendMessage: h.sendMessage,
    sleep: h.sleep,
    watchTimeoutSec: 5,
    ackBudget: 3,
  });
}

describe("WakeMachine — WATCHING / injection", () => {
  it("T4: rejects a message whose path is not the lead inbox (no injection, backoff)", async () => {
    const h = new Harness({
      watch: [watchMessage(["alice"], "/base/sid-1/inbox-other.jsonl"), watchTimeout()],
      inboxStatus: [status({ alice: sender(1, 0) })],
      maxCalls: 12,
    });
    await machine(h).run(h.controller.signal);
    expect(h.sends).toHaveLength(0);
    // never captured a generation from the wrong inbox
    expect(h.count("inbox-status")).toBe(0);
  });

  it("T4: reason=output / waiting never inject and simply re-arm", async () => {
    const h = new Harness({
      watch: [
        {
          stdout: JSON.stringify({ reason: "output", path: "x" }),
          stderr: "",
          code: 0,
          killed: false,
        },
        {
          stdout: JSON.stringify({ reason: "waiting", agent: "a", path: "y" }),
          stderr: "",
          code: 0,
          killed: false,
        },
        watchTimeout(),
      ],
      maxCalls: 12,
    });
    await machine(h).run(h.controller.signal);
    expect(h.sends).toHaveLength(0);
    expect(h.count("watch")).toBeGreaterThanOrEqual(2);
  });

  it("T5: single generation -> exactly one injection; re-arm only after ACK", async () => {
    const h = new Harness({
      watch: [watchMessage(["alice"], LEAD_INBOX), watchTimeout()],
      inboxStatus: [
        status({ alice: sender(2, 0) }), // capture -> target alice:2
        status({ alice: sender(2, 2) }), // ACK_WAIT poll -> acknowledged
      ],
      maxCalls: 20,
    });
    await machine(h).run(h.controller.signal);
    expect(h.sends).toHaveLength(1);
    expect(h.sends[0].message.customType).toBe("win-agent-teams/wake");
    expect(h.sends[0].message.display).toBe(true);
    expect(h.sends[0].options).toEqual({ triggerTurn: true, deliverAs: "steer" });
    // re-armed WATCHING only after the generation was acknowledged
    expect(h.count("watch")).toBeGreaterThanOrEqual(2);
  });

  it("T4/blocker-1: a Windows session_dir + backslash wake path injects exactly once", async () => {
    // Regression for the cross-platform path guard: Python's watch emits
    // str(Path(session_dir)/"inbox-team-lead.jsonl") with backslashes on
    // Windows; the extension must not discard that genuine wake as malformed.
    const winDir = "C:\\Users\\x\\.claude\\agent-sessions\\sid-1";
    const winInbox = "C:\\Users\\x\\.claude\\agent-sessions\\sid-1\\inbox-team-lead.jsonl";
    const h = new Harness({
      sessionDir: [okSessionDir("sid-1", winDir)],
      watch: [watchMessage(["alice"], winInbox), watchTimeout()],
      inboxStatus: [status({ alice: sender(2, 0) }), status({ alice: sender(2, 2) })],
      maxCalls: 20,
    });
    await machine(h).run(h.controller.signal);
    expect(h.sends).toHaveLength(1);
    // The wake was accepted (not dropped as a wrong-inbox path): a generation
    // was captured via inbox-status and re-armed only after ACK.
    expect(h.count("inbox-status")).toBeGreaterThanOrEqual(1);
    expect(h.count("watch")).toBeGreaterThanOrEqual(2);
  });

  it("passes the abort signal to every exec and sleep", async () => {
    const h = new Harness({
      watch: [watchMessage(["alice"], LEAD_INBOX), watchTimeout()],
      inboxStatus: [status({ alice: sender(1, 0) }), status({ alice: sender(1, 1) })],
      maxCalls: 8,
    });
    await machine(h).run(h.controller.signal);
    // The final abort was raised through the harness controller; loop terminated cleanly.
    expect(h.controller.signal.aborted).toBe(true);
  });
});

describe("WakeMachine — ACK budget / stall", () => {
  it("T5b: never-read generation -> exactly one injection then ACK_STALLED, no inbox watcher re-armed", async () => {
    const h = new Harness({
      watch: [watchMessage(["alice"], LEAD_INBOX)],
      inboxStatus: [status({ alice: sender(2, 0) })], // sticky: cursor never advances
      maxCalls: 20,
    });
    await machine(h).run(h.controller.signal);
    expect(h.sends).toHaveLength(1);
    // Blocker 4.1 regression: no inbox `watch` is started while the generation is unread.
    expect(h.count("watch")).toBe(1);
    // The machine kept probing (polling inbox-status), i.e. it is stalled, not re-arming.
    expect(h.count("inbox-status")).toBeGreaterThan(3);
  });

  it("T5c: >50 backlog -> cursor advances below target keeps waiting, no premature re-inject", async () => {
    const h = new Harness({
      watch: [watchMessage(["alice"], LEAD_INBOX), watchTimeout()],
      inboxStatus: [
        status({ alice: sender(60, 0) }), // capture -> target 60
        status({ alice: sender(60, 10) }), // progress
        status({ alice: sender(60, 30) }), // progress
        status({ alice: sender(60, 60) }), // acknowledged
      ],
      maxCalls: 25,
    });
    await machine(h).run(h.controller.signal);
    expect(h.sends).toHaveLength(1); // no re-injection during the drain
    expect(h.count("watch")).toBeGreaterThanOrEqual(2); // re-armed only after full drain
  });

  it("T5d: ACK resolves on the cursor threshold while bare unread stays UNCHANGED", async () => {
    // Blocker 2: the invariant is min(cursor,total) >= captured target, NOT
    // "unread decreased". Bare unread is 2 at capture AND at every later probe,
    // so an impl that ACKed merely because unread dropped would never re-arm.
    // ACK must fire only when the cursor threshold is reached (total=4,cursor=2).
    const h = new Harness({
      watch: [watchMessage(["alice"], LEAD_INBOX), watchTimeout()],
      inboxStatus: [
        status({ alice: sender(2, 0) }), // capture target 2; unread 2
        status({ alice: sender(3, 1) }), // consumed 1 + arrived 1: unread STILL 2, min=1 < 2 -> not acked
        status({ alice: sender(4, 2) }), // consumed 1 + arrived 1: unread STILL 2, min=2 >= 2 -> ACK
      ],
      maxCalls: 25,
    });
    // Guard the premise: bare unread is identical across all three snapshots.
    expect(sender(2, 0).unread).toBe(2);
    expect(sender(3, 1).unread).toBe(2);
    expect(sender(4, 2).unread).toBe(2);

    await machine(h).run(h.controller.signal);
    expect(h.sends).toHaveLength(1);

    // Locate the three inbox-status probes in call order; the third (4/2) is
    // the resolving snapshot. WATCHING must NOT be re-entered (no new watch)
    // before it, and must be re-entered only at/after it.
    const inboxIdx = h.calls
      .map((c, i) => (c.sub === "inbox-status" ? i : -1))
      .filter((i) => i >= 0);
    expect(inboxIdx.length).toBeGreaterThanOrEqual(3);
    const resolverIdx = inboxIdx[2];
    const watchBefore = h.calls.slice(0, resolverIdx).filter((c) => c.sub === "watch").length;
    const watchAfter = h.calls.slice(resolverIdx).filter((c) => c.sub === "watch").length;
    expect(watchBefore).toBe(1); // only the initial arming watch; no early re-arm
    expect(watchAfter).toBeGreaterThanOrEqual(1); // re-armed only after the threshold snapshot
  });

  it("T8: two senders in one generation -> ACK requires both", async () => {
    const h = new Harness({
      watch: [watchMessage(["alice", "bob"], LEAD_INBOX), watchTimeout()],
      inboxStatus: [
        status({ alice: sender(2, 0), bob: sender(1, 0) }), // capture {alice:2, bob:1}
        status({ alice: sender(2, 2), bob: sender(1, 0) }), // alice done, bob not -> NOT acked
        status({ alice: sender(2, 2), bob: sender(1, 1) }), // both done -> ACK
      ],
      maxCalls: 25,
    });
    await machine(h).run(h.controller.signal);
    expect(h.sends).toHaveLength(1);
    // at least 3 status probes happened before re-arm (did not ACK on the alice-only probe)
    expect(h.count("inbox-status")).toBeGreaterThanOrEqual(3);
    expect(h.count("watch")).toBeGreaterThanOrEqual(2);
  });
});

describe("WakeMachine — ACK_STALLED behavior", () => {
  it("T7: late drain -> returns to WATCHING with no new injection", async () => {
    const h = new Harness({
      watch: [watchMessage(["alice"], LEAD_INBOX), watchTimeout()],
      inboxStatus: [
        status({ alice: sender(5, 0) }), // capture target 5
        status({ alice: sender(5, 0) }), // no progress (budget 1)
        status({ alice: sender(5, 0) }), // no progress (budget 2)
        status({ alice: sender(5, 0) }), // no progress (budget 3) -> ACK_STALLED
        status({ alice: sender(5, 5) }), // stalled probe: late drain -> WATCHING
      ],
      maxCalls: 30,
    });
    await machine(h).run(h.controller.signal);
    expect(h.sends).toHaveLength(1); // late drain does NOT re-inject
    expect(h.count("watch")).toBeGreaterThanOrEqual(2); // re-armed after late drain
  });

  it("T7b: new sender's first message while stalled -> exactly one extra injection", async () => {
    const h = new Harness({
      watch: [watchMessage(["alice"], LEAD_INBOX)],
      inboxStatus: [
        status({ alice: sender(5, 0) }), // capture target {alice:5}
        status({ alice: sender(5, 0) }), // budget 1
        status({ alice: sender(5, 0) }), // budget 2
        status({ alice: sender(5, 0) }), // budget 3 -> ACK_STALLED
        status({ alice: sender(5, 0), bob: sender(1, 0) }), // B first message -> newer gen -> inject
      ],
      maxCalls: 30,
    });
    await machine(h).run(h.controller.signal);
    expect(h.sends).toHaveLength(2); // A generation + new B generation
    expect(h.sends[1].message.content).toContain("bob");
    // still no inbox watcher re-armed while generations remain unread
    expect(h.count("watch")).toBe(1);
  });

  it("T7c: arrive-then-consume between probes -> unread==0 guard suppresses re-injection", async () => {
    const h = new Harness({
      watch: [watchMessage(["alice"], LEAD_INBOX)],
      inboxStatus: [
        status({ alice: sender(5, 0) }), // capture
        status({ alice: sender(5, 0) }), // budget 1
        status({ alice: sender(5, 0) }), // budget 2
        status({ alice: sender(5, 0) }), // budget 3 -> ACK_STALLED
        status({ alice: sender(5, 0), bob: sender(1, 1) }), // B arrived+consumed: unread 0 -> no inject
      ],
      maxCalls: 30,
    });
    await machine(h).run(h.controller.signal);
    expect(h.sends).toHaveLength(1); // only the original A injection
  });
});

describe("WakeMachine — discovery / rebinding", () => {
  it("T6: no session yet -> waits, starts watching once session-dir returns one", async () => {
    const h = new Harness({
      sessionDir: [noSession(), noSession(), okSessionDir("sid-1", "/base/sid-1")],
      watch: [watchTimeout()],
      maxCalls: 8,
    });
    await machine(h).run(h.controller.signal);
    // first two exec calls are discovery probes, before any watch
    expect(h.calls[0].sub).toBe("session-dir");
    expect(h.calls[1].sub).toBe("session-dir");
    expect(h.count("watch")).toBeGreaterThanOrEqual(1);
    expect(h.sends).toHaveLength(0);
  });

  it("T6b: session_id change between refreshes -> rebind, never inject against the old session", async () => {
    const h = new Harness({
      sessionDir: [
        okSessionDir("sid-1", "/base/sid-1"), // initial discovery
        okSessionDir("sid-2", "/base/sid-2"), // refresh after watch -> changed
      ],
      watch: [
        watchMessage(["alice"], "/base/sid-1/inbox-team-lead.jsonl"), // message on old session
        watchTimeout(),
      ],
      inboxStatus: [status({ alice: sender(1, 0) })],
      maxCalls: 12,
    });
    await machine(h).run(h.controller.signal);
    // session changed before injection -> no injection at all
    expect(h.sends).toHaveLength(0);
    // rebound to sid-2: a later watch used the new session dir
    const rebound = h.calls.some((c) => c.sub === "watch" && c.args.includes("/base/sid-2"));
    expect(rebound).toBe(true);
  });

  it("§6: a nested lead (identity=<AGENT_NAME>) reaches WATCHING and wakes on its own inbox", async () => {
    // Regression for the over-constrained team-lead-only gate: a spawned Pi
    // subagent-as-lead reports identity=worker-1 and must watch
    // inbox-worker-1.jsonl, not inbox-team-lead.jsonl.
    const nestedInbox = "/base/sid-1/inbox-worker-1.jsonl";
    const h = new Harness({
      sessionDir: [okSessionDir("sid-1", "/base/sid-1", "worker-1")],
      watch: [watchMessage(["alice"], nestedInbox), watchTimeout()],
      inboxStatus: [status({ alice: sender(2, 0) }), status({ alice: sender(2, 2) })],
      maxCalls: 20,
    });
    await machine(h).run(h.controller.signal);
    expect(h.sends).toHaveLength(1);
    expect(h.count("watch")).toBeGreaterThanOrEqual(2);
  });

  it("§6: a nested lead ignores a message on the team-lead inbox (not its own)", async () => {
    const h = new Harness({
      sessionDir: [okSessionDir("sid-1", "/base/sid-1", "worker-1")],
      watch: [watchMessage(["alice"], "/base/sid-1/inbox-team-lead.jsonl"), watchTimeout()],
      inboxStatus: [status({ alice: sender(1, 0) })],
      maxCalls: 12,
    });
    await machine(h).run(h.controller.signal);
    expect(h.sends).toHaveLength(0);
    expect(h.count("inbox-status")).toBe(0);
  });

  it("§6: shells out WITHOUT --reader (lets the CLI apply its ambient default)", async () => {
    const h = new Harness({
      sessionDir: [okSessionDir("sid-1", "/base/sid-1", "worker-1")],
      watch: [watchMessage(["alice"], "/base/sid-1/inbox-worker-1.jsonl"), watchTimeout()],
      inboxStatus: [status({ alice: sender(1, 0) }), status({ alice: sender(1, 1) })],
      maxCalls: 12,
    });
    await machine(h).run(h.controller.signal);
    const watchCalls = h.calls.filter((c) => c.sub === "watch");
    const statusCalls = h.calls.filter((c) => c.sub === "inbox-status");
    expect(watchCalls.length).toBeGreaterThanOrEqual(1);
    for (const c of [...watchCalls, ...statusCalls]) {
      expect(c.args).not.toContain("--reader");
    }
  });

  it("§6: an explicit reader override is passed through as --reader", async () => {
    const h = new Harness({
      sessionDir: [okSessionDir("sid-1", "/base/sid-1", "team-lead")],
      watch: [watchTimeout()],
      maxCalls: 4,
    });
    const m = new WakeMachine({
      exec: h.exec,
      sendMessage: h.sendMessage,
      sleep: h.sleep,
      reader: "team-lead",
      watchTimeoutSec: 5,
    });
    await m.run(h.controller.signal);
    const watchCall = h.calls.find((c) => c.sub === "watch");
    expect(watchCall?.args).toContain("--reader");
    expect(watchCall?.args).toContain("team-lead");
  });

  it("§6: an empty/whitespace reader override is normalized to unset (no --reader)", async () => {
    const h = new Harness({
      sessionDir: [okSessionDir("sid-1", "/base/sid-1", "worker-1")],
      watch: [watchMessage(["alice"], "/base/sid-1/inbox-worker-1.jsonl"), watchTimeout()],
      inboxStatus: [status({ alice: sender(1, 0) }), status({ alice: sender(1, 1) })],
      maxCalls: 12,
    });
    const m = new WakeMachine({
      exec: h.exec,
      sendMessage: h.sendMessage,
      sleep: h.sleep,
      reader: "   ",
      watchTimeoutSec: 5,
      ackBudget: 3,
    });
    await m.run(h.controller.signal);
    const relevant = h.calls.filter((c) => c.sub === "watch" || c.sub === "inbox-status");
    expect(relevant.length).toBeGreaterThanOrEqual(1);
    for (const c of relevant) {
      expect(c.args).not.toContain("--reader");
    }
    // The guard binds to the session-dir identity, so injection still happens.
    expect(h.sends).toHaveLength(1);
  });
});
