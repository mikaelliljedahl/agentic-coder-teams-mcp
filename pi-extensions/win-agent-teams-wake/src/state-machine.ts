/**
 * Single-flight cursor-generation state machine (plan §3.1).
 *
 * DISCOVERING -> WATCHING -> WAKE_PENDING -> ACK_WAIT -> (ACK_STALLED) -> ...
 *
 * The loop is strictly sequential: exactly one `pi.exec` child is awaited at a
 * time (single-flight). The owned AbortSignal is threaded into every exec and
 * every backoff wait so `session_shutdown` tears the loop down promptly.
 */
import {
  isOwnInboxPath,
  runInboxStatus,
  runSessionDir,
  runWatch,
  type Discovery,
  type InboxStatusResult,
} from "./cli";
import {
  captureGeneration,
  consumedTotal,
  hasNewerGeneration,
  isAcknowledged,
  type Generation,
} from "./generation";
import type { PiExec, PiSendMessage } from "./types";
import { Backoff, isAbortError } from "./util";

export interface WakeMachineOptions {
  exec: PiExec;
  sendMessage: PiSendMessage;
  sleep: (ms: number, signal?: AbortSignal) => Promise<void>;
  /**
   * Optional explicit reader override. When omitted (the normal case) the
   * extension watches the agent's OWN identity: it shells out WITHOUT `--reader`
   * so the CLI applies its ambient default (`AGENT_NAME` or `team-lead`), and
   * binds the watched inbox to the identity `session-dir` reports. Set this only
   * to force a specific reader; do NOT default it to `team-lead`.
   */
  reader?: string;
  /** `watch --timeout` value, seconds. */
  watchTimeoutSec?: number;
  /** Consecutive no-progress ACK probes tolerated before ACK_STALLED. */
  ackBudget?: number;
  backoffBaseMs?: number;
  backoffCapMs?: number;
}

type State = "DISCOVERING" | "WATCHING" | "ACK_WAIT" | "ACK_STALLED";

type RefreshOutcome = "same" | "changed" | "unavailable";

export class WakeMachine {
  private readonly exec: PiExec;
  private readonly sendMessage: PiSendMessage;
  private readonly sleep: (ms: number, signal?: AbortSignal) => Promise<void>;
  private readonly readerOverride: string | undefined;
  private readonly watchTimeoutSec: number;
  private readonly ackBudget: number;

  private readonly discoverBackoff: Backoff;
  private readonly watchBackoff: Backoff;
  private readonly ackBackoff: Backoff;
  private readonly stalledBackoff: Backoff;

  private discovery: Discovery | null = null;
  private generation: Generation | null = null;
  private lastProgress = 0;
  private noProgressProbes = 0;

  constructor(opts: WakeMachineOptions) {
    this.exec = opts.exec;
    this.sendMessage = opts.sendMessage;
    this.sleep = opts.sleep;
    this.readerOverride = opts.reader;
    this.watchTimeoutSec = opts.watchTimeoutSec ?? 55;
    this.ackBudget = opts.ackBudget ?? 5;
    const base = opts.backoffBaseMs ?? 500;
    const cap = opts.backoffCapMs ?? 30_000;
    this.discoverBackoff = new Backoff(base, cap);
    this.watchBackoff = new Backoff(base, cap);
    this.ackBackoff = new Backoff(base, cap);
    this.stalledBackoff = new Backoff(base, cap);
  }

  /** Run the loop until the signal aborts. Resolves cleanly on abort. */
  async run(signal: AbortSignal): Promise<void> {
    let state: State = "DISCOVERING";
    try {
      while (!signal.aborted) {
        switch (state) {
          case "DISCOVERING":
            state = await this.discovering(signal);
            break;
          case "WATCHING":
            state = await this.watching(signal);
            break;
          case "ACK_WAIT":
            state = await this.ackWait(signal);
            break;
          case "ACK_STALLED":
            state = await this.ackStalled(signal);
            break;
        }
      }
    } catch (err) {
      if (isAbortError(err) || signal.aborted) {
        return;
      }
      throw err;
    }
  }

  /**
   * The identity whose inbox this machine watches: the explicit override when
   * set, else the identity the current discovery reports (the agent's OWN
   * identity — `AGENT_NAME` for a spawned agent / nested lead, `team-lead` for
   * the root lead).
   */
  private activeReader(): string {
    return this.readerOverride ?? this.discovery!.identity;
  }

  private async discovering(signal: AbortSignal): Promise<State> {
    const r = await runSessionDir(this.exec, signal);
    if (r.kind === "ok") {
      // A live session exists. Bind to the reported identity (own inbox) rather
      // than comparing against a hardcoded `team-lead`: a spawned agent / nested
      // lead reports identity=<AGENT_NAME> and must be watched too (§6).
      this.discovery = r.discovery;
      this.generation = null;
      this.discoverBackoff.reset();
      return "WATCHING";
    }
    // No session yet or an internal error: keep polling — never watch or inject
    // without a live session.
    await this.sleep(this.discoverBackoff.next(), signal);
    return "DISCOVERING";
  }

  /**
   * Re-run discovery. Rebinds `this.discovery` when the same session is still
   * present; reports `changed` when the session id moved (or the session became
   * unavailable), which the caller turns into DISCOVERING.
   */
  private async refresh(signal: AbortSignal): Promise<RefreshOutcome> {
    const r = await runSessionDir(this.exec, signal);
    if (r.kind === "ok") {
      const changed = this.discovery === null || r.discovery.sessionId !== this.discovery.sessionId;
      this.discovery = r.discovery;
      return changed ? "changed" : "same";
    }
    return "unavailable";
  }

  private async watching(signal: AbortSignal): Promise<State> {
    const dir = this.discovery!.sessionDir;
    const w = await runWatch(this.exec, dir, this.readerOverride, this.watchTimeoutSec, signal);

    // Refresh discovery after every watch exit (plan §3.3).
    const ref = await this.refresh(signal);
    if (ref !== "same") {
      this.generation = null;
      return "DISCOVERING";
    }

    if (w.kind === "message" && isOwnInboxPath(w.path, dir, this.activeReader())) {
      const st = await runInboxStatus(this.exec, dir, this.readerOverride, signal);
      if (st.kind !== "ok") {
        await this.sleep(this.watchBackoff.next(), signal);
        return "WATCHING";
      }
      const gen = captureGeneration(st.senders);
      if (Object.keys(gen).length === 0) {
        // Arrive-then-consume before capture: nothing unread, just re-arm.
        this.watchBackoff.reset();
        return "WATCHING";
      }
      // Refresh immediately before injection; never inject against a moved session.
      const ref2 = await this.refresh(signal);
      if (ref2 !== "same") {
        this.generation = null;
        return "DISCOVERING";
      }
      this.inject(gen, st);
      this.watchBackoff.reset();
      return "ACK_WAIT";
    }

    if (w.kind === "timeout" || w.kind === "output" || w.kind === "waiting") {
      // Re-arm. output/waiting are ignored in v1 (OQ6).
      this.watchBackoff.reset();
      return "WATCHING";
    }

    // Malformed line, non-0/non-2 exit, or a message on the wrong inbox path:
    // back off (no tight loop) and re-arm.
    await this.sleep(this.watchBackoff.next(), signal);
    return "WATCHING";
  }

  private async ackWait(signal: AbortSignal): Promise<State> {
    const ref = await this.refresh(signal);
    if (ref !== "same") {
      this.generation = null;
      return "DISCOVERING";
    }
    const st = await runInboxStatus(
      this.exec,
      this.discovery!.sessionDir,
      this.readerOverride,
      signal,
    );
    if (st.kind !== "ok") {
      await this.sleep(this.ackBackoff.next(), signal);
      return "ACK_WAIT";
    }
    if (isAcknowledged(this.generation!, st.senders)) {
      this.generation = null;
      this.ackBackoff.reset();
      return "WATCHING";
    }
    const progress = consumedTotal(this.generation!, st.senders);
    if (progress > this.lastProgress) {
      this.lastProgress = progress;
      this.noProgressProbes = 0; // progress refreshes the budget
    } else {
      this.noProgressProbes += 1;
    }
    if (this.noProgressProbes >= this.ackBudget) {
      this.ackBackoff.reset();
      return "ACK_STALLED";
    }
    await this.sleep(this.ackBackoff.next(), signal);
    return "ACK_WAIT";
  }

  private async ackStalled(signal: AbortSignal): Promise<State> {
    // Session change is handled first via the discovery refresh (rule 3);
    // rebinding sooner is safe and never injects against the old session.
    const ref = await this.refresh(signal);
    if (ref !== "same") {
      this.generation = null;
      return "DISCOVERING";
    }
    const st = await runInboxStatus(
      this.exec,
      this.discovery!.sessionDir,
      this.readerOverride,
      signal,
    );
    if (st.kind !== "ok") {
      await this.sleep(this.stalledBackoff.next(), signal);
      return "ACK_STALLED";
    }
    // Rule 1: late drain -> WATCHING (no new injection).
    if (isAcknowledged(this.generation!, st.senders)) {
      this.generation = null;
      this.stalledBackoff.reset();
      return "WATCHING";
    }
    // Rule 2: strictly-newer generation -> capture fresh targets, inject once.
    if (hasNewerGeneration(this.generation!, st.senders)) {
      const gen = captureGeneration(st.senders);
      const ref2 = await this.refresh(signal);
      if (ref2 !== "same") {
        this.generation = null;
        return "DISCOVERING";
      }
      this.inject(gen, st);
      this.stalledBackoff.reset();
      return "ACK_WAIT";
    }
    await this.sleep(this.stalledBackoff.next(), signal);
    return "ACK_STALLED";
  }

  /** Inject exactly one wake for a captured generation and begin its ACK budget. */
  private inject(gen: Generation, st: InboxStatusResult & { kind: "ok" }): void {
    const senders = Object.keys(gen);
    this.sendMessage(
      {
        customType: "win-agent-teams/wake",
        content:
          `📨 New inbox message(s) from ${senders.join(", ")}. ` +
          "Call read_messages and keep draining while has_more, then act on each.",
        display: true,
      },
      { triggerTurn: true, deliverAs: "steer" },
    );
    this.generation = gen;
    this.lastProgress = consumedTotal(gen, st.senders);
    this.noProgressProbes = 0;
  }
}
