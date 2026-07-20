/**
 * `win-agent-teams` Pi extension entry point.
 *
 * Loaded by Pi as an ExtensionFactory (`(pi: ExtensionAPI) => void`). It wakes
 * any Pi lead — at any nesting level — when a child posts to its OWN inbox
 * (`inbox-<AGENT_NAME>.jsonl`, or `inbox-team-lead.jsonl` for the root lead), by
 * running a single-flight cursor-generation state machine that shells out to the
 * read-only `win-agent-teams` CLI (`session-dir` / `inbox-status` / `watch`).
 * Activation is guarded (see `activate` below): a plain `pi` is a true no-op.
 *
 * Written against the pinned upstream API (see plan §10; devDependency
 * `@earendil-works/pi-coding-agent` is pinned to an exact version). Only the
 * `exec`, `sendMessage`, and `on(session_start|session_shutdown)` surface is used.
 */
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { createLifecycle } from "./src/lifecycle";
import { WakeMachine } from "./src/state-machine";
import type { PiExec, PiSendMessage } from "./src/types";
import { sleep } from "./src/util";

export { createLifecycle } from "./src/lifecycle";
export { WakeMachine } from "./src/state-machine";

export default function activate(pi: ExtensionAPI): void {
  // Activation guard (§2): only run inside a win-agent-teams session. Every
  // spawned agent (worker or nested subagent-as-lead) gets the extension via
  // the backend's `-e`, and the backend injects WIN_AGENT_TEAMS_SESSION_DIR —
  // that env var is the fast, subprocess-free signal. A bare human-launched root
  // lead has no such env, so it opts in explicitly with WIN_AGENT_TEAMS_LEAD=1
  // (set by the pi-lead launcher). Neither present → a plain `pi` → true no-op:
  // no handlers, no loop, no dependency on the CLI being on PATH.
  const inSession = !!process.env.WIN_AGENT_TEAMS_SESSION_DIR;
  const rootLeadOptIn = process.env.WIN_AGENT_TEAMS_LEAD === "1";
  if (!inSession && !rootLeadOptIn) {
    return;
  }

  const exec: PiExec = (command, args, options) => pi.exec(command, args, options);
  const sendMessage: PiSendMessage = (message, options) => pi.sendMessage(message, options);

  const lifecycle = createLifecycle({
    createController: () => new AbortController(),
    startLoop: (signal) => new WakeMachine({ exec, sendMessage, sleep }).run(signal),
  });

  // Owned controller created on session_start; aborted + awaited on shutdown.
  pi.on("session_start", async () => {
    await lifecycle.start();
  });

  pi.on("session_shutdown", async () => {
    await lifecycle.shutdown();
  });
}
