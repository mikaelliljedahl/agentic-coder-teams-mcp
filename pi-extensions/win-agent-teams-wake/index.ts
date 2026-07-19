/**
 * `win-agent-teams` Pi extension entry point.
 *
 * Loaded by Pi as an ExtensionFactory (`(pi: ExtensionAPI) => void`). It wakes a
 * team-lead Pi session when a worker posts to its inbox, by running a
 * single-flight cursor-generation state machine that shells out to the read-only
 * `win-agent-teams` CLI (`session-dir` / `inbox-status` / `watch`).
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
