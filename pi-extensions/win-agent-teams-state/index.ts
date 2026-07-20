/**
 * win-agent-teams state-reporting extension for the pi coding agent.
 *
 * The win-agent-teams coordinator tracks a spawned agent's coarse state by
 * watching a `state-<agent>.json` marker file in the team session directory
 * (schema shared with `claude_teams.hooks`: `{state, event, ts}` where state is
 * "running" | "waiting"). Claude Code and Codex write it via native lifecycle
 * hooks; pi has no hook CLI, so this extension writes it from pi's own
 * lifecycle events instead.
 *
 * It is a no-op unless launched by win-agent-teams (i.e. when the
 * WIN_AGENT_TEAMS_SESSION_DIR + AGENT_NAME env vars are present), so a user
 * running pi normally is unaffected. Marker writes are best-effort: a failure
 * to write status must never break the agent.
 *
 * Zero external dependencies (only node builtins) so it loads via `pi -e <dir>`
 * with no install step.
 */
import { mkdirSync, renameSync, writeFileSync } from "node:fs";
import { join } from "node:path";

type State = "running" | "waiting";

function writeMarker(state: State, event: string): void {
  const dir = process.env.WIN_AGENT_TEAMS_SESSION_DIR;
  const name = process.env.AGENT_NAME;
  if (!dir || !name) return;
  const target = join(dir, `state-${name}.json`);
  const tmp = `${target}.${process.pid}.tmp`;
  const payload = JSON.stringify({ state, event, ts: Date.now() / 1000 });
  try {
    mkdirSync(dir, { recursive: true });
    // Atomic replace: write to a pid-scoped temp then rename over the target,
    // so a concurrent reader never sees a half-written marker.
    writeFileSync(tmp, payload, "utf8");
    renameSync(tmp, target);
  } catch {
    /* best-effort: status reporting must never break the agent */
  }
}

// deno-lint-ignore no-explicit-any -- pi's ExtensionAPI, typed loosely to keep
// this extension dependency-free.
export default function (pi: any): void {
  // Any of these means the agent is actively working.
  pi.on("session_start", () => writeMarker("running", "session_start"));
  pi.on("turn_start", () => writeMarker("running", "turn_start"));
  pi.on("tool_call", () => writeMarker("running", "tool_call"));
  // Fired when pi will NOT continue automatically — the agent is idle and
  // waiting for input (a follow-up / message from the coordinator).
  pi.on("agent_settled", () => writeMarker("waiting", "agent_settled"));
}
