/**
 * Thin wrappers + strict-but-tolerant parsers for the read-only
 * `win-agent-teams` CLI surface the extension shells out to (plan §3.2):
 *   - `session-dir`                          -> discovery
 *   - `inbox-status <dir> --reader NAME`     -> non-consuming generation probe
 *   - `watch <dir> --reader NAME --timeout T`-> block until an inbox edge
 */
import path from "node:path";
import type { ExecResult, PiExec } from "./types";

export const CLI_COMMAND = "win-agent-teams";
export const DEFAULT_READER = "team-lead";

/**
 * Normalize path separators so a POSIX `/` path and a Windows `\` path compare
 * equal. The Python CLI emits `str(Path(session_dir)/"inbox-...jsonl")`, whose
 * separator follows the host OS (`\` on Windows, `/` on POSIX); the extension
 * must not reject a genuine wake purely because of separator style.
 */
function normalizeSeparators(p: string): string {
  return p.replace(/\\/g, "/").replace(/\/+$/, "");
}

/**
 * The inbox file the discovered session owns for a given identity (path guard
 * for watch). `identity` is the agent's own identity — `AGENT_NAME` for a
 * spawned agent / nested lead, or `team-lead` for the root lead — as reported by
 * `session-dir`. It is NOT hardcoded to `team-lead`.
 */
export function ownInboxPath(sessionDir: string, identity: string): string {
  return path.join(normalizeSeparators(sessionDir), `inbox-${identity}.jsonl`);
}

/**
 * Cross-platform equality check for the watch wake `path` against the agent's
 * own inbox for `identity`. Both sides are separator-normalized so a Windows
 * `session_dir` (`C:\\Users\\x\\...\\<id>`) and its backslash wake path match the
 * same way a POSIX pair does. Prevents dropping a real wake on Windows
 * (blocker 1) and, driven by the reported identity, matches a nested lead's
 * `inbox-<AGENT_NAME>.jsonl` rather than only `inbox-team-lead.jsonl`.
 */
export function isOwnInboxPath(candidate: string, sessionDir: string, identity: string): boolean {
  const expected = `${normalizeSeparators(sessionDir)}/inbox-${identity}.jsonl`;
  return normalizeSeparators(candidate) === expected;
}

export interface Discovery {
  sessionId: string;
  sessionDir: string;
  identity: string;
}

export type SessionDirResult =
  | { kind: "ok"; discovery: Discovery }
  | { kind: "no-session" }
  | { kind: "error"; message: string };

export function parseSessionDir(res: ExecResult): SessionDirResult {
  if (res.code === 3) {
    return { kind: "no-session" };
  }
  if (res.code !== 0) {
    return { kind: "error", message: res.stderr.trim() || `exit ${res.code}` };
  }
  const line = res.stdout.trim();
  const parts = line.split("\t");
  if (parts.length !== 3 || parts.some((p) => p.length === 0)) {
    return { kind: "error", message: `malformed session-dir line: ${JSON.stringify(line)}` };
  }
  const [sessionId, sessionDir, identity] = parts;
  return { kind: "ok", discovery: { sessionId, sessionDir, identity } };
}

export interface SenderStatus {
  total: number;
  cursor: number;
  unread: number;
}

export type InboxStatusResult =
  | { kind: "ok"; reader: string; senders: Record<string, SenderStatus> }
  | { kind: "bad-dir"; message: string }
  | { kind: "error"; message: string };

function isSenderStatus(value: unknown): value is SenderStatus {
  if (typeof value !== "object" || value === null) {
    return false;
  }
  const v = value as Record<string, unknown>;
  return (
    typeof v.total === "number" && typeof v.cursor === "number" && typeof v.unread === "number"
  );
}

export function parseInboxStatus(res: ExecResult): InboxStatusResult {
  if (res.code === 4) {
    return { kind: "bad-dir", message: res.stderr.trim() || "bad session dir" };
  }
  if (res.code !== 0) {
    return { kind: "error", message: res.stderr.trim() || `exit ${res.code}` };
  }
  let payload: unknown;
  try {
    payload = JSON.parse(res.stdout.trim());
  } catch {
    return { kind: "error", message: "inbox-status stdout is not JSON" };
  }
  if (typeof payload !== "object" || payload === null) {
    return { kind: "error", message: "inbox-status payload is not an object" };
  }
  const obj = payload as Record<string, unknown>;
  if (obj.schema !== "inbox-status/1") {
    return { kind: "error", message: `unexpected schema ${JSON.stringify(obj.schema)}` };
  }
  const rawSenders = obj.senders;
  if (typeof rawSenders !== "object" || rawSenders === null) {
    return { kind: "error", message: "inbox-status senders is not an object" };
  }
  const senders: Record<string, SenderStatus> = {};
  for (const [name, value] of Object.entries(rawSenders as Record<string, unknown>)) {
    if (!isSenderStatus(value)) {
      return { kind: "error", message: `malformed sender status for ${JSON.stringify(name)}` };
    }
    senders[name] = { total: value.total, cursor: value.cursor, unread: value.unread };
  }
  const reader = typeof obj.reader === "string" ? obj.reader : DEFAULT_READER;
  return { kind: "ok", reader, senders };
}

export type WatchResult =
  | { kind: "message"; from: string[]; path: string }
  | { kind: "output" }
  | { kind: "waiting" }
  | { kind: "timeout" }
  | { kind: "malformed"; detail: string };

export function parseWatch(res: ExecResult): WatchResult {
  if (res.code === 2) {
    return { kind: "timeout" };
  }
  if (res.code !== 0) {
    return { kind: "malformed", detail: res.stderr.trim() || `exit ${res.code}` };
  }
  // Plan §3.1: stderr noise on an exit-0 watch is not authoritative output —
  // treat it as malformed so the loop backs off instead of trusting a
  // warning-contaminated / version-skewed invocation.
  const noise = res.stderr.trim();
  if (noise.length > 0) {
    return { kind: "malformed", detail: `stderr on exit 0: ${noise}` };
  }
  const line = res.stdout.trim();
  if (line.length === 0) {
    return { kind: "malformed", detail: "empty stdout on exit 0" };
  }
  let payload: unknown;
  try {
    payload = JSON.parse(line);
  } catch {
    return { kind: "malformed", detail: "watch stdout is not JSON" };
  }
  if (typeof payload !== "object" || payload === null) {
    return { kind: "malformed", detail: "watch payload is not an object" };
  }
  const obj = payload as Record<string, unknown>;
  switch (obj.reason) {
    case "message": {
      const from = Array.isArray(obj.from)
        ? obj.from.filter((x): x is string => typeof x === "string")
        : [];
      const path = typeof obj.path === "string" ? obj.path : "";
      if (path.length === 0) {
        return { kind: "malformed", detail: "message wake without a path" };
      }
      return { kind: "message", from, path };
    }
    case "output":
      return { kind: "output" };
    case "waiting":
      return { kind: "waiting" };
    default:
      return { kind: "malformed", detail: `unknown reason ${JSON.stringify(obj.reason)}` };
  }
}

// ---- async runners -------------------------------------------------------

export function runSessionDir(exec: PiExec, signal: AbortSignal): Promise<SessionDirResult> {
  return exec(CLI_COMMAND, ["session-dir"], { signal }).then(parseSessionDir);
}

/**
 * `--reader NAME` args, or none. Omitting `--reader` lets the CLI apply its own
 * ambient default (`AGENT_NAME` or `team-lead`) — the agent's own identity —
 * which is what a spawned agent / nested lead needs. An explicit `reader` is
 * only ever an opt-in override.
 */
function readerArgs(reader: string | undefined): string[] {
  return reader ? ["--reader", reader] : [];
}

export function runInboxStatus(
  exec: PiExec,
  sessionDir: string,
  reader: string | undefined,
  signal: AbortSignal,
): Promise<InboxStatusResult> {
  return exec(CLI_COMMAND, ["inbox-status", sessionDir, ...readerArgs(reader)], { signal }).then(
    parseInboxStatus,
  );
}

export function runWatch(
  exec: PiExec,
  sessionDir: string,
  reader: string | undefined,
  timeoutSec: number,
  signal: AbortSignal,
): Promise<WatchResult> {
  return exec(
    CLI_COMMAND,
    ["watch", sessionDir, ...readerArgs(reader), "--timeout", String(timeoutSec)],
    { signal },
  ).then(parseWatch);
}
