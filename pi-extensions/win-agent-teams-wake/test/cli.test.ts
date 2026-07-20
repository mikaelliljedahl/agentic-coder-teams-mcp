import { describe, expect, it } from "vitest";
import {
  DEFAULT_READER,
  isLeadInboxPath,
  leadInboxPath,
  parseInboxStatus,
  parseSessionDir,
  parseWatch,
} from "../src/cli";
import type { ExecResult } from "../src/types";

function res(partial: Partial<ExecResult>): ExecResult {
  return { stdout: "", stderr: "", code: 0, killed: false, ...partial };
}

describe("parseSessionDir", () => {
  it("parses a tab-separated ok line (exit 0)", () => {
    const r = parseSessionDir(res({ code: 0, stdout: "sid-1\t/base/sid-1\tteam-lead\n" }));
    expect(r).toEqual({
      kind: "ok",
      discovery: { sessionId: "sid-1", sessionDir: "/base/sid-1", identity: "team-lead" },
    });
  });

  it("reports no-session on exit 3", () => {
    expect(parseSessionDir(res({ code: 3, stdout: "" })).kind).toBe("no-session");
  });

  it("reports error on exit 1", () => {
    const r = parseSessionDir(res({ code: 1, stderr: "boom" }));
    expect(r.kind).toBe("error");
  });

  it("treats a malformed ok line as an error, not a crash", () => {
    const r = parseSessionDir(res({ code: 0, stdout: "onlyonecolumn\n" }));
    expect(r.kind).toBe("error");
  });
});

describe("parseInboxStatus", () => {
  it("parses the inbox-status/1 schema (exit 0)", () => {
    const payload = {
      schema: "inbox-status/1",
      reader: "team-lead",
      senders: { alice: { total: 3, cursor: 1, unread: 2 } },
    };
    const r = parseInboxStatus(res({ code: 0, stdout: JSON.stringify(payload) + "\n" }));
    expect(r).toEqual({
      kind: "ok",
      reader: "team-lead",
      senders: { alice: { total: 3, cursor: 1, unread: 2 } },
    });
  });

  it("parses an empty inbox", () => {
    const payload = { schema: "inbox-status/1", reader: "team-lead", senders: {} };
    const r = parseInboxStatus(res({ code: 0, stdout: JSON.stringify(payload) }));
    expect(r).toEqual({ kind: "ok", reader: "team-lead", senders: {} });
  });

  it("reports bad-dir on exit 4", () => {
    expect(parseInboxStatus(res({ code: 4, stderr: "outside base" })).kind).toBe("bad-dir");
  });

  it("reports error on exit 1", () => {
    expect(parseInboxStatus(res({ code: 1, stderr: "boom" })).kind).toBe("error");
  });

  it("reports error on malformed json / wrong schema", () => {
    expect(parseInboxStatus(res({ code: 0, stdout: "not json" })).kind).toBe("error");
    const wrong = { schema: "inbox-status/2", reader: "team-lead", senders: {} };
    expect(parseInboxStatus(res({ code: 0, stdout: JSON.stringify(wrong) })).kind).toBe("error");
  });
});

describe("parseWatch", () => {
  it("parses reason=message and captures senders + path", () => {
    const line = JSON.stringify({
      reason: "message",
      from: ["alice", "bob"],
      path: "/base/sid-1/inbox-team-lead.jsonl",
    });
    const r = parseWatch(res({ code: 0, stdout: line + "\n" }));
    expect(r).toEqual({
      kind: "message",
      from: ["alice", "bob"],
      path: "/base/sid-1/inbox-team-lead.jsonl",
    });
  });

  it("classifies reason=output and reason=waiting", () => {
    const out = parseWatch(
      res({ code: 0, stdout: JSON.stringify({ reason: "output", path: "x" }) }),
    );
    expect(out.kind).toBe("output");
    const wait = parseWatch(
      res({ code: 0, stdout: JSON.stringify({ reason: "waiting", agent: "a", path: "y" }) }),
    );
    expect(wait.kind).toBe("waiting");
  });

  it("maps exit 2 to timeout", () => {
    expect(parseWatch(res({ code: 2, stdout: "" })).kind).toBe("timeout");
  });

  it("maps a malformed json line to malformed (no tight loop)", () => {
    expect(parseWatch(res({ code: 0, stdout: "{not-json" })).kind).toBe("malformed");
  });

  it("maps stderr noise / other exit codes to malformed", () => {
    expect(parseWatch(res({ code: 7, stderr: "weird" })).kind).toBe("malformed");
  });

  it("treats exit 0 with non-empty stderr as malformed (backoff, no wake)", () => {
    // Blocker 3 / plan §3.1: a warning-contaminated exit-0 watch must not be
    // trusted as authoritative output, even when stdout carries a valid line.
    const line = JSON.stringify({ reason: "message", from: ["a"], path: "p" });
    const r = parseWatch(res({ code: 0, stdout: line, stderr: "DeprecationWarning: noise" }));
    expect(r.kind).toBe("malformed");
    // Empty stdout + stderr noise is likewise malformed, not silently accepted.
    expect(parseWatch(res({ code: 0, stdout: "", stderr: "boom" })).kind).toBe("malformed");
  });

  it("tolerates extra whitespace and surrounding blank lines", () => {
    const line = JSON.stringify({ reason: "message", from: ["a"], path: "p" });
    const r = parseWatch(res({ code: 0, stdout: "\n  " + line + "  \n\n" }));
    expect(r.kind).toBe("message");
  });
});

describe("leadInboxPath", () => {
  it("derives the reader inbox path from the session dir", () => {
    expect(leadInboxPath("/base/sid-1")).toBe("/base/sid-1/inbox-team-lead.jsonl");
    expect(leadInboxPath("/base/sid-1/")).toBe("/base/sid-1/inbox-team-lead.jsonl");
    expect(DEFAULT_READER).toBe("team-lead");
  });
});

describe("isLeadInboxPath (cross-platform guard, blocker 1)", () => {
  it("matches a POSIX wake path against its POSIX session dir", () => {
    expect(isLeadInboxPath("/base/sid-1/inbox-team-lead.jsonl", "/base/sid-1")).toBe(true);
    expect(isLeadInboxPath("/base/sid-1/inbox-other.jsonl", "/base/sid-1")).toBe(false);
  });

  it("matches a Windows backslash wake path against its Windows session dir", () => {
    // Python emits str(Path(session_dir)/"inbox-team-lead.jsonl"), which uses
    // backslashes on Windows; the guard must still accept it.
    const dir = "C:\\Users\\x\\.claude\\agent-sessions\\sid-1";
    const wake = "C:\\Users\\x\\.claude\\agent-sessions\\sid-1\\inbox-team-lead.jsonl";
    expect(isLeadInboxPath(wake, dir)).toBe(true);
    expect(
      isLeadInboxPath("C:\\Users\\x\\.claude\\agent-sessions\\sid-1\\inbox-other.jsonl", dir),
    ).toBe(false);
  });

  it("is separator-agnostic across mixed styles and trailing separators", () => {
    const dir = "C:\\Users\\x\\sid-1\\";
    expect(isLeadInboxPath("C:/Users/x/sid-1/inbox-team-lead.jsonl", dir)).toBe(true);
  });
});
