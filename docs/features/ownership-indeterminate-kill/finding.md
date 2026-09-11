# Finding: indeterminate ownership abandons a live agent

Surfaced by the Herdr launcher's implementation review (round 2,
`docs/features/herdr-launcher/implementation-review-2.md`, BLOCKER 1). **Not
caused by that work** — it lives in shared lifecycle code that predates it and
applies to every backend (Windows, tmux, Linux terminal, Herdr alike). Recorded
here so it is not lost; the fix belongs in its own feature with its own plan and
review.

## The defect

`_PidOwnershipMixin` deliberately keeps three answers apart
(`process_manager.py:311-365`): `OURS`, `NOT_OURS`, and `INDETERMINATE` — the
last meaning "the PID is alive but its creation token could not be read, so we
cannot tell ownership from reuse". Its own docstring says callers gating a
*reclaim* must treat `INDETERMINATE` as "still held".

Two lifecycle callers collapse it to a boolean too early:

1. **`kill_agent`** (`server_simple.py:5876-5895`):

   ```python
   owned = process_manager.owns_process(str(agent.get("pid")), _agent_create_token(agent))
   if owned:
       process_manager.kill_process(str(agent["pid"]))
   remaining = [a for a in agents if a.get("name") != name]
   agents[:] = remaining          # <- unconditional
   _save_agents_transaction(session_id, agents)
   ...
   return {"success": True, "name": name}
   ```

   `owns_process` collapses `INDETERMINATE` to `False`, so the kill is skipped —
   correctly, since we must never signal a PID we cannot prove is ours. But the
   durable record is deleted anyway and success is returned. The process keeps
   running with no agent record left to address, retry or report it.

2. **Follow-up / resume** (`server_simple.py:4518-4527`): when the agent is
   alive but `owns_process` is false *solely* because ownership is
   indeterminate, both graceful and force shutdown are skipped and
   `backend.resume` runs anyway — so the old and new workers can run
   concurrently on one conversation.

## Why it is not a Herdr problem

`HerdrProcessManager._force_kill_settled` already refuses to claim success in
this state and raises `HerdrOwnershipUnprovenError` while keeping its record.
That covers the case where the first probe succeeds and the token becomes
unreadable later. It cannot cover *initial* indeterminacy, because
`kill_process` is never called at all. The gate is above the backend.

## Suggested fix (for the follow-up feature)

Use the three-valued `ownership_probe` at both decisions:

- `OURS` → graceful/kill as today.
- `NOT_OURS` → the original process is gone or reused; cleanup/resume may
  proceed.
- `INDETERMINATE` → return a structured, retriable refusal; **preserve** the
  durable record and the lease, and do not resume.

Catch a backend's ownership-unproven exception into the same result, ideally by
promoting `HerdrOwnershipUnprovenError` to a backend-neutral error.

This changes the `kill_agent` MCP tool's contract (success is no longer
unconditional) and the follow-up path's behaviour for every backend, which is
why it wants its own plan, its own review and its own PR rather than riding
along with a launcher change.

## Tests the fix should carry

- `kill_agent` retains `agents.json` and the lease, and reports a retriable
  refusal, when the initial probe is `INDETERMINATE`.
- Follow-up does not invoke `backend.resume` when the initial probe is
  `INDETERMINATE`.
- The transition case: first probe `OURS`, token unreadable by kill time —
  the backend raises, the durable record survives, no partial deletion.
