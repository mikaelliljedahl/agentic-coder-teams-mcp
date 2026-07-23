# External agent join implementation

Implemented the approved v4 token-carried external-member design on
`feature/external-agent-join`. The working tree is intentionally uncommitted.

## Red → green evidence

Commands used the worktree virtual environment through `uv run`.

1. Ticket store, safe names, strict TTL/retention parsing, and fenced prompt
   construction (matrix 1, 2, 27, 29):

   - Red:
     `uv run pytest -q tests/test_join_team.py -k 'create_ticket_token_prompt_exact or ticket_name_safe_and_dedup or ttl_and_retention_parsers or join_prompt_delimiter_injection'`
     → `8 failed` (missing tool/helpers).
   - Intermediate: `1 failed, 7 passed` exposed the required lowercase literal
     `save` in the generated prompt.
   - Green: the same command → `8 passed`.

2. Join record, deterministic member token, retention, reconciliation crash
   windows, conservative marker repair, marker retry, and corruption
   precedence (matrix 4–11):

   - Red:
     `uv run pytest -q tests/test_join_team.py -k 'join_happy_path or join_replay_idempotent or used_ticket_retention or crash_window or marker_write_failure or registry_corrupt'`
     → `11 failed` (missing `join_team`/marker writer).
   - Green: the same command → `11 passed`.

3. Exact bearer grammar, lock-scoped member operations, cursor semantics,
   heartbeat warnings, ambient-state independence, leave, and replay
   revocation (matrix 9b, 12, 14, 15, 17, 19):

   - Red:
     `uv run pytest -q tests/test_join_team.py -k 'token_grammar or external_read_full or external_send_and_heartbeat or no_ambient_reads or replay_after_leave or leave_idempotent'`
     → `15 failed` (missing token-bearing tools).
   - An all-numeric fixture UUID initially made the uppercase-UUID case
     ineffective; changing it to a lowercase UUID containing `a`–`f` exposed
     that grammar edge.
   - Green: the same command → `15 passed`.

4. Spawn/ticket reservations and concurrent join/read serialization (matrix 3,
   7, 16):

   - After correcting the spawn fixture's missing `mcp/` setup, red:
     `uv run pytest -q tests/test_join_team.py -k 'spawn_ticket_reservation or concurrent_join or two_process_external_read'`
     → `1 failed, 2 passed`; spawn reused an open ticket's name.
   - Green: the same command → `3 passed`.

5. Lead inbox delivery, external kill safety, credential redaction, and honest
   status/binding (matrix 13, 21, 23, 24):

   - Red:
     `uv run pytest -q tests/test_join_team.py -k 'lead_send_to_external or kill_external or no_credential or agent_status_backend'`
     → `2 failed, 2 passed`; full list leaked the digest and binding resolved
     as legacy.
   - Green: the same command → `4 passed`.

6. Guaranteed-delivery guards and authoritative row settlement (matrix 20,
   20b):

   - Red:
     `uv run pytest -q tests/test_join_team.py -k 'guaranteed_guards or prepare_race'`
     → `2 failed`; `deliver_pending` reached the claim and the authoritative
     helper did not exist.
   - Green: the same command → `2 passed`.
   - The final test also covers byte-identical pre-existing rows, released
     claims, immediate key reuse after a persisted discard, and
     `record_discarded:false` when discard persistence fails.

7. Discoverable-but-not-silently-adopted recovery (matrix 22):

   - Red:
     `uv run pytest -q tests/test_join_team.py -k external_only_session_discoverable`
     → `1 failed`; the external-only candidate was silently adopted.
   - Green: the same command → `1 passed`.

8. Import-time restricted registration (matrix 26):

   - The initial subprocess harness used a FastMCP method absent from this
     version and was corrected to `list_tools`.
   - Behavioral red:
     `uv run pytest -q tests/test_join_team.py -k external_only_mode`
     → `1 failed`; all normal tools were still registered.
   - Green: the same command → `1 passed`.

9. The complete new matrix suite:

   - `uv run pytest -q tests/test_join_team.py` → `51 passed`.
   - The four revocation races, real-binding subprocess restart, and
     barrier-based 10k-inbox contention tests were added as hardening after the
     shared token/locking implementation was already green; their first focused
     run was `7 passed` and required no further production change. This is a
     procedural deviation from one-red-per-matrix-cell, but the underlying
     behaviors were introduced only in response to the earlier failing member
     operation and recovery slices.

10. Whole-repository integration:

    - First full run after the feature implementation:
      `uv run pytest -q` → `73 failed, 1051 passed, 3 skipped`. The failures
      shared one environmental cause: this managed implementation process had
      imported `server_simple` with `IDENTITY="impl-external-join"`, while the
      existing autouse fixture cleared `AGENT_*` after import without resetting
      the captured globals. Completing that fixture's documented isolation
      reduced the targeted integration run to one intentional
      `agent_status` schema mismatch.
    - The targeted integration rerun was then `1 failed, 271 passed`; the sole
      failure was the existing exact `agent_status` schema assertion. It was
      updated for approved `backend` and centralized `binding`, then verified
      by the focused and full runs below.
    - Post-refactor focused integration:
      `uv run pytest -q tests/test_join_team.py tests/test_read_messages.py tests/test_tool_descriptions.py tests/test_agent_status.py`
      → `114 passed`.

## Final design as built

- `join-tickets.json` is an atomic JSON-array store sharing the cross-process
  agents lock. Strict positive finite TTL and retention parsers default to 24
  hours and seven days. Writes retain auditable used/expired rows within
  retention and prune older rows. Only open, unexpired tickets reserve names.
- `create_join_ticket` validates the hooks safe-name grammar, de-duplicates
  against registry records and ticket reservations, and returns a deterministic
  fenced paste prompt. Spawn uses the same reservations under the same lock.
- `join_team` validates a directly nested UUID session, reconciles by
  `join_ticket_id` before applying expiry, validates all immutable fields, and
  implements every v4 `(ticket status, record presence, record status)` cell.
  Unknown/corrupt states fail closed without mutation. Joined-marker repair is
  schema-aware and preserves every valid newer marker byte-for-byte.
- Member tokens use
  `wam1:<canonical-lowercase-hyphenated-uuid>:<64-lowercase-hex>`.
  The secret is derived from the retained ticket and only its SHA-256 digest is
  stored in `agents.json`; verification uses fixed-type
  `hmac.compare_digest`.
- `_member_operation` parses before locking, then holds the session's
  cross-process agents lock through membership re-resolution and every inbox,
  cursor, marker, or status side effect. The lock order is agents lock then
  per-inbox process lock. `external_read` and ambient `read_messages` share one
  `_read_inbox` implementation.
- External send/read heartbeat failures return successful inbox/cursor outcomes
  with `heartbeat_warning:true`. Leave is idempotent through `allow_left` and
  never rewrites an already-left marker.
- Lead `send_message` re-resolves external children in one agents transaction
  and appends pull-only inbox delivery. External kill removes registry/artifact
  state without any PID ownership/liveness probe or signal.
- Guaranteed delivery has guards before new-row creation, before pending-row
  reconciliation/claim, and authoritatively inside `_prepare` before backend
  lookup or lease reservation. Current-call rows are discarded under the new
  C2 refusal; pre-existing rows remain audit-identical with claims released;
  `deliver_pending` reports attempt refusals separately.
- `_public_agent_record` strips credential fields on full/compact list,
  `check_agent`, and `agent_status` construction. `agent_status` now reports
  `backend` and `binding`; external binding is centrally
  `not_applicable`, and external PIDs are informational rather than probed.
- External-only sessions remain in candidate discovery and explicit
  `resume_session`, but the single-candidate silent fallback requires a live
  non-external record.
- One `_register_tool` gate wraps every registration. With
  `WIN_AGENT_TEAMS_EXTERNAL_ONLY=1`, only `join_team`, `external_send`,
  `external_read`, `leave_team`, and `list_backends` are registered, with the
  same client-visible descriptions as normal mode.

## Deviations and environment limitations

- No live Claude Desktop profile/process smoke test was possible in this
  headless worktree. The import-time restricted surface and fresh-process
  restart are automated, but Desktop's per-profile MCP scoping was not
  observed here. README and INSTALL therefore state the honest release gate:
  use a separate profile/client instance when supported; if the client cannot
  scope MCP configuration, ambient-tool isolation is unavailable and a
  dual-entry setup is degraded mode.
- The managed-agent identity fix in `tests/conftest.py` was not in the planned
  file list. It is test-only, behavior-preserving isolation needed for the
  mandated full-repository gate when pytest itself runs inside a spawned team
  agent.
- As noted in the evidence, matrix 18/25/28 were green when first added because
  they validate locking/restart behavior already implemented by preceding red
  slices. No design behavior deviates from plan v4.

## Final validation

Commands and final outputs:

```text
uv run ruff check
All checks passed!

uv run ty check
All checks passed!

uv run pytest -q
1124 passed, 3 skipped
```

## Review-1 finding #1 resolution

Chose option (a). External members have no transcript binding by design, so
both `_agent_status_row` and compact `_list_agents_row` now initialize
`binding` to `not_applicable` before the live-marker shortcut. This keeps the
field stable regardless of whether transcript fallback runs, while preserving
`None` as “not evaluated on this call” for marker-present managed agents.
`test_agent_status_backend_and_binding_na` now covers marker-present
`agent_status` and compact `list_agents` rows as well as the existing fallback
and full-resolution paths.

Focused TDD evidence:

```text
uv run pytest -q tests/test_join_team.py -k agent_status_backend_and_binding_na
FAILED: marker_status["binding"] was None (1 failed, 50 deselected)

uv run pytest -q tests/test_join_team.py -k agent_status_backend_and_binding_na
1 passed, 50 deselected
```

Post-review whole-repository gates:

```text
uv run ruff check
All checks passed!

uv run ty check
All checks passed!

uv run pytest -q
1124 passed, 3 skipped
```

## Live smoke test (2026-07-23)

**Result: PASSED.** Run from a real Claude Desktop deployment against a live
lead session (`5c170d7e-…`) with a separately-started Desktop QA session as the
external member `visual-qa`:

- Lead minted a join ticket via `create_join_ticket`; the QA session called
  `join_team(session_id, token)` and received its `member_token`.
- **Downstream (lead → external):** lead `send_message` to the member resolved
  to `delivery:"inbox"` (pull-only, as designed); QA drained it with
  `external_read`.
- **Upstream (external → lead):** QA `external_send` landed in the lead inbox
  and the lead read it with `read_messages`.

**Process model — question answered.** Each interactive session gets its **own**
win-agent-teams MCP server process (host process count went 2 → 3 when the QA
session joined). There is no shared-process-per-Desktop concern; the
token-carried identity design is safe regardless, and the earlier
process-global-rebind blocker is moot by construction.

**Wake follow-ups (out of this feature's scope):** upstream external→lead
auto-wake is wired by running `install_lead_wake` on the lead. Downstream
lead→external hands-free pickup is deferred to
[`external-member-wake`](../external-member-wake/design.md) — pull-only remains
correct (a Desktop conversation cannot be resumed like a spawned CLI child).
