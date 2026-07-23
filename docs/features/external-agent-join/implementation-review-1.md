# Implementation review 1 — external-agent-join

**Verdict.** The implementation is a faithful, disciplined realization of approved
plan v4. Every committed §-item is present and behaves as specified; the 32 new
tests in `tests/test_join_team.py` assert exact values (records, tokens, marker
schemas, byte-identical stores, cursor windows) rather than smoke-testing for the
absence of exceptions, and they cover the full §5 matrix including the adversarial
cells (concurrent join, both crash windows incl. expired crash-A, two-process
exactly-once read, all four revocation-race combinations, prepare-race row
settlement, corruption precedence, delimiter injection). I reproduced the gates:
`ruff` clean, `ty` clean, `1124 passed / 3 skipped`. No existing test was deleted
or softened. The identity/locking/delivery invariants that sank v1–v3 are held
correctly: no ambient state is mutated, the cross-process registry lock is held
through every side effect, the guaranteed-delivery guards sit at both early call
sites plus the authoritative `_prepare` recheck, and the credential sanitizer is
applied at all four surfaces. I found no blockers.

**Score: 94/100 — APPROVED.**

## Blockers (must fix)

None.

## Non-blocking findings

1. **`agent_status` `binding` is `None` (not `"not_applicable"`) for an external
   record that still has a valid marker.** `_agent_status_row`
   (`server_simple.py:5710`–5729) only sets `binding_outcome` inside the
   `if last_activity_ts is None:` transcript-fallback branch; when a marker is
   present the field stays `None`. This is internally consistent — `binding` is
   resolved for *no* agent when a marker exists — and the plan's promise (test 24
   asserts `not_applicable` specifically on the fallback path) is met. But a reader
   of a running external member with a fresh activity marker sees `binding: null`,
   which reads slightly less "honest" than the docstring's blanket
   `binding="not_applicable"` claim (`agent_status` docstring, line 5766). Consider
   short-circuiting external records to `"not_applicable"` unconditionally. Purely
   cosmetic; no behavior depends on it.

2. **External inbox appends are not taken under `_inbox_lock`.** `external_send`
   (`server_simple.py:3129`) and the lead→external branch of `send_message`
   (`:3121`) append to the target inbox with a bare `open("a")` while holding only
   the registry lock, not `_inbox_lock(parent)`. The registry lock serializes them
   against kill/leave/other member ops (which is what §2.4 requires and what the
   race tests exercise), but it does **not** serialize against the parent's own
   ambient `read_messages`, which takes only `_inbox_lock`. This is identical to
   the pre-existing SPAWNER append path (`:3168`) and relies on the same
   single-line O_APPEND atomicity + malformed-line-skipping the repo already
   depends on, so it is not a regression and is within the documented same-process
   premise. Noted for completeness only.

3. **`create_join_ticket` legitimately calls `_active_session_id(create=True)`
   (`:2679`).** This is correct — it is a lead-side tool, not one of the four
   token-bearing external tools — but worth recording so a future reader does not
   mistake it for an ambient-state leak. The `test_no_ambient_reads` guard covers
   only `join_team`/`external_send`/`external_read`/`leave_team`, which is the
   right scope.

## Plan-faithfulness matrix

| §-item | Status | Evidence |
|---|---|---|
| §0 token grammar (`wam1:<uuid>:<64hex>`, canonical UUID, no exception escapes) | IMPLEMENTED | `_parse_member_token` `:349`–373: field-count, `wam1` literal, `str(uuid.UUID())==x` round-trip, 64-char lowercase-hex, `isascii`; all deviations → `None` → `invalid_member_token`. |
| §0 digest-only at rest + `hmac.compare_digest` | IMPLEMENTED | `_member_digest` `:280`; record stores `member_token_digest` only (`:730`); compare via `hmac.compare_digest(computed, stored_bytes)` `:409`; deterministic re-derivation `_member_secret` `:274`. |
| §2.1 ticket store, strict TTL/retention, name-reservation, retention pruning | IMPLEMENTED | `_strict_positive_seconds` `:614` (rejects 0/neg/nan/inf/non-numeric); `_ticket_retained` `:670`; `_save_join_tickets_unlocked` prunes `:657`; `_ticket_name_reserved` requires open+unexpired `:715`. |
| §2.2 external record shape | IMPLEMENTED | `:720`–734 matches the committed JSON exactly incl. `spawned_by`/`spawned_by_source:"join_ticket"`. |
| §2.2 discoverable-not-auto-adopted | IMPLEMENTED | `_non_terminal_agents` UNCHANGED (grep: sole consumer unmodified); new `_has_autoadoptable_agent` `:1113` gates ONLY `_recover_session_id`'s single-candidate branch `:1184`–1189. Test 22 confirms both halves. |
| §2.4 `_member_operation` holds lock through side effects | IMPLEMENTED | `:376`–432: parse+session check outside lock (refusals), then `with _agents_file_lock(sid)` wraps record location AND the `yield` — all caller side effects run inside. Lock order agents→inbox honored in `external_read`→`_read_inbox`→`_inbox_lock`. |
| §2.4 `leave_team` `allow_left` idempotency, no marker rewrite | IMPLEMENTED | `:1244`–1269; already-left → `already_left:true`, no `_write_state_marker`. Test 19 asserts marker mtime unchanged. |
| §3.3 reconciliation state machine (all cells, expiry-after-reconcile, corruption precedence, schema-aware marker repair, immutable-field incl. backend/session_id/spawned_by/source) | IMPLEMENTED | `join_team._join` `:653`–753: dup→`registry_corrupt`; `_external_record_matches` `:310` validates all 7 immutable fields; ticket-status-unknown check `:691` precedes left-rule `:698` (precedence correct); `record.status` outside {running,left}→corrupt `:693`; crash-A skips expiry `:738`; expiry only in `record is None` branch `:709`. `_ensure_joined_marker`+`_marker_schema_valid` `:92`–111 write only when missing/unparseable/schema-invalid. |
| §3.5 lead→external delivery under one transaction | IMPLEMENTED | `send_message` `:3092`–3133: re-resolve inside `_agents_transaction`, recheck backend+`spawned_by==IDENTITY`+status; left→`member_left`, non-running→`membership_revoked`, running→inbox append. |
| §3.6 guards: two early sites + authoritative `_prepare` recheck | IMPLEMENTED | `_guaranteed_send` pre-`_open_delivery_record` `:4945`; `deliver_pending` pre-`_claim` `:5203`; `_prepare` external check `:3852` BEFORE `registry.get` `:3906` and lease reservation. `external_agent_pull_only` ∈ `_C2_REFUSAL_REASONS` `:5001`; current-call row discarded `:4989`–4994, `record_discarded:false` preserved on persist failure. `deliver_pending` exposes separate `refusals` list; durable statuses stay queued/delivered/failed. |
| §3.7 sanitizer at all four surfaces | IMPLEMENTED | `_public_agent_record` `:5595`; applied in full-list splat `:5670`, compact row `:5622`, `_agent_check_payload` `:1715`, `_agent_status_row` `:5710`. Full-list docstring updated to "sanitized registry fields". Test 23 serializes all four and asserts digest/token/secret absent. |
| §3.8 EXTERNAL_ONLY gate wrapping ALL registrations | IMPLEMENTED | `_register_tool` `:353`–363 evaluated at decoration time; restricted set = {join_team, external_send, external_read, leave_team, list_backends}; `@_register_tool()` sits above `@_with_disk_note` preserving ordering. Test 26 asserts names AND identical descriptions per mode. |
| kill_agent external branch never probes/signals PID | IMPLEMENTED | `:5391`–5405: removes record, drops lease, cleans artifacts, returns `killed_process:false`/`external_agent_deregistered`; no `owns_process`/`kill_process`. Test 21 patches both to raise. |
| `_agent_alive`/binding for external | IMPLEMENTED | `:1666` external liveness = `status=="running"` (no PID probe); `_resolve_agent_binding` `:1640` → `not_applicable`. |

## Test-quality assessment

Strong across the board. Representative rigor: `test_join_happy_path` asserts the
**entire** record dict field-by-field (`test_join_team.py:175`–191);
`test_registry_corrupt_states` byte-compares both stores after each of five
corruption variants (`:375`–382); `test_two_process_external_read_exactly_once`
uses real forked processes + a barrier and asserts the union of consumed messages
equals exactly the 20 written with no duplication (`:711`–744);
`test_prepare_race_row_settlement` drives the current-call-discard,
pre-existing-row-retained, and discard-persist-failure sub-cases with exact
`list_for_sender` and byte comparisons (`:905`–991);
`test_large_inbox_read_contention_bounded` proves `agent_status` actually blocks
on the held lock (`assert status.is_alive()`) then completes under the 30 s
deadline (`:1209`–1256). Adversarial monkeypatches raise `AssertionError` on the
forbidden call (backend lookup, claim, PID probe/signal, ambient helpers) rather
than merely asserting a return value.

No tautological, no-op, or over-mocked tests were found. Every §5 matrix cell 1–29
has a real test; the implementation notes (matrix 18/25/28 green on first add) are
disclosed honestly and the behaviors they assert are genuinely exercised. The only
minor gap: `test_agent_status_backend_and_binding_na` verifies `not_applicable`
only via the marker-unlinked fallback path, which is why non-blocking finding #1
(marker-present → `binding:null`) is not caught — but that path is consistent with
all agents and not a plan violation.

## Verification log

- `uv run ruff check` → **All checks passed!**
- `uv run ty check` → **All checks passed!**
- `uv run pytest -q` → **1124 passed, 3 skipped** (32.1 s).
- `git diff` (uncommitted) is the feature; `git diff main...HEAD` is docs/PRD/plan only. No test deletions (`git diff -- tests/ | grep '^-.*assert\|def test'` empty).
- `_prepare` external guard ordering confirmed at `server_simple.py:3852` preceding `registry.get` `:3906` and lease reservation — satisfies the "`registry.get("external")` never called" requirement (test 20/20b patch `registry.get` to raise and pass).
- C2 rollback confirmed: `external_agent_pull_only` ∈ `_C2_REFUSAL_REASONS` (`:5001`) and `created and reason in _C2 and not _discard_delivery_record(...)` (`:4989`).
- `_member_operation` lock scope confirmed: `_agents_file_lock` wraps record location through the `yield` (`:396`–432); no validate-then-release TOCTOU.
- conftest.py change is test-only isolation (resets `IDENTITY`/`_IDENTITY_UNRESOLVED`/`_AGENT_PARENT_NAME` after import for pytest-inside-spawned-agent runs); behavior-preserving, disclosed in implementation.md.
