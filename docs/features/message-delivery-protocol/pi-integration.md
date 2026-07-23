# Pi under the message-delivery protocol

Pi arrived on `origin/main` (`279a171`, `56a8ff9`) while this branch was in
flight. It declares `supports_resume() -> True` but was written against the
pre-protocol world: no correlation marker, no binder, no named receipt record.

## What that meant on the merged tree

Nothing crashed, which is the problem. A Pi agent spawned cleanly, got a
`correlation_id` written to its record (the server writes one for every
backend), and then `agent_output._make_binder` returned `None` for `"pi"`,
so the binding ladder answered `unverified` and `follow_up_agent` refused.

That refusal was *accidentally* correct — it fell out of an unhandled branch,
not a decision. Had `_make_binder` had any fallback for unknown backends, Pi
would instead have resumed against an unconfirmable transcript and reported
`delivered`, which is precisely the false receipt R6 exists to prevent.

## Decision: Pi supports the full protocol

Pi turned out to satisfy every structural precondition, so the honest outcome
is support, not refusal. Evidence, in the order it was checked:

| Requirement | Pi's answer |
| --- | --- |
| Prompt reaches the agent verbatim, so it can carry a marker | Yes — `PiBackend._prompt_args` passes `request.prompt` as a single positional arg on the `node` launch path |
| Transcripts are locatable per agent | Yes, and more strongly than Codex/Claude: the **server** chooses the location (`--session-dir <session>/pi-sessions/<agent>`) |
| A record class means "the agent received this as input" | Yes — `{"type": "message", "message": {"role": "user", ...}}` |

Implemented as:

1. **Correlation marker** — `PiBackend._correlated_prompt`, reading
   `extra[CORRELATION_FIELD]`, applied by both `build_command` and
   `build_resume_command`. Pi follows the Codex pattern (backend appends)
   rather than the Claude pattern (server appends), because its prompt is
   verbatim; `_materialize_prompt` therefore leaves Pi prompts alone and
   double-marking cannot occur.
2. **`_PiBinder`** — enumerates the one server-chosen directory, with no mtime
   window, since every file there belongs to this agent name by construction.
   The directory path is persisted on the record (`PI_SESSION_DIR_FIELD`,
   written by `server_simple._pi_binding_extra` at spawn and backfilled on
   resume) rather than re-derived, so the storage layout stays owned by one
   place and a record predating the field is refused rather than guessed.
3. **Receipt record** — `delivery._pi_receipt_texts`, dispatched from
   `receipt_nonces`.

## What was deliberately NOT decided here

**Pi keys its session storage on the agent name** (`--session-id <agent>`,
`pi-sessions/<agent>`). A killed agent's name can be reused, and the
replacement then writes into *the same directory* as its predecessor and can
`--continue` the dead agent's conversation.

This is the same name-reuse collision that motivated per-spawn correlation ids
in the first place — but for Pi it lives in the backend's on-disk layout, where
a marker cannot prevent it. The marker does make it **detectable**: two
transcripts carrying one token is `ambiguous`, and the ladder refuses to pick
(`test_pi_two_marker_matches_is_ambiguous`).

Preventing it would mean changing Pi's session id to something per-spawn (e.g.
including the correlation id), which changes Pi's on-disk layout and its
`--continue` semantics, and would orphan sessions written by the current
scheme. **That is a product decision and is left open**, not silently taken.
Recommendation: fold it into whatever change first touches Pi's resume path,
and until then rely on the `ambiguous` outcome, which is safe but means a
name-reused Pi agent stops being follow-up-able rather than resuming wrongly.

## Tests

`tests/test_pi_delivery_protocol.py` — 13 tests covering marker presence and
uniqueness on spawn and resume, absence for a legacy record, receipt
recognition (and non-recognition of assistant echo), and the four binder
outcomes: `bound`, `unverified` (no marker — explicitly *not* newest-mtime),
`ambiguous` (name reuse), and `legacy`. Two further tests assert the server
writes `pi_session_dir` for Pi and only for Pi.
