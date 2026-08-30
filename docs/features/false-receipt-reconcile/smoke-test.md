# Smoke test: request-scoped receipt reconciliation (Linux)

Manual verification for the change in this feature directory. The automated
suite proves the branches; this proves the same behaviour through the real MCP
tools, a real spawned agent, a real transcript and a real delivery store, on the
platform CI actually runs.

Budget: ~20 minutes, most of it waiting for a real agent to start.

## What you are looking for

The defect was a **false receipt**: `follow_up_agent` answering
`{"status": "delivered", "reconciled": true}` for a prompt it never sent, while
`delivery_status(<that same key>)` answered `queued / pending / attempts: 0`.
Every check below is a variation on "does the answer describe the request I
actually made, and does the store agree with the answer".

## Phase 0 — isolated environment

```bash
git clone https://github.com/mikaelliljedahl/agentic-coder-teams-mcp.git /tmp/smoke
cd /tmp/smoke
git fetch origin claude/false-receipt-delivery-status-ad3d77
git checkout claude/false-receipt-delivery-status-ad3d77
uv sync
```

Run the four gates on Linux before anything else — a Windows-green tree is not
a Linux-green tree:

```bash
uv run ruff format --check . && uv run ruff check . && uv run ty check && uv run pytest -q
```

Expected: all four clean. Two known Windows-only artifacts should NOT appear
here — `test_kill_agent_proceeds_when_the_holder_token_no_longer_matches` and
the `ty` diagnostic in `tests/test_join_team.py`. If either fails on Linux too,
stop: that is new information and it is not what this branch documented.

> **Do not override `HOME`.** Session state lives under
> `~/.claude/agent-sessions/<session-id>/`, but the spawned CLI also reads its
> own credentials from `~`. Isolation here comes from a fresh clone, a fresh
> session id, and a dedicated agent name — not from a fake home.

## Phase 1 — the happy path still works

Everything below runs in one Python process so the lead session stays adopted
between calls. The MCP tools are plain async functions; calling them directly is
exactly what the server does.

```bash
cd /tmp/smoke
cat > /tmp/smoke1.py <<'PY'
import asyncio, json
from claude_teams import server_simple as s

async def main():
    spawn = await s.spawn_agent(
        prompt="You are a smoke-test target. Reply with the single word READY and then wait.",
        backend="claude-code",          # or "codex"
        name="smoke-target",
        cwd="/tmp/smoke",
    )
    print("SESSION", spawn["session_id"])
    print("SESSION_DIR", spawn["session_dir"])

    await asyncio.sleep(45)             # let the child actually start

    first = await s.follow_up_agent(
        "smoke-target", "Write the word ALPHA to /tmp/smoke-alpha.txt", "smoke-a"
    )
    print("FOLLOW_UP", json.dumps(first, indent=2))
    print("STATUS   ", json.dumps(await s.delivery_status("smoke-a"), indent=2))

asyncio.run(main())
PY
uv run python /tmp/smoke1.py
```

**PASS** when `status` is `delivered` in both outputs, and
`/tmp/smoke-alpha.txt` appears within a minute or so.

If the follow-up returns `queued/unconfirmed` instead, that is *not* a failure —
it is the state this whole feature is about. Retry with the **same** key; it
must then answer `delivered` without the agent receiving the instruction twice.

Note the session id and session dir; the phases below need them.

## Phase 2 — the four reconcile arms

These need a `pending_delivery` marker on the agent record whose nonce is
already present in the target's transcript. That state arises naturally when a
transcript write is buffered past the scan bound; here it is injected, because a
smoke test cannot wait for a race. Injection uses the documented on-disk
contract only.

Helper — re-arms the marker from the nonce of a delivered row:

```bash
cat > /tmp/rearm.py <<'PY'
import json, sys, time
from pathlib import Path

session_dir = Path(sys.argv[1])
key = sys.argv[2]                       # the idempotency key whose nonce to reuse
rows = json.loads((session_dir / "deliveries.json").read_text())
row = next(r for r in rows.values() if r["idempotency_key"] == key)
agents_path = session_dir / "agents.json"
agents = json.loads(agents_path.read_text())
target = next(a for a in agents if a["name"] == "smoke-target")
target["pending_delivery"] = {
    "nonce": row["nonce"],
    "operation_id": row.get("operation_id", ""),
    "attempted_at": time.time(),
    "prompt_file": "",
}
agents_path.write_text(json.dumps(agents, indent=2))
print("re-armed marker with nonce", row["nonce"][:12], "from", key)
PY
```

`SD=<the session_dir printed in phase 1>` for the rest of this section.

### 2a. Identical request under a NEW key → aliased, not re-sent

```bash
uv run python /tmp/rearm.py "$SD" smoke-a
cat > /tmp/smoke2a.py <<'PY'
import asyncio, json
from claude_teams import server_simple as s
async def main():
    await s.resume_session("SESSION_ID_HERE")
    r = await s.follow_up_agent(
        "smoke-target", "Write the word ALPHA to /tmp/smoke-alpha.txt", "smoke-b"
    )
    print("ANSWER", json.dumps(r, indent=2))
    print("OWN KEY", json.dumps(await s.delivery_status("smoke-b"), indent=2))
    print("OLD KEY", json.dumps(await s.delivery_status("smoke-a"), indent=2))
asyncio.run(main())
PY
uv run python /tmp/smoke2a.py
```

**PASS**: the answer is `delivered` with `reconciled_from_key: "smoke-a"`;
`delivery_status("smoke-b")` is **also** `delivered`, carries the same
`reconciled_from_key`, and has an **empty** `nonce`; `smoke-a` is `delivered`.

**FAIL (the original bug)**: the answer says `delivered` but
`delivery_status("smoke-b")` says `queued / pending / attempts: 0`.

### 2b. DIFFERENT request under a new key → actually sent

```bash
uv run python /tmp/rearm.py "$SD" smoke-a
```

Then run 2a's script again with the prompt changed to
`Write the word BRAVO to /tmp/smoke-bravo.txt` and the key to `smoke-c`.

**PASS**: `/tmp/smoke-bravo.txt` appears, and the answer does **not** contain
`reconciled: true`. This is the arm that used to drop the prompt on the floor.

**FAIL (the original bug)**: an immediate `delivered / reconciled: true` and no
`/tmp/smoke-bravo.txt` — ever.

### 2c. Damaged store → honest barrier, marker kept

```bash
uv run python /tmp/rearm.py "$SD" smoke-a
python - <<PY
import json; from pathlib import Path
p = Path("$SD") / "deliveries.json"
rows = json.loads(p.read_text())
p.write_text(json.dumps({k: v for k, v in rows.items()
                         if v["idempotency_key"] != "smoke-a"}, indent=2))
PY
```

Then run 2a's script with key `smoke-d` and any prompt.

**PASS**: `success: false`, `reason: "pending_attempt_unresolvable"`,
`retriable: true`, `status: "queued"`; nothing new reaches the agent; and
`pending_delivery` is **still** in `agents.json` — it is the only remaining
guard against re-sending a prompt that already landed:

```bash
grep -c pending_delivery "$SD/agents.json"     # expect 1
```

**FAIL**: `delivered` on no evidence, or the marker gone.

Restore the store afterwards: `git`-style, just re-run phase 1 with fresh keys,
or accept that `smoke-a` is now absent from the audit trail.

### 2d. Row already settled `failed` → not resurrected

```bash
uv run python /tmp/rearm.py "$SD" smoke-c     # any delivered row with a nonce
python - <<PY
import json; from pathlib import Path
p = Path("$SD") / "deliveries.json"
rows = json.loads(p.read_text())
for r in rows.values():
    if r["idempotency_key"] == "smoke-c":
        r.update(status="failed", phase="settled", reason="not_delivered")
p.write_text(json.dumps(rows, indent=2))
PY
```

Then run 2a's script with key `smoke-e` and the *same* prompt as `smoke-c`.

**PASS**: `reason: "prior_attempt_settled_failed"`, nothing sent, marker kept,
and `delivery_status("smoke-c")` still says `failed`. A settled outcome is never
withdrawn, and a second key must not be told a different story about one
attempt.

## Phase 3 — the shared entry points

The same machinery serves two other tools; a routing mistake would not show up
in phase 2.

```bash
uv run python /tmp/rearm.py "$SD" smoke-b
```

- **`send_message` to a spawned child**: call
  `await s.send_message("<the smoke-b prompt verbatim>", "smoke-target", "smoke-f")`.
  PASS: `delivered` with `reconciled_from_key`, nothing re-sent.
- **`deliver_pending`**: re-arm, then `await s.deliver_pending()`. PASS: it
  settles rather than re-sends, and no row is left claiming a receipt it does
  not have.

## Phase 4 — cleanup

```bash
uv run python -c "
import asyncio
from claude_teams import server_simple as s
async def m():
    await s.resume_session('SESSION_ID_HERE')
    print(await s.kill_agent('smoke-target'))
asyncio.run(m())"
rm -rf "$SD" /tmp/smoke-alpha.txt /tmp/smoke-bravo.txt /tmp/smoke*.py /tmp/rearm.py
```

Confirm no stray CLI process survived: `pgrep -af 'claude|codex' | grep smoke`.

## Out of scope for this smoke test

Deliberately not covered, and documented as deferred in `plan.md`:

1. A row returned to `pending` by a retriable failure keeps its nonce, and the
   next same-key attempt overwrites it. Unreachable while a marker exists.
2. Scan outcomes other than "found" still fall through to a normal send.
3. There is no repair path for a damaged delivery store beyond `kill_agent`.

## Reporting

If any PASS above does not hold, capture and attach: the full tool answer JSON,
`deliveries.json`, `agents.json`, and the target's transcript path from
`agent_watch_paths("smoke-target")`. Those four make the state reconstructible
without the machine.
