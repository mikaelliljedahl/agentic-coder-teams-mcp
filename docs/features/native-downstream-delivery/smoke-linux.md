# Native downstream delivery: Linux smoke tests

Manual checks to run on a Linux machine with a real `codex` and a logged-in
`claude` CLI. The Windows test VM cannot run them: its `claude` CLI is not
logged in, and a busy-turn test there is unreliable.

- **Part 1** (L-1 to L-4) can run **now**. It tests what Codex and Claude Code
  themselves do, plus what `main` already propagates, and needs none of the
  unfinished code.
- **Part 2** (L-5 onwards) needs parts A and B of the feature and runs before
  the PR.

Record every result in the table at the end, together with the exact command
output that proves it, and send the table to the lead (or paste it into
`spikes.md`). Never paste a `CLAUDE_CODE_MESSAGING_TOKEN` value: the scripts
below print only whether it is present.

## 0. Setup

```bash
codex --version
claude --version
claude -p "say ok"        # proves the CLI is logged in
```

`jq` and `python3` are needed. For the scripts that import the package, use a
checkout of the feature branch:

```bash
git clone https://github.com/mikaelliljedahl/agentic-coder-teams-mcp ~/nd-smoke
cd ~/nd-smoke
git checkout feat/native-downstream-delivery
uv sync
```

The MCP entry for win-agent-teams (in both `~/.claude.json` or the project
`.mcp.json`, and in `~/.codex/config.toml`) needs
`WIN_AGENT_TEAMS_NATIVE_WAKE=1` in its `env`, as INSTALL.md describes.

A helper that lists the environment of every running win-agent-teams MCP
server, showing only variable **names** for secrets:

```bash
wat_env() {
  for p in $(pgrep -f 'win-agent-teams|claude_teams'); do
    echo "== pid $p  parent: $(ps -o comm= -p "$(ps -o ppid= -p "$p")")"
    tr '\0' '\n' < "/proc/$p/environ" \
      | grep -E '^(WIN_AGENT_TEAMS_|AGENT_|CODEX_THREAD_ID|CODEX_HOME|CLAUDE_CODE_MESSAGING_)' \
      | sed -E 's/^(CLAUDE_CODE_MESSAGING_TOKEN)=.*/\1=<present>/'
  done
}
```

## Part 1: run now

### L-1 (N2): `codex queue` into a **busy** Codex thread

**Question.** What happens when a message is queued while the thread is in the
middle of a turn? The answer decides whether condition E6 ("Codex target must
be idle") can be dropped in this PR.

1. Start an interactive Codex TUI in a scratch directory: `codex`.
2. Ask it: *Run `echo $CODEX_THREAD_ID` in the shell and tell me the output.*
   Copy the id into a second terminal: `TID=<id>`.
3. Give the TUI a long turn: *Run `sleep 90 && echo SLEEP-DONE` in the shell,
   then reply with exactly TURN-1-DONE.*
4. While it is sleeping (within about 20 s), in the second terminal:

   ```bash
   MARK="N2-$(date +%s)"
   codex queue --thread "$TID" --message "Reply with exactly QUEUED-ACK $MARK"
   echo "exit=$?"
   ```

5. Wait until the TUI is idle again, then find the rollout and see where the
   marker landed:

   ```bash
   R=$(ls -t ~/.codex/sessions/*/*/*/rollout-*"$TID"*.jsonl | head -1)
   grep -n "$MARK\|SLEEP-DONE\|TURN-1-DONE\|QUEUED-ACK" "$R" | cut -c1-200
   grep -c "$MARK" "$R"
   jq -c 'select(.type=="event_msg") | .payload.type' "$R" | uniq -c | tail -20
   ```

**Record:**
- the `codex queue` output and exit code;
- whether the sleep finished and TURN-1-DONE was produced (the running turn
  was **not** aborted);
- whether the queued text was presented **once** (count of `response_item` user
  records with the marker), and whether it arrived **after** the turn ended (a
  new turn) or **inside** the running turn;
- whether QUEUED-ACK was produced.

**Pass for dropping E6:** the running turn completes normally and the queued
message is presented exactly once, either way. Anything else (turn aborted,
message lost, message presented twice) means E6 stays.

**Also measure (same thread, idle):** a 20 000-character message.

```bash
BIG=$(python3 -c 'print("x" * 20000)')
codex queue --thread "$TID" --message "Reply with the length you received. $BIG N2BIG-$(date +%s)"
```

Record whether it queued and whether the rollout holds the full text.

### L-2 (S-6): does a Codex-hosted MCP server see `CODEX_THREAD_ID`?

With the TUI from L-1 still open (its win-agent-teams MCP server is running):

```bash
wat_env
```

**Record:** whether the MCP server started by that `codex` has
`CODEX_THREAD_ID`, and whether it equals `$TID`; whether `CODEX_HOME` is set.
Yes means part D can register a Codex lead by itself; no means the lead must
run the shell command and call the registration tool.

### L-3 (S-2): environment of a Claude **child's** MCP server

1. In a Claude Code lead session with win-agent-teams connected, spawn an
   interactive Claude child: *spawn_agent a claude-code child named `s2probe`
   with the prompt "reply READY and then wait".*
2. Once it has replied, run `wat_env` and find the block whose
   `AGENT_NAME=s2probe`.
3. Find the child's own `claude` process and compare:

   ```bash
   ps -eo pid,ppid,comm,args | grep -E '[c]laude' | cut -c1-160
   ```

**Record, for the `s2probe` MCP server:**
- `WIN_AGENT_TEAMS_NATIVE_WAKE` (and any `WIN_AGENT_TEAMS_NATIVE_*` switches)
  present, with their values;
- `AGENT_NAME`, `AGENT_SESSION_ID`, `AGENT_PARENT_NAME`;
- `CLAUDE_CODE_MESSAGING_SOCKET`: its path, and whether the number in
  `<pid>.sock` is the **child's** `claude` PID (required for part B) or the
  lead's (a leak that must be scrubbed);
- `CLAUDE_CODE_MESSAGING_TOKEN` present or not.

Keep `s2probe` running for L-4.

### L-4 (S-3): how a socket-posted message is stored, and the size limit

This posts into the **throwaway child** `s2probe`, never into the session you
are working in. The child must be idle.

1. Save as `~/nd-smoke/s3_post.py`:

   ```python
   """Post one user line to a child's channel, reading its credentials from /proc."""
   import json, socket, sys, uuid
   from claude_teams import delivery

   pid, size = int(sys.argv[1]), int(sys.argv[2])
   env = dict(
       line.split("=", 1)
       for line in open(f"/proc/{pid}/environ").read().split("\0")
       if "=" in line
   )
   nonce = uuid.uuid4().hex
   body = "Reply with exactly S3-ACK and the length of this message. " + "y" * size
   text = delivery.delivered_prompt(body, nonce, single_line=False)
   wire = json.dumps({"type": "auth", "token": env["CLAUDE_CODE_MESSAGING_TOKEN"]}) + "\n"
   wire += json.dumps({"type": "user", "message": {"role": "user", "content": text}}) + "\n"
   with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
       s.settimeout(5)
       s.connect(env["CLAUDE_CODE_MESSAGING_SOCKET"])
       s.sendall(wire.encode())
       s.shutdown(socket.SHUT_WR)
   print(f"nonce={nonce} bytes={len(text.encode())}")
   ```

2. Run it with the PID of the `s2probe` **MCP server** from L-3, first small,
   then at the inline limit, then above it:

   ```bash
   cd ~/nd-smoke
   uv run python s3_post.py <mcp-pid> 100
   uv run python s3_post.py <mcp-pid> 16000
   uv run python s3_post.py <mcp-pid> 60000
   ```

3. After each post, find the child's transcript and check the record:

   ```bash
   T=$(grep -l "<nonce>" ~/.claude/projects/*/*.jsonl | head -1)
   grep "<nonce>" "$T" | jq -c '{type, role: .message.role, ctype: (.message.content|type)}'
   uv run python -c "
   import json, sys; from claude_teams import delivery
   for l in open('$T'):
       r = json.loads(l)
       if '<nonce>' in l: print(r.get('type'), delivery.receipt_nonces(r, 'claude-code'))"
   ```

**Record, per size:** whether the post succeeded; whether the child took a turn
and replied S3-ACK; the transcript record's `type` and content shape; whether
`receipt_nonces` finds the nonce (it must, for part B's receipts to work); and
whether the text arrived intact.

Kill `s2probe` afterwards.

## Part 2: after parts A and B are implemented

Run with `WIN_AGENT_TEAMS_NATIVE_WAKE=1` and
`WIN_AGENT_TEAMS_NATIVE_DOWNSTREAM=1` in the lead's MCP env. Detailed steps
will be added here when the code lands; the checks are:

| Id | Plan | Check |
|---|---|---|
| L-5 | N1 | Idle Codex child: `follow_up_agent` delivers in place. Same PID before and after, `method: codex_queue`, status `delivered`. |
| L-6 | N3 | Idle Claude child: delivered in place, same PID, `method: claude_mailbox`, no approval prompt in bypass mode. |
| L-7 | N4 | Busy Claude child: the message is held while the turn runs and posted once the child goes idle, exactly once. |
| L-8 | N5 | Killed child ⇒ the next follow-up resumes. An unresolved queued item blocks later follow-ups with `prior_native_attempt_unresolved` until it is found or `win-agent-teams deliveries release-native <key>` releases it. |
| L-9 | N7 | Codex lead woken by `codex queue` when a child replies. |
| L-10 | N8 | Flag baselines: master off ⇒ behaviour as on `main`; downstream off ⇒ every follow-up resumes. |

## Results

| Id | Date | Versions (codex / claude) | Result | Evidence |
|---|---|---|---|---|
| L-1 (N2) | | | | |
| L-1 20 000 chars | | | | |
| L-2 (S-6) | | | | |
| L-3 (S-2) | | | | |
| L-4 (S-3) 100 | | | | |
| L-4 (S-3) 16 000 | | | | |
| L-4 (S-3) 60 000 | | | | |
