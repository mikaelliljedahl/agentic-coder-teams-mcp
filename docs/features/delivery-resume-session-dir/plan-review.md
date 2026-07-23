# Plan review: pi resume session directory

## Verdict

**CHANGES-REQUESTED**

The proposed fallback would compute the expected disk-contract path for requests
created by the current server, and it would leave the normal explicit-extra
spawn path unchanged. However, the plan does not establish that the fallback is
reachable in production: the current `follow_up_agent` implementation already
puts `session_dir=str(_session_dir(session_id))` into the resume request. The
proposed RED test deliberately constructs a request that the production server
does not currently construct. Before changing backend layering, the plan needs
to identify the actual path that dropped `session_dir` (or revise the fix to
enforce and test the existing server-side invariant).

## Findings

### 1. Major — the claimed production request shape contradicts the current server code

The plan says resume `extra["session_dir"]` “can be absent/empty,” but
`server_simple.follow_up_agent` currently builds resume `extra` with:

```python
"session_dir": str(_session_dir(session_id)),
```

The spawn path does the same. These are the only production calls to
`backend.resume` found under `src/`. The plan itself is also internally
inconsistent: its proposed-design section acknowledges that the server's
`follow_up_agent` resume path supplies the value, while its current-behaviour
section attributes the observed failure to that value being absent.

Consequently, the proposed backend test demonstrates how `_pi_session_dir`
behaves for a synthetic/incomplete `SpawnRequest`, but it does not reproduce the
current production path or explain the cited smoke failure. The implementation
could turn that test green while leaving the real defect untouched.

Required disposition: trace or capture the actual failing resume request and
document why its `extra` lacks `session_dir` despite the code at
`server_simple.py` around the follow-up request construction. If the smoke came
from an older commit or a different resume entry point, say so and identify the
current reachable entry point. If no current path omits the value, reframe this
as invariant hardening rather than the fix for the observed production bug.

### 2. Major — backend-to-server import is the wrong layering and `lead_wake.py` is not precedent for it

A lazy import avoids the immediate module-initialization cycle when
`server_simple` imports the backend registry, so it is likely to work in the
normal server process. It is still an undesirable dependency:

- `PiBackend` is otherwise usable independently of the MCP server, whereas the
  fallback would import the full `server_simple` module, construct its `FastMCP`
  object, and initialize server globals merely to compute a path.
- Calling the backend standalone or from a test before `server_simple` is
  loaded now gains heavyweight server import side effects.
- The server already depends on the backend layer; making a backend depend back
  on the server, even lazily, creates a conceptual cycle that static tooling and
  future refactors cannot see cleanly.

`lead_wake.py` is not an equivalent precedent. It is an application-level hook
module that imports `server_simple` at module scope specifically to use server
session discovery. It does not sit below `server_simple` in the dependency
graph, and its module docstring explicitly discusses keeping that dependency
out of shared lower-level modules.

Preferred design: keep the request invariant server-side and make missing
`session_dir` explicit in the backend (for example, fail clearly rather than
silently using `cwd`). If fallback derivation is genuinely required outside the
server, extract the disk path contract into a small import-light module used by
both `server_simple` and the backend. Do not duplicate the home-directory
constant. Any extraction must also preserve the testability currently provided
by monkeypatching `_SESSION_BASE`.

### 3. Minor — `team_name == session_id` is true for current server requests, not guaranteed by the backend contract

For both current server constructions, `team_name=session_id`; spawn and resume
also use the same local `session_id`, and `_session_dir(session_id)` therefore
reproduces the exact spawn base in production. On that narrow question, the
derivation is correct.

But `SpawnRequest.team_name` is only documented as a backend-agnostic team name,
not as the canonical on-disk session ID. A backend fallback would turn a
server-specific convention into an undeclared backend contract. Direct backend
callers could provide a human team name, and a spawn with a custom explicit
session base cannot later be reconstructed from `team_name` if resume loses the
extra. If the design retains this fallback, update the `SpawnRequest` contract
or add a dedicated session-directory/session-id field rather than relying on an
incidental field meaning.

### 4. Major — the RED test is not a faithful production regression test

The proposed parity test is useful as a focused unit test of the proposed
fallback, but not as the sole RED reproduction. It manually gives spawn the
correct explicit base and manually removes it from resume, whereas the current
server supplies it in both cases. It therefore cannot prove the real
spawn-to-follow-up pipeline was fixed.

The plan should add a server-level regression test that spies on the pi
backend's spawn/resume calls (or their built commands) through `spawn_agent` and
`follow_up_agent`, then verifies:

- both requests carry the exact `str(_session_dir(session_id))` base;
- both commands contain exactly one `--session-dir`;
- the values equal
  `str(_session_dir(session_id) / "pi-sessions" / agent_name)`;
- the resume value is independent of the stored agent `cwd`;
- the same `team_name/session_id` and agent name are retained.

If an independently reachable path genuinely omits `session_dir`, exercise that
actual path rather than constructing only a backend request by hand. For the
focused fallback test, cover both a missing key and an empty string because the
proposed implementation treats both identically. The “extra wins” regression
should use a custom base deliberately different from both `cwd` and the
team-derived path, and should assert the full exact value rather than only a
suffix.

### 5. Minor — server-side enforcement is lower-risk and should be the primary fix unless another caller is proven

Threading `session_dir` through resume is the clean ownership boundary:
`server_simple` owns `_SESSION_BASE`, active-session resolution, and request
construction. It also preserves nonstandard/test session bases exactly, without
reconstructing them from a team identifier.

In the current tree this threading is already present, so the useful change may
instead be an assertion/regression test protecting that invariant, coupled with
a clear backend error when a pi request omits `session_dir`. A backend fallback
can be additional defense only if there is a legitimate caller that cannot
carry the authoritative base. The plan should name that caller and explain why
fixing it at request construction is insufficient.

### 6. Minor — compatibility risk is small for normal spawn, but understated for incomplete/direct requests

Normal server-driven spawn behaviour is unchanged because explicit
`extra["session_dir"]` remains authoritative. Resume requests from the current
server are also unchanged. Other backends are untouched, so there is no direct
behavioural risk to Codex or Claude Code.

The changed fallback does affect pi requests with missing or empty
`session_dir`, including initial spawns made directly against the backend:
today they store under `<cwd>/pi-sessions/<name>`; the proposal moves them under
the server's home-based session tree. The plan calls `cwd` “never correct,” but
has not established that for every backend caller. This is another reason to
either formalize `session_dir` as required or identify and test all supported
fallback callers.

## Answers to the requested review questions

1. **Exact production path:** yes, for requests constructed by the current
   server, because both spawn and follow-up assign `team_name=session_id` and
   spawn's explicit base is `_session_dir(session_id)`. The unresolved issue is
   that those same production paths already supply `session_dir` on resume.
2. **Lazy import:** it probably avoids the runtime circular-import crash on the
   normal path, but it introduces poor layering and heavyweight side effects.
   `lead_wake.py` is not precedent for a backend importing the server. Prefer
   server-side ownership or an import-light shared path-contract module.
3. **Server-side fix:** yes, this should be primary because the server owns the
   authoritative path. In fact the current server already does it, so the plan
   must explain the observed mismatch before adding a second derivation.
4. **Test design:** the proposed test is a valid backend fallback test, not a
   faithful production reproduction. Add a server-level pipeline regression
   and stronger exact/occurrence/empty-value assertions.
5. **Breakage risk:** explicit-extra spawn and all other backends remain
   unchanged. Missing-extra direct pi requests change semantics and the lazy
   import couples standalone backend use to server initialization.

