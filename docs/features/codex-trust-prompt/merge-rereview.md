## Re-review (758e3cf)

Verdict: **approve; both original minor findings are resolved.** New findings: **0 blocker, 0 major, 0 minor, 0 nit**.

Scope: reviewed `git diff ef1b370 758e3cf`, including cee9d98, 72bc7ca and 758e3cf. HEAD was verified as 758e3cf2064c23fa7d1ad8ea7d8a8969906435ae. No tracked files were edited.

### Finding 1: resolved

`src/claude_teams/server_simple.py:3109` now delegates to the same resolver used by Codex launch environment construction and rollout lookup. Native metadata therefore records the absolute home required by native thread verification.

No regression found for the requested compatibility cases:

- Unset and empty values retain the default `Path.home() / ".codex"`; the launch builder continues to omit its explicit CODEX_HOME entry in those cases. The fix does not change child environment inheritance.
- Absolute values resolve consistently with the existing launch behavior, including Windows drive-root completion of `/new-home`. Updating that existing test's expected value is correct.
- Relative values now resolve against the server's cwd before being persisted.
- Whitespace-only and whitespace-containing values now follow the launch resolver instead of the old metadata-only `.strip()` rule. This intentionally changes metadata to match the existing child environment; it does not change how the child is launched. Whitespace-only input is not newly defined as the default home.

The added `tests/test_native_record_fields.py:69` test is not tautological: besides comparison with the resolver, it independently checks the expected directory and runs real `verify_codex_thread()` against a matching rollout created in that directory. Returning the former relative string would fail both the directory expectation and native verification. Its fake spawn/headless setup is sufficient for this metadata contract; existing backend tests separately check actual launch environment construction.

### Finding 2: resolved

The six cases in `tests/test_trust_cwd_native.py` meaningfully cover the five requested transitions:

- Native success is exercised with both changed launch mode and a direct-launch setting that would refuse resume. Actual delivery method, queue invocation, no discovery/resume/shutdown, unchanged PID/token/epoch and lease release are asserted.
- Stage-2 loss changes the thread-verification response rather than bypassing the production stage-2 control flow. The test checks re-evaluation, trust refusal before request construction, zero attempts, no queue/resume and released lease.
- Queue-construction failure goes through the real queue runner with a fake Popen. The persisted `native_not_enqueued` reason and one attempt distinguish it from an early refusal; pending status and `is_unresolved_native` ensure the reverted attempt does not retain the unresolved-native barrier.
- Successful fallback asserts all three trust extras, an epoch above the old one, the persisted new epoch/PID, resume method and two attempts. The fake backend does not itself generate these extras, so these assertions test production behavior.
- Changed binary at fallback commit checks two discoveries, the specific changed-binary refusal, no resume/shutdown, one attempt, pending state and lease release. Dropping the comparison would allow the fake resume and fail these assertions.

The helpers inspect real persisted agent/delivery/lease state. Spies delegate to production request construction. Process execution, binding and eligibility dependencies are simulated, which is appropriate for these integration boundaries; the tests do not claim to verify live Codex parsing or Windows Terminal transport. The recorded mutation results are consistent with the assertions and control flow; I did not independently rerun mutations because this review must not edit tracked files. No issue found in the added implementation notes.

### Validation

With `WIN_AGENT_TEAMS_NATIVE_WAKE` and `WIN_AGENT_TEAMS_NATIVE_DOWNSTREAM` unset, ran:

```text
uv run pytest tests/test_trust_cwd_native.py tests/test_native_record_fields.py tests/test_native_selection.py tests/test_native_codex_dispatch.py tests/test_trust_cwd.py tests/test_follow_up_delivery.py tests/test_backends/test_codex.py tests/test_agent_output.py -q
477 passed in 37.61s
```

An additional read-only Python probe compared `_effective_codex_home()`, the shared resolver, independent expected paths and real `CodexBackend.build_env()` output for eight cases: unset, empty, absolute, relative, whitespace-only, leading space, internal space and trailing space. **All eight passed.** These probes verify consistent resolution/environment construction, not whether every whitespace-named directory is usable by a real Codex installation on every filesystem.
