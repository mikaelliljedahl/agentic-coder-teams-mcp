# Merge-resolution review

Verdict: approve with non-blocking issues. No blocker or major defect found in the conflict resolutions or the ef1b370 Windows launch correction. Two minor findings: a CODEX_HOME integration mismatch in an auto-merged path, and missing committed coverage of the trust/native transitions.

Scope: detached ef1b370e3793cc56071df0ce9ae6b9913cdf9ba1; merge 45c89cd compared with both 3c19d20 and 1388c37 using the remerge diff and parent diffs. This is a review of their integration, not a fresh review of either feature. No tracked file was edited.

## Numbered findings

1. **Minor — native metadata does not use the merged canonical CODEX_HOME resolver.**

   **Location:** `src/claude_teams/server_simple.py:3109` and `:3211`; compare `src/claude_teams/backends/codex.py:704`, `src/claude_teams/codex_home.py:7`, and `src/claude_teams/native_wake.py:916`.

   **Concrete scenario:** Start the server with `CODEX_HOME=isolated`, enable native downstream delivery, and spawn an interactive Codex agent with `trust_cwd=True`. The PR's launch code exports an absolute home and rollout lookup uses that same resolved directory, so the agent can launch and bind correctly. Main's auto-merged `_effective_codex_home()` instead persists `isolated` in the native record. `verify_codex_thread()` unconditionally rejects a relative home as `home_missing`; `_native_candidate()` therefore drops the native carrier and follows the resume path. A busy agent waits unnecessarily; an idle agent may be restarted instead of receiving an in-place queue delivery. This also affects agents without trust_cwd.

   **Evidence / provenance:** An isolated probe created a matching rollout under the resolved home. Verification returned `(True, '')` for `str(codex_home())`, but `(False, 'home_missing')` for the home produced by `_native_record_fields()`. The raw-home helper and absolute-path requirement were inherited from main; the integration oversight is leaving that helper separate from the PR's newly shared resolver. This is not a newly introduced trust bypass.

   **Suggested fix:** Have `_effective_codex_home()` use `str(codex_home())`, so launch, readers and native carrier metadata agree. Add a relative-home spawn/resume test asserting the persisted native home equals the child environment and permits native thread verification. Preserve path characters consistently rather than separately stripping the environment value.

2. **Minor — the newly resolved trust/native branch boundaries lack committed regression tests.**

   **Location:** `tests/test_native_codex_dispatch.py:348`, `tests/test_native_selection.py:1032`, and `tests/test_follow_up_delivery.py:273`; production boundaries at `src/claude_teams/server_simple.py:5588`, `:5729`, and `:5893`.

   **Concrete regression scenario:** Removing the `native_method is None` guard would reject an otherwise valid in-place delivery after the server's launch mode changes. Conversely, bypassing `_prepare` after stage-2 eligibility loss or a provably unqueued native attempt would allow a resume without the required trust preflight. Existing native tests do not set `trust_cwd`; existing trust follow-up tests do not exercise an eligible native carrier. They therefore do not pin the behavior chosen during conflict resolution.

   **Suggested fix:** Commit focused cases for trusted native success despite an unsafe current resume environment; stage-2 native loss followed by trust refusal; provably-not-enqueued followed by trust refusal; successful fallback preserving both trust extras and a new dispatch epoch; and a changed binary at fallback commit. Assert old PID, lease release, pending delivery phase, attempt counts and preservation of trust/epoch metadata. Review-only temporary probes for these five cases all passed; this finding is a coverage gap, not an observed failure in those paths.

## Areas explicitly checked

- **Native eligibility before trust preflight: no functional issue found.** Native delivery does not relaunch the target or reapply project trust. The queue helper is a separate process, but it only submits to the existing session and carries no project-trust override. Both stage-2 eligibility loss and a provably unqueued submission set `native_allowed=False` and re-enter `_prepare`; if the budget expires first, they return pending without launching anything. A supported `trust_cwd=True` spawn is Codex-only, so a trusted Claude mailbox target is not reachable through the public spawn contract.
- **Commit-time recheck and durable state: no issue found.** The only production plan constructor pairs `METHOD_RESUME` with `_build_resume_request()`'s non-null request; native plans have no request and return before resume. The recheck precedes `_mark_attempt_sent` and PID shutdown. Its refusal executes the lease-release finalizer. After a reverted native attempt, the row remains pending, with the native attempt counted but no extra resume attempt or agent pending marker added. Temporary probes verified these states, including a changed binary at fallback commit. An unused allocated dispatch epoch is a harmless monotonic gap; it does not alter the live agent's epoch.
- **Lease-release signature: no issue found.** All five call sites pass `(session_id, agent_name, operation_id)`: request-build failure at 5740, the three stage-2 exits at 5825/5830/5841, and commit finalization at 6039. No old two-argument call remains.
- **Extras merging and dispatch epochs: no issue found.** `_dispatch_extra()` emits only `dispatch_epoch`, so it cannot overwrite `codex_trust_*`. The additional pinned trust fields and command validation are guarded by `request is not None`. Native finalization preserves the existing trust and launch facts and does not mint an epoch. Successful fallback carries both families of extras.
- **ef1b370 and WT quoting: no issue found.** On a real Windows host, the raw flag and `codex_direct_launch_enabled()` agree. The latter is checked in server preflight and the Codex command builder; the normal trusted launch cannot reach the direct WT branch with the flag enabled. WindowsProcessManager is selected only on Windows in production. The raw-flag correction restores Linux-hosted tests of that class without changing the Windows decision. Main's direct-launch blockers, `_wt_argv`, wrapper fallback and PowerShell quoting remain intact. No new deterministic direct-launch bypass was found.
- **Other auto-merged interactions:** no additional issue found in trust persistence, native finalization, N5 barriers, launch metadata or epoch propagation. The home-path discrepancy is finding 1. Real Codex/WT smoke testing was not performed; the pre-existing Windows smoke requirement is outside this conflict-resolution review.

## Validation

Five temporary trust/native integration probes passed, covering native success, stage-2 loss, queue construction failure, successful resume fallback and commit-time refusal. A sixth probe reproduced finding 1. The probes used existing repository fixtures and real delivery/lease stores with simulated processes, lived outside the worktree, and did not edit repository tests.

Focused repository suite: trust preflight, follow-up delivery, native selection, Codex/Claude dispatch, native record fields, Codex backend, WT command line and PowerShell quoting. The first run had 542 passed and one failure caused by inherited `WIN_AGENT_TEAMS_NATIVE_WAKE=1` / `WIN_AGENT_TEAMS_NATIVE_DOWNSTREAM=1` in a flag-off identity assertion. That assertion passed after disabling those ambient flags. Final clean-environment suite result recorded below.

Final clean-environment focused suite: **543 passed in 29.17s**. Temporary review probes: **6 passed in 3.13s**.

Finding counts: **0 blocker, 0 major, 2 minor, 0 nit**.
