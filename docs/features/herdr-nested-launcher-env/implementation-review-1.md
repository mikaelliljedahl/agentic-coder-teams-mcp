# Implementation review 1 — Claude Opus

The first independent review was run against the initial implementation before
finalization. It found the following issues:

1. **High:** Codex passed arbitrary launcher values through a restrictive TOML
   literal encoder, so apostrophes or newlines aborted a spawn.
2. **Medium:** global test setup removed the inherited launcher after the
   process-manager singleton was selected, breaking tests run from Herdr.
3. **Medium:** tmux context omitted `TMUX`, allowing nested delegation to select
   a detached server.
4. **Low/medium:** the comment incorrectly attributed the loss to MCP env
   replacement instead of daemon-created panes.
5. **Low:** the global pi config needed an explicit rationale for remaining
   launcher-agnostic.
6. Tests and required feature documentation were incomplete.

All findings were accepted and addressed. A later plan review found an
additional astral-Unicode hole in the first TOML basic-string encoder; that was
also corrected before final review. The complete original review is retained in
the development session artifact `/tmp/claude-herdr-nested-review.md`; this file
records its actionable conclusions in the repository.
