# Plan review 1: OpenAI pricing tier remap

## Findings

### MAJOR — The test plan drops explicit coverage of the retained Sol hard-fail path

The proposed Luna mapping is internally consistent with the stated measurements: `medium` maps to Luna @ xhigh to replace Sol @ low, `high` maps to Luna @ max to replace Sol @ medium, and the two stronger Sol tiers remain unchanged (`plan.md:26-28`, `plan.md:36-42`). Codex, however, validates whichever concrete slug the tier selects and raises when that slug is absent (`src/claude_teams/backends/codex.py:209-220`, `src/claude_teams/backends/codex.py:237-246`).

The current suite explicitly proves that a Sol-backed tier fails when Sol is absent (`tests/test_backends/test_codex.py:139-143`). Once `high` becomes Luna-backed, that test must move to `xhigh` or `ultra`. The plan's enumerated cases mention only missing Luna for `medium` (`plan.md:79-80`), so following the plan literally can replace the old test and leave no assertion that the unchanged upper Sol tiers still hard-fail.

Revise the test matrix to require both catalog boundaries:

- Luna absent but Sol present: a Luna-backed tier such as default `medium` raises `BackendModelUnavailableError`.
- Sol absent but Luna present: `xhigh` or `ultra` raises `BackendModelUnavailableError`.

Retain the existing discovery-empty behavior separately; an empty discovery result intentionally skips validation (`src/claude_teams/backends/codex.py:224-245`, `tests/test_backends/test_codex.py:156-159`).

### MAJOR — Pi's new `max` fallback can cease to be a soft fallback, and the plan does not resolve or test that risk

Pi drops an unavailable tier model but retains the tier's thinking level (`src/claude_teams/backends/pi.py:180-193`). The new `high` tier therefore resolves to `("", "max")` when Luna is unavailable. Command construction then omits `--model` but still emits `--thinking max` for whatever provider/model Pi chooses as its default (`src/claude_teams/backends/pi.py:137-139`, `src/claude_teams/backends/pi.py:356-373`).

Membership of `max` in `_THINKING_OPTIONS` proves that this adapter accepts the spelling; it does not prove that an arbitrary fallback provider/default model accepts that level. This is materially riskier than the current fallback test, which retains only `low` (`tests/test_backends/test_pi.py:129-133`). The plan tests only the new `medium` fallback `("", "xhigh")` (`plan.md:79-80`) and never exercises the newly introduced `("", "max")` path.

The plan needs an explicit design decision for missing Luna on Pi:

- If keeping current semantics, state the compatibility assumption and add a `high` fallback test plus a command-level assertion that the fallback emits no `--model` and does emit `--thinking max`.
- If Pi/default-provider compatibility is not guaranteed, define a safe fallback effort (or omit `--thinking`) instead of claiming the operation soft-falls-back.

The direct Luna path should also have a command-level assertion for `openai-codex/gpt-5.6-luna` with `--thinking max`; tuple-only `resolve_launch` tests do not prove the new highest effort reaches Pi's argv.

### MINOR — The local option sets do not establish model/effort compatibility at the external launch boundary

The statement that no spec changes are needed is correct at the adapter-validation level: Codex includes `max` in its advertised effort set (`src/claude_teams/backends/codex.py:124-130`) and Pi includes it in `_THINKING_OPTIONS` (`src/claude_teams/backends/pi.py:137-139`). Codex's own comment also says the actual model/effort combination is validated only at launch (`src/claude_teams/backends/codex.py:127-129`).

Accordingly, the plan should distinguish “accepted by this repository” from “accepted by the installed CLI/model.” Add either a live/manual compatibility check for Luna @ high/xhigh/max on both backends or record that external compatibility is an explicit deployment precondition. At minimum, command-construction tests should prove the intended Luna slug and each new effort are emitted.

### MINOR — The TDD section has assertions but no executable verification gate

`plan.md:73-82` identifies useful behaviors, but it does not name the focused test commands or a full regression command. Add the exact commands for:

- the two backend test modules;
- the relevant server/spawn tests if the `server_simple.py` example is touched; and
- the repository's normal full test gate.

Require the initial focused run to fail on the old mapping and the same run to pass after the table/test updates. This makes “TDD” reproducible rather than descriptive.

### NIT — The `low` tier choice is plausible but not supported by the measurements cited in the plan

The evidence supports Luna @ xhigh versus Sol @ low and Luna @ max versus Sol @ medium (`plan.md:26-28`), which directly justifies `medium` and `high`. It does not explain why the low tier is specifically Luna @ high rather than another Luna effort. The mapping remains an ascending Luna ladder and is not contradicted by the quota rationale, but one sentence stating the quality/latency basis for Luna @ high would make the design fully traceable.

## Completeness and correctness summary

- The proposed five pairs match the stated pricing strategy and preserve the stronger Sol tiers.
- The affected production files are complete for the non-documentation code searched: the tier tables live only in `codex.py` and `pi.py`, and the stale `server_simple.py` example listed at `plan.md:57-59` is the only additional source wording tied to an old concrete pair.
- Keeping Terra raw-slug passthrough is consistent with both resolvers, which pass non-tier names through (`src/claude_teams/backends/codex.py:175-194`, `src/claude_teams/backends/pi.py:156-167`).
- Default `medium` remains unchanged as a tier name (`src/claude_teams/backends/codex.py:164-173`, `src/claude_teams/backends/pi.py:152-154`), but its new Luna availability dependency is correctly identified in the plan.

CHANGES-REQUIRED
