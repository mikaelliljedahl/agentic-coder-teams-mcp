# Pi CLI non-interactive invocation findings

Verified against the locally installed `@earendil-works/pi-coding-agent` (settings report `lastChangelogVersion: 0.83.0`), its installed extensions, `pi --help`, and runtime experiments on Windows.

## A1

**Answer**

Pi core ships **no LLM-callable human-question or permission tool**. Its built-in tool list is only `read`, `bash`, `edit`, `write`, `grep`, `find`, and `ls`. Core does have UI primitives (`select`, `confirm`, `input`, `editor`, `custom`) for extensions and interactive commands, plus the startup project-trust selector, but those are not tool names. The shipped README explicitly says “No permission popups.”

Mechanisms actually present in this installation/runtime are:

1. **`ask_user`** — a user-global extension tool registered by `C:\code\github\my-pi-setup\extensions\ask-user\index.ts`.
2. **MCP `elicitation/create` handler** — installed through `pi-mcp-adapter`; this is a protocol handler, **not an LLM tool**, so it has no tool name.
3. **MCP tool approval gate** — also in `pi-mcp-adapter`; it applies to whatever MCP tool is configured by `approveTools`, rather than having a separate tool name. In this run, the adapter exposes `mcp`, `mcpScript`, and direct `win_agent_teams_*` tools, but approval is conditional configuration, not an intrinsic property of those names.
4. **MCP sampling approval** — a protocol callback, not an LLM tool and therefore has no tool name.
5. **Project trust prompt** — a core startup mechanism, not an LLM tool. `-a` bypasses it by trusting; `-na` bypasses it by declining project resources.

No core tools named `ask_human`, `elicit`, or `request_input` were found. `ask_question` occurs only as an illustrative name in CLI help (`--exclude-tools ask_question`), not as a registered core tool.

**Evidence**

- Command: `node .../dist/cli.js --help` printed `Built-in Tool Names: read, bash, edit, write, grep, find, ls` and no question tool.
- Installed `README.md`, Philosophy: `**No permission popups.** ... build your own confirmation flow with extensions`.
- `C:\Users\mlilj\.pi\agent\settings.json` contains the global extension path `C:\code\github\my-pi-setup\extensions\ask-user\index.ts` and package `npm:pi-mcp-adapter`.
- Ask extension snippet:
  ```ts
  pi.registerTool({
    name: "ask_user",
  ```
- `pi-mcp-adapter/elicitation-handler.ts:29-31`:
  ```ts
  client.setRequestHandler("elicitation/create", request =>
    handleElicitationRequest(options, request));
  ```
- `pi-mcp-adapter/tool-approval.ts` calls `state.ui.select(... ["Allow once", "Allow for session", "Deny"])` for the target MCP tool.
- `dist/core/project-trust.js:9` calls `ctx.ui.select(...)`; it does not register a tool.
- Grep command: `rg 'ask_user|ask_question|ask_human|elicit|request_input|permission|approval' .../dist -g '*.js'`. The only `ask_question` hit was the help example in `dist/cli/args.js:324`.

**Confidence**: verified.

## A2

**Answer**

In `-p --mode json`, Pi chooses JSON mode and binds extensions with `ctx.mode === "json"` and `ctx.hasUI === false`. Lack of a TTY therefore does not itself cause a prompt wait.

Behavior by mechanism:

1. **`ask_user`**: remains in the tool list, but its own execute path checks `ctx.mode !== "tui"` and immediately returns a cancelled/no-UI tool result. It neither blocks nor errors. This is closest to **(c) auto-decline/return a default**, though the returned text is specifically “question could not be shown,” not an answer.
2. **Generic extension UI calls**: Pi supplies `noOpUIContext`: `select/input/editor/custom => undefined`, `confirm => false`, notifications/setters => no-op. Thus a well-behaved extension gets defaults immediately. Extensions can still implement their own blocking I/O, so that universal behavior is UNVERIFIED for arbitrary third-party code.
3. **MCP elicitation**: adapter initialization requires `hasUI`; in JSON/print mode it does not install elicitation config at all. This is effectively **(d) handler filtered/disabled**, not a tool-list change.
4. **Approval-gated MCP tool calls**: the MCP tool remains callable. If approval is required and no broker answers, `ensureToolCallApproved()` sees no UI and immediately returns `approval_required_headless`; the tool returns an error result saying it requires an interactive session. This is **(a) error/result refusal**, not a block.
5. **MCP sampling**: without UI it is not configured unless `samplingAutoApprove: true`. With auto-approve, it runs without prompting. Its defensive no-UI path throws “requires interactive approval.”
6. **Project trust**: no prompt is shown. After saved decision/default policy processing, `hasUI === false` falls back to `false`; project resources are ignored unless `-a` or `defaultProjectTrust: "always"` applies.

**Evidence**

- `dist/main.js:80-88` gives explicit `--mode json` precedence and otherwise selects print mode for `-p` or non-TTY streams.
- `dist/modes/print-mode.js:45-48` binds mode as `json` or `print`.
- `docs/extensions.md`, Mode Behavior table: JSON has `ctx.hasUI false`, UI methods no-op; print likewise; “Extensions run but can't prompt.”
- `dist/core/extensions/runner.js:87-119`:
  ```js
  select: async () => undefined,
  confirm: async () => false,
  input: async () => undefined,
  custom: async () => undefined,
  editor: async () => undefined,
  ```
- Ask extension:
  ```ts
  if (ctx.mode !== "tui") {
    return reply(buildAskUserResultMessage({ kind: "no-ui" }));
  }
  ```
  Its result text is: `No interactive UI is available, so the question could not be shown. Ask the user in plain text instead.`
- `pi-mcp-adapter/init.ts:135-140`: `elicitationEnabled = ... && hasUI`.
- `pi-mcp-adapter/tool-approval.ts`:
  ```ts
  if (!state.ui) return { ok: false, reason: "approval_required_headless" };
  ```
- `pi-mcp-adapter/init.ts:123-140` and `sampling-handler.ts:180-188` show `samplingAutoApprove` and the no-UI error.
- `dist/core/project-trust.js:41-50` processes `always`/`never`/`ask`, then returns false when `!hasUI`.

**Confidence**: verified.

## A3

**Answer**

Core CLI controls are:

- `--no-tools`, `-nt`: disable all tools by default, including extension/custom tools.
- `--no-builtin-tools`, `-nbt`: disable only built-ins.
- `--tools`, `-t <comma-list>`: allowlist exact tool names across built-in, extension, and custom tools. An autonomous launch can omit `ask_user` here.
- `--exclude-tools`, `-xt <comma-list>`: denylist exact tool names; for this installation use `--exclude-tools ask_user`.
- `--no-extensions`, `-ne`: disable discovered extensions while retaining explicit `-e` extensions.
- `--no-approve`, `-na`: ignore project-local resources for the run. This is project trust, not tool-call auto-answering.
- `--approve`, `-a`: trust project-local resources; it does **not** mean “yes to every question” or MCP tool approval.

There is no core `--yes`, `--non-interactive`, `--allowed-tools`, or `--disable-tool` flag in this installed help output. `-p`/`--print` and `--mode json` select non-interactive execution but do not remove interactive tools automatically. No core environment variable for auto-answering extension UI was listed by help or `docs/environment-variables.md`.

Relevant extension-specific config in `pi-mcp-adapter` is:

- `settings.elicitation: false` disables MCP elicitation.
- `settings.sampling: false` disables MCP sampling.
- `settings.samplingAutoApprove: true` permits sampling without UI.
- `settings.approveTools` and per-server `approveTools` select which MCP calls need approval; setting it false/omitting it means those adapter approval prompts are not required (subject to any external approval broker).

The installed `ask_user` extension itself has no auto-answer config or environment variable; exclude the tool or extension.

**Evidence**

- Exact spellings and descriptions are from `node .../dist/cli.js --help`.
- `dist/cli/args.js:70-93,119-149` parses the tool/resource switches listed above.
- `docs/environment-variables.md` lists Pi process settings and contains no prompt-auto-answer variable.
- `pi-mcp-adapter/types.ts:470-484` declares `approveTools`, `sampling`, `samplingAutoApprove`, and `elicitation`.
- `pi-mcp-adapter/init.ts:123-140` implements those sampling/elicitation switches.
- Full read of `C:\code\github\my-pi-setup\extensions\ask-user\index.ts` found no config/env lookup.

**Confidence**: verified.

## A4

**Answer**

By default, Pi resolves resources from both settings/packages and conventional locations:

- User: `~/.pi/agent/extensions`, `skills`, `prompts`, `themes`; user `settings.json` paths/packages; and `~/.agents/skills`.
- Project (only when trusted): `.pi/extensions`, `skills`, `prompts`, `themes`, project `.pi/settings.json` paths/packages, and ancestor `.agents/skills` up to the Git root.
- Context: one `AGENTS.md` or `CLAUDE.md` candidate from `~/.pi/agent`, then each directory from filesystem root down to cwd (with worktree de-duplication).
- System prompt files: trusted project `.pi/SYSTEM.md` / `.pi/APPEND_SYSTEM.md`, otherwise user-global `~/.pi/agent/SYSTEM.md` / `APPEND_SYSTEM.md`.
- Explicit CLI resources: `-e`, `--skill`, `--prompt-template`, and `--theme`.

`-a` means **trust project-local files for this run**. It opts into project `.pi/settings.json`, `.pi` resources/packages, ancestor project `.agents/skills`, and project extensions/system prompts. It does not control global resources and does not auto-approve later questions.

To load only explicitly supplied extension files and no discovered extensions, use:

```text
--no-extensions -e <ext1> -e <ext2>
```

For a truly isolated orchestrator launch with no user profile or project instructions/resources, set `PI_CODING_AGENT_DIR` in the child environment to a fresh empty directory and combine:

```text
-na -ne -ns -np --no-themes -nc -e <ext1> -e <ext2>
```

`-na` suppresses trusted project `.pi/SYSTEM.md`; the empty `PI_CODING_AGENT_DIR` prevents global settings, packages, extensions, `SYSTEM.md`, and `APPEND_SYSTEM.md` from being found. `-nc` removes `AGENTS.md`/`CLAUDE.md`. The resource-specific `--no-*` flags prevent other discovery while explicit matching flags remain accepted. If the `-e` source is itself a Pi package, it may contribute its manifest resources; use plain extension file paths when “extensions only” is literal.

**Evidence**

- `pi --help`: `--no-extensions ... explicit -e paths still work`; corresponding `--no-skills`, `--no-prompt-templates`, `--no-themes`, and `--no-context-files` flags.
- `dist/core/resource-loader.js:315-317`:
  ```js
  const extensionPaths = this.noExtensions
    ? cliEnabledExtensions
    : this.mergePaths(cliEnabledExtensions, enabledExtensions);
  ```
- `dist/core/resource-loader.js:329-375` has analogous no-skill/prompt/theme handling and `noContextFiles ? []`.
- `dist/core/package-manager.js:1918-1984` resolves user/project override arrays and the conventional `.pi`/`~/.pi/agent` directories; lines 1941-1966 include `~/.agents/skills` and trusted ancestor `.agents/skills`.
- `dist/core/resource-loader.js:81-110` loads global context then walks cwd ancestors.
- `dist/core/resource-loader.js:808-827` chooses project/global `SYSTEM.md` and `APPEND_SYSTEM.md`.
- `dist/core/project-trust.js:20-35` makes `trustOverride` authoritative.
- README CLI Resource Options: “Combine `--no-*` with explicit flags to load exactly what you need, ignoring settings.json.”

**Confidence**: verified.

## A5

**Answer**

The documented CLI hook is exactly:

```text
--append-system-prompt <text>
```

It may be repeated, and each value is treated as file contents when it names an existing file. For autonomous children, append an instruction such as: “Never call `ask_user`; when blocked on a decision, call MCP `send_message` to the parent and continue or stop safely.” Pair this with `--exclude-tools ask_user` for enforcement. In this runtime the parent communication tool is `win_agent_teams_send_message` (the MCP server’s original tool is `send_message`; the adapter/direct-tool prefix produces the exposed name).

There is also `--system-prompt <text>` to replace the default prompt, but appending is safer when only adding routing policy. Extensions can implement stronger policy through `before_agent_start`, `input`, or `tool_call`, but there is no built-in “redirect all questions to parent” switch.

**Evidence**

- `pi --help`: `--append-system-prompt <text>  Append text or file contents to the system prompt (can be used multiple times)`.
- `dist/cli/args.js:46-51` collects repeatable values.
- `dist/core/resource-loader.js:10-23` uses `existsSync(input)` and `readFileSync(input, "utf-8")`, otherwise returns literal input.
- `dist/core/system-prompt.js:8-10` appends the resolved section.
- `docs/extensions.md`, `before_agent_start`, documents per-turn system-prompt modification.
- Current runtime tool description for `win_agent_teams_send_message`: recipient defaults to the spawner/parent.

**Confidence**: verified.

## B1

**Answer**

Pi accepts all of these initial-input sources outside RPC mode:

1. positional argv message tokens;
2. piped stdin when stdin is not a TTY;
3. `@file` argv tokens, resolved by Pi itself, not the shell.

`@file` works in `-p` and JSON mode; it is explicitly rejected in RPC mode. Text files are not inserted raw: Pi wraps each in `<file name="absolute-path">\n...\n</file>\n`. Piped stdin is decoded as UTF-8 and trimmed. Pi combines stdin, file text, and the first positional message with `parts.join("")` (no added delimiter). Remaining positional messages are processed as later turns.

**Evidence**

- `pi --help`: usage `pi [options] [@files...] [messages...]`; examples include `pi -p @screenshot.png ...` and piped stdin.
- `dist/cli/args.js:172-173`: an argv token beginning `@` becomes `fileArgs.push(arg.slice(1))`.
- `dist/main.js:45-59`: non-TTY stdin is read with `process.stdin.setEncoding("utf8")`, accumulated, then `data.trim()`.
- `dist/cli/file-processor.js:15-50`: Pi resolves the path, reads text using `readFile(..., "utf-8")`, and adds the `<file>` wrapper.
- `dist/cli/initial-message.js:5-18`: pushes stdin/file/first message, shifts that first message, and uses `parts.join("")`.
- `dist/main.js:471-473`: `@file arguments are not supported in RPC mode`.

**Confidence**: verified.

## B2

**Answer**

No initial-prompt character limit, slice, or truncation was found in Pi’s CLI/message path. The complete strings are accumulated and passed to `session.prompt()`. Tool-output truncation constants (50 KiB/2,000 lines) do **not** apply to the user prompt. Model/provider context-window limits can still reject or trigger later compaction; that is not CLI character truncation.

**Evidence**

- Grep command over installed dist:
  ```text
  rg -i 'prompt.*slice|slice.*prompt|prompt.*substring|substring.*prompt|initialMessage.*slice|MAX.*PROMPT|max.*prompt|truncate.*prompt' dist -g '*.js'
  ```
  produced no initial-prompt truncation hit (only unrelated UI/session slices in broader searches).
- `dist/main.js:50-57`: stdin appends every chunk to `data` and resolves the accumulated string.
- `dist/cli/initial-message.js:5-18`: concatenates complete source strings without slicing.
- `dist/core/agent-session.js:849-859`: creates user content with `text: expandedText` unchanged.
- `dist/core/tools/index.js` exports `DEFAULT_MAX_BYTES`/`DEFAULT_MAX_LINES`; these belong to tool output, not prompt ingestion.

**Confidence**: verified for Pi’s installed CLI; external OS/provider limits are outside this claim.

## B3

**Answer**

A multiline prompt supplied as **one positional argv token** stays one JavaScript string and becomes one user turn. Pi does not split it on newlines. It splits turns only at argv-token boundaries: after stdin/files are combined with the first message, each remaining `parsed.messages` entry is sent via a separate `session.prompt(message)` call.

Therefore the orchestrator’s shown launch — one whole prompt as one positional argv token — produces one turn, even with many lines.

**Evidence**

- `dist/cli/args.js:10-15,188-190` initializes `messages: []` and pushes each non-option argv token as one entry; there is no newline split.
- `dist/cli/initial-message.js:13-15` takes only `parsed.messages[0]` and shifts that element.
- `dist/modes/print-mode.js:94-98` calls `session.prompt(initialMessage)` once, then loops over remaining message elements.
- Windows runtime experiment passed this as one subprocess-list argument:
  ```text
  line1
  ```ts
  const x = "— åäö ${HOME}";
  ```
  @later !later /later #later
  ```
  Node reported exact `ARGV_MATCH True`.

**Confidence**: verified.

## B4

**Answer**

Special handling is position- and layer-specific:

- **Leading `@` on an entire argv token**: CLI file include. An `@` on a later line of the same token is ordinary text because parsing tests only `arg.startsWith("@")`.
- **Leading `/` on the final combined prompt string**: `session.prompt()` first tries an extension command, then skill (`/skill:name`) and prompt-template expansion. Unknown commands pass through as normal model text. A `/` on a later line does not trigger this because checks use `text.startsWith("/")` or more specific leading prefixes.
- **Leading `!`**: shell-command syntax exists only in the interactive editor submit path. Print/JSON calls `session.prompt()` directly, so a leading `!` there is ordinary model text.
- **`#`, backticks, and `${...}`**: no CLI interpretation in an ordinary positional prompt. Direct process spawning also bypasses shell interpolation. `${...}`, `$1`, `$@`, etc. are interpreted only inside a prompt template after a leading `/template` invocation; replacement operates on the template content.
- **Leading `-`** is also noteworthy: a separate argv token beginning `--` is parsed as a flag/extension flag, and a single-dash unknown option errors. Keep the prompt after `-p` (whose parser can consume the immediate next token, including a token beginning `---`) or otherwise avoid a separate prompt that begins with `-`.

Tokens merely contained on later lines (`@`, `/`, `!`, `#`, backticks, `${...}`) are passed through unchanged.

**Evidence**

- `dist/cli/args.js:109-114,172-190` shows `-p` immediate-message handling, `arg.startsWith("@")`, flag parsing, and ordinary message push.
- `dist/core/agent-session.js:799-821` checks `text.startsWith("/")`, then expands skill/template commands; lines 921-958 parse only leading slash forms.
- `dist/modes/interactive/interactive-mode.js:2237-2241` alone handles editor input beginning `!`/`!!` as bash.
- `dist/core/prompt-templates.js:29-71` documents and implements `$1`, `$@`, `$ARGUMENTS`, and `${...}` replacement on template content.
- Runtime argv experiment preserved code fences, `${HOME}`, `@later`, `!later`, `/later`, `#later`, em dash, and Swedish characters exactly.

**Confidence**: verified.

## B5

**Answer**

Ranked for a 5–20 KiB multiline prompt:

1. **Direct positional argv via `spawn`/`subprocess` argument array, one token** — best verified option for content arriving unchanged. Do not use a shell and do not use the Windows `.cmd` shim. The actual orchestrator approach (`node cli.js ... prompt` as an argv array) is correct. A Windows runtime test preserved a 19,200-character/21,200-byte Unicode multiline argument exactly by SHA-256. There is still an external Windows process-command-line ceiling; 20 KiB fit this test, while duplicating that payload into two argv arguments failed with WinError 206.
2. **Binary UTF-8 stdin pipe** — robust against shell quoting and command-line length, but Pi applies `.trim()`, so leading/trailing whitespace is not verbatim. It is a good fallback when boundary whitespace is irrelevant. Avoid a launcher’s text-mode newline conversion if LF exactness matters.
3. **`@file`** — robust for length and explicitly supported in `-p`/JSON, reads UTF-8, but Pi adds XML-like file tags, the absolute path, and newlines, so the model does not receive the file bytes verbatim as the user-message string.
4. **`--session-dir` seeding / config key** — UNVERIFIED as a supported initial-prompt transport; no such documented mechanism was found. `--session-dir` selects session storage, not prompt input.

If the prompt may approach the Windows command-line ceiling, prefer a binary UTF-8 stdin pipe and tolerate/compensate for boundary trimming, or use `@file` and tolerate the wrapper. Pi concatenates stdin and a positional instruction without adding a delimiter, so include the desired delimiter yourself in one of the inputs (while remembering stdin’s outer trim).

**Evidence**

- Runtime test result:
  ```text
  EXPECTED_CHARS 19200 EXPECTED_BYTES 21200
  EXPECTED_SHA 5deaf664...e418
  NODE {"chars":19200,"bytes":21200,"sha":"5deaf664...e418"}
  ```
- A larger two-copy argv experiment failed in `_winapi.CreateProcess` with `[WinError 206] The filename or extension is too long`, proving the relevant failure is before Pi.
- Binary UTF-8 stdin experiment reported `BINARY_UTF8_STDIN_MATCH True` for LF plus `— åäö`.
- `dist/main.js:52-57` sets stdin UTF-8 but trims it.
- `dist/cli/file-processor.js:39-50` reads UTF-8 and adds `<file name=...>` wrappers.
- `dist/cli/initial-message.js:18` uses `parts.join("")`.
- `pi --help` describes `--session-dir` only as “Directory for session storage and lookup.”

**Confidence**: verified, with the stated external Windows ceiling.

## B6

**Answer**

Inside Pi:

- argv already arrives as JavaScript strings in `process.argv`; Pi performs no encode/decode conversion on positional messages.
- piped stdin is explicitly decoded as **UTF-8** (`setEncoding("utf8")`).
- `@file`, system prompt files, skills, templates, and context files are read as UTF-8.

For a Windows launcher, use direct process creation with an argv array and Unicode strings. Do not route through `cmd.exe`, a `.cmd` shim, or manually construct a shell command line. For stdin, write UTF-8 **bytes** and close stdin. `PYTHONIOENCODING` is not relevant to Node/Pi itself; it only affects a Python launcher’s own stdio behavior. No Pi-specific `NODE_OPTIONS` encoding switch exists. `chcp 65001` matters only when a console/shell performs byte conversion; it is unnecessary for direct Unicode argv and does not fix incorrectly encoded pipe bytes.

The observed em-dash corruption would therefore occur before Pi’s `parseArgs`: likely shell/code-page conversion, a launcher encoding a command string/pipe incorrectly, or a file written in a non-UTF-8 encoding. Exact origin in the reported incident is **UNVERIFIED** without the launcher bytes/code path.

One additional verbatim caveat is newlines: a Python `subprocess.run(..., text=True, input=str)` experiment on Windows changed LF to CRLF before Node read stdin, while passing `input=utf8_bytes` preserved LF exactly. That conversion was in the launcher’s text mode, not Pi.

**Evidence**

- `dist/cli.js:17`: `main(process.argv.slice(2))`; `dist/cli/args.js` stores argv strings without decoding.
- `dist/main.js:52`: `process.stdin.setEncoding("utf8")`.
- `dist/cli/file-processor.js:46`: `readFile(absolutePath, "utf-8")`.
- `dist/core/resource-loader.js:14-17,31-42`: prompt/context files use `readFileSync(..., "utf-8")`.
- Direct Windows subprocess argv test preserved `— åäö` and all multiline punctuation exactly.
- Runtime stdin tests:
  - Python text input produced CRLF (`STDIN_MATCH False`).
  - Explicit UTF-8 bytes produced `BINARY_UTF8_STDIN_MATCH True`.
- `pi --help` and `docs/environment-variables.md` contain no argv/stdin encoding environment setting.

**Confidence**: verified for Pi and the local direct-spawn experiments; root cause of the previously observed corruption is UNVERIFIED.
