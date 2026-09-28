# agentic-coder-teams-mcp (win-agent-teams)

> **This project is no longer maintained.** It is superseded by
> **[AgentTeamForge](https://github.com/PRFactory-app/agent-team-forge)**, a local
> daemon that owns agent jobs durably and exposes the same workflows over MCP.
>
> - Migration guide (tool-by-tool mapping):
>   [docs/migrating-from-win-agent-teams.md](https://github.com/PRFactory-app/agent-team-forge/blob/main/docs/migrating-from-win-agent-teams.md)
> - The agent skills that used to live here (`agent-orchestration`,
>   `external-member-invite`, `external-member-join`) now live in AgentTeamForge's
>   [`.claude/skills`](https://github.com/PRFactory-app/agent-team-forge/tree/main/.claude/skills)
>   (added in [PR #8](https://github.com/PRFactory-app/agent-team-forge/pull/8)).
>
> No new features or fixes will be made here.

## What it was

A minimal Python MCP server for spawning and communicating with Claude Code,
Codex and Pi agents on Windows or Linux: fire-and-forget `spawn_agent`,
`follow_up_agent` to resume a worker's native session, 1:1 messaging with
`send_message`/`read_messages`, state-marker watching for idle wake, restart
recovery via `session_info`/`resume_session`, and external members
(`create_join_ticket`/`join_team`) for manually started interactive sessions.

Existing installs keep working as-is; see [INSTALL.md](INSTALL.md) and
[docs/](docs/) for the historical documentation.

## License

[MIT](./LICENSE)
