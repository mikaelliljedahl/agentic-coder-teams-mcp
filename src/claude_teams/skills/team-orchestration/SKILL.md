---
name: team-orchestration
description: Lead a team of agentic coding agents — create teams, spawn teammates across any backend, assign tasks, coordinate work, and monitor health.
---

# Team Orchestration

You are leading a team of agentic coding agents through the `claude-teams` MCP server. This skill covers the full lifecycle: setup, delegation, coordination, and teardown.

## Quick start

1. **Create a team** with `team_create` — give it a name and your role.
2. **Spawn teammates** with `spawn_teammate` — pick the backend (`claude-code`, `gemini`, `codex`, etc.) and model tier (`fast`, `balanced`, `powerful`).
3. **Assign work** with `task_create` — describe what each teammate should do.
4. **Coordinate** with `send_message` — direct teammates, relay context, unblock work.
5. **Monitor** with `health_check` and `check_teammate` — catch stalls early.
6. **Wrap up** with `team_delete` when the work is done.

## Team lifecycle

### Creating and joining teams

- `team_create(name, lead_name)` — start a new team. You become the lead.
- `team_attach(name, principal_name, capability)` — rejoin an existing team. Use the capability token from team creation.
- `read_config(name)` — inspect team membership, backend assignments, and state.
- `list_backends()` — see which agentic coders are installed and available for spawning.

### Spawning teammates

- `spawn_teammate(team, name, backend, model, prompt)` — launch a new agent on any installed backend.
  - `backend` selects the agentic coder (`claude-code`, `gemini`, `codex`, `aider`, `goose`, etc.).
  - `model` accepts generic tiers (`fast`, `balanced`, `powerful`) or backend-specific model names.
  - The prompt is the teammate's initial instruction — be specific about scope and deliverables.
- Teammates run as independent processes. They have their own context and tools.

### Task management

- `task_create(team, title, description, assignee)` — create a tracked task.
- `task_update(team, task_id, fields)` — update task state. `fields` is a `TaskUpdateFields` payload, e.g. `{"status": "in_progress"}` or `{"owner": "worker-1"}`. Status values: `pending`, `in_progress`, `completed`, `deleted`.
- `task_list(team)` — see all tasks and their statuses.
- `task_get(team, task_id)` — get full task details including results.

### Communication

- `send_message(team, to, message)` — send a message to a specific teammate.
- `read_inbox(team)` — check messages sent to you.
- `poll_inbox(team)` — teammates use this to check for new messages.

### Monitoring and health

- `health_check(team)` — get health status for all teammates in the team.
- `check_teammate(team, name)` — check a specific teammate's status and recent output.
- `force_kill_teammate(team, name)` — terminate a stalled or misbehaving teammate.
- `process_shutdown_approved(team, name)` — approve a teammate's graceful shutdown request.

## Coordination patterns

### Parallel fan-out

Spawn multiple teammates on independent tasks, then collect results:

1. Create tasks for each work item.
2. Spawn a teammate per task with a focused prompt.
3. Poll `task_list` periodically to track progress.
4. Use `send_message` to relay context between teammates if their work overlaps.

### Sequential pipeline

Chain teammates where each depends on the previous:

1. Spawn teammate A with the initial task.
2. When A completes, read the result from `task_get`.
3. Spawn teammate B with A's output as context in the prompt.

### Mixed-backend teams

Different backends excel at different tasks. Use `list_backends` to see what is available, then assign based on strengths:

- High-reasoning tasks → `powerful` tier on capable backends.
- Fast iteration / linting → `fast` tier.
- Specialized tools → pick the backend whose upstream tool fits best.

## Troubleshooting

- **Teammate not responding**: Run `check_teammate` first. If stalled, `force_kill_teammate` and re-spawn.
- **Task stuck as `in_progress`**: Message the assignee with `send_message` to request a status update.
- **Wrong backend**: You cannot change a running teammate's backend. Kill and re-spawn on the correct one.
- **Team state unclear**: `read_config` shows the full membership and `task_list` shows all work items.
