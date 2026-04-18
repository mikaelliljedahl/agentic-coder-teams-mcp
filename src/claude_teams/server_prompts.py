"""Templated MCP prompts for common team-lead operations.

These prompts are tagged ``team`` and become visible after ``team_create``
or ``team_attach`` via FastMCP's progressive disclosure mechanism.  Each
prompt returns an assistant-prefilled conversation to skip preamble and
drive the model straight into action.
"""

from fastmcp import FastMCP
from fastmcp.prompts import Message

from claude_teams.server_schema import AgentName, Hint, MessageContent, TeamName


def register_prompts(mcp: FastMCP) -> None:
    """Register team-lead prompt templates on *mcp*."""

    @mcp.prompt(
        tags={"team"},
        description="Check a teammate's health, current task, and blockers.",
    )
    def status_check(team: TeamName, teammate: AgentName) -> list[Message]:
        return [
            Message(
                f"Check on teammate '{teammate}' in team '{team}'.\n"
                "1. Run check_teammate to get process health\n"
                "2. Run task_list and filter for their assignments\n"
                "Report: health status, current task + progress, any blockers."
            ),
            Message("Checking now.", role="assistant"),
        ]

    @mcp.prompt(
        tags={"team"},
        description="Run health check across all teammates and flag issues.",
    )
    def health_sweep(team: TeamName) -> list[Message]:
        return [
            Message(
                f"Sweep team '{team}' health.\n"
                "1. Run health_check for all teammates\n"
                "2. For any unhealthy or stalled, run check_teammate for details\n"
                "Report: table of teammate, status, action needed."
            ),
            Message("Running sweep.", role="assistant"),
        ]

    @mcp.prompt(
        tags={"team"},
        description="Hand off completed work from one teammate to another.",
    )
    def task_handoff(
        team: TeamName,
        from_teammate: AgentName,
        to_teammate: AgentName,
        context: MessageContent = "",
    ) -> list[Message]:
        ctx_line = f"\nAdditional context: {context}" if context else ""
        return [
            Message(
                f"Hand off work in team '{team}' from '{from_teammate}' "
                f"to '{to_teammate}'.{ctx_line}\n"
                "1. Get completed task results from the source via task_get\n"
                "2. Send relevant results to the target via send_message\n"
                "3. Create or update task assignment for the target\n"
                "Confirm when handoff is complete."
            ),
            Message("Starting handoff.", role="assistant"),
        ]

    @mcp.prompt(
        tags={"team"},
        description="Collect all task results and prepare a team summary.",
    )
    def wrap_up(team: TeamName) -> list[Message]:
        return [
            Message(
                f"Wrap up team '{team}'.\n"
                "1. Run task_list — identify done, in-progress, blocked\n"
                "2. For each done task, collect result via task_get\n"
                "3. For in-progress or blocked, check teammate status\n"
                "Produce: summary of deliverables, incomplete work, "
                "recommended next steps."
            ),
            Message("Collecting results.", role="assistant"),
        ]

    @mcp.prompt(
        tags={"team"},
        description="Send a targeted message to unblock a stuck teammate.",
    )
    def unblock_teammate(
        team: TeamName, teammate: AgentName, hint: Hint = ""
    ) -> list[Message]:
        hint_line = f"\nHint from lead: {hint}" if hint else ""
        return [
            Message(
                f"Unblock teammate '{teammate}' in team '{team}'.{hint_line}\n"
                "1. Run check_teammate to understand current state\n"
                "2. Check their task for blockers via task_get\n"
                "3. Send a targeted message with guidance via send_message\n"
                "4. Update task status if needed"
            ),
            Message("Investigating blocker.", role="assistant"),
        ]
