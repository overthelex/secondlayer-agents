"""LegalIntern main loop engine.

Nine-agent pipeline for complex legal consultations:

1. Surveyor   -- maps the legal landscape (runs once)
2. Planner    -- produces/revises research strategy
3. Orchestrator -- dispatches tasks to researcher or analyst
4. Researcher -- finds case law, legislation, doctrine (via SecondLayer MCP)
5. Analyst    -- computes deadlines, penalties, procedural checks
6. Reviewer   -- adversarial verification of evidence (auto-triggered)
7. Critic     -- periodic strategy audit (every N iterations)
8. Adjudicator -- resolves inter-agent disagreements
9. Formatter  -- produces final consultation document

No agent carries conversation history. All state lives in ConsultationState.
The workspace is git-versioned for full reproducibility.
"""

from __future__ import annotations

import time
from pathlib import Path

from rich.console import Console
from rich.panel import Panel

from .agents import (
    AdjudicatorAgent,
    AnalystAgent,
    CriticAgent,
    FormatterAgent,
    OrchestratorAgent,
    PlannerAgent,
    ResearcherAgent,
    ReviewerAgent,
    SurveyorAgent,
)
from .core.config import Config
from .core.workspace import WorkspaceManager
from .state.loop_state import LoopState
from .state.research_state import ConsultationState, CritiqueStatus

console = Console()


class LegalIntern:
    """Main loop for the LegalIntern consultation system."""

    def __init__(self, question: str, config: Config | None = None) -> None:
        self.config = config or Config.from_env()
        self.workspace = WorkspaceManager(self.config)
        self.workspace.init(question[:60])

        self.state = ConsultationState()
        self.state.client_question = question.strip()
        self.state.title = question[:80]

        self._loop = LoopState()

        # Initialize all 9 agents
        self.surveyor = SurveyorAgent(self.config, self.workspace)
        self.planner = PlannerAgent(self.config, self.workspace)
        self.orchestrator = OrchestratorAgent(self.config, self.workspace)
        self.researcher = ResearcherAgent(self.config, self.workspace)
        self.analyst = AnalystAgent(self.config, self.workspace)
        self.reviewer = ReviewerAgent(self.config, self.workspace)
        self.critic = CriticAgent(self.config, self.workspace)
        self.adjudicator = AdjudicatorAgent(self.config, self.workspace)
        self.formatter = FormatterAgent(self.config, self.workspace)

    async def run(self) -> str:
        """Execute the full consultation pipeline. Returns the final answer."""
        console.print(Panel(f"[bold]LegalIntern[/bold]\n{self.state.client_question[:200]}"))

        # Phase 1: Survey
        console.print("[dim]Phase 1: Surveyor[/dim]")
        t0 = time.time()
        result = await self.surveyor.run(self.state)
        self._loop.record(0, "surveyor", result.summary, result.success, time.time() - t0)
        self._loop.survey_done = True
        self.workspace.snapshot("survey complete")
        console.print(f"  [green]{result.summary}[/green]")

        # Phase 2: Plan
        console.print("[dim]Phase 2: Planner[/dim]")
        t0 = time.time()
        result = await self.planner.run(self.state)
        self._loop.record(0, "planner", result.summary, result.success, time.time() - t0)
        self._loop.plan_done = True
        self.workspace.snapshot("plan complete")
        console.print(f"  [green]{result.summary}[/green]")

        # Phase 3: Main research loop
        for iteration in range(1, self.config.max_iterations + 1):
            self.state.iteration = iteration
            console.print(f"\n[bold cyan]--- Iteration {iteration} ---[/bold cyan]")

            # Orchestrator decides what to do next
            t0 = time.time()
            orch_result = await self.orchestrator.run(self.state)
            self._loop.record(
                iteration, "orchestrator", orch_result.summary, orch_result.success, time.time() - t0
            )

            if not orch_result.success:
                if self._loop.consecutive_failures >= self.config.max_consecutive_failures:
                    console.print("[red]Too many consecutive failures, stopping.[/red]")
                    break
                continue

            # Parse orchestrator dispatch
            dispatch = getattr(orch_result, "_dispatch", None) or {}
            task_type = "research"  # default
            task_desc = orch_result.summary

            # Execute researcher or analyst
            if task_type == "analysis":
                console.print(f"  [yellow]Analyst:[/yellow] {task_desc[:80]}")
                t0 = time.time()
                agent_result = await self.analyst.run(self.state, task_description=task_desc)
            else:
                console.print(f"  [yellow]Researcher:[/yellow] {task_desc[:80]}")
                t0 = time.time()
                agent_result = await self.researcher.run(self.state, task_description=task_desc)

            self._loop.record(
                iteration,
                agent_result.agent,
                agent_result.summary,
                agent_result.success,
                time.time() - t0,
            )
            console.print(f"  [green]{agent_result.summary}[/green]")

            # Auto-trigger reviewer
            console.print("  [dim]Reviewer...[/dim]")
            t0 = time.time()
            review_result = await self.reviewer.run(self.state)
            self._loop.record(
                iteration, "reviewer", review_result.summary, review_result.success, time.time() - t0
            )
            console.print(f"  [green]{review_result.summary}[/green]")

            # Periodic critic audit
            if iteration % self.config.critic_every_n == 0:
                console.print("  [dim]Senior Legal Advisor...[/dim]")
                t0 = time.time()
                critic_result = await self.critic.run(self.state)
                self._loop.record(
                    iteration, "critic", critic_result.summary, critic_result.success, time.time() - t0
                )
                self._loop.last_critic_iteration = iteration
                console.print(f"  [magenta]{critic_result.summary}[/magenta]")

                # Route critiques
                await self._route_critiques()

                # Check if critic says we can proceed
                if "можна завершувати: True" in critic_result.summary:
                    console.print("[green]Critic approved -- proceeding to answer.[/green]")
                    break

            # Check termination: no open questions and no active critiques
            if not self.state.open_questions() and not self.state.active_critiques():
                console.print("[green]All questions resolved -- proceeding to answer.[/green]")
                break

            self.workspace.snapshot(f"iteration {iteration}")

        # Phase 4: Format final answer
        console.print("\n[bold]Phase 4: Formatting consultation[/bold]")
        t0 = time.time()
        fmt_result = await self.formatter.run(self.state)
        self._loop.record(
            self.state.iteration, "formatter", fmt_result.summary, fmt_result.success, time.time() - t0
        )
        self.workspace.snapshot("answer formatted")

        # Write answer file
        answer_path = self.workspace.root / "CONSULTATION.md"
        answer_path.write_text(self.state.answer, encoding="utf-8")
        self.workspace.snapshot("final")

        console.print(Panel(f"[bold green]Done![/bold green] {answer_path}"))
        self._print_summary()

        return self.state.answer

    async def _route_critiques(self) -> None:
        """Route active critiques to appropriate agents."""
        for critique in self.state.active_critiques():
            if critique.type == "strategy":
                result = await self.planner.run(
                    self.state, revision_critique=critique.details
                )
                critique.status = CritiqueStatus.RESOLVED
            elif critique.type == "completeness":
                # Generate new research questions
                state_questions_before = len(self.state.questions)
                result = await self.orchestrator.run(self.state)
                if len(self.state.questions) > state_questions_before:
                    critique.status = CritiqueStatus.RESOLVED
            elif critique.target_hypothesis:
                result = await self.adjudicator.run(
                    self.state,
                    hypothesis_id=critique.target_hypothesis,
                    critique_id=critique.id,
                )

    def _print_summary(self) -> None:
        console.print(f"\n[dim]Iterations: {self.state.iteration}[/dim]")
        console.print(f"[dim]Hypotheses: {len(self.state.hypotheses)} "
                       f"(established: {len(self.state.established_hypotheses())})[/dim]")
        console.print(f"[dim]Evidence: {len(self.state.evidence)} "
                       f"(refuted: {sum(1 for e in self.state.evidence if e.refuted)})[/dim]")
        console.print(f"[dim]Total tokens: {self.state.total_tokens:,}[/dim]")
