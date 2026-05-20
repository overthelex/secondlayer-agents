"""Base agent -- all agents inherit from this.

Each agent call starts from a fresh context (no conversation history).
The agent reads ConsultationState, performs its task, and mutates state via tools.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..core.config import Config
    from ..core.workspace import WorkspaceManager
    from ..state.research_state import ConsultationState


@dataclass
class AgentResult:
    success: bool
    agent: str
    summary: str = ""
    tokens_used: int = 0
    duration_sec: float = 0.0
    error: str = ""


class BaseAgent(ABC):
    """Base class for all LegalIntern agents."""

    role: str = "base"
    description: str = ""

    def __init__(self, config: Config, workspace: WorkspaceManager) -> None:
        self.config = config
        self.workspace = workspace

    @abstractmethod
    async def run(self, state: ConsultationState, **kwargs: Any) -> AgentResult:
        """Execute the agent's task and mutate state."""
        ...

    def _build_state_context(self, state: ConsultationState) -> str:
        """Render current state as a text summary for the LLM prompt."""
        parts = []
        parts.append(f"# Запит клієнта\n{state.client_question}")

        if state.jurisdiction:
            parts.append(f"**Юрисдикція**: {state.jurisdiction}")

        if state.survey_summary:
            parts.append(f"# Огляд правового ландшафту\n{state.survey_summary}")

        if state.strategy.approach:
            parts.append(f"# Стратегія\n{state.strategy.approach}")
            if state.strategy.key_questions:
                parts.append("**Ключові питання**: " + "; ".join(state.strategy.key_questions))
            if state.strategy.relevant_legislation:
                parts.append(
                    "**Релевантне законодавство**: "
                    + "; ".join(state.strategy.relevant_legislation)
                )

        if state.hypotheses:
            parts.append("# Правові позиції")
            for h in state.hypotheses:
                parts.append(f"- {h.short()}")

        if state.evidence:
            parts.append("# Зібрані докази")
            for ev in state.evidence:
                parts.append(f"- {ev.short()}")

        if state.open_questions():
            parts.append("# Відкриті питання")
            for q in state.open_questions():
                parts.append(f"- [{q.id}] {q.question}")

        if state.active_critiques():
            parts.append("# Активні зауваження")
            for c in state.active_critiques():
                parts.append(f"- [{c.id}] ({c.severity}) {c.summary}")

        return "\n\n".join(parts)
