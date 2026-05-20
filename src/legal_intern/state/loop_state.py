"""Loop state -- tracks iteration progress and dispatch history."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class DispatchRecord:
    iteration: int
    agent: str
    task_summary: str
    success: bool
    duration_sec: float = 0.0
    tokens_used: int = 0
    error: str = ""


@dataclass
class LoopState:
    """Tracks the main loop's progress (separate from ConsultationState)."""

    dispatch_history: list[DispatchRecord] = field(default_factory=list)
    consecutive_failures: int = 0
    survey_done: bool = False
    plan_done: bool = False
    last_critic_iteration: int = -1
    formatter_attempts: int = 0

    def record(
        self,
        iteration: int,
        agent: str,
        task_summary: str,
        success: bool,
        duration_sec: float = 0.0,
        tokens_used: int = 0,
        error: str = "",
    ) -> None:
        self.dispatch_history.append(
            DispatchRecord(
                iteration=iteration,
                agent=agent,
                task_summary=task_summary,
                success=success,
                duration_sec=duration_sec,
                tokens_used=tokens_used,
                error=error,
            )
        )
        if success:
            self.consecutive_failures = 0
        else:
            self.consecutive_failures += 1
