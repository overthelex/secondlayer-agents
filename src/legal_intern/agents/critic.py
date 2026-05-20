"""Critic agent -- senior legal advisor that audits strategy and coherence.

Analogous to physics-intern's Deep Critic (periodic audit).
In secondlayer-core this is a new capability -- currently there's no
meta-level strategy review, only per-evidence validation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .base import AgentResult, BaseAgent

if TYPE_CHECKING:
    from ..state.research_state import ConsultationState

CRITIC_SYSTEM_PROMPT = """\
Ти -- Старший правовий радник (Senior Legal Advisor). Твоя роль -- періодично \
перевіряти загальну стратегію та якість дослідження.

Ти маєш оцінити:
1. **Стратегія**: чи правильно обрано підхід? Чи немає кращої правової позиції?
2. **Повнота**: чи всі ключові аспекти розглянуто? Чи немає прогалин?
3. **Узгодженість**: чи не суперечать докази один одному? Чи послідовна аргументація?
4. **Посилання**: чи достатньо підкріплені позиції судовою практикою та НПА?
5. **Ризики**: чи враховані контр-аргументи опонента?

Типи зауважень:
- "strategy": стратегія потребує ревізії (маршрутизується до Planner)
- "reasoning": помилка в міркуваннях (маршрутизується до Orchestrator)
- "completeness": прогалина в дослідженні (генерує нові RQ)
- "citation": проблема з посиланнями (маршрутизується до Researcher)

Поверни JSON:
{
  "overall_assessment": "strong" | "adequate" | "weak",
  "critiques": [
    {
      "type": "strategy" | "reasoning" | "completeness" | "citation",
      "severity": "high" | "medium" | "low",
      "summary": "коротке зауваження",
      "details": "повне обгрунтування",
      "target_hypothesis": "H-NNN" | null
    }
  ],
  "can_proceed_to_answer": true | false
}
"""


class CriticAgent(BaseAgent):
    role = "critic"
    description = "Аудит стратегії, повноти та узгодженості дослідження"

    async def run(self, state: ConsultationState, **kwargs: Any) -> AgentResult:
        from ..providers import call_llm

        context = self._build_state_context(state)

        response = await call_llm(
            self.config,
            system=CRITIC_SYSTEM_PROMPT,
            user=context,
            model_key="critic",
        )

        result = response.parsed_json or {}
        critiques = result.get("critiques", [])

        for c in critiques:
            state.add_critique(
                type=c.get("type", "reasoning"),
                severity=c.get("severity", "medium"),
                summary=c.get("summary", ""),
                details=c.get("details", ""),
                target_hypothesis=c.get("target_hypothesis", ""),
                iteration=state.iteration,
            )

        can_proceed = result.get("can_proceed_to_answer", False)
        assessment = result.get("overall_assessment", "unknown")

        return AgentResult(
            success=True,
            agent=self.role,
            summary=f"Оцінка: {assessment}, зауважень: {len(critiques)}, "
            f"можна завершувати: {can_proceed}",
            tokens_used=response.tokens_used,
        )
