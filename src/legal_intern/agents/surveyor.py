"""Surveyor agent -- maps the legal landscape before the main loop.

Analogous to physics-intern's Surveyor: identifies relevant areas of law,
applicable legislation, court practice trends, and potential pitfalls.
Runs once at the start.

In secondlayer-core this corresponds to: IntentClassifier + QueryPlanner
(but here it's a full LLM agent that produces a structured survey).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .base import AgentResult, BaseAgent

if TYPE_CHECKING:
    from ..state.research_state import ConsultationState

SURVEYOR_SYSTEM_PROMPT = """\
Ти -- Правовий оглядач (Legal Surveyor). Твоя роль -- провести початковий аналіз \
правового питання клієнта та скласти карту релевантного правового ландшафту.

Ти отримуєш запит клієнта і маєш:
1. Визначити юрисдикцію (цивільна, господарська, адміністративна, кримінальна)
2. Ідентифікувати галузі права, що застосовуються
3. Перелічити ключові нормативні акти та статті
4. Визначити основні правові позиції ВС/КС, що стосуються питання
5. Зазначити потенційні ризики та слабкі сторони
6. Окреслити темпоральні аспекти (строки давності, процесуальні строки)

Поверни структурований огляд українською мовою у форматі Markdown.
НЕ давай відповідь на питання клієнта -- лише картуй ландшафт.
"""


class SurveyorAgent(BaseAgent):
    role = "surveyor"
    description = "Початковий огляд правового ландшафту"

    async def run(self, state: ConsultationState, **kwargs: Any) -> AgentResult:
        from ..providers import call_llm

        prompt = f"{SURVEYOR_SYSTEM_PROMPT}\n\n# Запит клієнта\n{state.client_question}"

        response = await call_llm(
            self.config,
            system=SURVEYOR_SYSTEM_PROMPT,
            user=state.client_question,
            model_key="surveyor",
        )

        state.survey_summary = response.content
        if response.jurisdiction:
            state.jurisdiction = response.jurisdiction

        return AgentResult(
            success=True,
            agent=self.role,
            summary=f"Огляд завершено: {len(response.content)} символів",
            tokens_used=response.tokens_used,
        )
