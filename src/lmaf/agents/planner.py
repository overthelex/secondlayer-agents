"""Planner agent -- produces and revises the legal research strategy.

Analogous to physics-intern's Planner: creates the initial strategy and
revises it when critiques demand. In secondlayer-core this maps to the
QueryPlanner + execution plan generation in ChatService.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .base import AgentResult, BaseAgent

if TYPE_CHECKING:
    from ..state.research_state import ConsultationState

PLANNER_SYSTEM_PROMPT = """\
Ти -- Стратег правової консультації (Legal Strategy Planner). Твоя роль -- \
розробити план дослідження правового питання клієнта.

На основі огляду правового ландшафту ти маєш:
1. Сформулювати підхід до вирішення (стратегію)
2. Визначити ключові правові питання, що потребують дослідження
3. Пріоритезувати питання за важливістю
4. Вказати, які джерела потрібно дослідити (судова практика, НПА, доктрина)
5. Оцінити ризики обраної стратегії
6. Визначити, які обчислення потрібні (строки, суми, пеня, інфляція)

Якщо ти отримуєш зауваження від Senior Legal Advisor -- переглянь стратегію \
відповідно до зауважень та поясни що змінилось.

Поверни структуровану стратегію у форматі JSON з полями:
- approach: str (опис підходу)
- legal_domains: list[str]
- key_questions: list[str]
- relevant_legislation: list[str]
- risk_factors: list[str]
"""


class PlannerAgent(BaseAgent):
    role = "planner"
    description = "Розробка та ревізія стратегії дослідження"

    async def run(
        self,
        state: ConsultationState,
        revision_critique: str = "",
        **kwargs: Any,
    ) -> AgentResult:
        from ..providers import call_llm

        context = self._build_state_context(state)
        user_msg = context
        if revision_critique:
            user_msg += f"\n\n# Зауваження до стратегії (потребує ревізії)\n{revision_critique}"

        response = await call_llm(
            self.config,
            system=PLANNER_SYSTEM_PROMPT,
            user=user_msg,
            model_key="planner",
            json_mode=True,
        )

        strategy_data = response.parsed_json or {}
        state.strategy.approach = strategy_data.get("approach", response.content)
        state.strategy.legal_domains = strategy_data.get("legal_domains", [])
        state.strategy.key_questions = strategy_data.get("key_questions", [])
        state.strategy.relevant_legislation = strategy_data.get("relevant_legislation", [])
        state.strategy.risk_factors = strategy_data.get("risk_factors", [])

        if revision_critique:
            state.strategy.revision_history.append(
                f"Iteration {state.iteration}: revised due to critique"
            )

        for q in state.strategy.key_questions:
            if not any(rq.question == q for rq in state.questions):
                state.add_question(q, iteration=state.iteration)

        return AgentResult(
            success=True,
            agent=self.role,
            summary=f"Стратегія: {len(state.strategy.key_questions)} питань, "
            f"{len(state.strategy.relevant_legislation)} НПА",
            tokens_used=response.tokens_used,
        )
