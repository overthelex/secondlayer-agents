"""Orchestrator agent -- dispatches research tasks to researcher or analyst.

Analogous to physics-intern's Orchestrator: decides what to investigate next,
dispatches to Researcher (case law / legislation lookup) or Analyst (calculations),
and formulates working hypotheses from the evidence.

In secondlayer-core this maps to the ChatService agentic loop that selects
tool calls and routes between search, legislation, and analysis tools.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .base import AgentResult, BaseAgent

if TYPE_CHECKING:
    from ..state.research_state import ConsultationState

ORCHESTRATOR_SYSTEM_PROMPT = """\
Ти -- Оркестратор правового дослідження (Legal Research Orchestrator). Твоя роль -- \
координувати хід дослідження, визначаючи наступні кроки.

На основі поточного стану (стратегія, зібрані докази, відкриті питання, зауваження) \
ти маєш:

1. Обрати наступне питання для дослідження (або відповісти на зауваження)
2. Визначити тип завдання:
   - "research": пошук судової практики, аналіз законодавства, доктрини
   - "analysis": обчислення (строки, суми, пеня, інфляція, 3% річних)
3. Сформулювати конкретне завдання для агента-виконавця
4. Оновити або сформулювати правову позицію (гіпотезу) на основі нових доказів

Поверни JSON:
{
  "task_type": "research" | "analysis",
  "question_id": "RQ-NNN" (якщо прив'язано до питання),
  "task_description": "конкретне завдання",
  "hypothesis_update": {  // optional
    "id": "H-NNN" або null для нової,
    "statement": "формулювання позиції"
  },
  "reasoning": "чому саме це завдання зараз"
}
"""


class OrchestratorAgent(BaseAgent):
    role = "orchestrator"
    description = "Координація ходу дослідження та формулювання гіпотез"

    async def run(self, state: ConsultationState, **kwargs: Any) -> AgentResult:
        from ..providers import call_llm

        context = self._build_state_context(state)

        response = await call_llm(
            self.config,
            system=ORCHESTRATOR_SYSTEM_PROMPT,
            user=context,
            model_key="orchestrator",
            json_mode=True,
        )

        dispatch = response.parsed_json or {}

        # Update hypothesis if suggested
        hyp_update = dispatch.get("hypothesis_update")
        if hyp_update and hyp_update.get("statement"):
            hyp_id = hyp_update.get("id")
            if hyp_id:
                for h in state.hypotheses:
                    if h.id == hyp_id:
                        h.statement = hyp_update["statement"]
                        break
            else:
                state.add_hypothesis(hyp_update["statement"], iteration=state.iteration)

        return AgentResult(
            success=True,
            agent=self.role,
            summary=f"Dispatch: {dispatch.get('task_type', '?')} -- "
            f"{dispatch.get('task_description', '')[:60]}",
            tokens_used=response.tokens_used,
        )
