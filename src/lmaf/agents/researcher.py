"""Researcher agent -- finds and analyzes case law, legislation, doctrine.

Analogous to physics-intern's Researcher (analytical reasoning).
In secondlayer-core this maps to: tool calls for search_court_decisions,
get_legislation, search_echr, plus the evidence extraction logic.

This agent has access to SecondLayer MCP tools via the tool bridge.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .base import AgentResult, BaseAgent

if TYPE_CHECKING:
    from ..state.research_state import ConsultationState

RESEARCHER_SYSTEM_PROMPT = """\
Ти -- Правовий дослідник (Legal Researcher). Твоя роль -- знаходити та аналізувати \
правові джерела для відповіді на конкретне дослідницьке питання.

Ти маєш доступ до інструментів:
- search_court_decisions: пошук рішень суду в ЄДРСР (100М+ документів)
- get_legislation: отримання тексту НПА з zakon.rada.gov.ua
- search_echr: пошук рішень ЄСПЛ
- search_supreme_court: пошук правових позицій ВС

Для кожного знайденого джерела ти маєш:
1. Оцінити його релевантність до питання
2. Витягнути ключову правову позицію або норму
3. Сформулювати як це підтримує або спростовує робочу гіпотезу
4. Надати точне посилання (номер справи, стаття закону)

Поверни JSON з полями:
{
  "findings": [
    {
      "type": "case_law" | "legislation" | "doctrine" | "echr",
      "source": "edrsr" | "rada" | "echr",
      "citation": "точне посилання",
      "summary": "ключовий висновок",
      "relevance": "як стосується питання",
      "supports_hypothesis": "H-NNN" | null,
      "opposes_hypothesis": "H-NNN" | null,
      "confidence": "high" | "medium" | "low"
    }
  ],
  "conclusion": "загальний висновок з дослідження"
}
"""


class ResearcherAgent(BaseAgent):
    role = "researcher"
    description = "Пошук та аналіз судової практики, законодавства, доктрини"

    async def run(
        self,
        state: ConsultationState,
        task_description: str = "",
        question_id: str = "",
        **kwargs: Any,
    ) -> AgentResult:
        from ..providers import call_llm

        context = self._build_state_context(state)
        user_msg = f"{context}\n\n# Поточне завдання\n{task_description}"
        if question_id:
            user_msg += f"\n(Питання: {question_id})"

        response = await call_llm(
            self.config,
            system=RESEARCHER_SYSTEM_PROMPT,
            user=user_msg,
            model_key="researcher",
            tools=self._get_tools(),
            json_mode=True,
        )

        findings = (response.parsed_json or {}).get("findings", [])
        for f in findings:
            ev = state.add_evidence(
                type=f.get("type", "case_law"),
                source=f.get("source", "edrsr"),
                citation=f.get("citation", ""),
                summary=f.get("summary", ""),
                relevance=f.get("relevance", ""),
                confidence=f.get("confidence", "medium"),
                iteration=state.iteration,
            )
            # Link evidence to hypotheses
            if f.get("supports_hypothesis"):
                for h in state.hypotheses:
                    if h.id == f["supports_hypothesis"]:
                        h.supporting_evidence.append(ev.id)
            if f.get("opposes_hypothesis"):
                for h in state.hypotheses:
                    if h.id == f["opposes_hypothesis"]:
                        h.opposing_evidence.append(ev.id)

        # Resolve research question if assigned
        if question_id:
            for q in state.questions:
                if q.id == question_id:
                    q.answer = (response.parsed_json or {}).get("conclusion", "")
                    q.evidence_ids = [ev.id for ev in state.evidence[-len(findings) :]]
                    from ..state.research_state import RQStatus

                    q.status = RQStatus.RESOLVED

        return AgentResult(
            success=True,
            agent=self.role,
            summary=f"Знайдено {len(findings)} джерел",
            tokens_used=response.tokens_used,
        )

    def _get_tools(self) -> list[dict]:
        """Return SecondLayer MCP tool definitions available to this agent."""
        return [
            {
                "name": "search_court_decisions",
                "description": "Семантичний пошук рішень суду в ЄДРСР",
                "parameters": {
                    "query": {"type": "string"},
                    "jurisdiction": {"type": "string", "enum": ["civil", "commercial"]},
                    "limit": {"type": "integer", "default": 10},
                },
            },
            {
                "name": "get_legislation",
                "description": "Отримати текст НПА за назвою або номером",
                "parameters": {
                    "query": {"type": "string"},
                    "article": {"type": "string"},
                },
            },
            {
                "name": "search_supreme_court",
                "description": "Пошук правових позицій Верховного Суду",
                "parameters": {
                    "query": {"type": "string"},
                    "category": {"type": "string"},
                },
            },
        ]
