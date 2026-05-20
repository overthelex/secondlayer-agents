"""Consultation state -- structured representation of legal analysis.

Mirrors physics-intern's ResearchState: agents mutate state via tools,
Markdown is rendered from it for git snapshots. No agent carries
conversation history; all context lives here.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any


class HypothesisStatus(StrEnum):
    WORKING = "working"
    ESTABLISHED = "established"
    REFUTED = "refuted"
    ABANDONED = "abandoned"


class Verdict(StrEnum):
    VERIFIED = "verified"
    REFUTED = "refuted"
    INCONCLUSIVE = "inconclusive"


class Severity(StrEnum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class CritiqueStatus(StrEnum):
    ACTIVE = "active"
    RESOLVED = "resolved"
    WITHDRAWN = "withdrawn"


class RQStatus(StrEnum):
    OPEN = "open"
    RESOLVED = "resolved"
    ABANDONED = "abandoned"


@dataclass
class LegalEvidence:
    """Evidence produced by a researcher or analyst agent."""

    id: str = ""
    type: str = ""  # "case_law", "legislation", "doctrine", "computation"
    source: str = ""  # "edrsr", "rada", "echr", "manual"
    summary: str = ""
    full_text: str = ""
    citation: str = ""  # e.g. "Справа №757/12345/22" or "ст. 625 ЦК України"
    relevance: str = ""
    confidence: str = ""  # "high", "medium", "low"
    iteration: int | None = None
    refuted: bool = False

    def short(self) -> str:
        return f"[{self.id}] {self.citation}: {self.summary[:80]}"


@dataclass
class ReviewResult:
    """Review produced by the legal reviewer agent."""

    verdict: str = ""
    summary: str = ""
    details: str = ""
    iteration: int | None = None


@dataclass
class LegalHypothesis:
    """A legal position or argument being developed."""

    id: str = ""
    statement: str = ""
    status: HypothesisStatus = HypothesisStatus.WORKING
    supporting_evidence: list[str] = field(default_factory=list)  # evidence IDs
    opposing_evidence: list[str] = field(default_factory=list)
    reviews: list[ReviewResult] = field(default_factory=list)
    iteration_created: int | None = None
    iteration_resolved: int | None = None

    def short(self) -> str:
        return f"[{self.id}] ({self.status}) {self.statement[:80]}"


@dataclass
class Critique:
    """Critique from the senior legal advisor."""

    id: str = ""
    type: str = ""  # "strategy", "reasoning", "completeness", "citation"
    severity: Severity = Severity.MEDIUM
    status: CritiqueStatus = CritiqueStatus.ACTIVE
    summary: str = ""
    details: str = ""
    target_hypothesis: str = ""
    iteration: int | None = None


@dataclass
class ResearchQuestion:
    """An open question that needs investigation."""

    id: str = ""
    question: str = ""
    status: RQStatus = RQStatus.OPEN
    assigned_to: str = ""  # agent role
    answer: str = ""
    evidence_ids: list[str] = field(default_factory=list)
    iteration_created: int | None = None


@dataclass
class LegalStrategy:
    """The consultation strategy produced by the planner."""

    approach: str = ""
    legal_domains: list[str] = field(default_factory=list)
    key_questions: list[str] = field(default_factory=list)
    relevant_legislation: list[str] = field(default_factory=list)
    risk_factors: list[str] = field(default_factory=list)
    revision_history: list[str] = field(default_factory=list)


@dataclass
class ConsultationState:
    """Central state object for a legal consultation.

    All agents read from and write to this object.
    No agent carries its own conversation history.
    """

    # Problem
    client_question: str = ""
    jurisdiction: str = ""  # "civil", "commercial", "administrative", "criminal"
    title: str = ""

    # Strategy
    strategy: LegalStrategy = field(default_factory=LegalStrategy)

    # Hypotheses (legal positions)
    hypotheses: list[LegalHypothesis] = field(default_factory=list)
    _hyp_counter: int = 0

    # Evidence
    evidence: list[LegalEvidence] = field(default_factory=list)
    _ev_counter: int = 0

    # Open research questions
    questions: list[ResearchQuestion] = field(default_factory=list)
    _rq_counter: int = 0

    # Critiques
    critiques: list[Critique] = field(default_factory=list)
    _crit_counter: int = 0

    # Survey results (from the initial legal landscape survey)
    survey_summary: str = ""

    # Final answer
    answer: str = ""
    answer_template: str = ""

    # Metadata
    iteration: int = 0
    total_tokens: int = 0
    total_cost_usd: float = 0.0

    # --- Mutation helpers ---

    def add_hypothesis(self, statement: str, iteration: int | None = None) -> LegalHypothesis:
        self._hyp_counter += 1
        h = LegalHypothesis(
            id=f"H-{self._hyp_counter:03d}",
            statement=statement,
            iteration_created=iteration,
        )
        self.hypotheses.append(h)
        return h

    def add_evidence(self, **kwargs: Any) -> LegalEvidence:
        self._ev_counter += 1
        ev = LegalEvidence(id=f"EV-{self._ev_counter:03d}", **kwargs)
        self.evidence.append(ev)
        return ev

    def add_question(self, question: str, iteration: int | None = None) -> ResearchQuestion:
        self._rq_counter += 1
        rq = ResearchQuestion(
            id=f"RQ-{self._rq_counter:03d}",
            question=question,
            iteration_created=iteration,
        )
        self.questions.append(rq)
        return rq

    def add_critique(self, **kwargs: Any) -> Critique:
        self._crit_counter += 1
        c = Critique(id=f"CR-{self._crit_counter:03d}", **kwargs)
        self.critiques.append(c)
        return c

    def open_questions(self) -> list[ResearchQuestion]:
        return [q for q in self.questions if q.status == RQStatus.OPEN]

    def active_critiques(self) -> list[Critique]:
        return [c for c in self.critiques if c.status == CritiqueStatus.ACTIVE]

    def working_hypotheses(self) -> list[LegalHypothesis]:
        return [h for h in self.hypotheses if h.status == HypothesisStatus.WORKING]

    def established_hypotheses(self) -> list[LegalHypothesis]:
        return [h for h in self.hypotheses if h.status == HypothesisStatus.ESTABLISHED]

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2))

    @classmethod
    def load(cls, path: Path) -> ConsultationState:
        data = json.loads(path.read_text())
        state = cls()
        state.client_question = data.get("client_question", "")
        state.jurisdiction = data.get("jurisdiction", "")
        state.title = data.get("title", "")
        state.survey_summary = data.get("survey_summary", "")
        state.answer = data.get("answer", "")
        state.iteration = data.get("iteration", 0)
        state.total_tokens = data.get("total_tokens", 0)
        state.total_cost_usd = data.get("total_cost_usd", 0.0)
        # Hydrate nested objects
        if "strategy" in data:
            state.strategy = LegalStrategy(**data["strategy"])
        for h in data.get("hypotheses", []):
            hyp = LegalHypothesis(**{k: v for k, v in h.items() if k != "reviews"})
            hyp.reviews = [ReviewResult(**r) for r in h.get("reviews", [])]
            state.hypotheses.append(hyp)
        for ev in data.get("evidence", []):
            state.evidence.append(LegalEvidence(**ev))
        for q in data.get("questions", []):
            state.questions.append(ResearchQuestion(**q))
        for c in data.get("critiques", []):
            state.critiques.append(Critique(**c))
        return state
