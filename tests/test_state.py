"""Tests for ConsultationState and its data structures."""

import json
import tempfile
from pathlib import Path

import pytest

from lmaf.state.research_state import (
    ConsultationState,
    Critique,
    CritiqueStatus,
    HypothesisStatus,
    LegalEvidence,
    LegalHypothesis,
    LegalStrategy,
    RQStatus,
    ResearchQuestion,
    Severity,
)


class TestConsultationState:
    def test_empty_state(self):
        state = ConsultationState()
        assert state.client_question == ""
        assert state.hypotheses == []
        assert state.evidence == []
        assert state.questions == []
        assert state.critiques == []
        assert state.iteration == 0

    def test_add_hypothesis(self):
        state = ConsultationState()
        h = state.add_hypothesis("Позовна давність становить 3 роки", iteration=1)
        assert h.id == "H-001"
        assert h.status == HypothesisStatus.WORKING
        assert len(state.hypotheses) == 1

    def test_add_multiple_hypotheses(self):
        state = ConsultationState()
        state.add_hypothesis("First")
        state.add_hypothesis("Second")
        state.add_hypothesis("Third")
        assert len(state.hypotheses) == 3
        assert state.hypotheses[2].id == "H-003"

    def test_add_evidence(self):
        state = ConsultationState()
        ev = state.add_evidence(
            type="case_law",
            source="edrsr",
            summary="Рішення ВС щодо строків давності",
            citation="Справа №757/12345/22",
        )
        assert ev.id == "EV-001"
        assert ev.type == "case_law"
        assert not ev.refuted

    def test_add_question(self):
        state = ConsultationState()
        rq = state.add_question("Чи застосовується ст. 625 ЦК?", iteration=2)
        assert rq.id == "RQ-001"
        assert rq.status == RQStatus.OPEN

    def test_add_critique(self):
        state = ConsultationState()
        c = state.add_critique(
            type="strategy",
            severity=Severity.HIGH,
            summary="Не враховано зустрічний позов",
        )
        assert c.id == "CR-001"
        assert c.status == CritiqueStatus.ACTIVE

    def test_open_questions(self):
        state = ConsultationState()
        q1 = state.add_question("Open question")
        q2 = state.add_question("Resolved question")
        q2.status = RQStatus.RESOLVED
        assert len(state.open_questions()) == 1
        assert state.open_questions()[0].id == q1.id

    def test_active_critiques(self):
        state = ConsultationState()
        c1 = state.add_critique(type="strategy", summary="Active")
        c2 = state.add_critique(type="reasoning", summary="Resolved")
        c2.status = CritiqueStatus.RESOLVED
        assert len(state.active_critiques()) == 1
        assert state.active_critiques()[0].id == c1.id

    def test_working_hypotheses(self):
        state = ConsultationState()
        h1 = state.add_hypothesis("Working")
        h2 = state.add_hypothesis("Established")
        h2.status = HypothesisStatus.ESTABLISHED
        assert len(state.working_hypotheses()) == 1
        assert len(state.established_hypotheses()) == 1

    def test_serialization_roundtrip(self):
        state = ConsultationState()
        state.client_question = "Тестове питання"
        state.jurisdiction = "civil"
        state.survey_summary = "Огляд правового ландшафту"
        state.strategy = LegalStrategy(
            approach="Аналіз строків давності",
            legal_domains=["цивільне право"],
            key_questions=["Який строк?"],
        )
        state.add_hypothesis("Гіпотеза 1", iteration=1)
        state.add_evidence(type="legislation", citation="ст. 257 ЦК")
        state.add_question("Питання 1")
        state.add_critique(type="completeness", summary="Прогалина")

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = Path(f.name)

        state.save(path)
        loaded = ConsultationState.load(path)

        assert loaded.client_question == "Тестове питання"
        assert loaded.jurisdiction == "civil"
        assert loaded.strategy.approach == "Аналіз строків давності"
        assert len(loaded.hypotheses) == 1
        assert len(loaded.evidence) == 1
        assert len(loaded.questions) == 1
        assert len(loaded.critiques) == 1
        path.unlink()

    def test_to_dict(self):
        state = ConsultationState()
        state.client_question = "Test"
        d = state.to_dict()
        assert isinstance(d, dict)
        assert d["client_question"] == "Test"


class TestLegalEvidence:
    def test_short(self):
        ev = LegalEvidence(id="EV-001", citation="ст. 625 ЦК", summary="Пеня за прострочення")
        assert "EV-001" in ev.short()
        assert "ст. 625 ЦК" in ev.short()

    def test_defaults(self):
        ev = LegalEvidence()
        assert ev.refuted is False
        assert ev.iteration is None


class TestLegalHypothesis:
    def test_short(self):
        h = LegalHypothesis(id="H-001", statement="Позов обгрунтований", status=HypothesisStatus.WORKING)
        assert "H-001" in h.short()
        assert "working" in h.short()
