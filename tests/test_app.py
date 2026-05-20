"""Tests for Gradio app construction."""

from pathlib import Path

import pytest

gradio = pytest.importorskip("gradio")


def test_architecture_html_exists():
    path = Path(__file__).parent.parent / "architecture.html"
    assert path.exists(), "architecture.html not found"
    content = path.read_text()
    assert "lmaf-arch" in content
    assert "<script>" in content
    assert "d3" in content


def test_architecture_html_is_valid_document():
    path = Path(__file__).parent.parent / "architecture.html"
    content = path.read_text()
    assert content.strip().startswith("<!DOCTYPE html>")
    assert "</html>" in content


def test_app_builds():
    import app
    demo = app.build_app()
    assert demo is not None


def test_app_has_iframe_srcdoc():
    import app
    assert "iframe" in app.ARCHITECTURE_HTML
    assert "srcdoc" in app.ARCHITECTURE_HTML
