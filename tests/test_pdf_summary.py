from pathlib import Path

import pdf_summary as ps


def test_clean_text_removes_extra_whitespace():
    text = "A   B\n\n\nC"
    assert ps.clean_text(text) == "A B\nC"


def test_chunk_by_words_splits_as_expected():
    text = "one two three four five"
    chunks = ps.chunk_by_words(text, 2)
    assert chunks == ["one two", "three four", "five"]


def test_heuristic_summary_handles_empty():
    assert "No extractable" in ps.heuristic_summary("", max_bullets=3)


def test_run_direct_with_monkeypatch(tmp_path, monkeypatch):
    fake_pdf = tmp_path / "a.pdf"
    fake_pdf.write_text("stub")

    monkeypatch.setattr(ps, "extract_pdf_text", lambda _: "Sentence one. Sentence two.")

    cfg = ps.SummaryConfig(mode="direct", backend="heuristic", max_bullets=2)
    out = ps.run(Path(fake_pdf), cfg)
    assert out.startswith("-")
