from pathlib import Path

import pytest
from research_copilot.ingestion import load_documents, split_documents


def test_load_and_split_markdown(tmp_path: Path) -> None:
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "sample.md").write_text(
        "# Evaluation\nPrecision measures positive prediction quality.",
        encoding="utf-8",
    )

    documents = load_documents(docs_dir)
    chunks = split_documents(documents)

    assert len(documents) == 1
    assert documents[0].metadata["source"] == "sample.md"
    assert documents[0].metadata["title"] == "sample"
    assert chunks
    assert "Precision" in chunks[0].page_content


def test_load_documents_fails_when_directory_is_missing(tmp_path: Path) -> None:
    missing = tmp_path / "missing"

    with pytest.raises(FileNotFoundError, match=str(missing)):
        load_documents(missing)
