from types import SimpleNamespace

import pytest
from langchain_core.documents import Document

from research_copilot.rag import answer_question


class FakeVectorStore:
    def __init__(self, documents: list[Document]) -> None:
        self.documents = documents

    def similarity_search(self, query: str, k: int = 4) -> list[Document]:
        assert query
        return self.documents[:k]


class FakeChatModel:
    def __init__(self) -> None:
        self.messages = []

    def invoke(self, messages):
        self.messages = messages
        return SimpleNamespace(content="不要反覆依測試集調參。[S1]")


def test_answer_uses_retrieved_evidence_and_returns_sources() -> None:
    documents = [
        Document(
            page_content="測試集只用於最終評估。",
            metadata={"source": "experiment_basics.md", "title": "Experiment basics"},
        )
    ]
    model = FakeChatModel()

    answer, sources = answer_question(
        "測試集應如何使用？",
        FakeVectorStore(documents),
        model,
    )

    assert "[S1]" in answer
    assert sources == documents
    assert "測試集只用於最終評估" in model.messages[1].content


def test_answer_rejects_empty_question() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        answer_question("  ", FakeVectorStore([]), FakeChatModel())


def test_answer_handles_no_retrieved_documents() -> None:
    answer, sources = answer_question(
        "沒有文件的問題",
        FakeVectorStore([]),
        FakeChatModel(),
    )

    assert "找不到相關文件" in answer
    assert sources == []
