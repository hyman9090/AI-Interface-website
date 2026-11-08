from pathlib import Path

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from research_copilot.config import Settings
from research_copilot.rag import create_chroma_store, open_vector_store


class FakeEmbeddings(Embeddings):
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[float(len(text)), float(text.count(" "))] for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return [float(len(text)), float(text.count(" "))]


def test_chroma_index_persists_between_clients(tmp_path: Path) -> None:
    settings = Settings(
        hf_token="test-token",
        chat_model="test-chat",
        chat_provider="auto",
        embedding_model="test-embeddings",
        embedding_provider="hf-inference",
        chroma_persist_directory=tmp_path / "chroma",
        chroma_collection_name="test_collection",
    )
    embeddings = FakeEmbeddings()
    first_client = create_chroma_store(settings, embeddings)
    first_client.add_documents(
        [
            Document(
                page_content="Keep the test set for final evaluation.",
                metadata={"source": "experiment_basics.md"},
            )
        ]
    )

    reopened_client = open_vector_store(settings, embeddings)
    results = reopened_client.similarity_search("final evaluation", k=1)

    assert len(results) == 1
    assert results[0].metadata["source"] == "experiment_basics.md"
    assert "test set" in results[0].page_content
