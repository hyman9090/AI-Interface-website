import argparse
from pathlib import Path

from research_copilot.config import Settings
from research_copilot.ingestion import load_documents, split_documents
from research_copilot.rag import create_chroma_store, create_embeddings

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create or explicitly rebuild the persistent Chroma index."
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Replace the existing collection with a fresh index.",
    )
    args = parser.parse_args()

    settings = Settings.from_env()
    documents = load_documents(ROOT / "data" / "docs")
    chunks = split_documents(documents)
    embeddings = create_embeddings(settings)
    store = create_chroma_store(settings, embeddings)
    existing_ids = store.get(limit=1)["ids"]

    if existing_ids:
        if not args.rebuild:
            raise RuntimeError(
                "A Chroma index already exists. Use "
                "`python scripts/index_documents.py --rebuild` to replace it."
            )
        store.delete_collection()
        store = create_chroma_store(settings, embeddings)

    store.add_documents(chunks)
    print(
        f"Indexed {len(documents)} source documents into {len(chunks)} chunks. "
        f"Chroma directory: {settings.chroma_persist_directory}"
    )


if __name__ == "__main__":
    main()
