import json
from pathlib import Path

from research_copilot.config import Settings
from research_copilot.rag import create_embeddings, open_vector_store


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    settings = Settings.from_env()
    store = open_vector_store(settings, create_embeddings(settings))

    rows = [
        json.loads(line)
        for line in (ROOT / "tests" / "eval_questions.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]

    # The tutorial corpus has two source files, so top-5 would retrieve all
    # files and make the result uninformative. Start with top-1 instead.
    k = 1
    recalls: list[float] = []
    for row in rows:
        expected = set(row["expected_sources"])
        if not expected:
            raise ValueError("Every retrieval evaluation row needs expected_sources.")

        retrieved_docs = store.similarity_search(row["question"], k=k)
        retrieved = {doc.metadata["source"] for doc in retrieved_docs}
        recall = len(expected & retrieved) / len(expected)
        recalls.append(recall)
        print(
            f"Recall@{k}={recall:.2f} | "
            f"expected={sorted(expected)} | retrieved={sorted(retrieved)}"
        )

    mean_recall = sum(recalls) / len(recalls)
    print(f"\nMean Recall@{k}: {mean_recall:.3f} ({len(rows)} questions)")
    print("Initial target: >= 0.80. Treat this as a starting target, not a universal SLA.")


if __name__ == "__main__":
    main()
