from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

SUPPORTED_SUFFIXES = {".md", ".txt", ".pdf"}


def load_documents(data_dir: Path) -> list[Document]:
    if not data_dir.exists():
        raise FileNotFoundError(f"Document directory does not exist: {data_dir}")

    documents: list[Document] = []
    for path in sorted(data_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue

        source = path.relative_to(data_dir).as_posix()
        title = path.stem.replace("_", " ")

        if path.suffix.lower() == ".pdf":
            reader = PdfReader(str(path))
            for page_number, page in enumerate(reader.pages, start=1):
                text = (page.extract_text() or "").strip()
                if text:
                    documents.append(
                        Document(
                            page_content=text,
                            metadata={
                                "source": source,
                                "title": title,
                                "page": page_number,
                            },
                        )
                    )
        else:
            text = path.read_text(encoding="utf-8").strip()
            if text:
                documents.append(
                    Document(
                        page_content=text,
                        metadata={"source": source, "title": title},
                    )
                )

    if not documents:
        raise ValueError(
            f"No readable .md, .txt, or .pdf documents found in {data_dir}"
        )
    return documents


def split_documents(documents: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=150,
        add_start_index=True,
    )
    chunks = splitter.split_documents(documents)
    if not chunks:
        raise ValueError("Document splitting produced no chunks.")
    return chunks
