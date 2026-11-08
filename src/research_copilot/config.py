from dataclasses import dataclass
import os
from pathlib import Path

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    hf_token: str
    chat_model: str
    chat_provider: str
    embedding_model: str
    embedding_provider: str
    chroma_persist_directory: Path
    chroma_collection_name: str

    @classmethod
    def from_env(cls) -> "Settings":
        load_dotenv()

        required = {
            "HUGGINGFACEHUB_API_TOKEN": os.getenv(
                "HUGGINGFACEHUB_API_TOKEN", ""
            ).strip(),
            "HF_CHAT_MODEL": os.getenv("HF_CHAT_MODEL", "").strip(),
            "HF_EMBEDDING_MODEL": os.getenv("HF_EMBEDDING_MODEL", "").strip(),
        }
        missing = [name for name, value in required.items() if not value]
        if missing:
            raise RuntimeError(
                "Missing required environment variables: "
                + ", ".join(missing)
                + ". Copy .env.example to .env and fill in the values."
            )

        return cls(
            hf_token=required["HUGGINGFACEHUB_API_TOKEN"],
            chat_model=required["HF_CHAT_MODEL"],
            chat_provider=os.getenv("HF_CHAT_PROVIDER", "auto").strip() or "auto",
            embedding_model=required["HF_EMBEDDING_MODEL"],
            embedding_provider=os.getenv(
                "HF_EMBEDDING_PROVIDER", "hf-inference"
            ).strip()
            or "hf-inference",
            chroma_persist_directory=Path(
                os.getenv("CHROMA_PERSIST_DIRECTORY", ".chroma")
            ).expanduser(),
            chroma_collection_name=os.getenv(
                "CHROMA_COLLECTION_NAME", "research_copilot"
            ).strip()
            or "research_copilot",
        )
