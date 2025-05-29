from __future__ import annotations

from pathlib import Path

import typer
from langsmith import Client
from pydantic import BaseModel

try:
    from beir import util
    from beir.datasets.data_loader import GenericDataLoader
except ModuleNotFoundError:  # pragma: no cover - beir optional
    util = None  # type: ignore
    GenericDataLoader = None  # type: ignore


class Example(BaseModel):
    """Represents a single dataset example."""

    inputs: dict[str, str]
    outputs: dict[str, str]
    metadata: dict[str, str] | None = None


def download_dataset(name: str, cache_dir: Path) -> Path:
    """Download *name* dataset via BEIR into *cache_dir*."""
    if util is None:
        raise RuntimeError("beir package is not installed")
    url = (
        f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{name}.zip"
    )
    return Path(util.download_and_unzip(url, cache_dir))


def parse_examples(dataset_path: Path) -> list[Example]:
    """Load BEIR dataset located at *dataset_path* into ``Example`` objects."""
    if GenericDataLoader is None:
        raise RuntimeError("beir package is not installed")

    data = GenericDataLoader(dataset_path)
    corpus, queries, qrels = data.load(split="test")

    examples: list[Example] = []
    for qid, doc_dict in qrels.items():
        query = queries[qid]
        for doc_id in doc_dict.keys():
            doc = corpus[doc_id]
            examples.append(
                Example(
                    inputs={"query": query},
                    outputs={"answer": doc.get("text", "")},
                    metadata={"doc_id": doc_id},
                )
            )
    return examples


DEFAULT_CACHE_DIR = Path("beir_data")

app = typer.Typer(add_completion=False)


@app.command()
def main(
    dataset_name: str,
    cache_dir: Path = typer.Option(DEFAULT_CACHE_DIR),  # noqa: B008
) -> None:
    """Download and upload a BEIR dataset to LangSmith."""
    client = Client()
    if client.has_dataset(dataset_name=dataset_name):
        typer.echo(f"Dataset '{dataset_name}' already exists on LangSmith")
        raise typer.Exit(0)

    dataset_path = download_dataset(dataset_name, cache_dir)
    examples = parse_examples(dataset_path)

    client.create_dataset(dataset_name)
    client.create_examples(
        dataset_name=dataset_name, examples=[e.model_dump() for e in examples]
    )
    typer.echo(f"Uploaded {len(examples)} examples to '{dataset_name}'")


if __name__ == "__main__":
    app()
