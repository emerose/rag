"""Generate synthetic question-answer pairs from text files."""

from __future__ import annotations

import re
from collections.abc import Iterable
from pathlib import Path

import typer
from pydantic import BaseModel


class QAPair(BaseModel):
    """Represents a single question-answer pair."""

    question: str
    answer: str
    source: str


def extract_sentences(text: str) -> list[str]:
    """Split text into individual sentences."""
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in sentences if s.strip()]


def generate_pairs(file_path: Path, num_pairs: int = 3) -> list[QAPair]:
    """Generate synthetic QA pairs from *file_path*.

    The first sentence of each pair forms the question context and the
    subsequent sentence becomes the answer.
    """
    text = file_path.read_text(encoding="utf-8")
    sentences = extract_sentences(text)
    pairs: list[QAPair] = []
    for i in range(min(num_pairs, len(sentences) - 1)):
        question = f"What does the text say about: {sentences[i][:50]}?"
        answer = sentences[i + 1]
        pairs.append(QAPair(question=question, answer=answer, source=str(file_path)))
    return pairs


def generate_from_dir(input_dir: Path, pairs_per_file: int = 3) -> Iterable[QAPair]:
    """Yield QA pairs for all ``*.txt`` files in ``input_dir``."""
    for path in sorted(input_dir.glob("*.txt")):
        yield from generate_pairs(path, pairs_per_file)


app = typer.Typer(add_completion=False)


@app.command()
def main(
    input_dir: Path = typer.Argument(..., exists=True, file_okay=False),  # noqa: B008
    output_file: Path = typer.Option("qa_pairs.jsonl", "--output"),  # noqa: B008
    pairs_per_file: int = typer.Option(3, min=1, help="Pairs to generate per file"),
) -> None:
    """Generate synthetic QA pairs from text files."""
    pairs = list(generate_from_dir(input_dir, pairs_per_file))
    with output_file.open("w", encoding="utf-8") as f:
        for pair in pairs:
            f.write(pair.model_dump_json() + "\n")
    typer.echo(f"Generated {len(pairs)} pairs to {output_file}")


if __name__ == "__main__":
    app()
