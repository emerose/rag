from pathlib import Path

from scripts.generate_synthetic_qa import generate_pairs


def test_generate_pairs(tmp_path: Path) -> None:
    sample = tmp_path / "doc.txt"
    sample.write_text("Question one. Answer one. Question two. Answer two.")
    pairs = generate_pairs(sample, num_pairs=2)
    assert len(pairs) == 2
    assert pairs[0].answer == "Answer one."
    assert pairs[1].answer == "Question two."
