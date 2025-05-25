from pathlib import Path

from rag.embeddings.model_map import get_model_for_path, load_model_map


def test_load_model_map(tmp_path: Path) -> None:
    content = '"docs/*.md": text-embedding-3-small\nother.pdf: text-embedding-3-large\n'
    yaml_file = tmp_path / "embeddings.yaml"
    yaml_file.write_text(content)
    model_map = load_model_map(yaml_file)
    assert model_map["docs/*.md"] == "text-embedding-3-small"
    assert model_map["other.pdf"] == "text-embedding-3-large"


def test_get_model_for_path() -> None:
    mapping = {"docs/*.md": "small", "reports/*": "large"}
    assert get_model_for_path("docs/file.md", mapping, "default") == "small"
    assert get_model_for_path("reports/annual.pdf", mapping, "default") == "large"
    assert get_model_for_path("other.txt", mapping, "default") == "default"
