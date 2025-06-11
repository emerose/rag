"""Type stubs for the datasets library."""

from collections.abc import Iterator, Mapping, Sequence
from typing import (
    Any,
    Literal,
    Union,
    overload,
)

# Type aliases for common patterns
Features = dict[str, Any]
Split = Union[str, "NamedSplit"]

class NamedSplit:
    """Named split constants."""

    TRAIN: Literal["train"]
    TEST: Literal["test"]
    VALIDATION: Literal["validation"]

class Dataset:
    """HuggingFace Dataset class."""

    def __getitem__(
        self, key: int | str | slice
    ) -> dict[str, Any] | list[dict[str, Any]]: ...
    def __iter__(self) -> Iterator[dict[str, Any]]: ...
    def __len__(self) -> int: ...
    @property
    def features(self) -> Features: ...
    @property
    def num_rows(self) -> int: ...
    @property
    def column_names(self) -> list[str]: ...
    def to_dict(self) -> dict[str, list[Any]]: ...
    def to_list(self) -> list[dict[str, Any]]: ...
    def select(self, indices: list[int] | range) -> Dataset: ...
    def filter(self, function: Any, **kwargs: Any) -> Dataset: ...
    def map(self, function: Any, **kwargs: Any) -> Dataset: ...

class IterableDataset:
    """HuggingFace IterableDataset class."""

    def __iter__(self) -> Iterator[dict[str, Any]]: ...
    @property
    def features(self) -> Features: ...

class DatasetDict(dict[str, Dataset]):
    """Dictionary of datasets by split name."""

    def __getitem__(self, key: str) -> Dataset: ...
    def __setitem__(self, key: str, value: Dataset) -> None: ...
    @property
    def column_names(self) -> dict[str, list[str]]: ...
    def map(self, function: Any, **kwargs: Any) -> DatasetDict: ...
    def filter(self, function: Any, **kwargs: Any) -> DatasetDict: ...

class IterableDatasetDict(dict[str, IterableDataset]):
    """Dictionary of iterable datasets by split name."""

    def __getitem__(self, key: str) -> IterableDataset: ...

class DownloadConfig:
    """Configuration for dataset downloads."""

    ...

class DownloadMode:
    """Download mode constants."""

    REUSE_DATASET_IF_EXISTS: str
    REUSE_CACHE_IF_EXISTS: str
    FORCE_REDOWNLOAD: str

class VerificationMode:
    """Verification mode constants."""

    ...

class Version:
    """Dataset version."""
    def __init__(self, version_str: str) -> None: ...

@overload
def load_dataset(
    path: str,
    name: str | None = None,
    *,
    split: str | Split | None = None,
    cache_dir: str | None = None,
    features: Features | None = None,
    download_config: DownloadConfig | None = None,
    download_mode: DownloadMode | str | None = None,
    verification_mode: VerificationMode | str | None = None,
    keep_in_memory: bool | None = None,
    save_infos: bool = False,
    revision: str | Version | None = None,
    token: bool | str | None = None,
    streaming: Literal[False] = False,
    num_proc: int | None = None,
    storage_options: dict[str, Any] | None = None,
    trust_remote_code: bool | None = None,
    **config_kwargs: Any,
) -> DatasetDict | Dataset: ...
@overload
def load_dataset(
    path: str,
    name: str | None = None,
    *,
    split: str | Split | None = None,
    cache_dir: str | None = None,
    features: Features | None = None,
    download_config: DownloadConfig | None = None,
    download_mode: DownloadMode | str | None = None,
    verification_mode: VerificationMode | str | None = None,
    keep_in_memory: bool | None = None,
    save_infos: bool = False,
    revision: str | Version | None = None,
    token: bool | str | None = None,
    streaming: Literal[True],
    num_proc: int | None = None,
    storage_options: dict[str, Any] | None = None,
    trust_remote_code: bool | None = None,
    **config_kwargs: Any,
) -> IterableDatasetDict | IterableDataset: ...
def load_dataset(
    path: str,
    name: str | None = None,
    data_dir: str | None = None,
    data_files: str | Sequence[str] | Mapping[str, str | Sequence[str]] | None = None,
    split: str | Split | None = None,
    cache_dir: str | None = None,
    features: Features | None = None,
    download_config: DownloadConfig | None = None,
    download_mode: DownloadMode | str | None = None,
    verification_mode: VerificationMode | str | None = None,
    keep_in_memory: bool | None = None,
    save_infos: bool = False,
    revision: str | Version | None = None,
    token: bool | str | None = None,
    streaming: bool = False,
    num_proc: int | None = None,
    storage_options: dict[str, Any] | None = None,
    trust_remote_code: bool | None = None,
    **config_kwargs: Any,
) -> DatasetDict | Dataset | IterableDatasetDict | IterableDataset: ...
