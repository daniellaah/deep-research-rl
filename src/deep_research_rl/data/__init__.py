"""Reproducible dataset conversion and integrity contracts."""

from deep_research_rl.data.hotpotqa import (
    DataPipelineError,
    HotpotQABuildResult,
    build_hotpotqa,
    verify_hotpotqa_build,
)
from deep_research_rl.data.models import (
    CorpusDocument,
    HotpotQAExample,
    SupportingFact,
    hotpotqa_example_from_dict,
)
from deep_research_rl.data.source import (
    DataSourceConfig,
    SourceSplit,
    download_source_files,
    load_source_config,
)

__all__ = [
    "CorpusDocument",
    "DataPipelineError",
    "DataSourceConfig",
    "HotpotQABuildResult",
    "HotpotQAExample",
    "SourceSplit",
    "SupportingFact",
    "build_hotpotqa",
    "download_source_files",
    "hotpotqa_example_from_dict",
    "load_source_config",
    "verify_hotpotqa_build",
]
