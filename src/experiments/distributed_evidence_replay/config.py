"""Configuration for the distributed_evidence_replay experiment.

Thin adapter: reuses figbrowse.config's BenchmarkConfig/RetrieverConfig
verbatim, and points at the completed no-injection run (condition A) as its
source trajectories. Retriever/preview settings here MUST match the ones
used to produce that source run, since search-result reconstruction depends
on hitting the same retrieval cache keys.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

from figbrowse.config import BenchmarkConfig, RetrieverConfig


class ModelConfig(BaseModel):
    name: str = "USE_EXISTING_FIGBROWSE_DEFAULT"
    temperature: float = 0.0
    provider: str = "azure_openai"  # "azure_openai" (default, unchanged) or "cerebras"

    def resolved_name(self) -> str:
        if self.name and self.name != "USE_EXISTING_FIGBROWSE_DEFAULT":
            return self.name
        if self.provider == "cerebras":
            return os.environ.get("CEREBRAS_MODEL", "gpt-oss-120b")
        return os.environ.get("FIGBROWSE_AGENT_MODEL", "gpt-4o")


class SourceConfig(BaseModel):
    """Where the frozen no-injection trajectories come from."""

    runs_path: str = "outputs/initial_gold_injection_confirmatory_no_injection/runs.jsonl"
    injection_position: str = "no_injection"  # which position within that file to read
    document_view_tokens: int = 4096  # must match the source run's agent.document_view_tokens


class ReplayConfig(BaseModel):
    placement_seeds: list[int] = Field(default_factory=lambda: [42, 43, 44])
    prefer_evidence_documents: bool = True
    fallback_to_gold_documents: bool = True


class DistributedEvidenceReplayConfig(BaseModel):
    experiment_name: str = "distributed_evidence_replay"
    seed: int = 42
    sample_limit: int = 100

    figbrowse_path: str = "../FigBrowse"

    model: ModelConfig = Field(default_factory=ModelConfig)
    source: SourceConfig = Field(default_factory=SourceConfig)
    replay: ReplayConfig = Field(default_factory=ReplayConfig)

    benchmark: BenchmarkConfig = Field(default_factory=BenchmarkConfig)
    retriever: RetrieverConfig = Field(default_factory=RetrieverConfig)

    output_dir: str = "outputs/distributed_evidence_replay"
    resume: bool = True

    def figbrowse_root(self) -> Path:
        return Path(self.figbrowse_path).resolve()

    def resolve_path(self, relative: str) -> Path:
        p = Path(relative)
        if p.is_absolute():
            return p
        return (self.figbrowse_root() / p).resolve()

    def resolve_own_path(self, relative: str) -> Path:
        """Resolve a path relative to this repo (stopBrowse), not figbrowse_path."""
        p = Path(relative)
        if p.is_absolute():
            return p
        return (Path(__file__).parent.parent.parent.parent / p).resolve()

    def resolved_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


def load_config(path: str | Path) -> DistributedEvidenceReplayConfig:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open() as f:
        raw = yaml.safe_load(f) or {}
    return DistributedEvidenceReplayConfig.model_validate(raw)
