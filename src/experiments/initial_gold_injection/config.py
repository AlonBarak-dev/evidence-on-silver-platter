"""Configuration for the initial_gold_injection experiment.

Thin adapter: reuses FigBrowse's own ``BenchmarkConfig`` and
``RetrieverConfig`` blocks unchanged (same fields, same defaults) so the
experiment stays wired to the same corpus/index/model conventions FigBrowse
already uses. Only the experiment-specific ``agent``/``injection`` blocks are
new.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

from figbrowse.config import BenchmarkConfig, RetrieverConfig

# Split name -> FigBrowse manifest file (relative to figbrowse_path).
# FigBrowse ships three frozen manifests (CLAUDE.md §3.2); this experiment
# does not define its own splits, so "validation"/"dev" map to the 15-query
# development pool (safe to use freely) and "confirmatory"/"test" map to the
# frozen 100-query set. See the completion report for this assumption.
SPLIT_TO_MANIFEST = {
    "validation": "data/manifests/development_15.jsonl",
    "dev": "data/manifests/development_15.jsonl",
    "development": "data/manifests/development_15.jsonl",
    "pilot": "data/manifests/pilot_10.jsonl",
    "confirmatory": "data/manifests/confirmatory_100.jsonl",
    "test": "data/manifests/confirmatory_100.jsonl",
}


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


class AgentConfig(BaseModel):
    max_post_injection_actions: int = 8
    enable_search: bool = True
    enable_get_document: bool = True
    max_output_tokens: int = 2048
    document_view_tokens: int = 4096


class InjectionConfig(BaseModel):
    prefer_evidence_documents: bool = True
    fallback_to_gold_documents: bool = True
    shuffle_documents: bool = True
    expose_full_text_immediately: bool = False

    # Which injection position(s) to run as separate conditions in one
    # invocation. "beginning" = the original single-condition design (turn
    # 0). "mid_trajectory" = evidence appears only after the agent has
    # already taken `mid_trajectory_delay` real actions on its own, sharing
    # the same total action budget so the two conditions stay comparable.
    # "no_injection" = a control condition: the plain, unmodified FigBrowse
    # search loop with no synthetic evidence at all (see
    # `no_injection_max_actions` for its own action ceiling).
    positions: list[str] = Field(default_factory=lambda: ["beginning"])
    mid_trajectory_delay: int = 2
    no_injection_max_actions: int = 10


class ExperimentConfig(BaseModel):
    experiment_name: str = "initial_gold_injection"
    dataset: str = "browsecomp_plus"
    split: str = "validation"
    sample_limit: int = 10
    seed: int = 42

    figbrowse_path: str = "../FigBrowse"

    model: ModelConfig = Field(default_factory=ModelConfig)
    agent: AgentConfig = Field(default_factory=AgentConfig)
    injection: InjectionConfig = Field(default_factory=InjectionConfig)

    # Reused verbatim from FigBrowse (same corpus/index/retriever wiring).
    benchmark: BenchmarkConfig = Field(default_factory=BenchmarkConfig)
    retriever: RetrieverConfig = Field(default_factory=RetrieverConfig)

    output_dir: str = "outputs/initial_gold_injection"
    resume: bool = True

    def figbrowse_root(self) -> Path:
        return Path(self.figbrowse_path).resolve()

    def resolve_path(self, relative: str) -> Path:
        """Resolve a path that is relative to figbrowse_path (matches the
        conventions in FigBrowse's own configs/pilot.yaml)."""
        p = Path(relative)
        if p.is_absolute():
            return p
        return (self.figbrowse_root() / p).resolve()

    def manifest_path(self) -> Path:
        rel = SPLIT_TO_MANIFEST.get(self.split)
        if rel is None:
            raise ValueError(
                f"Unknown split {self.split!r}; expected one of {sorted(SPLIT_TO_MANIFEST)}"
            )
        return self.resolve_path(rel)

    def resolved_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


def load_config(path: str | Path) -> ExperimentConfig:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open() as f:
        raw = yaml.safe_load(f) or {}
    return ExperimentConfig.model_validate(raw)
