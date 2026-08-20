"""Local offline fakes for stopBrowse tests — no network, no index loading.

Small, test-only duplicates of the shapes FigBrowse's own fixtures use
(figbrowse/tests/fixtures is not part of the installed `figbrowse` package,
so it isn't importable from here); the real production code under test is
still 100% FigBrowse's ``figbrowse.agent``/``figbrowse.schemas``.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from figbrowse.llm import LLMCallRecord
from figbrowse.schemas import BenchmarkExample, RetrievedDocument


class FakeCorpusStore:
    def __init__(self, docs: dict[str, str] | None = None) -> None:
        self._docs = docs or {
            f"doc_{i:03d}": f"Full document text for doc_{i:03d}. " * 20 for i in range(6)
        }

    def get(self, doc_id: str) -> dict[str, str] | None:
        text = self._docs.get(doc_id)
        return {"text": text, "url": ""} if text is not None else None


class FakeRetriever:
    retriever_id = "fake-retriever"

    def search(self, query: str, corpus_store: Any = None, top_k: int | None = None) -> list[RetrievedDocument]:
        k = top_k or 5
        return [
            RetrievedDocument(doc_id=f"other_doc_{i:03d}", rank=i + 1, score=1.0 - i * 0.1, preview=f"preview {i}")
            for i in range(k)
        ]


class FakeLLMClient:
    """Returns scripted JSON responses in order, one per call (cycles if exhausted)."""

    def __init__(self, responses: list[str] | None = None) -> None:
        self._responses = list(responses or [])
        self._call_count = 0

    def _model(self) -> str:
        return "fake-model"

    def generate(self, *, messages, prompt_template_id, temperature=None, model=None):
        raw = self._responses[self._call_count % len(self._responses)] if self._responses else (
            '{"thought":"x","action":"final","answer":"fake","confidence":0.5,'
            '"cited_doc_ids":[],"brief_support":"x"}'
        )
        self._call_count += 1
        return LLMCallRecord(
            model="fake-model", provider="fake", prompt_template_id=prompt_template_id,
            raw_response=raw, latency_ms=1.0, timestamp_utc=datetime.now(timezone.utc).isoformat(),
        )

    def generate_structured(self, *, messages, schema, prompt_template_id, temperature=None, model=None, max_schema_retries=2):
        rec = self.generate(messages=messages, prompt_template_id=prompt_template_id, temperature=temperature, model=model)
        parsed = json.loads(rec.raw_response)
        validated = schema.model_validate(parsed)
        return validated, [rec]


def fake_benchmark_example(query_id: str = "q001") -> BenchmarkExample:
    return BenchmarkExample(
        query_id=query_id,
        question="What was the name of the ship that sank in 1912?",
        reference_answers=["Titanic"],
        evidence_qrels={"doc_001": 1.0, "doc_002": 1.0},
        gold_qrels={"doc_002": 1.0},
    )


class FakeLoader:
    """Stand-in for figbrowse.data.BrowseCompPlusLoader — ignores its __init__ kwargs."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    def get(self, query_id: str) -> BenchmarkExample:
        return fake_benchmark_example(query_id)
