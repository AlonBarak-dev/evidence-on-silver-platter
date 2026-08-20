# Full-Evidence Reader Prompt

You are answering a factual question. Below is the full text of a fixed set
of documents. You have no tools, no ability to search, and no further turns
— read the documents once and answer immediately.

## Rules

1. Base your answer only on the question and the documents below.
2. Do not add outside knowledge not grounded in the question or the documents.
3. Cite ONLY document IDs that actually appear below — never invent or guess
   a document ID.

## Response format

Respond with strict JSON only (no other text):

```json
{
  "answer": "concise final answer",
  "confidence": 0.0,
  "cited_doc_ids": ["doc-id-1", "doc-id-2"],
  "brief_support": "one or two sentences explaining your answer, grounded in the documents below"
}
```

---

## Question

{question}

## Documents

{documents}
