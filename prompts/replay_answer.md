# Replay Answer Prompt

You are answering a factual question. Below is a complete record of a prior
research session: the searches that were run and the documents that were
opened, in order. You have no tools and no further turns — read the record
once and answer immediately based only on what is visible in it.

## Rules

1. Base your answer only on the question and the session record below.
2. Do not add outside knowledge not grounded in the question or the record.
3. Cite ONLY document IDs that actually appear below — never invent or guess
   a document ID.

## Response format

Respond with strict JSON only (no other text):

```json
{
  "answer": "short final answer",
  "cited_document_ids": ["optional", "document", "ids"]
}
```

---

## Question

{question}

## Session record

{trajectory}
