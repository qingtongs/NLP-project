"""Measure document recall against public annotations, outside the RAG run."""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

from rag_project import BM25, chunk_corpus, retrieve


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    corpus = json.loads(args.corpus.read_text(encoding="utf-8"))
    questions = json.loads(args.questions.read_text(encoding="utf-8"))
    index = BM25(chunk_corpus(corpus, 240, 40))
    counts = collections.defaultdict(lambda: collections.Counter())
    for question in questions[: args.limit or len(questions)]:
        reference_urls = {e["url"] for e in question["evidence_list"] if e.get("url")}
        if not reference_urls:
            continue
        _, selected = retrieve(index, question["query"], args.top_k)
        retrieved_urls = {item["source"] for item in selected}
        for group in ("overall", question["question_type"]):
            counts[group]["questions"] += 1
            counts[group]["any_hit"] += bool(reference_urls & retrieved_urls)
            counts[group]["all_hit"] += reference_urls <= retrieved_urls
            counts[group]["reference_documents"] += len(reference_urls)
            counts[group]["retrieved_reference_documents"] += len(reference_urls & retrieved_urls)
    for group, values in counts.items():
        n = values["questions"]
        print(group, "n=", n,
              "any=", round(values["any_hit"] / n, 4),
              "all=", round(values["all_hit"] / n, 4),
              "doc_recall=", round(values["retrieved_reference_documents"] /
                                    values["reference_documents"], 4))


if __name__ == "__main__":
    main()
