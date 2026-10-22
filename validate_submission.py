"""Validate the complete predictions and question-linked execution record."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--record", type=Path, required=True)
    args = parser.parse_args()
    predictions = json.loads(args.predictions.read_text(encoding="utf-8"))
    record = json.loads(args.record.read_text(encoding="utf-8"))
    rows = record["question_records"]
    expected = [f"q{i:04d}" for i in range(2556)]
    if len(predictions) != 2556 or len(rows) != 2556:
        raise ValueError("Predictions and records must each contain 2556 questions")
    if [row["question_id"] for row in predictions] != expected:
        raise ValueError("Prediction IDs are missing, duplicated, or out of order")
    if [row["question_id"] for row in rows] != expected:
        raise ValueError("Execution record IDs are missing, duplicated, or out of order")
    for prediction, row in zip(predictions, rows):
        if set(prediction) != {"question_id", "model_answer"}:
            raise ValueError(f"Unexpected prediction fields at {row['question_id']}")
        if not isinstance(prediction["model_answer"], str) or \
                prediction["model_answer"] != row["model_answer"]:
            raise ValueError(f"Prediction/record answer mismatch at {row['question_id']}")
        if not row["question"] or not row["rounds"]:
            raise ValueError(f"Missing question or rounds at {row['question_id']}")
        for round_data in row["rounds"]:
            if not round_data["retrieval_calls"] or not round_data["generation_calls"]:
                raise ValueError(f"Missing retrieval or generation at {row['question_id']}")
            for call in round_data["retrieval_calls"]:
                if not call["query"] or not isinstance(call["retrieved_documents"], list):
                    raise ValueError(f"Missing retrieval query/results at {row['question_id']}")
                if any(not doc["source"] or not doc["text"]
                       for doc in call["retrieved_documents"]):
                    raise ValueError(f"Missing source or passage at {row['question_id']}")
            for call in round_data["generation_calls"]:
                if not isinstance(call["actual_context"], str) or not call["messages"]:
                    raise ValueError(f"Missing actual context/messages at {row['question_id']}")
        if "gold_answer" in row or "evidence_list" in row:
            raise ValueError(f"Reference data leaked into record at {row['question_id']}")
    info = record["run_info"]
    if info["dataset"]["question_count"] != 2556 or \
            info["dataset"]["corpus_document_count"] != 609:
        raise ValueError("Run metadata has the wrong dataset counts")
    if any(model["total_parameters"] > 10_000_000_000 for model in info["models"]):
        raise ValueError("A model exceeds the 10B limit")
    print("Validated 2556 predictions and matching question-linked execution records.")
    print(f"Failed questions: {info['resource_usage']['failed_question_count']}; "
          f"retries: {info['resource_usage']['retry_count']}")


if __name__ == "__main__":
    main()
