"""Reproducible lexical-retrieval RAG run for the MultiHop-RAG course project.

Only ``query`` is read from the question file. Reference answers and annotated
evidence are never used for retrieval or generation.
"""

from __future__ import annotations

import argparse
import collections
import ctypes
import datetime as dt
import hashlib
import json
import math
import os
import platform
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:['-][A-Za-z0-9]+)*")
STOP = set("""a an the and or but if then than is are was were be been being to of in
on at by for from with without about as into over under after before between
who what which when where why how does do did has have had can could would
should this that these those it its their his her they them he she there
according reported report reports article articles question answer both same
also not""".split())


def terms(text: str) -> list[str]:
    return [t.lower() for t in TOKEN_RE.findall(text) if len(t) > 1 and t.lower() not in STOP]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def chunk_corpus(corpus: list[dict], words_per_chunk: int, overlap: int) -> list[dict]:
    chunks = []
    stride = words_per_chunk - overlap
    if stride < 1:
        raise ValueError("overlap must be smaller than words_per_chunk")
    for document_id, doc in enumerate(corpus):
        words = str(doc.get("body") or "").split()
        for start in range(0, len(words), stride):
            passage = " ".join(words[start : start + words_per_chunk])
            if not passage:
                continue
            chunks.append({
                "document_id": f"d{document_id:04d}",
                "chunk_id": f"d{document_id:04d}_c{start // stride:04d}",
                "source": doc.get("url") or doc.get("title") or "",
                "title": doc.get("title") or "",
                "publisher": doc.get("source") or "",
                "published_at": doc.get("published_at") or "",
                "text": passage,
            })
            if start + words_per_chunk >= len(words):
                break
    return chunks


class BM25:
    def __init__(self, chunks: list[dict], k1: float = 1.5, b: float = 0.75):
        self.chunks = chunks
        self.k1, self.b = k1, b
        self.postings: dict[str, list[tuple[int, int]]] = collections.defaultdict(list)
        self.lengths = []
        for cid, chunk in enumerate(chunks):
            # Title repetition boosts document discovery without altering shown evidence.
            tokens = terms((chunk["title"] + " ") * 3 + (chunk["publisher"] + " ") * 2 + chunk["text"])
            counts = collections.Counter(tokens)
            self.lengths.append(len(tokens))
            for token, count in counts.items():
                self.postings[token].append((cid, count))
        self.avg_length = sum(self.lengths) / max(len(self.lengths), 1)

    def search(self, query: str, limit: int = 12) -> list[tuple[int, float]]:
        n = len(self.chunks)
        scores: dict[int, float] = collections.defaultdict(float)
        for token in set(terms(query)):
            posting = self.postings.get(token, ())
            df = len(posting)
            if not df:
                continue
            idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
            for cid, tf in posting:
                normalizer = tf + self.k1 * (1 - self.b + self.b * self.lengths[cid] / self.avg_length)
                scores[cid] += idf * tf * (self.k1 + 1) / normalizer
        return sorted(scores.items(), key=lambda pair: pair[1], reverse=True)[:limit]


def retrieval_queries(question: str) -> list[str]:
    queries = [question]
    # Clause searches recover facts that a long multi-hop question can bury.
    clauses = re.split(r",|\b(?:while|whereas|and|as reported by|according to)\b", question, flags=re.I)
    candidates = sorted((s.strip(" ?.") for s in clauses), key=lambda s: len(terms(s)), reverse=True)
    for clause in candidates:
        if len(terms(clause)) >= 4 and clause.casefold() != question.casefold():
            queries.append(clause)
        if len(queries) == 3:
            break
    return queries


def retrieve(index: BM25, question: str, top_k: int) -> tuple[list[dict], list[dict]]:
    calls = []
    fused: dict[int, float] = collections.defaultdict(float)
    for query in retrieval_queries(question):
        hits = index.search(query, limit=15)
        calls.append({
            "query": query,
            "retrieved_documents": [
                {**{key: index.chunks[cid][key] for key in
                    ("document_id", "chunk_id", "source", "text")},
                 "rank": rank, "score": round(score, 6)}
                for rank, (cid, score) in enumerate(hits, 1)
            ],
        })
        for rank, (cid, _) in enumerate(hits, 1):
            fused[cid] += 1 / (60 + rank)
    selected, doc_counts = [], collections.Counter()
    for cid, score in sorted(fused.items(), key=lambda pair: pair[1], reverse=True):
        doc_id = index.chunks[cid]["document_id"]
        if doc_counts[doc_id] >= 2:
            continue
        selected.append({**index.chunks[cid], "fused_score": round(score, 8)})
        doc_counts[doc_id] += 1
        if len(selected) >= top_k:
            break
    return calls, selected


def context_text(selected: list[dict]) -> str:
    return "\n\n".join(
        f"[{i}] {item['title']} | {item['publisher']} | {item['published_at']} | {item['source']}\n{item['text']}"
        for i, item in enumerate(selected, 1)
    )


SYSTEM_PROMPT = (
    "Answer the question using only the retrieved article excerpts. Find the relevant fact in "
    "each article before combining facts. When asked for a person, company, place, date, number, "
    "or other entity, give that entity; do not reply Yes or No. When the question is explicitly "
    "yes/no, answer Yes if the asked statements are supported, No if contradicted, and pay "
    "attention to negation and whether it asks for similarity or difference. If a necessary "
    "fact is absent, answer exactly: Insufficient Information. Return only a short direct answer "
    "in English, with no explanation or citations."
)

PROMPT_COMPARE = (
    "Use only the retrieved article excerpts to answer. Resolve each part of a multi-hop question "
    "against its article, then combine the facts. For yes/no questions, answer Yes when all asked "
    "conditions hold, and No when a condition is contradicted. Pay attention to negations and "
    "whether the question asks for similarity or difference. Do not treat different topics as a "
    "contradiction when the question explicitly asks whether they differ. For dates, compare the "
    "actual dates in the excerpts. If a necessary fact is absent, answer exactly: Insufficient "
    "Information. Return only the shortest direct answer in English, with no explanation or citations."
)


def generate(endpoint: str, messages: list[dict], max_tokens: int, timeout: int,
             model_name: str, temperature: float, top_p: float) -> dict:
    payload = {
        "model": model_name,
        "messages": messages,
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
        "seed": 42,
    }
    request = urllib.request.Request(
        endpoint.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def clean_answer(text: str, question: str) -> str:
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[-1]
    elif "<think>" in text:
        return ""
    answer = text.strip().splitlines()[0].strip() if text.strip() else ""
    answer = re.sub(r"^(?:Answer|Final answer)\s*:\s*", "", answer, flags=re.I)
    answer = answer.strip(' .,!?:;"\'')
    binary_question = re.match(r"^(?:Do|Does|Did|Is|Are|Was|Were|Has|Have|Can|Could|Would|Should)\b", question, flags=re.I)
    binary = re.match(r"^(Yes|No)\b", answer, flags=re.I)
    if binary_question and binary:
        return binary.group(1).title()
    return answer


def hardware_description() -> str:
    gpu = "unavailable"
    try:
        gpu = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            text=True, timeout=10,
        ).strip()
    except (OSError, subprocess.SubprocessError):
        pass
    cpu = platform.processor()
    ram = "unavailable"
    if os.name == "nt":
        try:
            import winreg
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                                r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as key:
                cpu = winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
            class MemoryStatus(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                    *[(name, ctypes.c_ulonglong) for name in (
                        "ullTotalPhys", "ullAvailPhys", "ullTotalPageFile", "ullAvailPageFile",
                        "ullTotalVirtual", "ullAvailVirtual", "ullAvailExtendedVirtual")],
                ]
            status = MemoryStatus()
            status.dwLength = ctypes.sizeof(status)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                ram = f"{status.ullTotalPhys / 1024 ** 3:.2f} GiB"
        except (OSError, AttributeError):
            pass
    return f"GPU: {gpu}; CPU: {cpu} ({os.cpu_count()} logical); RAM: {ram}; OS: {platform.platform()}"


def git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-c", f"safe.directory={Path.cwd().as_posix()}", "rev-parse", "HEAD"],
            text=True, timeout=10,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--model-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--endpoint", default="http://127.0.0.1:8080")
    parser.add_argument("--model-name", default="Qwen3-8B-Q4_K_M")
    parser.add_argument("--model-revision", default="Qwen GGUF 7c41481f57cb95916b40956ab2f0b139b296d974; base Qwen3-8B b968826d9c46dd6066d109eabc6255188de91218")
    parser.add_argument("--model-parameters", type=int, default=8200000000)
    parser.add_argument("--github-repository", default=None)
    parser.add_argument("--words-per-chunk", type=int, default=240)
    parser.add_argument("--overlap", type=int, default=40)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--no-think", action="store_true")
    parser.add_argument("--prompt-mode", choices=("entity", "comparison"), default="entity")
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--limit", type=int, default=0, help="Diagnostic run only; zero runs all questions")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    corpus = json.loads(args.corpus.read_text(encoding="utf-8"))
    question_data = json.loads(args.questions.read_text(encoding="utf-8"))
    if len(corpus) != 609 or len(question_data) != 2556:
        raise ValueError("Expected the full official corpus and all 2556 questions")
    chunks = chunk_corpus(corpus, args.words_per_chunk, args.overlap)
    index = BM25(chunks)
    record_path = args.output_dir / "execution_records.jsonl"
    existing = {}
    if record_path.exists():
        for line in record_path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            existing[row["question_id"]] = row
    run_meta_path = args.output_dir / "run_meta.json"
    if run_meta_path.exists():
        started_at = json.loads(run_meta_path.read_text(encoding="utf-8"))["started_at"]
    else:
        started_at = dt.datetime.now().astimezone().isoformat()
        run_meta_path.write_text(json.dumps({"started_at": started_at}) + "\n", encoding="utf-8")
    count = min(args.limit, len(question_data)) if args.limit else len(question_data)
    with record_path.open("a", encoding="utf-8") as output:
        for i in range(count):
            question_id = f"q{i:04d}"
            if question_id in existing:
                continue
            question = question_data[i]["query"]
            tic = time.perf_counter()
            retrieval_calls, selected = retrieve(index, question, args.top_k)
            context = context_text(selected)
            system_prompt = SYSTEM_PROMPT if args.prompt_mode == "entity" else PROMPT_COMPARE
            messages = [
                {"role": "system", "content": system_prompt + (" /no_think" if args.no_think else "")},
                {"role": "user", "content": f"Question: {question}\n\nRetrieved excerpts:\n{context}"},
            ]
            generation_calls, failures, answer = [], [], ""
            for attempt in range(1, 3):
                call_start = time.perf_counter()
                try:
                    response = generate(args.endpoint, messages, args.max_tokens, args.timeout,
                                        args.model_name, args.temperature, args.top_p)
                    raw = response["choices"][0]["message"]["content"] or ""
                    answer = clean_answer(raw, question)
                    if not answer:
                        raise ValueError("Model returned an empty answer")
                    usage = response.get("usage") or {}
                    generation_calls.append({
                        "model_name": args.model_name,
                        "actual_context": context,
                        "messages": messages,
                        "settings_overrides": {"temperature": args.temperature, "top_p": args.top_p,
                                               "max_tokens": args.max_tokens, "seed": 42,
                                               "no_think": args.no_think},
                        "output_text": raw,
                        "input_tokens": usage.get("prompt_tokens"),
                        "output_tokens": usage.get("completion_tokens"),
                        "elapsed_seconds": round(time.perf_counter() - call_start, 3),
                        "status": "success",
                        "error": None,
                    })
                    break
                except (OSError, ValueError, KeyError, IndexError) as error:
                    failures.append(str(error))
                    generation_calls.append({
                        "model_name": args.model_name,
                        "actual_context": context, "messages": messages,
                        "settings_overrides": {"temperature": args.temperature, "top_p": args.top_p,
                                               "max_tokens": args.max_tokens, "seed": 42,
                                               "no_think": args.no_think},
                        "output_text": "", "input_tokens": None, "output_tokens": None,
                        "elapsed_seconds": round(time.perf_counter() - call_start, 3),
                        "status": "failed", "error": str(error),
                    })
                    time.sleep(attempt)
            record = {
                "question_id": question_id,
                "question": question,
                "status": "success" if generation_calls[-1]["status"] == "success" else "failed",
                "rounds": [{"round": 1, "retrieval_calls": retrieval_calls, "generation_calls": generation_calls}],
                "model_answer": answer,
                "failures_and_retries": failures,
                "elapsed_seconds": round(time.perf_counter() - tic, 3),
            }
            output.write(json.dumps(record, ensure_ascii=False) + "\n")
            output.flush()
            existing[question_id] = record
            if i % 25 == 0 or i + 1 == count:
                print(f"{i + 1}/{count}: {question_id} -> {answer!r}", flush=True)
    if count != len(question_data):
        return
    rows = [existing[f"q{i:04d}"] for i in range(len(question_data))]
    predictions = [{"question_id": row["question_id"], "model_answer": row["model_answer"]} for row in rows]
    (args.output_dir / "predictions.json").write_text(json.dumps(predictions, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    calls = [call for row in rows for rnd in row["rounds"] for call in rnd["generation_calls"]]
    run_info = {
        "group_id": "Group07",
        "run_id": started_at,
        "github_repository": args.github_repository,
        "git_commit": git_commit(),
        "started_at": started_at,
        "finished_at": dt.datetime.now().astimezone().isoformat(),
        "dataset": {"name": "MultiHop-RAG", "question_count": len(question_data),
                    "corpus_document_count": len(corpus), "questions_file_sha256": sha256(args.questions),
                    "corpus_file_sha256": sha256(args.corpus)},
        "models": [{"role": "generator", "name": args.model_name,
                    "revision": args.model_revision,
                    "total_parameters": args.model_parameters, "precision_or_quantization": "Q4_K_M",
                    "weights_frozen": True, "training_details_if_any": "not applicable"}],
        "hardware": hardware_description(),
        "software_versions": {"python": platform.python_version(), "llama.cpp": "b11804 Vulkan"},
        "rag_method": "BM25 over overlapping article passages, clause queries, reciprocal-rank fusion, and one frozen-model answer call",
        "settings": {"chunking": {"words_per_chunk": args.words_per_chunk, "overlap_words": args.overlap,
                                  "chunk_count": len(chunks)},
                     "retrieval": {"algorithm": "BM25", "k1": 1.5, "b": 0.75, "top_k": args.top_k,
                                   "max_chunks_per_document": 2, "search_depth_per_query": 15},
                     "reranking": {"algorithm": "reciprocal rank fusion", "constant": 60},
                     "generation": {"temperature": args.temperature, "top_p": args.top_p,
                                    "max_tokens": args.max_tokens, "seed": 42,
                                    "no_think": args.no_think, "prompt_mode": args.prompt_mode},
                     "other": {"model_file_sha256": sha256(args.model_file)}},
        "prediction_command": subprocess.list2cmdline([sys.executable, *sys.argv]),
        "resource_usage": {"model_calls_by_model": {args.model_name: len(calls)},
                           "input_tokens": sum(call["input_tokens"] or 0 for call in calls),
                           "output_tokens": sum(call["output_tokens"] or 0 for call in calls),
                           "elapsed_seconds": sum(row["elapsed_seconds"] for row in rows),
                           "failed_question_count": sum(row["status"] == "failed" for row in rows),
                           "retry_count": sum(max(0, len(row["rounds"][0]["generation_calls"]) - 1) for row in rows),
                           "notes": "Token counts are from llama.cpp chat-completion usage; elapsed is sum of per-question wall times."},
    }
    (args.output_dir / "execution_record.json").write_text(
        json.dumps({"run_info": run_info, "question_records": rows}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
