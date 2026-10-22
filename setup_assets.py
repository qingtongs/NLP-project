"""Download the pinned public data, model, and Windows inference runtime."""

from __future__ import annotations

import argparse
import hashlib
import shutil
import urllib.request
import zipfile
from pathlib import Path


ASSETS = {
    "questions": (
        "https://huggingface.co/datasets/yixuantt/MultiHopRAG/resolve/71ac0d0bd1f951d2d6b70311f7d2ae404e1ffa82/MultiHopRAG.json",
        Path("NLP_官方评测教程/eval_tutorial/MultiHopRAG.json"),
        "03cfb4926461f868684903aadc8024447bdda5bb3f6804741424cce338515bff",
    ),
    "corpus": (
        "https://huggingface.co/datasets/yixuantt/MultiHopRAG/resolve/71ac0d0bd1f951d2d6b70311f7d2ae404e1ffa82/corpus.json",
        Path("corpus.json"),
        "20b61b5ab84de84a927420c5d265b7ec8d859ae49980699958a787ade9e4d28f",
    ),
    "evaluator": (
        "https://raw.githubusercontent.com/yixuantt/MultiHop-RAG/c1c1287aa60a94acf9c4d20c891c9cd611a0f6e8/qa_evaluate.py",
        Path("NLP_官方评测教程/eval_tutorial/qa_evaluate.py"),
        "a518bc37d94d2e84cc9012b41a758567e308dceea92aba14cda44a1364626ae7",
    ),
    "model": (
        "https://huggingface.co/Qwen/Qwen3-8B-GGUF/resolve/7c41481f57cb95916b40956ab2f0b139b296d974/Qwen3-8B-Q4_K_M.gguf",
        Path("models/Qwen3-8B-Q4_K_M.gguf"),
        "d98cdcbd03e17ce47681435b5150e34c1417f50b5c0019dd560e4882c5745785",
    ),
    "runtime": (
        "https://github.com/ggml-org/llama.cpp/releases/download/b11804/llama-b11804-bin-win-vulkan-x64.zip",
        Path("runtime/llama-b11804-bin-win-vulkan-x64.zip"),
        "db9ab31997e93bf713d3dfec6eb1cf1b5a6a5f2940a3d4d111ad0ffab54c87d5",
    ),
}


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def download(name: str) -> None:
    url, path, expected = ASSETS[name]
    if path.exists():
        actual = file_hash(path)
        if actual != expected:
            raise ValueError(f"Checksum mismatch for {path}: {actual}")
        print(f"Verified {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    print(f"Downloading {name} to {path}", flush=True)
    request = urllib.request.Request(url, headers={"User-Agent": "MultiHop-RAG-course-project"})
    with urllib.request.urlopen(request, timeout=600) as response, temporary.open("wb") as target:
        shutil.copyfileobj(response, target, length=1 << 20)
    actual = file_hash(temporary)
    if actual != expected:
        raise ValueError(f"Checksum mismatch for {name}: {actual}")
    temporary.replace(path)
    print(f"Saved {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("names", nargs="*", choices=tuple(ASSETS))
    args = parser.parse_args()
    names = args.names or list(ASSETS)
    for name in names:
        download(name)
    if "runtime" in names:
        destination = Path("runtime/llama-b11804")
        if not destination.exists():
            with zipfile.ZipFile(ASSETS["runtime"][1]) as archive:
                archive.extractall(destination)
            print(f"Extracted {destination}")


if __name__ == "__main__":
    main()
