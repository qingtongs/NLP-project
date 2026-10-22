"""Package the non-source course deliverables as Group07_NLP_RAG.zip."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--record", type=Path, required=True)
    parser.add_argument("--qa-submission", type=Path, required=True)
    parser.add_argument("--official-scores", type=Path, required=True)
    parser.add_argument("--raw-official-scores", type=Path)
    parser.add_argument("--report-docx", type=Path, required=True)
    parser.add_argument("--report-pdf", type=Path, required=True)
    parser.add_argument("--report-payload", type=Path, required=True)
    parser.add_argument("--output-parent", type=Path, default=Path("."))
    args = parser.parse_args()
    folder = args.output_parent / "Group07_NLP_RAG"
    archive = args.output_parent / "Group07_NLP_RAG.zip"
    if folder.exists() or archive.exists():
        raise FileExistsError("Package output already exists; choose a fresh output parent")
    mapping = {
        "predictions.json": args.predictions,
        "execution_record/execution_record.json": args.record,
        "evaluation/qa_submission.json": args.qa_submission,
        "evaluation/official_scores.txt": args.official_scores,
        "report/Group07_project_report.docx": args.report_docx,
        "report/Group07_project_report.pdf": args.report_pdf,
    }
    if args.raw_official_scores:
        mapping["evaluation/raw_official_scores.txt"] = args.raw_official_scores
    files = {}
    for relative, source in mapping.items():
        target = folder / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        files[relative] = {"bytes": target.stat().st_size, "sha256": sha256(target)}
    payload = json.loads(args.report_payload.read_text(encoding="utf-8"))
    (folder / "evaluation/scores.json").write_text(
        json.dumps(payload["scores"], ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    scores_path = folder / "evaluation/scores.json"
    files["evaluation/scores.json"] = {"bytes": scores_path.stat().st_size,
                                        "sha256": sha256(scores_path)}
    manifest = {"group_id": "Group07", "member": payload["member_name"],
                "student_id": payload["student_id"], "github_repository": payload["github_url"],
                "files": files}
    (folder / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with ZipFile(archive, "w", compression=ZIP_DEFLATED, compresslevel=6) as output:
        for path in sorted(folder.rglob("*")):
            if path.is_file():
                output.write(path, (Path("Group07_NLP_RAG") / path.relative_to(folder)).as_posix())
    print(f"Created {archive} ({archive.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
