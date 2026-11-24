#!/usr/bin/env python3
"""그래프 생성부터 합성과 정량 평가까지 한 번에 실행한다."""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from dataset_io import read_jsonl, write_json, write_jsonl
from evaluate_synthetic_quality import evaluate_dataset, print_summary
from review_dataset import finalize, prepare as prepare_review, review_stats
from schema import SPLIT_COUNTS
from score_outputs import prepare as prepare_extraction, score
from step1_generate_graphs import generate_records
from step2_3_openai_synthesize import PIPELINE_VERSION, add_synthesis_options, run_lock, synthesize, validate_synthesis_options
from validate_dataset import validate_records


def run_pipeline(args):
    records = generate_records(args.split, args.seed)
    split, pair_count = validate_records(records)
    axis_counts = dict(Counter(record["axis"] for record in records))
    print(f"[Step 1] {split} 그래프 {len(records)}개 · 대조쌍 {pair_count}개 검증 | {axis_counts}")
    if args.dry_run:
        synthesize(records, args)
        return

    run_dir = args.run_dir or Path(__file__).resolve().parent / "data" / "runs" / f"{split}_seed{args.seed}_v{PIPELINE_VERSION}"
    graph_path = run_dir / "graphs.jsonl"
    draft_path = run_dir / "synthesized.jsonl"
    stats_path = run_dir / "synthesis_stats.json"
    report_path = run_dir / "report.json"
    final_path = run_dir / "final.jsonl"
    jobs_path = run_dir / "extraction_jobs.jsonl"
    lock_path = run_dir / ".pipeline.lock"
    reserved = {
        path.resolve() for path in (
            graph_path, draft_path, stats_path, report_path, final_path, jobs_path, lock_path,
            run_dir / "synthesized_state.json", run_dir / "synthesized_state.json.lock",
        )
    }
    for source in (args.author_review, args.predictions, args.extractions):
        if source and source.resolve() in reserved:
            raise ValueError(f"입력 파일과 실행 산출물의 경로가 겹칩니다. {source}")
    reviews = read_jsonl(args.author_review) if args.author_review else None
    predictions = read_jsonl(args.predictions) if args.predictions else None
    extractions = read_jsonl(args.extractions) if args.extractions else None

    with run_lock(lock_path):
        if graph_path.exists():
            if read_jsonl(graph_path) != records:
                raise ValueError("실행 폴더에 다른 그래프가 있습니다. 새 --run-dir을 지정하세요.")
        else:
            write_jsonl(graph_path, records)

        args.input, args.output, args.stats_out = graph_path, draft_path, stats_path
        drafts, stats = synthesize(records, args)
        partial = len(drafts) < len(records)
        validate_records(drafts, "synthesis", partial=partial)
        print(f"[Step 3] 합성 초안 {len(drafts)}개 검증 완료")

        report = {
            "split": split,
            "seed": args.seed,
            "scope": {
                "graph_scenarios": len(records), "graph_axes": axis_counts, "graph_pairs": pair_count,
                "requested_limit": args.limit, "evaluated_scenarios": len(drafts), "partial": partial,
            },
            "synthesis": {
                key: stats[key] for key in (
                    "api_endpoint", "gen_model", "rt_model", "generation_request_settings",
                    "roundtrip_request_settings", "models_resolved",
                    "scenario_statuses", "total_generated_texts", "total_candidates_checked", "first_try_style_slots",
                    "first_try_checked", "first_try_passed", "first_try_pass_rate", "roundtrip_first_try_pass_rate",
                    "final_passed", "rejection_reasons", "tokens",
                )
            },
            "draft_quality": evaluate_dataset(draft_path, records=drafts),
            "artifacts": {"graphs": graph_path.name, "drafts": draft_path.name, "synthesis_stats": stats_path.name},
        }
        supported = [row for row in drafts if row["expect_supported"]]
        ready = sum(len(row["references"]) >= 2 for row in supported)
        report["scope"]["drafts_with_at_least_two_references"] = ready
        print(f"[Step 4] 참조가 2개 이상인 지원 시나리오 {ready}/{len(supported)}개")

        if reviews is None:
            template = list(prepare_review(drafts))
            # 초안이 달라지면 새 양식을 만들고, 이미 작성 중인 양식은 보존한다.
            digest = hashlib.sha256(json.dumps(template, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:12]
            template_path = run_dir / f"author_review_{digest}.jsonl"
            if not template_path.exists():
                write_jsonl(template_path, template)
            report["author_review"] = {"status": "pending"}
            report["artifacts"]["author_review_template"] = template_path.name
            print(f"[Step 4] 저자 검수 양식 {template_path}")
        else:
            final = finalize(drafts, reviews, partial=partial)
            write_jsonl(final_path, final)
            report["author_review"] = {"status": "verified", **review_stats(final)}
            report["final_quality"] = evaluate_dataset(final_path, records=final)
            report["artifacts"]["final"] = final_path.name
            print(f"[Step 4] 저자 검수 최종본 {final_path}")

        if predictions is not None:
            write_jsonl(jobs_path, prepare_extraction(drafts, predictions))
            report["artifacts"]["extraction_jobs"] = jobs_path.name
            print(f"[독립 역추출] 작업 파일 {jobs_path}")
        if extractions is not None:
            report["independent_scores"] = {"extractor": args.extractor, **score(drafts, extractions)}
            print(f"[독립 채점] {json.dumps(report['independent_scores']['overall'], ensure_ascii=False)}")

        write_json(report_path, report)
        if report["draft_quality"]["counts"]["total_verified_references"]:
            print_summary(report["draft_quality"])
        else:
            print("[정량 평가] 합성 초안의 통과 참조가 0개입니다. 탈락 사유는 리포트에 기록했습니다.")
        if "final_quality" in report:
            print_summary(report["final_quality"])
        print(f"\n[완료] 통합 정량 리포트 {report_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=SPLIT_COUNTS, default="pilot30")
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--run-dir", type=Path, help="그래프, 합성 결과, 체크포인트, 리포트를 저장할 폴더")
    parser.add_argument("--author-review", type=Path, help="저자가 작성한 검수 JSONL")
    parser.add_argument("--predictions", type=Path, help="독립 역추출 작업을 만들 실험 모델 발화 JSONL")
    parser.add_argument("--extractions", type=Path, help="독립 채점에 사용할 역추출 결과 JSONL")
    parser.add_argument("--extractor", help="독립 역추출기의 이름과 버전")
    add_synthesis_options(parser)
    args = parser.parse_args()
    validate_synthesis_options(parser, args)
    if args.extractions and not (args.extractor and args.extractor.strip()):
        parser.error("--extractions를 사용할 때는 --extractor도 지정하세요.")
    try:
        run_pipeline(args)
    except (ValueError, OSError, RuntimeError) as exc:
        parser.exit(1, f"실행 실패: {exc}\n")


if __name__ == "__main__":
    main()
