#!/usr/bin/env python3
# 독립적으로 역추출한 사건과 역할을 그래프 정답과 비교해 점수를 집계한다.
"""Score independently extracted predictions against graph gold."""

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from dataset_io import read_jsonl, validate_paths, write_json, write_jsonl
from semantic import comparison_errors, unique_index
from step2_3_openai_synthesize import RT_SYSTEM_PROMPT, format_roundtrip_prompt
from validate_dataset import require, validate_records


def prepare(records, predictions):
    by_id = unique_index(predictions, "id")
    require(by_id.keys() == {record["id"] for record in records}, "Prediction IDs must match the input split")
    for record in records:
        prediction = by_id[record["id"]]
        if prediction.get("status") == "unsupported":
            yield {"id": record["id"], "status": "unsupported"}
            continue
        text = prediction.get("text")
        require(isinstance(text, str) and text.strip(), f"{record['id']}: missing text or unsupported status")
        # 독립 추출용 입력은 모델 발화와 개체 표기로 구성한다.
        yield {"id": record["id"], "system_prompt": RT_SYSTEM_PROMPT,
               "user_prompt": format_roundtrip_prompt(text, record)}


def score(records, predictions):
    by_id = unique_index(predictions, "id")
    require(by_id.keys() == {record["id"] for record in records}, "Extraction IDs must match the input split")
    rows, buckets, pairs = [], defaultdict(list), defaultdict(list)
    binding_errors = {"missing_event", "hallucinated_event", "missing_role", "hallucinated_role",
                      "role_swap", "role_value_mismatch", "invalid_extraction", "unexpected_unsupported"}
    for record in records:
        output = by_id[record["id"]]
        refusal = output.get("status") == "unsupported"
        if refusal:
            require("extraction" not in output, f"{record['id']}: unsupported row has an extraction")
        else:
            require(isinstance(output.get("extraction"), dict), f"{record['id']}: extraction object is required")
        # 미지원 사례의 판단 기준은 모델이 미지원으로 반환했는지 여부다.
        if not record["expect_supported"]:
            errors = [] if refusal else ["should_be_unsupported"]
        elif refusal:
            errors = ["unexpected_unsupported"]
        else:
            addressee_id = record["utterance_goal"]["addressee"]
            addressee = next(ent["surface"] for ent in record["input_graph"]["entities"] if ent["id"] == addressee_id)
            errors = comparison_errors(record["gold_tuples"], output["extraction"], addressee)
        flags = set(errors)
        row = {"id": record["id"], "size": len(record["utterance_goal"]["target_events"]),
               "supported": record["expect_supported"], "exact": not errors, "errors": errors}
        if row["supported"]:
            row.update({
                "binding_exact": not bool(flags & binding_errors),
                "role_swap": "role_swap" in flags,
                "omission": bool(flags & {"missing_event", "missing_role"}),
                "hallucination": bool(flags & {"hallucinated_event", "hallucinated_role", "extra_fact"}),
                "speech_act_match": not bool(flags & {"speech_act_mismatch", "invalid_extraction", "unexpected_unsupported"}),
                "speech_level_match": not bool(flags & {"speech_level_mismatch", "invalid_extraction", "unexpected_unsupported"}),
            })
        rows.append(row)
        buckets[(record["axis"], row["size"])].append(row)
        if record.get("pair_id"):
            pairs[record["pair_id"]].append((record.get("pair_variant"), row))

    complete_pairs = [
        members for members in pairs.values()
        if len(members) == 2 and {variant for variant, _ in members} == {"a", "b"}
        and all(row["supported"] for _, row in members)
    ]

    def summarize(group):
        supported = [row for row in group if row["supported"]]
        unsupported = [row for row in group if not row["supported"]]
        # 의미 정확도와 오류율의 분모는 지원 사례이며 미지원 탐지율은 별도로 계산한다.
        rates = {}
        for key in ("binding_exact", "role_swap", "omission", "hallucination", "speech_act_match", "speech_level_match"):
            rates[f"{key}_rate"] = sum(row[key] for row in supported) / len(supported) if supported else None
        return {
            "scenarios": len(group), "supported": len(supported), "unsupported": len(unsupported),
            "exact_rate": sum(row["exact"] for row in supported) / len(supported) if supported else None,
            "unsupported_detection_rate": sum(row["exact"] for row in unsupported) / len(unsupported) if unsupported else None,
            **rates, "errors": dict(Counter(error for row in group for error in row["errors"])),
        }

    return {
        "overall": summarize(rows),
        "by_axis_and_size": [
            {"axis": axis, "size": size, **summarize(group)} for (axis, size), group in sorted(buckets.items())
        ],
        "contrast_pairs": {
            "total": len(complete_pairs),
            "both_bindings_exact": sum(all(row["binding_exact"] for _, row in pair) for pair in complete_pairs),
            "both_exact": sum(all(row["exact"] for _, row in pair) for pair in complete_pairs),
        },
        "rows": rows,
    }


def prediction_status(prediction):
    status = prediction.get("status", "ok")
    text = prediction.get("text")
    if status not in ("ok", "empty", "unsupported"):
        raise ValueError(f'{prediction["id"]}: unknown prediction status {status!r}')
    if text is not None and not isinstance(text, str):
        raise ValueError(f'{prediction["id"]}: prediction text must be a string')
    has_text = bool(text and text.strip())
    if status != "ok" and has_text:
        raise ValueError(f'{prediction["id"]}: {status} prediction contains a nonempty text')
    if status == "unsupported":
        return status
    return "ok" if has_text else "empty"


def paired_scenarios(records):
    pairs = defaultdict(dict)
    for record in records:
        pair_id = record.get("pair_id")
        if pair_id is None:
            continue
        if not isinstance(pair_id, str) or not pair_id.strip():
            raise ValueError(f'{record["id"]}: invalid pair ID')
        variant = record.get("pair_variant")
        if not record["expect_supported"] or variant not in ("a", "b") or variant in pairs[pair_id]:
            raise ValueError(f'{record["id"]}: invalid or duplicate variant for pair {pair_id}')
        pairs[pair_id][variant] = record["id"]
    return pairs


def select_records(ref_records, pred_records, partial=False):
    by_id = unique_index(ref_records, "id")
    predictions = unique_index(pred_records, "id")
    if not by_id or not predictions:
        raise ValueError("Reference and prediction files must each contain at least one scenario")
    foreign_ids = sorted(predictions.keys() - by_id.keys())
    if foreign_ids:
        raise ValueError(f"Unknown prediction IDs ({len(foreign_ids)}): {foreign_ids[:3]}")
    missing_ids = sorted(by_id.keys() - predictions.keys())
    if missing_ids and not partial:
        raise ValueError(f"Missing prediction IDs ({len(missing_ids)}): {missing_ids[:3]}")
    # Validate the reference pair metadata even when only one variant is evaluated.
    paired_scenarios(ref_records)
    return [record for record in ref_records if record["id"] in predictions], predictions


def prepare_jobs(records, predictions):
    by_id = unique_index(predictions, "id")
    if by_id.keys() != unique_index(records, "id").keys():
        raise ValueError("Prediction IDs must match the evaluated scenarios")
    statuses = {sid: prediction_status(pred) for sid, pred in by_id.items()}
    with_text = [record for record in records if statuses[record["id"]] == "ok"]
    jobs = {
        job["id"]: job for job in prepare(with_text, [by_id[record["id"]] for record in with_text])
    }
    for record in records:
        sid = record["id"]
        job = jobs.get(sid, {"id": sid, "status": statuses[sid]})
        yield {**job, "input_sha256": job_input_sha256(job)}


def job_input_sha256(job):
    """Hash the extraction job payload without its stored digest."""
    return _fingerprint({field: value for field, value in job.items() if field != "input_sha256"})


def _fingerprint(payload):
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def score_input_sha256(record, job):
    """Bind an automatic score to the extraction job and the answer it was scored against."""
    return _fingerprint({
        "input_sha256": job["input_sha256"],
        "expect_supported": record["expect_supported"],
        "gold_tuples": record["gold_tuples"],
    })


def evaluate_extractions(records, predictions, extractions):
    jobs = {job["id"]: job for job in prepare_jobs(records, predictions)}
    rows = unique_index(extractions, "id")
    if rows.keys() != jobs.keys():
        raise ValueError("Extraction IDs must match the generated predictions")
    texts = {row["id"]: row.get("text") for row in predictions}

    verified = []
    for sid, row in rows.items():
        job = jobs[sid]
        status = job.get("status", "ok")
        if row.get("status", "ok") != status:
            raise ValueError(f"{sid}: extraction status does not match the prediction")
        if row.get("input_sha256") != job["input_sha256"]:
            raise ValueError(f"{sid}: extraction input_sha256 does not match the job for this prediction")
        if "text" in row and row["text"] != texts[sid]:
            raise ValueError(f"{sid}: extraction text does not match the prediction")
        if status != "ok" and "extraction" in row:
            raise ValueError(f"{sid}: {status} prediction cannot have a semantic extraction")

        if status == "unsupported":
            verified.append({"id": sid, "status": "unsupported"})
        elif status == "empty":
            # The model supplied no events; there was no extraction job to fail.
            verified.append({"id": sid, "extraction": {
                "speech_act": "unknown", "speech_level": "unknown", "events": [], "discourse": [], "extra_facts": [],
            }})
        else:
            verified.append({"id": sid, "extraction": row.get("extraction")})

    paired_scenarios(records)
    report = score(records, verified)
    for record, row in zip(records, report["rows"]):
        row["score_input_sha256"] = score_input_sha256(record, jobs[row["id"]])
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "score"))
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True, help="Original model predictions JSONL")
    parser.add_argument("--extractions", type=Path, help="Completed extraction JSONL retaining input_sha256")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--extractor", help="Name and revision of the independent human or local extractor")
    parser.add_argument("--partial", action="store_true", help="Allow evaluating a subset of scenarios")
    args = parser.parse_args()
    try:
        validate_paths((args.input, args.predictions, args.extractions), (args.output,))
        if args.action == "score":
            require(args.extractions is not None, "--extractions is required for scoring")
            require(args.extractor and args.extractor.strip(), "--extractor is required for scoring")
        else:
            require(args.extractions is None and args.extractor is None, "--extractions and --extractor require score")

        input_records, predictions = read_jsonl(args.input), read_jsonl(args.predictions)
        validate_records(input_records, partial=args.partial, allow_incomplete_pairs=args.partial)
        records, _ = select_records(input_records, predictions, args.partial)
        if args.action == "prepare":
            write_jsonl(args.output, prepare_jobs(records, predictions))
        else:
            report = evaluate_extractions(records, predictions, read_jsonl(args.extractions))
            write_json(args.output, {"extractor": args.extractor, **report})
            print(json.dumps(report["overall"], ensure_ascii=False, indent=2))
    except (ValueError, OSError) as exc:
        parser.exit(1, f"채점 실패: {exc}\n")


if __name__ == "__main__":
    main()
