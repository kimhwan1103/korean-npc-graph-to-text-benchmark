#!/usr/bin/env python3
# 독립적으로 역추출한 사건과 역할을 그래프 정답과 비교해 점수를 집계한다.
"""Score independently extracted pilot or benchmark predictions against graph gold."""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from dataset_io import read_jsonl, write_json, write_jsonl
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
        if record["pair_id"]:
            pairs[record["pair_id"]].append(row["binding_exact"])

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
        "contrast_pairs": {"total": len(pairs), "both_bindings_exact": sum(all(pair) for pair in pairs.values())},
        "rows": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "score"))
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--extractor", help="Name and revision of the independent human or local extractor")
    parser.add_argument("--partial", action="store_true")
    args = parser.parse_args()
    try:
        require(len({path.resolve() for path in (args.input, args.predictions, args.output)}) == 3,
                "Input, predictions and output must differ")
        records, predictions = read_jsonl(args.input), read_jsonl(args.predictions)
        validate_records(records, partial=args.partial)
        if args.action == "prepare":
            write_jsonl(args.output, prepare(records, predictions))
        else:
            require(args.extractor and args.extractor.strip(), "--extractor is required for scoring")
            report = score(records, predictions)
            write_json(args.output, {"extractor": args.extractor, **report})
            print(json.dumps(report["overall"], ensure_ascii=False, indent=2))
    except (ValueError, OSError) as exc:
        parser.exit(1, f"채점 실패: {exc}\n")


if __name__ == "__main__":
    main()
