#!/usr/bin/env python3
# 저자 검수 양식을 만들고 초안과 채택·수정 근거를 보존한 최종본을 내보낸다.
"""Prepare author review forms and publish a verified benchmark with an edit audit."""

import argparse
from pathlib import Path

from dataset_io import read_jsonl, write_json, write_jsonl
from schema import STYLE_KEYS
from semantic import text_key, unique_index
from validate_dataset import require, validate_records


def reviewed_references(draft, review):
    require(isinstance(review.get("reviewer"), str) and review["reviewer"].strip(),
            "A reviewer is required")
    require(review.get("verified") is True, "Author has not marked the review verified")
    originals = unique_index(draft, "style")
    decisions = unique_index(review["references"], "style")
    require(originals.keys() <= decisions.keys() <= set(STYLE_KEYS), "Every draft style needs a review decision")

    final = []
    for style in STYLE_KEYS:
        if style not in decisions:
            continue
        entry = decisions[style]
        original = originals[style]["text"] if style in originals else None
        # 양식을 만든 뒤 초안이 바뀌었다면 이전 판정을 적용할 수 없다.
        require("original_text" in entry and entry["original_text"] == original,
                f"{style}: draft has changed since review was prepared")
        decision, text = entry.get("decision"), entry.get("text")
        if decision == "reject":
            require(original is not None and text is None, f"{style}: reject requires a draft and null text")
            continue
        require(isinstance(text, str) and text.strip(), f"{style}: final text is required")
        require(
            (decision == "keep" and original == text)
            or (decision == "edit" and original is not None and text_key(original) != text_key(text))
            or (decision == "add" and original is None),
            f"{style}: invalid keep/edit/add decision",
        )
        final.append({"style": style, "text": text})
    require(2 <= len(final) <= 3, "Expected 2–3 accepted references")
    require(len({text_key(ref["text"]) for ref in final}) == len(final), "Duplicate final references")
    return final


def prepare(records):
    # 저자가 의미를 확인할 수 있도록 정답 구조와 문체별 원문을 양식에 넣는다.
    for record in records:
        if not record["expect_supported"]:
            continue
        yield {
            "id": record["id"], "reviewer": "", "verified": False,
            "gold_tuples": record["gold_tuples"],
            "references": [
                {"style": ref["style"], "original_text": ref["text"], "decision": "pending",
                 "text": ref["text"], "note": ""}
                for ref in record["references"]
            ],
        }


def finalize(records, reviews, *, partial=False):
    by_id = unique_index(reviews, "id")
    supported = {record["id"] for record in records if record["expect_supported"]}
    require(by_id.keys() == supported, "Review must cover exactly the supported scenario IDs")
    final = []
    for record in records:
        if record["id"] not in by_id:
            # 미지원 시나리오는 참조 대사 없이도 최종 분할에 포함한다.
            final.append(record)
            continue
        review = by_id[record["id"]]
        if "gold_tuples" in review:
            require(review["gold_tuples"] == record["gold_tuples"], f"{record['id']}: review graph has changed")
        try:
            refs = reviewed_references(record["references"], review)
        except (ValueError, TypeError, KeyError, AttributeError) as exc:
            raise ValueError(f"{record['id']}: {exc}") from exc
        audit = {key: review[key] for key in ("reviewer", "verified", "references")}
        final.append({**record, "draft_references": record["references"], "references": refs,
                      "author_review": audit, "author_verified": True})
    validate_records(final, "final", partial=partial)
    return final


def review_stats(records):
    verified = [record for record in records if record["expect_supported"]]
    decisions = [entry["decision"] for record in verified for entry in record["author_review"]["references"]]
    refs = [ref["text"] for record in verified for ref in record["references"]]
    # 저자가 새로 쓴 대사는 초안 대비 수정률의 분모에서 뺀다.
    original_count = len(decisions) - decisions.count("add")
    return {
        "verified_scenarios": len(verified), "draft_references": original_count, "final_references": len(refs),
        "edited": decisions.count("edit"), "rejected": decisions.count("reject"), "added": decisions.count("add"),
        "edit_rate": decisions.count("edit") / original_count if original_count else None,
        "mean_chars": sum(map(len, refs)) / len(refs),
        "mean_eojeol": sum(len(text.split()) for text in refs) / len(refs),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    for name in ("prepare", "finalize"):
        command = commands.add_parser(name)
        command.add_argument("--input", type=Path, required=True)
        command.add_argument("--output", type=Path, required=True)
    command = commands.choices["finalize"]
    command.add_argument("--review", type=Path, required=True)
    command.add_argument("--stats-out", type=Path)
    args = parser.parse_args()

    try:
        records = read_jsonl(args.input)
        validate_records(records, "synthesis")
        paths = [args.input.resolve(), args.output.resolve()]
        if args.action == "prepare":
            require(len(set(paths)) == len(paths), "Review output must differ from dataset input")
            require(not args.output.exists(), "Review form already exists")
            write_jsonl(args.output, prepare(records))
        else:
            stats_path = args.stats_out or args.output.with_name(args.output.stem + "_stats.json")
            paths.extend([args.review.resolve(), stats_path.resolve()])
            require(len(set(paths)) == len(paths), "Dataset, review and stats paths must differ")
            final = finalize(records, read_jsonl(args.review))
            write_jsonl(args.output, final)
            write_json(stats_path, review_stats(final))
        print(f"[Step 4] {args.output}")
    except (ValueError, OSError) as exc:
        parser.exit(1, f"검수 실패: {exc}\n")


if __name__ == "__main__":
    main()
