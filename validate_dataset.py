#!/usr/bin/env python3
# 그래프, 합성 초안, 저자 검수 최종본의 데이터 규약을 검사한다.
"""Validate complete graph, synthesis, and author-verified benchmark datasets."""

import argparse
from collections import Counter, defaultdict
from itertools import product
from pathlib import Path

from dataset_io import read_jsonl
from schema import (
    AXIS1, AXIS2, AXIS3, AXIS4, AXIS2_PAIR_COUNTS, AXIS4_STRESS_COUNTS, BOUNDARIES,
    CHARACTERS, CODAS, DIAG_DISCOURSE, DIAG_ROLE, FOCAL_ROLES, FRAME_OPTIONAL_ROLES,
    FRAME_REQUIRED_ROLES, ITEMS, LOCATIONS, MAX_DISCOURSE_DEPTH, PAIR_SWAP_ROLES,
    PLAYER_ENTITY, ROLE_TYPES, SCHEMA_VERSION, SPEECH_ACTS, SPEECH_LEVELS,
    SPLIT_COUNTS, STYLE_KEYS, SUPPORTED_FRAMES, UNSUPPORTED_FRAMES, compute_coda,
    derive_gold,
)
from semantic import comparison_errors, discourse_tree, event_signatures, text_key, unique_index


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate_lexicon():
    for pool, count in ((CHARACTERS, 6), (ITEMS, 8), (LOCATIONS, 4)):
        require(Counter(entity.coda for entity in pool) == dict.fromkeys(CODAS, count),
                "Lexicon coda imbalance")
        require(all(compute_coda(entity.surface) == entity.coda for entity in pool),
                "Incorrect lexicon coda")


def infer_boundary(events, relations, act):
    # 상충 화행 표시는 따로 검사하고, 나머지 담화 구조에서 트리의 깊이를 구한다.
    conflicts = [rel["conflicting_acts"] for rel in relations if "conflicting_acts" in rel]
    clean = [{key: value for key, value in rel.items() if key != "conflicting_acts"} for rel in relations]
    _, depth = discourse_tree({event["id"]: ("event", event["id"]) for event in events}, clean)
    boundary = []
    if any(event["frame"] not in FRAME_REQUIRED_ROLES for event in events):
        boundary.append("unsupported_frame")
    if any(set(FRAME_REQUIRED_ROLES.get(event["frame"], ())) - event["roles"].keys() for event in events):
        boundary.append("missing_required_role")
    if depth > MAX_DISCOURSE_DEPTH:
        boundary.append("excessive_nesting_depth")
    if conflicts:
        acts, = conflicts
        require(isinstance(acts, list) and len(acts) >= 2 and len(set(acts)) == len(acts)
                and set(acts) <= set(SPEECH_ACTS) and act == "+".join(acts), "Invalid conflicting acts")
        boundary.append("conflicting_speech_acts")
    else:
        require(act in SPEECH_ACTS, "Invalid speech act")
    return boundary


def validate_graph(record):
    require(record.get("schema_version") == SCHEMA_VERSION, "Regenerate Step 1 graphs for schema v2")
    split = record["split"]
    require(split in SPLIT_COUNTS and record["axis"] in SPLIT_COUNTS[split], "Unexpected split or axis")
    require(record["id"].startswith(split + "_") and type(record["seed"]) is int, "Invalid ID or seed")
    require(type(record["expect_supported"]) is bool and isinstance(record["subtype"], str),
            "Invalid support flag or subtype")

    graph, goal = record["input_graph"], record["utterance_goal"]
    entities = unique_index(graph["entities"], "id")
    for entity in entities.values():
        typ = entity["type"]
        require(typ in {"character", "item", "location"}, "Unknown entity type")
        require(entity["animacy"] == ("animate" if typ == "character" else "inanimate"), "Wrong animacy")
        require(entity["coda"] == compute_coda(entity["surface"]), "Wrong entity coda")
    require(len({entity["surface"] for entity in entities.values()}) == len(entities), "Ambiguous entity surfaces")

    for role in ("speaker", "addressee"):
        require(goal[role] in entities and entities[goal[role]]["type"] == "character", f"Invalid {role}")
    require(goal["speech_level"] in SPEECH_LEVELS, "Invalid speech level")

    events = unique_index(graph["events"], "id")
    targets = goal["target_events"]
    require(isinstance(targets, list) and targets and len(set(targets)) == len(targets)
            and set(targets) <= events.keys(), "Invalid target events")
    for event in events.values():
        frame, roles = event["frame"], event["roles"]
        require(isinstance(roles, dict), "Invalid event roles")
        if frame in FRAME_REQUIRED_ROLES:
            allowed = FRAME_REQUIRED_ROLES[frame] + FRAME_OPTIONAL_ROLES.get(frame, ())
            require(roles.keys() <= set(allowed), f"Unexpected role for {frame}")
        else:
            require(frame in UNSUPPORTED_FRAMES, f"Unknown frame {frame}")
        for role, value in roles.items():
            require(value in entities, f"Dangling entity {value!r}")
            require(ROLE_TYPES.get(role) == entities[value]["type"], f"Invalid entity type for {role}")

    relations = goal["discourse"]
    require(isinstance(relations, list) and all(isinstance(rel, dict) for rel in relations), "Invalid discourse")
    # 저장된 지원 여부를 그대로 신뢰하지 않고 대상 사건과 담화 구조에서 다시 판정한다.
    boundary = infer_boundary([events[target] for target in targets], relations, goal["speech_act"])
    label = record["unsupported_reason"]
    if record["expect_supported"]:
        require(not boundary and label is None, "Supported label conflicts with graph")
    else:
        require(record["axis"] == AXIS4 and boundary == [label], "Unsupported label conflicts with graph")

    # 그래프와 발화 목적에서 정답을 다시 만들어 저장된 정답과 같은지 확인한다.
    require(record["gold_tuples"] == derive_gold(graph, goal), "Gold does not match graph and goal")
    if record["axis"] in {AXIS2, DIAG_ROLE, DIAG_DISCOURSE}:
        require(isinstance(record.get("pair_id"), str) and record["pair_id"].startswith(split + "_pair_")
                and record.get("pair_variant") in {"a", "b"}, "Contrastive axis requires a split-scoped pair and variant")
    else:
        require(record.get("pair_id") is None and record.get("pair_variant") is None, "Unexpected pair metadata")


def validate_pair(rows):
    variants = unique_index(rows, "pair_variant")
    require(set(variants) == {"a", "b"}, "Expected variants a and b")
    a, b = variants["a"], variants["b"]
    common_fields = ("split", "seed", "axis", "subtype", "expect_supported")
    require(all(a[key] == b[key] for key in common_fields), "Pair differs in metadata")
    require(unique_index(a["input_graph"]["entities"], "id") == unique_index(b["input_graph"]["entities"], "id"),
            "Pair has different entities")
    events_a = unique_index(a["input_graph"]["events"], "id")
    events_b = unique_index(b["input_graph"]["events"], "id")
    require(events_a.keys() == events_b.keys() == set(a["utterance_goal"]["target_events"]), "Pair has different events")

    if a["axis"] == DIAG_ROLE:
        require(a["utterance_goal"] == b["utterance_goal"], "Role-direction pair differs in utterance_goal")
        require(len(events_a) == 1, "Diag_RoleDirection requires one event")
        ev_a, ev_b = next(iter(events_a.values())), next(iter(events_b.values()))
        require(ev_a["frame"] == ev_b["frame"] == "Deliver", "Diag_RoleDirection requires Deliver")
        ra, rb = ev_a["roles"], ev_b["roles"]
        require(ra.keys() == rb.keys(), "Role-direction pair changed role schema")
        require(ra["Agent"] != PLAYER_ENTITY.id and ra["Recipient"] != PLAYER_ENTITY.id, "Diag_RoleDirection requires non-player NPCs")
        require(ra["Agent"] == rb["Recipient"] and ra["Recipient"] == rb["Agent"] and ra["Agent"] != ra["Recipient"],
                "Role-direction pair must flip Agent and Recipient")
        for role in ra.keys() - {"Agent", "Recipient"}:
            require(ra[role] == rb[role], f"Role-direction pair changed {role}")
        return

    if a["axis"] == DIAG_DISCOURSE:
        require(len(events_a) == 2, "Discourse-contrast pair requires exactly two events")
        require(a["input_graph"] == b["input_graph"], "Discourse-contrast pair must share identical events")
        ga, gb = a["utterance_goal"], b["utterance_goal"]
        for key in ("speaker", "addressee", "speech_level", "speech_act", "target_events"):
            require(ga[key] == gb[key], f"Discourse-contrast pair differs in {key}")
        signatures = event_signatures(a["gold_tuples"]["events"], ga["speech_act"], None)
        require(len(set(signatures.values())) == 2, "Discourse-contrast pair requires distinct events")
        cause, effect = ga["target_events"]
        require(ga["discourse"] == [{"type": "Coord", "args": ga["target_events"]}]
                and gb["discourse"] == [{"type": "Reason", "cause": cause, "effect": effect}],
                "Discourse-contrast pair must contrast Coord(a) vs Reason(b)")
        return

    require(a["utterance_goal"] == b["utterance_goal"], "Pair differs outside event bindings")
    frames = {event["frame"] for event in events_a.values()}
    require(len(frames) == 1 and len(events_a) in (2, 3, 4), "Expected 2–4 events with the same frame")
    frame = next(iter(frames))
    require(frame in PAIR_SWAP_ROLES, f"No contrast role defined for {frame}")
    relations = a["utterance_goal"]["discourse"]
    require(len(relations) == 1 and relations[0]["type"] == "Coord", "Axis 2 requires one coordination")
    # 지정된 역할의 값 두 개가 서로 바뀌었는지 사건 ID를 기준으로 확인한다.
    swap_role = PAIR_SWAP_ROLES[frame]
    changed = []
    for event_id, event in events_a.items():
        other = events_b[event_id]
        require(event["frame"] == other["frame"] and event["roles"].keys() == other["roles"].keys(),
                "Pair changed frame or role schema")
        for role, value in event["roles"].items():
            other_value = other["roles"][role]
            if other_value != value:
                require(role == swap_role, f"Pair changed {role}, expected {swap_role}")
                changed.append((value, other_value))
    require(len(changed) == 2 and changed[0] == changed[1][::-1], "Pair must transpose exactly two bindings")


def validate_references(record, stage):
    refs = record["references"]
    styles = unique_index(refs, "style")
    require(styles.keys() <= set(STYLE_KEYS), "Unknown reference style")
    texts = [ref.get("text") for ref in refs]
    require(all(isinstance(text, str) and text.strip() for text in texts), "Empty or invalid reference")
    require(len({text_key(text) for text in texts}) == len(texts), "Duplicate reference text")
    if not record["expect_supported"]:
        require(not refs and record.get("author_verified") is False, "Unsupported row cannot have references")
        return
    if stage == "synthesis":
        # 합성 초안의 통과 표시는 원래 역추출 결과를 다시 비교해 확인한다.
        require(record.get("author_verified") is False, "Draft wrongly marked verified")
        log = record["roundtrip_log"]
        attempts = log["attempts"]
        require(isinstance(attempts, list) and log["passed_count"] == len(refs), "Invalid Round-Trip log")
        addressee_id = record["utterance_goal"]["addressee"]
        addressee = next(ent["surface"] for ent in record["input_graph"]["entities"] if ent["id"] == addressee_id)
        passed = {}
        for attempt in attempts:
            if attempt.get("ok") is True:
                style = attempt["style"]
                require(style not in passed, "Repeated passed style")
                require(not comparison_errors(record["gold_tuples"], attempt["extracted"], addressee),
                        "Passing attempt fails semantic recheck")
                passed[style] = attempt["text"]
        require(passed == {key: ref["text"] for key, ref in styles.items()},
                "References must match passing attempts")
    elif stage == "final":
        # 최종 참조는 저자의 채택·수정 기록에서 만들어지는 대사와 일치해야 한다.
        require(record.get("author_verified") is True and 2 <= len(refs) <= 3,
                "Supported row requires author verification and 2–3 references")
        from review_dataset import reviewed_references
        require(refs == reviewed_references(record["draft_references"], record["author_review"]),
                "Final references differ from the author review")


def validate_coverage(records, split, pairs):
    # 전체 분할에서는 총 행 수와 함께 프레임·화행·화체 조합, 사건 수 분포도 확인한다.
    counts = Counter(record["axis"] for record in records)
    require(counts == SPLIT_COUNTS[split], f"Wrong split counts: {dict(counts)}")
    axis1 = [record for record in records if record["axis"] == AXIS1]
    if axis1:
        cells = Counter()
        codas = Counter()
        for record in axis1:
            events = record["input_graph"]["events"]
            require(len(events) == 1, "Axis 1 must use one event")
            event, goal = events[0], record["utterance_goal"]
            frame = event["frame"]
            cells[(frame, goal["speech_act"], goal["speech_level"])] += 1
            focal = event["roles"][FOCAL_ROLES[frame]]
            entity = next(entity for entity in record["input_graph"]["entities"] if entity["id"] == focal)
            codas[(goal["speech_level"], entity["coda"])] += 1
        require(cells == dict.fromkeys(product(SUPPORTED_FRAMES, SPEECH_ACTS, SPEECH_LEVELS), 1),
                "Incomplete Axis 1 matrix")
        require(all(codas[(level, coda)] for level, coda in product(SPEECH_LEVELS, CODAS)),
                "Axis 1 confounds coda and speech level")
    if split in AXIS2_PAIR_COUNTS:
        axis2_pairs = [p for p in pairs.values() if p[0]["axis"] == AXIS2]
        require(Counter(len(pair[0]["utterance_goal"]["target_events"]) for pair in axis2_pairs)
                == AXIS2_PAIR_COUNTS[split], "Wrong Axis 2 pair-size distribution")
    if split == "diag40":
        require(Counter(p[0]["axis"] for p in pairs.values()) == {DIAG_ROLE: 10, DIAG_DISCOURSE: 10},
                "Wrong diag40 pair distribution")
    axis3 = [record for record in records if record["axis"] == AXIS3]
    for rec in axis3:
        require(any(rel["type"] == "Reason" for rel in rec["utterance_goal"]["discourse"]),
                "Axis 3 requires a reason relation")
        frames = {event["frame"] for event in rec["gold_tuples"]["events"]}
        require(frames & {"Need", "Located_At", "Danger"} and frames & {"Deliver", "Fetch"},
                "Axis 3 requires both state and action events")
    axis4 = [record for record in records if record["axis"] == AXIS4]
    if axis4:
        require(Counter(len(rec["utterance_goal"]["target_events"]) for rec in axis4 if rec["expect_supported"])
                == AXIS4_STRESS_COUNTS, "Wrong Axis 4 stress distribution")
        require(Counter(rec["unsupported_reason"] for rec in axis4 if not rec["expect_supported"])
                == dict.fromkeys(BOUNDARIES, 5), "Wrong Axis 4 failure distribution")
        require({event["frame"] for rec in axis4 if rec["unsupported_reason"] == "unsupported_frame"
                 for event in rec["gold_tuples"]["events"]} == set(UNSUPPORTED_FRAMES),
                "Axis 4 must cover every unsupported frame")


def validate_records(records, stage="graphs", partial=False, *, allow_incomplete_pairs=False):
    require(bool(records), "Empty dataset file")
    unique_index(records, "id")
    validate_lexicon()
    pairs = defaultdict(list)
    for record in records:
        try:
            validate_graph(record)
            if stage != "graphs":
                validate_references(record, stage)
            if record["pair_id"]:
                pairs[record["pair_id"]].append(record)
        except (ValueError, KeyError, TypeError, AttributeError) as exc:
            raise ValueError(f"{record['id']}: {exc}") from exc

    split = records[0]["split"]
    require(all((r["split"], r["seed"]) == (split, records[0]["seed"]) for r in records), "Mixed splits or seeds")
    for pair_id, members in pairs.items():
        if partial and allow_incomplete_pairs and len(members) == 1:
            continue
        try:
            validate_pair(members)
        except ValueError as exc:
            raise ValueError(f"{pair_id}: {exc}") from exc
    # 부분 실행은 전체 분할의 개수 조건을 생략한다.
    if not partial:
        validate_coverage(records, split, pairs)
    return split, len(pairs)


def validate_file(path: Path, stage=None, partial=False):
    records = read_jsonl(path)
    if stage is None:
        stage = ("final" if any("author_review" in row for row in records)
                 else "synthesis" if any("references" in row for row in records) else "graphs")
    split, pairs = validate_records(records, stage, partial)
    print(f"[검증 통과] {path.name} | {split} | {stage} | 시나리오 {len(records)}개 | 대조쌍 {pairs}개")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    parser.add_argument("--stage", choices=("graphs", "synthesis", "final"))
    parser.add_argument("--partial", action="store_true", help="Allow a subset while requiring complete contrast pairs")
    args = parser.parse_args()
    try:
        validate_file(args.path, args.stage, args.partial)
    except (ValueError, OSError) as exc:
        parser.exit(1, f"검증 실패: {exc}\n")


if __name__ == "__main__":
    main()
