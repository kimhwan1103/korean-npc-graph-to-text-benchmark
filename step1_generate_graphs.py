#!/usr/bin/env python3
# 프로토콜의 평가 축에 따라 같은 시드에서 같은 시나리오 그래프를 생성한다.
"""Step 1: Deterministic Code-Only Input Graph & Minimal Contrastive Pair Generator.

Follows DATASET_PROTOCOL_KO_20261123.md (CheckList + Contrast Sets principles):
- Pilot-30: Axis 2 contrastive pairs (20 = 10 pairs) + Axis 3 complex clauses (10)
- Full-240: Axis 1 (60) + Axis 2 (80 = 40 pairs) + Axis 3 (60) + Axis 4 (40 = 20 supported edge + 20 unsupported)
"""

import argparse
import random
from copy import deepcopy
from itertools import product
from pathlib import Path
from typing import Any, Dict, List, Optional

from dataset_io import write_jsonl
from schema import (
    AXIS1, AXIS2, AXIS3, AXIS4, AXIS4_STRESS_COUNTS, BOUNDARIES,
    CHARACTERS, CODAS, ENTITY_BY_ID, ITEMS, LOCATIONS, PAIR_SWAP_ROLES,
    PLAYER_ENTITY, SCHEMA_VERSION, SPEECH_ACTS, SPEECH_LEVELS, SUPPORTED_FRAMES,
    UNSUPPORTED_FRAMES, Entity, derive_gold,
)


def _group_by_coda(entities: List[Entity]) -> Dict[str, List[Entity]]:
    groups: Dict[str, List[Entity]] = {"none": [], "rieul": [], "other": []}
    for ent in entities:
        groups[ent.coda].append(ent)
    return groups


CHARS_BY_CODA = _group_by_coda(CHARACTERS)
ITEMS_BY_CODA = _group_by_coda(ITEMS)
LOCS_BY_CODA = _group_by_coda(LOCATIONS)
THREATS = [ENTITY_BY_ID[key] for key in ("char_rogue", "char_mage", "char_captain")]


def _pick_distinct(rng: random.Random, pool: List[Entity], k: int, exclude_ids: Optional[set] = None) -> List[Entity]:
    exclude_ids = exclude_ids or set()
    candidates = [e for e in pool if e.id not in exclude_ids]
    return rng.sample(candidates, k)


def _build_record(
    scenario_id: str,
    axis: str,
    subtype: str,
    speaker: Entity,
    events: List[Dict[str, Any]],
    discourse: List[Dict[str, Any]],
    speech_level: str,
    speech_act: str,
    expect_supported: bool = True,
    pair_id: Optional[str] = None,
    pair_variant: Optional[str] = None,
    unsupported_reason: Optional[str] = None,
) -> Dict[str, Any]:
    # 화자와 청자, 그리고 이번 발화의 사건에 등장하는 개체만 그래프에 넣는다.
    used_ids = {speaker.id, PLAYER_ENTITY.id}
    for event in events:
        used_ids.update(event["roles"].values())
    graph = {
        "entities": [ENTITY_BY_ID[eid].to_dict() for eid in sorted(used_ids)],
        "events": events,
    }
    goal = {
        "speaker": speaker.id, "addressee": PLAYER_ENTITY.id,
        "speech_level": speech_level, "speech_act": speech_act,
        "target_events": [event["id"] for event in events], "discourse": discourse,
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "id": scenario_id, "axis": axis, "subtype": subtype,
        "pair_id": pair_id, "pair_variant": pair_variant,
        "expect_supported": expect_supported, "unsupported_reason": unsupported_reason,
        "input_graph": graph, "utterance_goal": goal, "gold_tuples": derive_gold(graph, goal),
    }


def generate_axis1(rng: random.Random, count: int) -> List[Dict[str, Any]]:
    """Cover the complete frame × act × level matrix."""
    records = []
    cells = list(product(SUPPORTED_FRAMES, SPEECH_ACTS, SPEECH_LEVELS))
    for i, (frame, act, level) in enumerate(cells[:count]):
        # 화체가 같아도 평가 대상 개체의 받침 유형이 다양하게 나오도록 순환시킨다.
        coda = CODAS[(i // len(SPEECH_LEVELS) + i % len(SPEECH_LEVELS)) % len(CODAS)]
        speaker = rng.choice(CHARACTERS)
        actor = PLAYER_ENTITY.id
        if frame == "Deliver":
            roles = {"Agent": actor, "Theme": rng.choice(ITEMS_BY_CODA[coda]).id,
                     "Recipient": _pick_distinct(rng, CHARACTERS, 1, {speaker.id})[0].id}
            if (i // 3 + i % 3) % 2 == 0:
                roles["Source"] = rng.choice(LOCATIONS).id
        elif frame == "Fetch":
            roles = {"Agent": actor, "Theme": rng.choice(ITEMS_BY_CODA[coda]).id,
                     "Source": rng.choice(LOCATIONS).id}
        elif frame == "Need":
            roles = {"Experiencer": _pick_distinct(rng, CHARS_BY_CODA[coda], 1, {speaker.id})[0].id,
                     "Required_Entity": rng.choice(ITEMS).id}
        elif frame == "Located_At":
            roles = {"Entity": rng.choice(ITEMS_BY_CODA[coda]).id, "Location": rng.choice(LOCATIONS).id}
        else:
            threat = _pick_distinct(rng, THREATS, 1, {speaker.id})[0]
            roles = {"Location": rng.choice(LOCS_BY_CODA[coda]).id, "Threat": threat.id}
            if (i // 3 + i % 3) % 2 == 0:
                roles["Target"] = actor
        records.append(_build_record(
            f"ax1_{i+1:03d}", AXIS1, f"single_{frame.lower()}_{coda}", speaker,
            [{"id": "e1", "frame": frame, "roles": roles}], [], level, act,
        ))
    return records


def generate_axis2_pairs(rng: random.Random, num_pairs: int) -> List[Dict[str, Any]]:
    """Transpose one role across two events and preserve all other bindings."""
    records = []
    for p in range(num_pairs):
        size = 2 + p % 3
        frame = tuple(PAIR_SWAP_ROLES)[p % len(PAIR_SWAP_ROLES)]
        swap_role = PAIR_SWAP_ROLES[frame]
        cycle = p // len(PAIR_SWAP_ROLES)
        speaker = rng.choice(CHARACTERS)
        items = _pick_distinct(rng, ITEMS, size)
        chars = _pick_distinct(rng, CHARACTERS, size, {speaker.id})
        locations = _pick_distinct(rng, LOCATIONS, size)
        act = ("Request", "Inform")[cycle % 2] if frame in {"Deliver", "Fetch"} else "Inform"
        level = SPEECH_LEVELS[(cycle + p) % len(SPEECH_LEVELS)]
        events = []
        for i, (item, character, location) in enumerate(zip(items, chars, locations), start=1):
            if frame == "Deliver":
                roles = {"Agent": PLAYER_ENTITY.id, "Theme": item.id, "Recipient": character.id}
                if (cycle + p % 3) % 2 == 0:
                    roles["Source"] = location.id
            elif frame == "Fetch":
                roles = {"Agent": PLAYER_ENTITY.id, "Theme": item.id, "Source": location.id}
            elif frame == "Need":
                roles = {"Experiencer": character.id, "Required_Entity": item.id}
            else:
                roles = {"Entity": item.id, "Location": location.id}
            events.append({"id": f"e{i}", "frame": frame, "roles": roles})
        # b에서 두 사건의 지정 역할값만 바꿔 다른 차이 없이 역할 교차를 평가한다.
        swapped = deepcopy(events)
        first, second = (swapped[i]["roles"] for i in rng.sample(range(size), 2))
        first[swap_role], second[swap_role] = second[swap_role], first[swap_role]
        discourse = [{"type": "Coord", "args": [event["id"] for event in events]}]
        for variant, bindings in (("a", events), ("b", swapped)):
            records.append(_build_record(
                f"ax2_{p+1:03d}_{variant}", AXIS2, f"contrast_{frame.lower()}_{size}",
                speaker, bindings, discourse, level, act, pair_id=f"pair_{p+1:03d}", pair_variant=variant,
            ))
    return records


def generate_axis3_complex(rng: random.Random, count: int) -> List[Dict[str, Any]]:
    """Axis 3: Complex discourse clauses (Reason + Request, Reason + Coord)."""
    records: List[Dict[str, Any]] = []
    subtypes = (
        "reason_need_to_deliver",
        "reason_danger_to_fetch",
        "reason_located_to_fetch",
        "reason_need_to_coord_deliver",
    )

    for i in range(count):
        subtype = subtypes[i % len(subtypes)]
        speech_level = SPEECH_LEVELS[i % len(SPEECH_LEVELS)]
        speaker = rng.choice(CHARACTERS)

        if subtype == "reason_need_to_deliver":
            recipient = _pick_distinct(rng, CHARACTERS, 1, {speaker.id})[0]
            item = rng.choice(ITEMS)
            src = rng.choice(LOCATIONS)
            events = [
                {"id": "e1", "frame": "Need", "roles": {"Experiencer": recipient.id, "Required_Entity": item.id}},
                {
                    "id": "e2",
                    "frame": "Deliver",
                    "roles": {"Agent": PLAYER_ENTITY.id, "Theme": item.id, "Recipient": recipient.id, "Source": src.id},
                },
            ]
            discourse = [{"type": "Reason", "cause": "e1", "effect": "e2"}]
            speech_act = "Request"

        elif subtype == "reason_danger_to_fetch":
            # 위험한 장소와 물건을 가져올 장소를 구분해 두 장소의 역할을 평가한다.
            danger, source = _pick_distinct(rng, LOCATIONS, 2)
            threat = _pick_distinct(rng, THREATS, 1, {speaker.id})[0]
            item = rng.choice([ENTITY_BY_ID[key] for key in ("item_shield", "item_armor", "item_longsword")])
            events = [
                {"id": "e1", "frame": "Danger", "roles": {"Location": danger.id, "Threat": threat.id}},
                {"id": "e2", "frame": "Fetch", "roles": {"Agent": PLAYER_ENTITY.id, "Theme": item.id, "Source": source.id}},
            ]
            discourse = [{"type": "Reason", "cause": "e1", "effect": "e2"}]
            speech_act = "Request"

        elif subtype == "reason_located_to_fetch":
            loc = rng.choice(LOCATIONS)
            item = rng.choice(ITEMS)
            events = [
                {"id": "e1", "frame": "Located_At", "roles": {"Entity": item.id, "Location": loc.id}},
                {"id": "e2", "frame": "Fetch", "roles": {"Agent": PLAYER_ENTITY.id, "Theme": item.id, "Source": loc.id}},
            ]
            discourse = [{"type": "Reason", "cause": "e1", "effect": "e2"}]
            speech_act = "Request"

        else:
            # 두 필요 상태를 한 묶음으로, 두 전달 행동을 다른 묶음으로 만들어 인과로 잇는다.
            r1, r2 = _pick_distinct(rng, CHARACTERS, 2, {speaker.id})
            t1, t2 = _pick_distinct(rng, ITEMS, 2)
            events = [
                {"id": "e1", "frame": "Need", "roles": {"Experiencer": r1.id, "Required_Entity": t1.id}},
                {"id": "e2", "frame": "Need", "roles": {"Experiencer": r2.id, "Required_Entity": t2.id}},
                {"id": "e3", "frame": "Deliver", "roles": {"Agent": PLAYER_ENTITY.id, "Theme": t1.id, "Recipient": r1.id}},
                {"id": "e4", "frame": "Deliver", "roles": {"Agent": PLAYER_ENTITY.id, "Theme": t2.id, "Recipient": r2.id}},
            ]
            discourse = [
                {"type": "Coord", "id": "d1", "args": ["e1", "e2"]},
                {"type": "Coord", "id": "d2", "args": ["e3", "e4"]},
                {"type": "Reason", "cause": "d1", "effect": "d2"},
            ]
            speech_act = "Request"

        records.append(
            _build_record(
                scenario_id=f"ax3_{i+1:03d}",
                axis=AXIS3,
                subtype=subtype,
                speaker=speaker,
                events=events,
                discourse=discourse,
                speech_level=speech_level,
                speech_act=speech_act,
            )
        )
    return records


def generate_axis4_boundary(rng: random.Random, count: int) -> List[Dict[str, Any]]:
    """Axis 4: High-density supported stress cases (half) + explicit UNSUPPORTED boundary cases (half)."""
    records: List[Dict[str, Any]] = []
    n_supported = count // 2
    n_unsupported = count - n_supported

    # 사건 수를 늘리면서도 병렬 관계의 깊이는 유지해 지원 범위 안의 난이도를 높인다.
    for i in range(n_supported):
        size = tuple(AXIS4_STRESS_COUNTS)[i % len(AXIS4_STRESS_COUNTS)]
        speech_level = SPEECH_LEVELS[(i // 3 + i) % len(SPEECH_LEVELS)]
        speaker = rng.choice(CHARACTERS)
        items = _pick_distinct(rng, ITEMS, size)
        recipients = _pick_distinct(rng, CHARACTERS, size, {speaker.id})
        events = [
            {"id": f"e{j}", "frame": "Deliver", "roles": {
                "Agent": PLAYER_ENTITY.id, "Theme": item.id, "Recipient": recipient.id,
            }}
            for j, (item, recipient) in enumerate(zip(items, recipients), start=1)
        ]
        discourse = [{"type": "Coord", "args": [event["id"] for event in events]}]
        records.append(
            _build_record(
                scenario_id=f"ax4_sup_{i+1:03d}",
                axis=AXIS4,
                subtype=f"stress_{size}event_coord",
                speaker=speaker,
                events=events,
                discourse=discourse,
                speech_level=speech_level,
                speech_act="Request",
                expect_supported=True,
            )
        )

    for j in range(n_unsupported):
        mode = BOUNDARIES[j % len(BOUNDARIES)]
        speech_level = SPEECH_LEVELS[j % len(SPEECH_LEVELS)]
        speaker = rng.choice(CHARACTERS)

        if mode == "unsupported_frame":
            uframe = UNSUPPORTED_FRAMES[(j // len(BOUNDARIES)) % len(UNSUPPORTED_FRAMES)]
            item = rng.choice(ITEMS)
            target_char = _pick_distinct(rng, CHARACTERS, 1, {speaker.id})[0]
            events = [{"id": "e1", "frame": uframe, "roles": {"Agent": PLAYER_ENTITY.id, "Theme": item.id, "Target": target_char.id}}]
            discourse = []
            speech_act = "Request"

        elif mode == "missing_required_role":
            target_char = _pick_distinct(rng, CHARACTERS, 1, {speaker.id})[0]
            # 전달 프레임에서 필수 역할인 물건(Theme)을 의도적으로 뺀다.
            events = [{"id": "e1", "frame": "Deliver", "roles": {"Agent": PLAYER_ENTITY.id, "Recipient": target_char.id}}]
            discourse = []
            speech_act = "Request"

        elif mode == "excessive_nesting_depth":
            # 인과 관계를 세 겹으로 중첩해 지원 깊이 2를 넘긴다.
            loc = rng.choice(LOCATIONS)
            item = rng.choice(ITEMS)
            recipient = _pick_distinct(rng, CHARACTERS, 1, {speaker.id})[0]
            events = [
                {"id": "e1", "frame": "Danger", "roles": {"Location": loc.id, "Threat": _pick_distinct(rng, THREATS, 1, {speaker.id})[0].id}},
                {"id": "e2", "frame": "Located_At", "roles": {"Entity": item.id, "Location": loc.id}},
                {"id": "e3", "frame": "Need", "roles": {"Experiencer": recipient.id, "Required_Entity": item.id}},
                {"id": "e4", "frame": "Deliver", "roles": {"Agent": PLAYER_ENTITY.id, "Theme": item.id, "Recipient": recipient.id}},
            ]
            discourse = [
                {"type": "Reason", "id": "d1", "cause": "e1", "effect": "e2"},
                {"type": "Reason", "id": "d2", "cause": "d1", "effect": "e3"},
                {"type": "Reason", "id": "d3", "cause": "d2", "effect": "e4"},
            ]
            speech_act = "Request"

        else:
            # 한 발화에 경고와 질문을 함께 지정한 미지원 사례다.
            loc = rng.choice(LOCATIONS)
            item = rng.choice(ITEMS)
            events = [
                {"id": "e1", "frame": "Danger", "roles": {"Location": loc.id, "Threat": _pick_distinct(rng, THREATS, 1, {speaker.id})[0].id}},
                {"id": "e2", "frame": "Fetch", "roles": {"Agent": PLAYER_ENTITY.id, "Theme": item.id, "Source": loc.id}},
            ]
            discourse = [{"type": "Coord", "args": ["e1", "e2"], "conflicting_acts": ["Warn", "Question"]}]
            speech_act = "Warn+Question"

        records.append(
            _build_record(
                scenario_id=f"ax4_unsup_{j+1:03d}",
                axis=AXIS4,
                subtype=mode,
                speaker=speaker,
                events=events,
                discourse=discourse,
                speech_level=speech_level,
                speech_act=speech_act,
                expect_supported=False,
                unsupported_reason=mode,
            )
        )
    return records


def generate_records(split: str, seed: int) -> List[Dict[str, Any]]:
    # 동일한 숫자 시드를 써도 분할과 스키마 버전에 따라 난수열을 구분한다.
    rng = random.Random(f"{seed}:{split}:v{SCHEMA_VERSION}")

    if split == "pilot30":
        records = generate_axis2_pairs(rng, num_pairs=10) + generate_axis3_complex(rng, count=10)
    elif split == "full240":
        records = (
            generate_axis1(rng, count=60)
            + generate_axis2_pairs(rng, num_pairs=40)
            + generate_axis3_complex(rng, count=60)
            + generate_axis4_boundary(rng, count=40)
        )
    else:
        raise ValueError(f"Unknown split: {split}")

    # 서로 다른 분할의 결과를 합쳐도 시나리오 ID와 대조쌍 ID가 충돌하지 않게 한다.
    for record in records:
        record["split"] = split
        record["seed"] = seed
        record["id"] = f"{split}_{record['id']}"
        if record["pair_id"]:
            record["pair_id"] = f"{split}_{record['pair_id']}"
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Step 1: Generate deterministic RPG event-role graphs.")
    parser.add_argument("--split", choices=["pilot30", "full240"], default="pilot30")
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    records = generate_records(args.split, args.seed)
    out_path = args.out or Path(__file__).parent / "data" / f"{args.split}_graphs.jsonl"
    write_jsonl(out_path, records)
    print(f"[Step 1 완료] split={args.split}, seed={args.seed}, count={len(records)} -> {out_path}")


if __name__ == "__main__":
    main()
