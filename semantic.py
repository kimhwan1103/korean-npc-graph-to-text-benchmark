# 역추출 결과를 비교 가능한 구조로 바꿔 정답과 다른 사건·역할·담화 관계를 찾는다.
"""Compare independently extracted events and discourse against the dataset contract."""

from collections import Counter
from unicodedata import normalize

from schema import SPEECH_ACTS, SPEECH_LEVELS


def text_key(text):
    # 한글 유니코드 표현과 공백의 차이로 같은 대사가 서로 다르게 판정되지 않게 한다.
    return normalize("NFC", " ".join(text.split()))


def unique_index(rows, id_key):
    if not isinstance(rows, list) or any(
        not isinstance(row, dict) or not isinstance(row.get(id_key), str) or not row[id_key]
        for row in rows
    ):
        raise ValueError(f"Expected objects with nonempty {id_key} values")
    index = {row[id_key]: row for row in rows}
    if len(index) != len(rows):
        raise ValueError(f"Duplicate {id_key}")
    return index


def discourse_tree(events, relations):
    """Return the canonical rooted tree and its depth, with event IDs resolved."""
    if not isinstance(relations, list):
        raise ValueError("Expected a discourse list")
    if not relations:
        if len(events) != 1:
            raise ValueError("Multiple events require explicit discourse")
        return next(iter(events.values())), 0

    # 하위 관계도 다른 담화 관계의 인자가 될 수 있도록 관계 ID를 먼저 등록한다.
    nodes = {}
    for i, rel in enumerate(relations):
        if not isinstance(rel, dict):
            raise ValueError("Expected a discourse object")
        rel_id = rel.get("id", f"@{i}")
        if not isinstance(rel_id, str) or not rel_id or rel_id in nodes or rel_id in events:
            raise ValueError("Invalid or duplicate discourse ID")
        relation_type = rel.get("type")
        keys = set(rel) - {"id"}
        if relation_type == "Coord" and keys == {"type", "args"}:
            args = rel["args"]
            if not isinstance(args, list) or len(args) < 2:
                raise ValueError("Coord requires at least two arguments")
        elif relation_type == "Reason" and keys == {"type", "cause", "effect"}:
            args = [rel["cause"], rel["effect"]]
        else:
            raise ValueError("Invalid discourse relation")
        if any(not isinstance(arg, str) or not arg for arg in args) or len(set(args)) != len(args):
            raise ValueError("Invalid or repeated discourse argument")
        nodes[rel_id] = (relation_type, args)

    # 트리에서는 루트 외 모든 노드가 한 부모에 속하며, 모든 대상 사건이 연결돼야 한다.
    uses = Counter(arg for _, args in nodes.values() for arg in args)
    if set(uses) - (events.keys() | nodes.keys()) or any(n != 1 for n in uses.values()):
        raise ValueError("Discourse contains unknown or shared arguments")
    roots = nodes.keys() - uses.keys()
    if len(roots) != 1 or events.keys() - uses.keys():
        raise ValueError("Discourse must cover all target events in one tree")
    seen = set()

    def visit(node_id):
        if node_id in events:
            return events[node_id], 0
        if node_id in seen:
            raise ValueError("Discourse cycle")
        seen.add(node_id)
        relation_type, arguments = nodes[node_id]
        children = [visit(child) for child in arguments]
        signatures = [child[0] for child in children]
        if relation_type == "Coord":
            # 병렬은 절 순서가 바뀌어도 같지만 인과는 원인과 결과의 순서를 보존한다.
            signatures.sort()
        return (relation_type, tuple(signatures)), 1 + max(child[1] for child in children)

    result = visit(next(iter(roots)))
    if len(seen) != len(nodes):
        raise ValueError("Disconnected discourse cycle")
    return result


def event_signatures(events, speech_act, addressee):
    signatures = {}
    for event_id, event in unique_index(events, "event_id").items():
        frame, roles = event.get("frame"), event.get("roles")
        if not isinstance(frame, str) or not frame.strip() or not isinstance(roles, dict):
            raise ValueError("Expected event frame and role object")
        if any(not isinstance(role, str) or not role.strip() or not isinstance(value, str) or not value.strip()
               for role, value in roles.items()):
            raise ValueError("Expected nonempty role names and surfaces")
        roles = {role: text_key(value) for role, value in roles.items()}
        if frame in {"Fetch", "Deliver"} and speech_act in {"Request", "Warn"} and addressee:
            # 요청·경고에서 생략된 주체만 청자로 해석한다. 명시된 주체는 그대로 둔다.
            roles.setdefault("Agent", text_key(addressee))
        signatures[event_id] = ("event", frame, tuple(sorted(roles.items())))
    return signatures


def comparison_errors(gold, extracted, addressee=None):
    """Report semantic errors without looking at reference dialogue or generation logs."""
    if not isinstance(extracted, dict):
        return ["invalid_extraction"]
    act, level = extracted.get("speech_act"), extracted.get("speech_level")
    facts = extracted.get("extra_facts")
    relations = extracted.get("discourse")
    if (not isinstance(act, str) or not act.strip() or not isinstance(level, str) or not level.strip()
            or not isinstance(relations, list) or not isinstance(facts, list)
            or any(not isinstance(fact, str) or not fact.strip() for fact in facts)):
        return ["invalid_extraction"]

    try:
        expected = event_signatures(gold["events"], gold["speech_act"], addressee)
        observed = event_signatures(extracted.get("events"), act, addressee)
    except ValueError:
        return ["invalid_extraction"]

    errors = []
    if act != gold["speech_act"] or act not in SPEECH_ACTS:
        errors.append("speech_act_mismatch")
    if level != gold["speech_level"] or level not in SPEECH_LEVELS:
        errors.append("speech_level_mismatch")
    if facts:
        errors.append("extra_fact")

    # 사건 ID와 배열 순서가 달라도 같은 사건의 개수와 역할값은 일치해야 한다.
    expected_events = Counter(expected.values())
    observed_events = Counter(observed.values())
    if expected_events != observed_events:
        expected_frames = Counter(frame for _, frame, _ in expected.values())
        observed_frames = Counter(frame for _, frame, _ in observed.values())
        if expected_frames - observed_frames:
            errors.append("missing_event")
        if observed_frames - expected_frames:
            errors.append("hallucinated_event")

        if expected_frames == observed_frames:
            expected_slots = Counter((frame, role) for _, frame, roles in expected.values() for role, _ in roles)
            observed_slots = Counter((frame, role) for _, frame, roles in observed.values() for role, _ in roles)
            if expected_slots - observed_slots:
                errors.append("missing_role")
            if observed_slots - expected_slots:
                errors.append("hallucinated_role")
            if expected_slots == observed_slots:
                # 프레임별 역할값의 구성은 같지만 사건에 배정된 값이 다르면 역할 교차다.
                expected_values = Counter((frame, role, value) for _, frame, roles in expected.values()
                                          for role, value in roles)
                observed_values = Counter((frame, role, value) for _, frame, roles in observed.values()
                                          for role, value in roles)
                reason = "role_swap" if expected_values == observed_values else "role_value_mismatch"
                errors.append(reason)
    else:
        # 사건이 일치하면 추출기가 붙인 ID를 실제 사건 내용으로 치환해 담화 구조를 비교한다.
        expected_tree, _ = discourse_tree(expected, gold["discourse"])
        try:
            observed_tree, _ = discourse_tree(observed, relations)
        except ValueError:
            errors.append("discourse_mismatch")
        else:
            if observed_tree != expected_tree:
                errors.append("discourse_mismatch")
    return errors
