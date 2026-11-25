# 역추출 결과를 비교 가능한 구조로 바꿔 정답과 다른 사건·역할·담화 관계를 찾는다.
"""Compare independently extracted events and discourse against the dataset contract."""

from collections import Counter, defaultdict
from unicodedata import normalize

from schema import FOCAL_ROLES, SPEECH_ACTS, SPEECH_LEVELS


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


def _minimum_cost_assignment(costs):
    """Match every row to a distinct column in a rectangular cost matrix."""
    n_rows, n_columns = len(costs), len(costs[0])
    row_potential, column_potential = [0] * (n_rows + 1), [0] * (n_columns + 1)
    matched_rows = [0] * (n_columns + 1)
    # Hungarian algorithm. Column zero stands for the row being added.
    for new_row in range(1, n_rows + 1):
        matched_rows[0] = new_row
        previous = [0] * (n_columns + 1)
        distances = [float("inf")] * (n_columns + 1)
        used_columns = set()
        column = 0
        while True:
            used_columns.add(column)
            row = matched_rows[column]
            delta, next_column = float("inf"), 0
            for candidate in range(1, n_columns + 1):
                if candidate in used_columns:
                    continue
                cost = costs[row - 1][candidate - 1] - row_potential[row] - column_potential[candidate]
                if cost < distances[candidate]:
                    distances[candidate], previous[candidate] = cost, column
                if distances[candidate] < delta:
                    delta, next_column = distances[candidate], candidate
            for candidate in range(n_columns + 1):
                if candidate in used_columns:
                    row_potential[matched_rows[candidate]] += delta
                    column_potential[candidate] -= delta
                else:
                    distances[candidate] -= delta
            column = next_column
            if not matched_rows[column]:
                break
        while column:
            parent = previous[column]
            matched_rows[column] = matched_rows[parent]
            column = parent
    return [(row - 1, column - 1) for column, row in enumerate(matched_rows[1:], start=1) if row]


def _match_roles(expected, observed, focal_role):
    if len(expected) > len(observed):
        return [(gold, extracted) for extracted, gold in _match_roles(observed, expected, focal_role)]
    # Prefer fewer role edits, using the frame's focal role to break ties.
    scale = len(expected) + 1
    costs = [
        [scale * sum(gold.get(role) != extracted.get(role) for role in gold.keys() | extracted.keys())
         + (gold.get(focal_role) != extracted.get(focal_role))
         for extracted in observed]
        for gold in expected
    ]
    return [(expected[row], observed[column]) for row, column in _minimum_cost_assignment(costs)]


def _remove_swaps(changes):
    """Consume closed cycles of changed values without reusing a changed slot."""
    found = False
    for slot, (before, after) in tuple(changes.items()):
        if slot not in changes:
            continue
        # Look for a return path for before -> after, using other role changes.
        route = {after: None}
        queue = [after]
        for value in queue:
            for candidate, (start, end) in changes.items():
                if start == value and end not in route:
                    route[end] = (value, candidate)
                    queue.append(end)
            if before in route:
                break
        if before in route:
            del changes[slot]
            value = before
            while value != after:
                value, used_slot = route[value]
                del changes[used_slot]
            found = True
    return found


def _role_errors(expected_events, observed_events):
    # Remove identical events, then align remaining same-frame events by content.
    expected, observed = defaultdict(list), defaultdict(list)
    for _, frame, roles in sorted((expected_events - observed_events).elements()):
        expected[frame].append(dict(roles))
    for _, frame, roles in sorted((observed_events - expected_events).elements()):
        observed[frame].append(dict(roles))

    errors = set()
    for frame in expected.keys() & observed.keys():
        by_role = defaultdict(dict)
        for index, (gold, extracted) in enumerate(_match_roles(expected[frame], observed[frame], FOCAL_ROLES.get(frame))):
            if gold.keys() - extracted.keys():
                errors.add("missing_role")
            if extracted.keys() - gold.keys():
                errors.add("hallucinated_role")
            changes = {role: (gold[role], extracted[role]) for role in sorted(gold.keys() & extracted.keys())
                       if gold[role] != extracted[role]}
            if _remove_swaps(changes):
                errors.add("role_swap")
            for role, values in changes.items():
                by_role[role][index] = values
        for changes in by_role.values():
            if _remove_swaps(changes):
                errors.add("role_swap")
            if changes:
                errors.add("role_value_mismatch")
    return [error for error in ("missing_role", "hallucinated_role", "role_swap", "role_value_mismatch")
            if error in errors]


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

        errors.extend(_role_errors(expected_events, observed_events))
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
