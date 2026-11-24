#!/usr/bin/env python3
# 여러 문체의 참조 대사를 생성하고, 역추출 결과와 정답을 비교해 통과 여부를 기록한다.
"""Generate and Round-Trip filter Korean references, with resumable audit records."""

import argparse
import hashlib
import json
import os
from collections import Counter, defaultdict
from contextlib import contextmanager
from pathlib import Path

from dataset_io import read_jsonl, write_json, write_jsonl
from openai_responses import RESPONSES_URL, call_openai_json, request_settings, strict_object, uses_reasoning
from schema import SCHEMA_VERSION, SPEECH_ACTS, SPEECH_LEVELS, STYLE_KEYS, selected_events
from semantic import comparison_errors, text_key
from validate_dataset import validate_records

PIPELINE_VERSION = 3
GEN_MODEL = "gpt-6.2-sol"
RT_MODEL = "gpt-6.1-terra"

GEN_SYSTEM_PROMPT = """당신은 한국어 RPG 게임 시나리오 라이터입니다. 사건과 목적에 맞는 NPC 대사를 JSON으로 작성하세요.
- 대상 사건, 모든 역할값, Reason의 원인·결과, Coord의 병렬 의미를 보존하세요. 입력 외 사실을 추가하지 마세요.
- 입력에 주어진 인물·물건·장소 표기를 그대로 유지하고 '님' 같은 임의 호칭 접미사를 덧붙이지 마세요.
- 화행과 화체를 각각 지키세요. Inform은 평서형으로 진술하고, Request는 요청하며, Question은 질문하고, Warn은 명시적으로 주의를 주세요.
- Deliver·Fetch의 Inform은 지정된 Agent가 해당 행동을 한다는 현재 또는 예정 사실을 서술하세요. Question은 그 행동 여부를 물으세요. 청자에게 명령하지 마세요. Request·Warn에서 청자가 Agent이면 한국어에서 주어를 생략할 수 있습니다.
- Need·Located_At·Danger의 Request는 해당 상태인지 확인해 달라는 요청입니다. Warn은 해당 상태를 경고의 내용이나 근거로 표현하세요. Deliver·Fetch의 Warn은 해당 행동을 할 때 주의하라는 뜻입니다. 입력에 없는 위험, 시간, 대상을 덧붙이지 마세요.
- Deliver.Source는 전달할 물건의 출처입니다. 값이 지정된 모든 사건에서 출처를 해당 물건과 함께 표현하세요. Danger.Threat는 위험을 유발하는 존재이며 위험 상태도 대사에 표현하세요.
- 요청받은 키마다 서로 다른 대사를 작성하세요. style_a_concise는 간결형, style_b_colloquial은 자연스러운 구어형, style_c_structural은 절 배치·성분 어순을 바꾼 구조 변형형입니다. 화체는 세 문체에서 같아야 합니다."""

GEN_FEW_SHOT = """[예시 1]
해요체, Request. Deliver(Agent=모험가, Theme=등불, Recipient=문지기, Source=정원), Deliver(Agent=모험가, Theme=책, Recipient=서기관, Source=자료실). 두 사건은 Coord.
{"style_a_concise":"정원의 등불은 문지기에게, 자료실의 책은 서기관에게 전해 주세요.","style_b_colloquial":"문지기한테는 정원에 있는 등불을 주고, 자료실의 책은 서기관한테 좀 갖다주세요.","style_c_structural":"문지기에게 정원의 등불을 전해 주세요. 서기관에게는 자료실의 책을 건네주세요."}

[예시 2]
해라체, Request. Need(Experiencer=사서, Required_Entity=책), Deliver(Agent=모험가, Theme=책, Recipient=사서). 앞 사건이 뒤 사건의 이유.
{"style_a_concise":"사서가 책을 필요로 하니 책을 사서에게 전해라.","style_b_colloquial":"사서한테 책이 필요하니까 사서한테 책 좀 갖다줘라.","style_c_structural":"책을 사서에게 가져다줘라. 사서에게 그 책이 필요하기 때문이다."}

[예시 3]
하십시오체, Inform. Fetch(Agent=모험가, Theme=등불, Source=정원).
{"style_a_concise":"모험가는 정원에서 등불을 가져옵니다.","style_b_colloquial":"모험가가 정원에서 등불을 챙겨 옵니다.","style_c_structural":"등불은 모험가가 정원에서 가져옵니다."}"""

RT_INSTRUCTIONS = """한국어 RPG 대사에 실제로 표현된 사건, 역할, 최상위 화행, 화체, 담화 관계를 JSON으로 추출하세요.
- 개체 목록은 표기와 지시 대상 확인을 위한 참고입니다. 목록에 없더라도 발화에 나온 개체는 그대로 추출하세요. 발화에 없는 역할이나 사실은 추정하지 마세요.
- Deliver: Agent, Theme, Recipient, Source(출처가 표현된 경우). Fetch: Agent, Theme, Source. Need: Experiencer, Required_Entity. Located_At: Entity, Location. Danger: Location, Threat, Target(위험을 받는 대상이 표현된 경우).
- 전달에 쓰인 '정원의 등불', '정원에 있는 등불', '정원에서 가져다가 문지기에게 전해 주세요'의 정원은 Deliver.Source입니다. 전달 대상 물건을 수식하는 관형 표현('~의 ~', '~에 있는 ~')은 별도 Located_At 사건으로 분리하지 마세요. 독립적인 회수 지시가 따로 표현되면 Fetch도 추출하세요.
- 표현된 Agent를 기록하세요. 청자를 가리키는 명시적 2인칭은 청자 이름으로 해석하세요. Agent가 실제로 생략됐고 앞 절에도 근거가 없으면 Agent 역할을 생략하세요.
- speech_act는 Inform, Request, Question, Warn 중 하나이며 판단하기 어려우면 unknown으로 반환하세요. 상태인지 확인해 달라는 말은 Request이며, 확인 대상 상태도 추출하세요. 경고 표현과 확인 요청 자체는 프레임에 추가하지 마세요.
- 화체는 독립적으로 끝나는 절의 종결 어미로 판단하세요. '가져와라', '있다', '필요하니?'는 haera, '가져와요', '전해 주세요'는 haeyo, '가져옵니다', '가져오십시오'는 hasipsio입니다. '-고', 이유절의 '-니까' 같은 연결 어미는 화체 판정에서 제외하세요. 독립된 절의 화체가 혼합되거나 판단하기 어려우면 unknown을 반환하세요.
- 사건마다 자체 ID를 부여하세요. 목록, 연결절, 별도 문장으로 나란히 진술하거나 요청한 사건은 Coord로 묶으세요. 공통 서술어가 마지막에만 나오는 경우에도 사건별 역할값을 추출하고 각 사건 ID를 Coord.args에 넣으세요.
- A가 B의 이유로 표현되면 Reason의 cause=A, effect=B입니다. 이유절이 뒤에 와도 방향은 유지하세요. 관계의 ID를 다른 관계의 인자로 참조할 수 있습니다. 표현되지 않은 이유는 추정하지 마세요.
- extra_facts는 프레임과 역할로 표현할 수 없는 추가 사실을 담는 문자열 배열입니다. 추가 사실이 없으면 []로 반환하세요. 공손 표현과 문체 차이는 제외하세요."""

# 독립 추출 작업 파일에서 쓰는 기존 역할 객체 형식을 제공한다.
RT_SYSTEM_PROMPT = RT_INSTRUCTIONS + """
- JSON 예: {"speech_act":"Request","speech_level":"haeyo","events":[{"event_id":"x1","frame":"Need","roles":{"Experiencer":"사서","Required_Entity":"책"}},{"event_id":"x2","frame":"Deliver","roles":{"Theme":"책","Recipient":"사서"}}],"discourse":[{"type":"Reason","cause":"x1","effect":"x2"}],"extra_facts":[]}"""

RT_API_SYSTEM_PROMPT = RT_INSTRUCTIONS + """
- events[].roles는 role과 value를 담는 객체의 배열입니다. 담화 관계에는 각각 고유한 id를 붙이세요. 사건 행과 관계 행을 모두 반환하고 단일 사건이면 discourse를 []로 반환하세요.
- 예시 발화: "정원에서 등불을, 창고에서 밧줄을 가져와라."
- JSON 정답: {"speech_act":"Request","speech_level":"haera","events":[{"event_id":"e1","frame":"Fetch","roles":[{"role":"Theme","value":"등불"},{"role":"Source","value":"정원"}]},{"event_id":"e2","frame":"Fetch","roles":[{"role":"Theme","value":"밧줄"},{"role":"Source","value":"창고"}]}],"discourse":[{"id":"d1","type":"Coord","args":["e1","e2"]}],"extra_facts":[]}"""

STRING = {"type": "string"}
RT_EVENT = strict_object({
    "event_id": STRING, "frame": STRING,
    "roles": {"type": "array", "items": strict_object({"role": STRING, "value": STRING})},
})
RT_SCHEMA = strict_object({
    "speech_act": {"type": "string", "enum": [*SPEECH_ACTS, "unknown"]},
    "speech_level": {"type": "string", "enum": [*SPEECH_LEVELS, "unknown"]},
    "events": {"type": "array", "items": RT_EVENT},
    "discourse": {"type": "array", "items": {"anyOf": [
        strict_object({"id": STRING, "type": {"type": "string", "enum": ["Coord"]},
                       "args": {"type": "array", "items": STRING}}),
        strict_object({"id": STRING, "type": {"type": "string", "enum": ["Reason"]},
                       "cause": STRING, "effect": STRING}),
    ]}},
    "extra_facts": {"type": "array", "items": STRING},
})


def load_dotenv_if_needed():
    # 이미 설정된 환경 변수를 우선하고, 없으면 로컬 파일에서 API 키만 읽는다.
    if os.environ.get("OPENAI_API_KEY"):
        return
    for path in (Path.cwd() / ".env", Path(__file__).parent / ".env"):
        if not path.is_file():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            key, sep, value = line.strip().removeprefix("export ").partition("=")
            if sep and key.strip() == "OPENAI_API_KEY" and value.strip():
                os.environ["OPENAI_API_KEY"] = value.strip().strip("\"'")
                return


def format_scenario_prompt(rec, styles=STYLE_KEYS):
    # 생성 모델에 전달할 사건과 역할은 입력 그래프와 발화 목적에서 가져온다.
    graph, goal = rec["input_graph"], rec["utterance_goal"]
    names = {entity["id"]: entity["surface"] for entity in graph["entities"]}
    task = {
        "speaker": names[goal["speaker"]], "addressee": names[goal["addressee"]],
        "speech_level": goal["speech_level"], "speech_act": goal["speech_act"],
        "events": selected_events(graph, goal), "discourse": goal["discourse"],
    }
    return (f"{GEN_FEW_SHOT}\n\n[실제 생성 과제]\n{json.dumps(task, ensure_ascii=False, indent=2)}"
            f"\n반환할 JSON 키: {', '.join(styles)}")


def format_roundtrip_prompt(utterance, rec):
    # 역추출 모델에는 발화와 개체 표기만 제공해 정답의 역할 연결이 드러나지 않게 한다.
    graph, goal = rec["input_graph"], rec["utterance_goal"]
    names = {entity["id"]: entity["surface"] for entity in graph["entities"]}
    context = {"speaker": names[goal["speaker"]], "addressee": names[goal["addressee"]],
               "candidate_entities": sorted(names.values()), "utterance": utterance}
    return json.dumps(context, ensure_ascii=False)


def decode_extraction(value):
    # API의 역할 배열을 기존 비교기가 사용하는 역할명–개체 표기 객체로 변환한다.
    if not isinstance(value, dict) or not isinstance(value.get("events"), list):
        return None
    events = []
    for event in value["events"]:
        if not isinstance(event, dict) or not isinstance(event.get("roles"), list):
            return None
        roles = {}
        for entry in event["roles"]:
            if not isinstance(entry, dict):
                return None
            role, surface = entry.get("role"), entry.get("value")
            if not isinstance(role, str) or not isinstance(surface, str) or role in roles:
                return None
            roles[role] = surface
        events.append({**event, "roles": roles})
    return {**value, "events": events}


def generation_schema(styles):
    return strict_object({style: STRING for style in styles})


def api_settings(args):
    temperature = args.temperature if args.temperature is not None else 0.7
    return (
        request_settings(args.model, args.reasoning_effort, temperature, args.max_output_tokens),
        request_settings(args.rt_model, args.rt_reasoning_effort, 0.0, args.rt_max_output_tokens),
    )


def verify_roundtrip(gold, extracted, addressee=None):
    errors = comparison_errors(gold, extracted, addressee)
    return not errors, errors[0] if errors else "ok"


def select_records(records, limit):
    """Fit whole contrast pairs in a sample with at most limit scenarios."""
    if limit is None:
        return records
    # 대조쌍을 하나의 묶음으로 취급해 제한 건수에서 한쪽만 선택되지 않게 한다.
    groups = defaultdict(list)
    for rec in records:
        groups[rec["pair_id"] or rec["id"]].append(rec)
    selected = []
    for group in groups.values():
        if len(selected) + len(group) <= limit:
            selected.extend(group)
    return selected


def run_manifest(records, args):
    # 입력, 모델, 프롬프트가 달라진 실행은 같은 체크포인트에 섞지 않는다.
    digest = hashlib.sha256(json.dumps(records, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    gen_settings, rt_settings = api_settings(args)
    return {
        "pipeline_version": PIPELINE_VERSION, "schema_version": SCHEMA_VERSION, "input_sha256": digest,
        "api_endpoint": RESPONSES_URL, "gen_model": args.model, "rt_model": args.rt_model,
        "temperature": gen_settings.get("temperature"), "rt_temperature": rt_settings.get("temperature"),
        "generation_request_settings": gen_settings, "roundtrip_request_settings": rt_settings,
        "generation_output_schema": generation_schema(STYLE_KEYS), "roundtrip_output_schema": RT_SCHEMA,
        "generation_system_prompt": GEN_SYSTEM_PROMPT, "generation_few_shot": GEN_FEW_SHOT,
        "roundtrip_system_prompt": RT_API_SYSTEM_PROMPT,
    }


@contextmanager
def run_lock(path):
    # 체크포인트 파일 자체는 교체되므로 별도 잠금 파일을 사용한다.
    import fcntl

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as file:
        try:
            fcntl.flock(file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError(f"Another synthesis run is using {path}") from exc
        yield


def iter_candidates(rounds):
    for attempt, round_data in enumerate(rounds, 1):
        for style, item in round_data["candidates"].items():
            yield {"attempt": attempt, "style": style, **item}


def accepted_references(rounds):
    accepted = {item["style"]: item["text"] for item in iter_candidates(rounds) if item.get("ok")}
    return [{"style": style, "text": accepted[style]} for style in STYLE_KEYS if style in accepted]


def process_record(rec, rounds, args, api_key, checkpoint):
    names = {ent["id"]: ent["surface"] for ent in rec["input_graph"]["entities"]}
    addressee = names[rec["utterance_goal"]["addressee"]]
    gen_settings, rt_settings = api_settings(args)
    for i in range(args.max_retries + 1):
        accepted = accepted_references(rounds)
        if i == len(rounds):
            # 새 시도에서는 아직 통과한 대사가 없는 문체만 모델에 요청한다.
            accepted_styles = {ref["style"] for ref in accepted}
            missing = [key for key in STYLE_KEYS if key not in accepted_styles]
            if not missing:
                break
            generated, usage = call_openai_json(
                api_key, args.model, GEN_SYSTEM_PROMPT, format_scenario_prompt(rec, missing),
                schema_name="reference_dialogue", schema=generation_schema(missing), settings=gen_settings,
            )
            candidates = {}
            for style in missing:
                value = generated.get(style) if isinstance(generated, dict) else None
                candidates[style] = {"text": value.strip() if isinstance(value, str) else ""}
            rounds.append({"generation_usage": usage, "candidates": candidates})
            # 생성 직후 저장해 역추출 중에 멈춰도 같은 대사를 다시 생성할 필요가 없게 한다.
            checkpoint()

        used_texts = {text_key(ref["text"]) for ref in accepted}
        for candidate in rounds[i]["candidates"].values():
            # ok 키는 판정 완료를 뜻한다. 재개할 때 통과·탈락 판정이 저장된 후보는 건너뛴다.
            if "ok" in candidate:
                continue
            text = candidate["text"]
            if not text:
                reason = rounds[i]["generation_usage"].get("api_error", "empty_or_invalid_output")
                candidate.update(ok=False, reason=reason)
            elif text_key(text) in used_texts:
                candidate.update(ok=False, reason="duplicate_reference")
            else:
                raw, usage = call_openai_json(
                    api_key, args.rt_model, RT_API_SYSTEM_PROMPT, format_roundtrip_prompt(text, rec),
                    schema_name="roundtrip_extraction", schema=RT_SCHEMA, settings=rt_settings,
                )
                extracted = decode_extraction(raw)
                api_error = usage.get("api_error")
                errors = [api_error] if api_error else comparison_errors(rec["gold_tuples"], extracted, addressee)
                candidate.update(ok=not errors, reason=errors[0] if errors else "ok",
                                 errors=errors, extracted=extracted, usage=usage)
                if extracted is None and raw is not None:
                    candidate["raw_extraction"] = raw
                if not errors:
                    used_texts.add(text_key(text))
            checkpoint()


def export_results(records, state, args, out_path, stats_path):
    # 이번 실행의 호출만 세면 재개 이후 통계가 줄어들 수 있어 저장된 전체 이력을 집계한다.
    output = []
    reasons, statuses = Counter(), Counter()
    models = {"generation": Counter(), "roundtrip": Counter()}
    tokens = Counter(dict.fromkeys((
        "gen_prompt", "gen_completion", "gen_reasoning", "gen_cached_prompt",
        "rt_prompt", "rt_completion", "rt_reasoning", "rt_cached_prompt",
    ), 0))
    generated = checked = first_slots = first_passed = first_checked = final_passed = 0
    for rec in records:
        if rec["id"] not in state["rows"]:
            continue
        rounds = state["rows"][rec["id"]]
        refs = accepted_references(rounds)
        attempts = list(iter_candidates(rounds))
        status = "unsupported"
        if rec["expect_supported"]:
            if len(refs) == len(STYLE_KEYS):
                status = "complete"
            elif len(rounds) >= args.max_retries + 1 and all("ok" in item for item in attempts):
                status = "retry_limit_reached"
            else:
                status = "pending"
        statuses[status] += 1
        for round_data in rounds:
            usage = round_data["generation_usage"]
            models["generation"][usage.get("model", args.model)] += 1
            tokens["gen_prompt"] += usage["prompt_tokens"]
            tokens["gen_completion"] += usage["completion_tokens"]
            tokens["gen_reasoning"] += usage.get("reasoning_tokens", 0)
            tokens["gen_cached_prompt"] += usage.get("cached_prompt_tokens", 0)
        for item in attempts:
            # 문체 슬롯 기준 통과율과 실제 역추출 호출 기준 통과율의 분모를 따로 센다.
            first_slots += item["attempt"] == 1
            generated += bool(item["text"])
            if "usage" in item:
                checked += 1
                first_checked += item["attempt"] == 1
                usage = item["usage"]
                models["roundtrip"][usage.get("model", args.rt_model)] += 1
                tokens["rt_prompt"] += usage["prompt_tokens"]
                tokens["rt_completion"] += usage["completion_tokens"]
                tokens["rt_reasoning"] += usage.get("reasoning_tokens", 0)
                tokens["rt_cached_prompt"] += usage.get("cached_prompt_tokens", 0)
            if item.get("ok") is True and item["attempt"] == 1:
                first_passed += 1
            elif item.get("ok") is False:
                reasons[item["reason"]] += 1
        final_passed += len(refs)
        output.append({**rec, "references": refs, "author_verified": False,
                       "roundtrip_log": {"status": status, "passed_count": len(refs), "attempts": attempts}})
    write_jsonl(out_path, output)
    stats = {
        **state["manifest"], "input_file": str(args.input), "max_retries": args.max_retries,
        "total_input_scenarios": len(records), "total_scenarios": len(output),
        "supported_scenarios": len(output) - statuses["unsupported"], "unsupported_scenarios": statuses["unsupported"],
        "scenario_statuses": dict(statuses), "total_generated_texts": generated, "total_candidates_checked": checked,
        "first_try_style_slots": first_slots, "first_try_checked": first_checked, "first_try_passed": first_passed,
        "first_try_pass_rate": first_passed / first_slots if first_slots else None,
        "roundtrip_first_try_pass_rate": first_passed / first_checked if first_checked else None,
        "final_passed": final_passed, "rejection_reasons": dict(reasons), "tokens": dict(tokens),
        "models_resolved": {stage: dict(counts) for stage, counts in models.items()},
    }
    write_json(stats_path, stats)
    return output, stats


def add_synthesis_options(parser):
    parser.add_argument("--model", default=GEN_MODEL, help=f"참조 대사 생성 모델 (기본 {GEN_MODEL})")
    parser.add_argument("--rt-model", default=RT_MODEL, help=f"Round-Trip 역추출 모델 (기본 {RT_MODEL})")
    effort_levels = ("low", "medium", "high")
    parser.add_argument("--reasoning-effort", choices=effort_levels, default="medium", help="생성 모델의 추론 강도")
    parser.add_argument("--rt-reasoning-effort", choices=effort_levels, default="medium", help="역추출 모델의 추론 강도")
    parser.add_argument("--temperature", type=float, help="비추론 생성 모델의 온도 (기본 0.7)")
    parser.add_argument("--max-output-tokens", type=int, default=4096, help="생성 응답의 토큰 상한")
    parser.add_argument("--rt-max-output-tokens", type=int, default=6144, help="역추출 응답의 토큰 상한")
    parser.add_argument("--max-retries", type=int, choices=range(3), default=2, help="문체당 최대 재생성 횟수")
    parser.add_argument("--limit", type=int, help="이번 실행에서 처리할 시나리오 수 상한")
    parser.add_argument("--dry-run", action="store_true", help="API 호출 없이 그래프와 예상 호출 수 확인")


def validate_synthesis_options(parser, args):
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    if args.temperature is not None:
        if not 0 <= args.temperature <= 2:
            parser.error("--temperature must be within [0, 2]")
        if uses_reasoning(args.model):
            parser.error("추론 모델에는 --temperature 대신 --reasoning-effort를 지정하세요.")
    if min(args.max_output_tokens, args.rt_max_output_tokens) < 16:
        parser.error("응답 토큰 상한은 16 이상이어야 합니다.")


def has_pending_work(rounds, max_retries):
    has_unchecked = any("ok" not in item for item in iter_candidates(rounds))
    can_regenerate = len(rounds) <= max_retries and len(accepted_references(rounds)) < len(STYLE_KEYS)
    return has_unchecked or can_regenerate


def synthesize(records, args):
    selected = select_records(records, args.limit)
    if not selected:
        raise ValueError("No complete scenario group fits --limit")
    supported = [rec for rec in selected if rec["expect_supported"]]
    if args.dry_run:
        print(f"[DRY-RUN] {len(selected)}개 | 생성 1차 {len(supported)}회 | 역추출 1차 {len(supported) * 3}회")
        gen_settings, rt_settings = api_settings(args)
        print(f"[Responses API] 생성 {args.model} {gen_settings} | 역추출 {args.rt_model} {rt_settings}")
        if supported:
            print(format_scenario_prompt(supported[0]))
        return None

    out_path = args.output or args.input.with_name(
        f"{args.input.stem.removesuffix('_graphs')}_v{PIPELINE_VERSION}_synthesized.jsonl"
    )
    stats_path = args.stats_out or out_path.with_name(out_path.stem + "_stats.json")
    state_path = out_path.with_name(out_path.stem + "_state.json")
    lock_path = out_path.with_name(out_path.stem + "_state.json.lock")
    paths = [args.input, out_path, stats_path, state_path, lock_path]
    if len({path.resolve() for path in paths}) != len(paths):
        raise ValueError("Input, output, stats and checkpoint paths must be different")
    manifest = run_manifest(records, args)
    with run_lock(lock_path):
        if state_path.exists():
            state = json.loads(state_path.read_text(encoding="utf-8"))
            if state["manifest"] != manifest:
                raise ValueError(f"체크포인트의 입력 또는 합성 설정이 다릅니다. 새 출력 경로를 사용하세요. {state_path}")
            if any(len(rounds) > args.max_retries + 1 for rounds in state["rows"].values()):
                raise ValueError("--max-retries would exclude recorded attempts")
        elif out_path.exists() or stats_path.exists():
            raise ValueError(f"기존 산출물의 체크포인트가 없습니다. 새 출력 경로를 사용하세요. {out_path}")
        else:
            state = {"manifest": manifest, "rows": {}}

        def checkpoint():
            write_json(state_path, state)

        # 재개 시 실제로 API 호출이 남아 있을 때만 키가 필요하다.
        api_key = None
        if any(has_pending_work(state["rows"].get(rec["id"], []), args.max_retries) for rec in supported):
            load_dotenv_if_needed()
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY is required")
        # 선택된 행을 먼저 등록해 중간에 멈춰도 대조쌍의 두 변형이 결과에 함께 남게 한다.
        for rec in selected:
            state["rows"].setdefault(rec["id"], [])
        checkpoint()
        try:
            for rec in supported:
                rounds = state["rows"][rec["id"]]
                process_record(rec, rounds, args, api_key, checkpoint)
                print(f"{rec['id']}: {len(accepted_references(rounds))}/3 통과", flush=True)
        finally:
            # 실행이 중단돼도 저장된 후보와 통계로 현재 상태를 확인할 수 있게 한다.
            output, stats = export_results(records, state, args, out_path, stats_path)
        print(f"[Step 2+3] {out_path} | {stats['scenario_statuses']}")
        return output, stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--stats-out", type=Path)
    add_synthesis_options(parser)
    args = parser.parse_args()
    validate_synthesis_options(parser, args)

    try:
        records = read_jsonl(args.input)
        validate_records(records)
        synthesize(records, args)
    except (ValueError, OSError, RuntimeError) as exc:
        parser.exit(1, f"합성 실패: {exc}\n")


if __name__ == "__main__":
    main()
