# 그래프 형식, 프레임별 역할, 조사 평가에 사용할 RPG 어휘를 정의한다.
"""Schema, Korean phonological coda utility, and balanced RPG lexicon for BFGR benchmark."""

from copy import deepcopy
from dataclasses import dataclass, asdict
from typing import Dict, List, Literal, Tuple

SCHEMA_VERSION = 2
CODAS = ("none", "rieul", "other")
STYLE_KEYS = ("style_a_concise", "style_b_colloquial", "style_c_structural")
AXIS1 = "Axis1_SingleEvent"
AXIS2 = "Axis2_ContrastiveBinding"
AXIS3 = "Axis3_ComplexDiscourse"
AXIS4 = "Axis4_FailureBoundary"
DIAG_ROLE = "Diag_RoleDirection"
DIAG_DISCOURSE = "Diag_DiscourseContrast"
# 생성기와 검증기가 같은 분할별 시나리오 수와 사건 수 분포를 사용한다.
SPLIT_COUNTS = {
    "pilot30": {AXIS2: 20, AXIS3: 10},
    "full240": {AXIS1: 60, AXIS2: 80, AXIS3: 60, AXIS4: 40},
    "diag40": {DIAG_ROLE: 20, DIAG_DISCOURSE: 20},
}
AXIS2_PAIR_COUNTS = {"pilot30": {2: 4, 3: 3, 4: 3}, "full240": {2: 14, 3: 13, 4: 13}}
AXIS4_STRESS_COUNTS = {5: 7, 6: 7, 8: 6}
BOUNDARIES = ("unsupported_frame", "missing_required_role", "excessive_nesting_depth", "conflicting_speech_acts")
MAX_DISCOURSE_DEPTH = 2

CodaType = Literal["none", "rieul", "other"]
AnimacyType = Literal["animate", "inanimate"]
EntityType = Literal["character", "item", "location"]
SpeechLevel = Literal["haeyo", "haera", "hasipsio"]
SpeechAct = Literal["Inform", "Request", "Question", "Warn"]

SUPPORTED_FRAMES: Tuple[str, ...] = (
    "Deliver",
    "Fetch",
    "Need",
    "Located_At",
    "Danger",
)

UNSUPPORTED_FRAMES: Tuple[str, ...] = (
    "Trade",
    "Craft",
    "Betray",
    "Enchant",
)

SPEECH_LEVELS: Tuple[SpeechLevel, ...] = ("haeyo", "haera", "hasipsio")
SPEECH_ACTS: Tuple[SpeechAct, ...] = ("Inform", "Request", "Question", "Warn")

# 프레임의 필수·선택 역할과 각 역할에 연결할 수 있는 개체 유형을 지정한다.
FRAME_REQUIRED_ROLES: Dict[str, Tuple[str, ...]] = {
    "Deliver": ("Agent", "Theme", "Recipient"),
    "Fetch": ("Agent", "Theme", "Source"),
    "Need": ("Experiencer", "Required_Entity"),
    "Located_At": ("Entity", "Location"),
    "Danger": ("Location", "Threat"),
}
FRAME_OPTIONAL_ROLES = {"Deliver": ("Source",), "Danger": ("Target",)}
ROLE_TYPES = {
    "Agent": "character", "Theme": "item", "Recipient": "character", "Source": "location",
    "Experiencer": "character", "Required_Entity": "item", "Entity": "item", "Location": "location",
    "Threat": "character", "Target": "character",
}
# 대조쌍에서 교환할 역할과 단일 사건의 받침 평가에 사용할 역할을 지정한다.
PAIR_SWAP_ROLES = {"Deliver": "Recipient", "Fetch": "Source", "Need": "Required_Entity", "Located_At": "Location"}
FOCAL_ROLES = {"Deliver": "Theme", "Fetch": "Theme", "Need": "Experiencer", "Located_At": "Entity", "Danger": "Location"}


def selected_events(graph, goal):
    # 발화 대상 사건만 목적에 지정된 순서대로 꺼내고, 역할의 개체 ID를 실제 표기로 바꾼다.
    surfaces = {entity["id"]: entity["surface"] for entity in graph["entities"]}
    events = {event["id"]: event for event in graph["events"]}
    return [
        {"event_id": event_id, "frame": events[event_id]["frame"],
         "roles": {role: surfaces[entity] for role, entity in sorted(events[event_id]["roles"].items())}}
        for event_id in goal["target_events"]
    ]


def derive_gold(graph, goal):
    # 정답의 담화 관계를 복사해 발화 목적의 내용을 나중에 바꿔도 정답이 함께 바뀌지 않게 한다.
    return {
        "speech_act": goal["speech_act"], "speech_level": goal["speech_level"],
        "events": selected_events(graph, goal), "discourse": deepcopy(goal["discourse"]),
    }


def compute_coda(surface: str) -> CodaType:
    """Compute Korean phonological coda category of the final Hangul syllable."""
    if not surface:
        raise ValueError("Empty surface string")
    last_char = surface[-1]
    code = ord(last_char)
    if 0xAC00 <= code <= 0xD7A3:
        # 한글 완성형 음절은 종성 28가지를 순서대로 배치하며 0은 받침 없음, 8은 ㄹ 받침이다.
        jong = (code - 0xAC00) % 28
        if jong == 0:
            return "none"
        if jong == 8:
            return "rieul"
        return "other"
    raise ValueError(f"Non-Hangul final character in surface: {surface!r}")


@dataclass(frozen=True)
class Entity:
    id: str
    surface: str
    coda: CodaType
    animacy: AnimacyType
    type: EntityType

    def to_dict(self) -> Dict[str, str]:
        return asdict(self)


def _make_entity(eid: str, surface: str, animacy: AnimacyType, etype: EntityType, expected_coda: CodaType) -> Entity:
    actual_coda = compute_coda(surface)
    assert actual_coda == expected_coda, (
        f"Coda mismatch for {surface}: expected {expected_coda}, got {actual_coda}"
    )
    return Entity(id=eid, surface=surface, coda=actual_coda, animacy=animacy, type=etype)


# 어휘 풀에서 받침 없음, ㄹ 받침, 기타 받침을 같은 수로 배치한다. 플레이어는 공통 청자다.
PLAYER_ENTITY = Entity(
    id="player",
    surface="모험가",
    coda=compute_coda("모험가"),
    animacy="animate",
    type="character",
)

CHARACTERS: List[Entity] = [
    # 받침 없음 6개
    _make_entity("char_healer", "치유사", "animate", "character", "none"),
    _make_entity("char_knight", "기사", "animate", "character", "none"),
    _make_entity("char_mage", "마법사", "animate", "character", "none"),
    _make_entity("char_blacksmith", "대장장이", "animate", "character", "none"),
    _make_entity("char_alchemist", "연금술사", "animate", "character", "none"),
    _make_entity("char_scout", "정찰자", "animate", "character", "none"),
    # ㄹ 받침 6개
    _make_entity("char_envoy", "사절", "animate", "character", "rieul"),
    _make_entity("char_soldier", "병졸", "animate", "character", "rieul"),
    _make_entity("char_rael", "라엘", "animate", "character", "rieul"),
    _make_entity("char_kael", "카엘", "animate", "character", "rieul"),
    _make_entity("char_haesol", "해솔", "animate", "character", "rieul"),
    _make_entity("char_hanul", "한울", "animate", "character", "rieul"),
    # 기타 받침 6개
    _make_entity("char_merchant", "상인", "animate", "character", "other"),
    _make_entity("char_chief", "촌장", "animate", "character", "other"),
    _make_entity("char_rogue", "도적", "animate", "character", "other"),
    _make_entity("char_general", "장군", "animate", "character", "other"),
    _make_entity("char_captain", "대장", "animate", "character", "other"),
    _make_entity("char_hunter", "사냥꾼", "animate", "character", "other"),
]

ITEMS: List[Entity] = [
    # 받침 없음 8개
    _make_entity("item_herb", "약초", "inanimate", "item", "none"),
    _make_entity("item_bandage", "붕대", "inanimate", "item", "none"),
    _make_entity("item_map", "지도", "inanimate", "item", "none"),
    _make_entity("item_key", "열쇠", "inanimate", "item", "none"),
    _make_entity("item_cloak", "망토", "inanimate", "item", "none"),
    _make_entity("item_ring", "반지", "inanimate", "item", "none"),
    _make_entity("item_shield", "방패", "inanimate", "item", "none"),
    _make_entity("item_holywater", "성수", "inanimate", "item", "none"),
    # ㄹ 받침 8개
    _make_entity("item_arrow", "화살", "inanimate", "item", "rieul"),
    _make_entity("item_treasure", "보물", "inanimate", "item", "rieul"),
    _make_entity("item_relic", "유물", "inanimate", "item", "rieul"),
    _make_entity("item_mirror", "거울", "inanimate", "item", "rieul"),
    _make_entity("item_rope", "밧줄", "inanimate", "item", "rieul"),
    _make_entity("item_steel", "강철", "inanimate", "item", "rieul"),
    _make_entity("item_orb", "수정구슬", "inanimate", "item", "rieul"),
    _make_entity("item_torch", "횃불", "inanimate", "item", "rieul"),
    # 기타 받침 8개
    _make_entity("item_potion", "물약", "inanimate", "item", "other"),
    _make_entity("item_dagger", "단검", "inanimate", "item", "other"),
    _make_entity("item_letter", "서신", "inanimate", "item", "other"),
    _make_entity("item_ore", "광석", "inanimate", "item", "other"),
    _make_entity("item_armor", "갑옷", "inanimate", "item", "other"),
    _make_entity("item_longsword", "장검", "inanimate", "item", "other"),
    _make_entity("item_rations", "식량", "inanimate", "item", "other"),
    _make_entity("item_crystal", "수정", "inanimate", "item", "other"),
]

LOCATIONS: List[Entity] = [
    # 받침 없음 4개
    _make_entity("loc_pass", "고개", "inanimate", "location", "none"),
    _make_entity("loc_citadel", "성채", "inanimate", "location", "none"),
    _make_entity("loc_dock", "부두", "inanimate", "location", "none"),
    _make_entity("loc_graveyard", "묘지", "inanimate", "location", "none"),
    # ㄹ 받침 4개
    _make_entity("loc_cave", "동굴", "inanimate", "location", "rieul"),
    _make_entity("loc_village", "마을", "inanimate", "location", "rieul"),
    _make_entity("loc_sanctum", "성물실", "inanimate", "location", "rieul"),
    _make_entity("loc_well", "우물", "inanimate", "location", "rieul"),
    # 기타 받침 4개
    _make_entity("loc_forest", "숲", "inanimate", "location", "other"),
    _make_entity("loc_mine", "광산", "inanimate", "location", "other"),
    _make_entity("loc_ruins", "유적", "inanimate", "location", "other"),
    _make_entity("loc_temple", "신전", "inanimate", "location", "other"),
]

ENTITY_BY_ID: Dict[str, Entity] = {
    e.id: e for e in ([PLAYER_ENTITY] + CHARACTERS + ITEMS + LOCATIONS)
}
