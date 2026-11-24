#!/usr/bin/env python3
# 합성 대사의 조사·화체, 문체 다양성, 대조쌍 구별 정도와 저자 검수 현황을 집계한다.
"""Quantitative Quality Evaluator for Synthesized Korean RPG Graph-to-Text Datasets.

Zero-dependency Python 3.9+ script measuring 5 paper-ready quality dimensions:
1. Semantic Fidelity & Curation Yield (Round-Trip pass rates, entity recall, lexicon intrusion)
2. Korean Morphosyntactic & Pragmatic Accuracy (coda-particle allomorphy, speech-level, speech-act)
3. Multi-Reference Diversity (intra-scenario 1-Jaccard, corpus Distinct-1/2, length stats)
4. Contrastive Pair Discriminability (Axis 2 pair_id a vs b non-collapse check)
5. Author Audit Progress (Step 4 verification & optional edit-rate tracking)
"""

import argparse
import json
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

from dataset_io import read_jsonl
from schema import CHARACTERS, ITEMS, LOCATIONS

ALL_LEXICON_SURFACES: Set[str] = {e.surface for e in (CHARACTERS + ITEMS + LOCATIONS)}

# 받침 유형별로 어울리지 않는 조사를 모아 둔다.
# ㄹ 받침에는 은·이·을을 붙이지만 방향 조사에는 로를 붙인다.
INVALID_PARTICLES_BY_CODA: Dict[str, Tuple[str, ...]] = {
    "none": ("은", "이", "을", "으로"),
    "rieul": ("는", "가", "를", "으로"),
    "other": ("는", "가", "를", "로"),
}

SPEECH_LEVEL_PATTERNS: Dict[str, re.Pattern] = {
    "haeyo": re.compile(r"(요|세요|주세요|드려요|해요|나요|까요|예요|이에요)[.?!…\"']*$"),
    "haera": re.compile(r"(다|라|니|냐|느냐|어|아|줘|해|야|지|거라|너라)[.?!…\"']*$"),
    "hasipsio": re.compile(r"(습니다|ㅂ니다|읍니다|습니까|ㅂ니까|십시오|주십시오|하시지요|하시오)[.?!…\"']*$"),
}


def check_particle_errors(text: str, entities: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """Detect deterministic Korean particle allomorphy errors immediately after known entity surfaces."""
    errors: List[Dict[str, str]] = []
    for ent in entities:
        if ent["id"] == "player":
            continue
        surface = ent["surface"]
        coda = ent["coda"]
        # 개체명 바로 뒤에 붙은 조사 후보를 공백이나 문장부호 경계로 확인한다.
        invalid_parts = INVALID_PARTICLES_BY_CODA[coda]
        for part in invalid_parts:
            pattern = re.compile(rf"(?<![가-힣]){re.escape(surface)}{re.escape(part)}(?=\s|[,.?!]|$)")
            if pattern.search(text):
                # 조사 뒤의 경계 조건 덕분에 '이다', '이라'의 첫 음절은 조사로 잡히지 않는다.
                errors.append({"entity": surface, "coda": coda, "bad_particle": part})
    return errors


def check_speech_level(text: str, level: str) -> bool:
    pat = SPEECH_LEVEL_PATTERNS.get(level)
    if not pat:
        return True
    cleaned = text.strip()
    if level == "hasipsio":
        # '합니다', '갑니다', '줍니다'처럼 ㅂ이 앞 음절 받침에 들어가는 종결형도 확인한다.
        if re.search(r"[가-힣](니다|니까|십시오|시오)[.?!…\"']*$", cleaned):
            return True
    return bool(pat.search(cleaned))


def check_speech_act_surface(text: str, speech_act: str) -> bool:
    # 질문의 물음표와 경고의 핵심어를 표면 단서로 확인한다.
    cleaned = text.strip()
    if speech_act == "Question":
        return "?" in cleaned
    if speech_act == "Warn":
        return any(kw in cleaned for kw in ("조심", "주의", "위험", "경계", "살피", "삼가"))
    return True


def char_ngrams(text: str, n: int = 3) -> Set[str]:
    compact = re.sub(r"\s+", "", text)
    if len(compact) < n:
        return {compact} if compact else set()
    return {compact[i : i + n] for i in range(len(compact) - n + 1)}


def jaccard_similarity(a: Set[str], b: Set[str]) -> float:
    if not a and not b:
        return 1.0
    union = a | b
    return len(a & b) / len(union) if union else 0.0


def evaluate_dataset(path: Path, *, records=None) -> Dict[str, Any]:
    if records is None:
        records = read_jsonl(path)
    supported = [r for r in records if r.get("expect_supported", True)]
    unsupported = [r for r in records if not r.get("expect_supported", True)]

    total_refs = 0
    roundtrip_refs = 0
    first_try_checked = 0
    first_try_passed = 0
    rejection_reasons: Counter = Counter()

    entity_recall_hits = 0
    entity_recall_total = 0
    intrusion_refs = 0

    particle_error_refs = 0
    particle_error_examples: List[Dict[str, Any]] = []
    speech_level_ok = 0
    speech_act_ok = 0

    all_tokens: List[str] = []
    all_bigrams: List[Tuple[str, str]] = []
    char_lengths: List[int] = []
    token_lengths: List[int] = []
    intra_diversities: List[float] = []
    style_lengths: Dict[str, List[int]] = defaultdict(list)

    refs_per_scenario: Counter = Counter()
    author_verified_count = 0

    # 같은 대조쌍에 속한 두 변형의 참조 대사를 모은다.
    pairs: Dict[str, Dict[str, List[str]]] = defaultdict(dict)

    for rec in supported:
        refs = rec.get("references", [])
        # 최종본에는 저자가 추가한 문장이 있을 수 있어 자동 필터의 수율은 초안에서 계산한다.
        roundtrip_refs += len(rec.get("draft_references", refs))
        refs_per_scenario[len(refs)] += 1
        if rec.get("author_verified"):
            author_verified_count += 1

        # v2에서는 실제 역추출 호출 기록을, v1에서는 당시 저장된 시도 기록을 센다.
        rt_log = rec.get("roundtrip_log", {})
        for att in rt_log.get("attempts", []):
            if att.get("attempt") == 1 and ("usage" in att or rec.get("schema_version") is None):
                first_try_checked += 1
                if att.get("ok"):
                    first_try_passed += 1
            if att.get("ok") is False:
                rejection_reasons[att.get("reason", "unknown")] += 1

        # 대사에서 표기 여부를 확인할 개체를 정답 역할에서 모은다. 행동 주체는 이 검사에서 뺀다.
        gold_entities = set()
        for ev in rec["gold_tuples"]["events"]:
            for role, surface in ev["roles"].items():
                if role != "Agent":
                    gold_entities.add(surface)

        non_gold_lexicon = ALL_LEXICON_SURFACES - gold_entities
        # '수정구슬' 속 '수정'처럼 정답 개체명의 일부인 어휘는 외부 개체로 세지 않는다.
        non_gold_lexicon = {w for w in non_gold_lexicon if not any(w in g for g in gold_entities)}

        goal = rec["utterance_goal"]
        ref_texts: List[str] = []

        for ref in refs:
            text = ref["text"].strip()
            style = ref.get("style", "unknown")
            ref_texts.append(text)
            total_refs += 1

            # 정답 개체의 표기 여부와 입력에 없던 어휘 풀 개체의 등장 여부를 센다.
            for g_ent in gold_entities:
                entity_recall_total += 1
                if g_ent in text:
                    entity_recall_hits += 1

            if any(re.search(rf"(?<![가-힣]){re.escape(w)}", text) for w in non_gold_lexicon):
                intrusion_refs += 1

            # 조사 결합, 종결 어미, 화행 표면 단서를 점검한다.
            p_errs = check_particle_errors(text, rec["input_graph"]["entities"])
            if p_errs:
                particle_error_refs += 1
                if len(particle_error_examples) < 10:
                    particle_error_examples.append({"id": rec["id"], "style": style, "text": text, "errors": p_errs})

            if check_speech_level(text, goal["speech_level"]):
                speech_level_ok += 1
            if check_speech_act_surface(text, goal["speech_act"]):
                speech_act_ok += 1

            # 대사별 글자 수와 어절 수를 세고 인접 어절쌍을 모은다.
            char_lengths.append(len(text))
            toks = text.split()
            token_lengths.append(len(toks))
            style_lengths[style].append(len(text))
            all_tokens.extend(toks)
            all_bigrams.extend(zip(toks, toks[1:]))

        # 같은 시나리오 참조끼리 비교한 문자 3-그램 평균 자카드 유사도를 1에서 뺀다.
        if len(ref_texts) >= 2:
            ng_sets = [char_ngrams(t, 3) for t in ref_texts]
            sims = []
            for i in range(len(ng_sets)):
                for j in range(i + 1, len(ng_sets)):
                    sims.append(jaccard_similarity(ng_sets[i], ng_sets[j]))
            intra_diversities.append(1.0 - (sum(sims) / len(sims)))

        if rec.get("pair_id"):
            pairs[rec["pair_id"]][rec["pair_variant"]] = ref_texts

    # 양쪽에 참조가 있는 대조쌍에서 같은 대사가 두 변형에 함께 쓰였는지 확인한다.
    complete_pairs = 0
    non_collapsed_pairs = 0
    pair_divergences: List[float] = []
    for pid, variants in pairs.items():
        if "a" in variants and "b" in variants and variants["a"] and variants["b"]:
            complete_pairs += 1
            set_a = set(variants["a"])
            set_b = set(variants["b"])
            if set_a.isdisjoint(set_b):
                non_collapsed_pairs += 1
            # 두 변형에서 첫 참조를 뽑아 문자 3-그램 차이를 비교한다.
            div = 1.0 - jaccard_similarity(char_ngrams(variants["a"][0], 3), char_ngrams(variants["b"][0], 3))
            pair_divergences.append(div)

    def _rate(num: int, den: int) -> float:
        return round((num / den) * 100.0, 2) if den > 0 else 0.0

    report: Dict[str, Any] = {
        "dataset_file": str(path),
        "counts": {
            "total_scenarios": len(records),
            "supported_scenarios": len(supported),
            "unsupported_scenarios": len(unsupported),
            "total_verified_references": total_refs,
            "refs_per_supported_scenario": dict(sorted(refs_per_scenario.items())),
        },
        "1_semantic_fidelity": {
            "roundtrip_first_try_pass_rate_pct": _rate(first_try_passed, first_try_checked),
            "roundtrip_final_yield_pct": _rate(roundtrip_refs, len(supported) * 3),
            "entity_surface_recall_pct": _rate(entity_recall_hits, entity_recall_total),
            "lexicon_intrusion_rate_pct": _rate(intrusion_refs, total_refs),
            "rejection_reasons": dict(rejection_reasons),
        },
        "2_korean_morphosyntax_pragmatics": {
            "particle_allomorphy_error_rate_pct": _rate(particle_error_refs, total_refs),
            "speech_level_adherence_pct": _rate(speech_level_ok, total_refs),
            "speech_act_surface_alignment_pct": _rate(speech_act_ok, total_refs),
            "particle_error_samples": particle_error_examples,
        },
        "3_diversity_and_length": {
            "intra_scenario_diversity_mean": round(statistics.mean(intra_diversities), 4) if intra_diversities else 0.0,
            "corpus_distinct_1": round(len(set(all_tokens)) / len(all_tokens), 4) if all_tokens else 0.0,
            "corpus_distinct_2": round(len(set(all_bigrams)) / len(all_bigrams), 4) if all_bigrams else 0.0,
            "char_length_mean": round(statistics.mean(char_lengths), 2) if char_lengths else 0.0,
            "char_length_std": round(statistics.pstdev(char_lengths), 2) if len(char_lengths) > 1 else 0.0,
            "token_length_mean": round(statistics.mean(token_lengths), 2) if token_lengths else 0.0,
            "mean_char_length_by_style": {
                k: round(statistics.mean(v), 2) for k, v in sorted(style_lengths.items())
            },
        },
        "4_contrastive_pairs": {
            "evaluated_pairs": complete_pairs,
            "non_collapse_rate_pct": _rate(non_collapsed_pairs, complete_pairs),
            "pair_surface_divergence_mean": round(statistics.mean(pair_divergences), 4) if pair_divergences else 0.0,
        },
        "5_author_audit": {
            "author_verified_scenarios": author_verified_count,
            "author_verified_rate_pct": _rate(author_verified_count, len(supported)),
        },
    }
    reviewed = [rec for rec in supported if rec.get("author_verified") and "author_review" in rec]
    if reviewed:
        from review_dataset import review_stats

        report["5_author_audit"].update(review_stats(reviewed))
    return report


def print_summary(report: Dict[str, Any]) -> None:
    c = report["counts"]
    f = report["1_semantic_fidelity"]
    k = report["2_korean_morphosyntax_pragmatics"]
    d = report["3_diversity_and_length"]
    p = report["4_contrastive_pairs"]
    a = report["5_author_audit"]

    print(f"\n=== 합성 데이터셋 정량 품질 리포트 ({Path(report['dataset_file']).name}) ===")
    print(f"총 시나리오: {c['total_scenarios']} (지원 {c['supported_scenarios']} / 미지원 {c['unsupported_scenarios']}), 평가 대상 참조: {c['total_verified_references']}문장")
    print(f"시나리오당 참조 분포: {c['refs_per_supported_scenario']}")
    print("\n[1] 의미 충실도 & 필터링 수율")
    print(f"  - Round-Trip 1차 통과율 : {f['roundtrip_first_try_pass_rate_pct']}%")
    print(f"  - Round-Trip 최종 수율 : {f['roundtrip_final_yield_pct']}%")
    print(f"  - 개체 표면형 재현율    : {f['entity_surface_recall_pct']}%")
    print(f"  - 외부 어휘 환각(혼입)률: {f['lexicon_intrusion_rate_pct']}%")
    if f["rejection_reasons"]:
        print(f"  - 탈락 사유 분포        : {f['rejection_reasons']}")
    print("\n[2] 한국어 형태·조사·화체 적합성")
    print(f"  - 조사 이형태 오류율    : {k['particle_allomorphy_error_rate_pct']}%")
    print(f"  - 화체 종결어미 일치율  : {k['speech_level_adherence_pct']}%")
    print(f"  - 화행 표면 표지 일치율 : {k['speech_act_surface_alignment_pct']}%")
    print("\n[3] 다중 참조 다양성 & 길이")
    print(f"  - 시나리오 내 다양성(1-Jaccard): {d['intra_scenario_diversity_mean']}")
    print(f"  - 코퍼스 Distinct-1 / 2        : {d['corpus_distinct_1']} / {d['corpus_distinct_2']}")
    print(f"  - 평균 길이 (글자 / 어절)      : {d['char_length_mean']} (±{d['char_length_std']}) / {d['token_length_mean']}")
    print(f"  - 문체별 평균 글자 수          : {d['mean_char_length_by_style']}")
    print("\n[4] 최소 대조쌍 변별력 (Axis 2)")
    print(f"  - 평가 쌍 수 / 비붕괴율        : {p['evaluated_pairs']}쌍 / {p['non_collapse_rate_pct']}%")
    print(f"  - 쌍 간 평균 표면 차이(1-Jacc) : {p['pair_surface_divergence_mean']}")
    print("\n[5] 저자 전수 검수 현황 (Step 4)")
    print(f"  - 검수 완료: {a['author_verified_scenarios']}/{c['supported_scenarios']} ({a['author_verified_rate_pct']}%)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute quantitative quality metrics for synthesized dataset.")
    parser.add_argument("input", type=Path, help="Synthesized JSONL file (output of step2_3_openai_synthesize.py)")
    parser.add_argument("--json-out", type=Path, default=None, help="Optional path to save JSON report")
    args = parser.parse_args()

    report = evaluate_dataset(args.input)
    print_summary(report)

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n[저장 완료] {args.json_out}")


if __name__ == "__main__":
    main()
