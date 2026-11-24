# 한국어 RPG 사건–발화 평가셋

한국어 · [English](README_EN.md)

한국어 RPG 대사가 주어진 사건, 역할, 화행, 화체, 담화 관계를 얼마나 정확히 표현하는지 평가하기 위한 데이터셋 구축 파이프라인입니다. [데이터셋 프로토콜](../DATASET_PROTOCOL_KO_20261123.md)과 [BFGR 알고리즘 명세](../ALGORITHM_SPEC_KO_20261123.md)를 바탕으로 입력 그래프를 만들고, OpenAI 모델로 참조 대사를 합성한 뒤 Round-Trip 검증과 저자 검수를 거칩니다. 실험 모델의 발화는 독립적으로 역추출해 그래프 정답과 비교할 수 있습니다.

## 빠른 시작

`dataset_pipeline/`에서 실행합니다. Python 3.9 이상과 표준 라이브러리를 사용합니다. 합성 실행의 파일 잠금은 macOS·Linux에서 동작합니다. 그래프 생성, 로컬 검증, 리포트 재계산에는 API 키가 필요 없습니다.

```bash
python3 --version
python3 main.py --help
python3 main.py --split pilot30 --limit 5 --dry-run
```

`--dry-run`은 그래프 검증 결과, 선택된 시나리오 수, 모델 설정, 최초 시도의 예상 API 호출 수와 프롬프트 예시를 출력합니다. API 호출과 파일 저장은 생략합니다. 실제 합성은 유료 API를 사용합니다. 셸 환경 변수 `OPENAI_API_KEY`를 우선 읽고, 값이 없으면 실행 디렉터리 또는 스크립트 디렉터리의 로컬 `.env`에서 키를 찾습니다. `.env`는 Git 무시 대상입니다.

```dotenv
OPENAI_API_KEY=여기에_API_키를_입력
```

### 현재 저장된 실행 이어가기

아래 명령은 해당 실행의 체크포인트가 로컬에 있을 때 사용합니다. 새로 내려받았다면 [새 실험 시작하기](#새-실험-시작하기)를 참고하세요.

```bash
# 완료된 Pilot-30의 체크포인트를 읽고 리포트와 검수 양식을 재사용
python3 main.py --split pilot30 --run-dir data/runs/pilot30_seed2027_v4

# Full-240 체크포인트의 남은 후보부터 합성
python3 main.py --split full240 --run-dir data/runs/full240_seed2027_v4
```

현재 코드의 `PIPELINE_VERSION`은 `3`이므로 `--run-dir`의 기본 경로는 `data/runs/<분할>_seed<시드>_v3/`입니다. 위 `_v4` 폴더는 현재 프롬프트로 저장한 실행의 디렉터리 이름이며, 두 체크포인트에는 합성 버전 `3`과 그래프 스키마 버전 `2`가 기록돼 있습니다. 기존 `pilot30_seed2027_v3`에는 이전 프롬프트로 시나리오 5개를 처리한 기록이 있어 현재 프롬프트로 그 체크포인트를 열면 설정 불일치 오류가 발생합니다.

### 새 실험 시작하기

```bash
# 별도 폴더에서 최대 5개만 실제 합성
python3 main.py --split pilot30 --limit 5 \
  --run-dir data/runs/pilot30_smoke_seed2027

# 같은 체크포인트에서 나머지까지 합성
python3 main.py --split pilot30 \
  --run-dir data/runs/pilot30_smoke_seed2027
```

`--limit`는 시나리오 수 상한입니다. 대조쌍의 두 변형을 한 묶음으로 선택하며, 남은 자리보다 묶음이 크면 다음 묶음을 확인합니다. 현재 Pilot에서 `--limit 5`는 Axis 2 대조쌍 2개와 Axis 3 시나리오 1개를 고릅니다. 전체 그래프는 항상 생성·검증하며, `report.json`의 `scope.evaluated_scenarios`는 해당 폴더의 체크포인트에 누적된 시나리오 수입니다.

| 실행 범위 | 지원 시나리오 | 첫 시도 생성 요청 | 첫 시도 역추출 요청 예상 |
| --- | ---: | ---: | ---: |
| Pilot 최대 5개 | 5 | 5 | 15 |
| Pilot-30 | 30 | 30 | 90 |
| Full-240 | 220 | 220 | 660 |

생성은 한 시나리오의 필요한 문체를 한 요청에 묶고, 역추출은 생성 후보별로 호출합니다. 빈 응답과 중복 후보는 역추출 전에 탈락합니다. 재생성과 HTTP 재시도에 따라 실제 호출 수가 달라집니다. 미지원 경계 사례는 대사 합성 없이 평가 분할에 포함됩니다.

## 파이프라인 흐름

| 단계 | 실행 내용 | 확인할 결과 |
| --- | --- | --- |
| 1. 그래프 생성 | 분할·시드·스키마 버전으로 난수열을 고정하고 사건, 발화 목적, 정답을 생성합니다. 대조쌍, 역할 참조, 담화 트리, 축별 개수를 검증합니다. | `graphs.jsonl` |
| 2. 참조 합성 | 생성 모델에 입력 그래프와 발화 목적을 전달합니다. 문체 A는 간결형, B는 구어형, C는 절 배치나 성분 어순을 바꾼 구조 변형형입니다. | 체크포인트의 문체별 후보 |
| 3. Round-Trip 검증 | 후보 대사를 사건·역할·화행·화체·담화로 역추출한 뒤 정답과 비교합니다. 통과한 문체를 보존하고, 부족한 문체를 기본 최대 2회 다시 생성합니다. | `synthesized.jsonl`, `synthesis_stats.json` |
| 4. 저자 검수 | 자동 통과 초안을 읽고 채택·수정·탈락을 기록합니다. 필요한 문체는 저자가 직접 추가할 수 있습니다. | `author_review_<해시>.jsonl`, 검수 입력을 제출하면 `final.jsonl` |
| 평가 | 초안과 최종본의 정량 지표를 저장합니다. 실험 발화의 독립 역추출 결과를 전달하면 실험 모델 점수도 집계합니다. | `report.json` |

생성과 역추출은 Responses API의 Structured Outputs를 사용합니다. 생성 스키마는 이번에 요청한 문체만 요구하고, 역추출 스키마는 `speech_act`, `speech_level`, `events`, `discourse`, `extra_facts`를 필수로 요구합니다. API의 역할 배열은 저장 전에 역할명과 개체 표기를 연결한 JSON 객체로 변환합니다.

역추출 모델에는 대사, 화자·청자, 후보 개체 표기가 주어집니다. 비교기는 요청·경고에서 생략된 `Deliver`·`Fetch`의 행동 주체를 청자로 처리하고, 명시된 주체는 그 값으로 비교합니다. `Coord`의 순서는 바뀔 수 있으며 `Reason`의 원인과 결과, 중첩 구조는 일치해야 합니다. 자료형과 별도로 사건 및 역할 배정, 화행·화체, 추가 사실까지 검사합니다.

## 분할과 그래프 스키마

| 평가 축 | Pilot-30 | Full-240 | 구성 |
| --- | ---: | ---: | --- |
| Axis 1 단일 사건 | 0 | 60 | 프레임 5종 × 화행 4종 × 화체 3종. 각 화체에서 받침 없음·ㄹ 받침·기타 받침을 모두 사용합니다. |
| Axis 2 역할 교차 | 20 | 80 | 대조쌍의 두 변형에서 두 사건의 지정 역할값만 교환합니다. 사건 수 2·3·4별 대조쌍 수는 Pilot 4·3·3개, Full 14·13·13개입니다. |
| Axis 3 복합 담화 | 10 | 60 | 상태가 행동의 이유인 사례와 병렬 관계를 중첩한 이유절을 다룹니다. |
| Axis 4 실패 경계 | 0 | 40 | 지원 사례 20개는 사건 수 5·6·8에 각각 7·7·6개입니다. 미지원 사례 20개는 미지원 프레임·필수 역할 누락·담화 깊이 초과·상충 화행이 각각 5개입니다. |
| 합계 | 30 | 240 | Pilot은 모두 지원 사례이며 Full은 지원 220개와 미지원 20개를 포함합니다. |

UTF-8 JSONL 파일은 한 줄에 한 시나리오를 저장합니다. 주요 키는 다음과 같습니다.

| 키 | 내용 |
| --- | --- |
| `id`, `split`, `seed`, `schema_version` | 시나리오 ID, 분할, 생성 시드, 그래프 스키마 버전 |
| `axis`, `subtype`, `pair_id`, `pair_variant` | 평가 축과 세부 유형, Axis 2 대조쌍 ID 및 변형 `a`·`b` |
| `input_graph.entities` | 개체 ID, 표기 `surface`, 개체 유형, 생물성, 받침 범주 |
| `input_graph.events` | 사건 ID, 프레임, 역할명에서 개체 ID로 연결되는 `roles` |
| `utterance_goal` | 화자·청자 ID, `speech_act`, `speech_level`, 대상 사건 ID, 담화 관계 |
| `gold_tuples` | 대상 사건의 개체 ID를 실제 표기로 바꾼 정답과 정답의 화행·화체·담화 관계 |
| `expect_supported`, `unsupported_reason` | 지원 여부와 미지원 사유 |

화행은 `Inform`·`Request`·`Question`·`Warn`, 화체는 `haeyo`(해요체)·`haera`(해라체)·`hasipsio`(하십시오체)를 사용합니다. 담화는 사건이나 하위 관계를 묶는 `Coord`와 원인·결과를 연결하는 `Reason`으로 구성하며 지원 깊이는 2입니다.

| 프레임 | 필수 역할 | 선택 역할 |
| --- | --- | --- |
| `Deliver` | `Agent`, `Theme`, `Recipient` | `Source` |
| `Fetch` | `Agent`, `Theme`, `Source` | |
| `Need` | `Experiencer`, `Required_Entity` | |
| `Located_At` | `Entity`, `Location` | |
| `Danger` | `Location`, `Threat` | `Target` |

## 실행 옵션과 산출물

| `main.py` 옵션 | 기본값 | 용도 |
| --- | --- | --- |
| `--split`, `--seed` | `pilot30`, `2027` | 그래프 분할과 재현용 시드 |
| `--run-dir` | `data/runs/<분할>_seed<시드>_v3/` | 입력 그래프, 체크포인트, 리포트 저장 폴더 |
| `--limit`, `--dry-run` | 전체, 실제 실행 | 대상 시나리오 수 상한, 호출 전 설정 및 범위 확인 |
| `--model`, `--rt-model` | `gpt-6.2-sol`, `gpt-6.1-terra` | 참조 대사 생성 모델과 Round-Trip 역추출 모델 |
| `--reasoning-effort`, `--rt-reasoning-effort` | 둘 다 `medium` | 각 추론 모델의 강도. `low`·`medium`·`high` |
| `--max-output-tokens`, `--rt-max-output-tokens` | `4096`, `6144` | 생성·역추출의 응답별 토큰 상한 |
| `--temperature` | 비추론 생성 모델에서 `0.7` | `gpt-4o` 같은 비추론 모델의 생성 온도. 기본 GPT-6.2 Sol에서는 추론 강도를 사용합니다. |
| `--max-retries` | `2` | 최초 시도 이후 문체별 재생성 횟수. `0`·`1`·`2` |
| `--author-review` | 생략 | 저자가 작성한 검수 JSONL. 전달하면 최종본과 최종 품질 지표를 저장합니다. |
| `--predictions` | 생략 | 실험 모델의 발화 JSONL. 독립 역추출용 작업 파일을 만듭니다. |
| `--extractions`, `--extractor` | 생략 | 독립 역추출 결과 JSONL과 추출기 이름·버전. 둘을 전달하면 점수를 리포트에 추가합니다. |

| 실행 폴더의 파일 | 확인할 내용 |
| --- | --- |
| `graphs.jsonl` | 분할 전체의 입력 그래프, 발화 목적, 정답 |
| `synthesized_state.json` | 체크포인트. `manifest`에는 입력 해시와 모델·프롬프트·스키마 설정, `rows`에는 후보 생성·역추출·판정 이력 |
| `synthesized.jsonl` | 시나리오별로 통과한 `references`와 후보별 `roundtrip_log.attempts` |
| `synthesis_stats.json` | 실제 응답 모델명과 요청 설정, 통과·탈락 건수, 입력·출력·추론·캐시 토큰 통계 |
| `report.json` | 실행 범위 `scope`, 합성 통계 `synthesis`, 초안 품질 `draft_quality`, 저자 검수 상태 `author_review`. 해당 옵션으로 실행하면 `final_quality`와 `independent_scores`도 추가됩니다. |
| `author_review_<해시>.jsonl` | 현재 초안에 대응하는 저자 검수 양식. 실제 파일명은 `report.json`의 `artifacts.author_review_template`에 기록합니다. |
| `final.jsonl`, `extraction_jobs.jsonl` | 저자 검수를 적용한 최종본, 독립 추출에 전달할 작업 파일 |

### 재개 규칙

- 같은 분할·시드·모델·프롬프트·응답 설정과 같은 `--run-dir`로 재실행하면 이미 저장한 판정을 재사용합니다. API 키는 남은 합성 작업이 있을 때 읽습니다.
- `--limit`로 시작한 실행은 같은 폴더에서 옵션을 빼고 확장할 수 있습니다. `--max-retries 0`으로 시작한 실행은 같은 폴더에서 `1` 또는 `2`로 늘릴 수 있습니다.
- 모델, 프롬프트, 응답 설정 또는 그래프가 달라진 실험은 별도의 `--run-dir`에 저장합니다. 이전 체크포인트는 조건과 결과를 재현하는 데 쓰입니다.
- 실행 중에는 `synthesized_state.json`에 진행이 저장됩니다. `synthesized.jsonl`과 `synthesis_stats.json`은 합성 루프를 빠져나올 때 저장하고, `report.json`은 검증과 리포트 단계까지 마쳤을 때 생성합니다.

## 결과 해석과 문제 해결

### `0/3`과 시나리오 상태

`0/3 통과`는 해당 시나리오의 세 문체 중 자동 검증을 통과한 참조가 0개라는 뜻입니다. 원문이 생성됐어도 역할 누락, 화행·화체 오판, 담화 불일치, 응답 형식 오류 등으로 탈락할 수 있습니다. 시나리오별로 `synthesized.jsonl`의 `roundtrip_log.attempts`에서 후보 `text`, `extracted`, 대표 사유 `reason`, 전체 검출 사유 `errors`를 함께 확인합니다. 실행 중에는 같은 정보를 체크포인트에서 확인할 수 있습니다.

| 상태 또는 사유 | 해석 |
| --- | --- |
| `complete` | 세 문체 모두 자동 검증을 통과했습니다. 저자 검수 완료 여부는 `author_verified`에서 확인합니다. |
| `retry_limit_reached` | 정해진 재생성 횟수를 모두 썼고 통과 참조가 3개보다 적습니다. 참조가 2개라면 저자 검수 후 최종본 기준을 충족할 수 있습니다. |
| `pending` | 아직 판정 또는 추가 시도가 남아 있습니다. |
| `unsupported` | 그래프 생성 단계에서 미지원으로 분류한 경계 사례입니다. |
| `speech_act_mismatch`, `speech_level_mismatch` | 역추출한 화행 또는 화체가 정답과 다릅니다. 후보 종결 어미와 역추출 결과를 함께 확인합니다. |
| `missing_event`, `missing_role`, `role_swap` | 사건·역할 누락 또는 사건 간 역할값 교차입니다. 후보 텍스트에서 해당 사건과 역할이 표현됐는지 확인합니다. |
| `hallucinated_event`, `hallucinated_role`, `extra_fact` | 역추출에서 추가 사건·역할 또는 프레임으로 표현할 수 없는 추가 사실이 발견됐습니다. |
| `discourse_mismatch`, `invalid_extraction` | 담화 구조가 일치하지 않거나 역추출 결과의 형식을 검사할 수 없습니다. |
| `api_incomplete`, `api_refusal` | API 응답 미완료 또는 거절입니다. 기록된 응답 상세는 `usage` 또는 해당 시도의 `generation_usage`에 있습니다. |

`rejection_reasons`는 후보별 대표 사유의 합계입니다. 후보 하나에서 여러 문제가 검출되면 `errors`에 함께 기록됩니다. 대사에 있는 내용이 역추출에서 빠진 사례와 대사 자체의 사실 누락을 구분하려면 후보 원문과 정답을 함께 읽습니다.

### 리포트의 주요 지표

| 지표 | 계산 기준 |
| --- | --- |
| `first_try_pass_rate` | 첫 응답에서 통과한 참조 수 ÷ 첫 생성 응답에 기록된 문체 슬롯 수 |
| `roundtrip_first_try_pass_rate` | 첫 응답에서 통과한 참조 수 ÷ 역추출 호출 기록이 남은 첫 후보 수 |
| `roundtrip_final_yield_pct` | 자동 통과 참조 수 ÷ (지원 시나리오 수 × 3) × 100. 재시도 결과를 포함하며 최종본 리포트에서도 합성 초안의 자동 통과 참조를 분자로 사용합니다. |
| `drafts_with_at_least_two_references` | 자동 통과 참조가 2개 이상인 지원 시나리오 수 |
| 표면형 재현율·어휘 혼입률 | 정답 개체 표기가 문장에 나타나는지, 입력 역할에 없는 어휘 풀 개체가 문장에 나타나는지 확인하는 표면 지표 |
| 조사·화체·화행 지표 | 지정 개체 뒤의 조사 이형태, 종결 어미, 질문·경고의 표면 단서를 규칙으로 점검한 값 |
| `intra_scenario_diversity_mean`, Distinct-1/2 | 같은 시나리오의 참조 간 문자 3-그램 차이와, 전체 참조에서 고유 어절·인접 어절쌍의 비율 |
| 대조쌍 비붕괴율 | 양쪽에 참조가 있는 대조쌍 중 두 변형의 참조 집합에 동일한 대사가 없는 쌍의 비율. 표면 차이는 각각의 첫 참조를 비교합니다. |

`*_pct`는 0–100 단위의 백분율이고, `synthesis`의 `*_rate`와 독립 채점의 비율은 0–1 단위입니다. 품질 스크립트는 분모가 없는 일부 백분율을 `0.0`으로 기록하므로 참조 수와 평가 대조쌍 수를 함께 확인합니다.

### 저장된 Pilot 결과 예시

2026-11-24에 로컬에 저장된 리포트를 확인한 결과입니다. 두 실행은 평가 범위와 프롬프트가 달라 통과율을 직접 비교할 때 이 조건을 함께 봐야 합니다. 각 리포트는 아래 실행 폴더의 `report.json`에 저장됩니다.

| 실행 폴더 | 평가 범위 | Round-Trip 1차 통과 | 자동 통과 참조 | 참조 2개 이상 |
| --- | ---: | ---: | ---: | ---: |
| `pilot30_seed2027_v3` | 5개 | 12/15 (80.00%) | 12/15 | 5/5 |
| `pilot30_seed2027_v4` | 30개 | 73/90 (81.11%) | 82/90 | 27/30 |

30건 실행의 참조 분포는 3개가 27건, 1개가 1건, 0개가 2건입니다. 대표 탈락 사유는 `extra_fact` 21건과 `speech_act_mismatch` 17건이었습니다. 이 수치는 자동 검증 초안 기준이며, 해당 실행의 리포트에는 저자 검수가 `pending`으로 기록돼 있습니다.

### 흔히 발생하는 실행 문제

| 문제 | 확인할 항목 |
| --- | --- |
| `OPENAI_API_KEY is required` | 새 API 작업이 남아 있습니다. 셸 환경 변수나 작업 폴더의 `.env`에 키를 설정합니다. |
| 체크포인트의 입력 또는 합성 설정이 다름 | `synthesis_stats.json` 또는 체크포인트의 `manifest`에서 모델·프롬프트·토큰 상한을 확인합니다. 기존 실험은 해당 설정으로 이어가고, 변경한 설정은 별도 폴더를 사용합니다. |
| `Another synthesis run is using ...` | 같은 실행 폴더를 사용하는 작업이 있습니다. 먼저 해당 작업이 끝난 뒤 재개합니다. |
| `api_incomplete`가 반복됨 | `usage.api_error_detail`이 `max_output_tokens`인지 확인합니다. 해당 단계의 토큰 상한을 변경해 새 폴더에서 점검할 수 있습니다. |
| 부분 실행의 합성 초안 검증이 실패함 | 실제로 부분 실행인 경우 `validate_dataset.py --stage synthesis --partial`을 사용합니다. 대조쌍의 두 변형은 포함돼야 합니다. |
| 검수 또는 독립 결과의 ID 불일치 | `report.json`의 `scope`와 이번 실행 폴더의 `synthesized.jsonl`에서 실제 ID 범위를 확인합니다. |

## 저자 검수와 최종본

실행 후 `report.json`의 `artifacts.author_review_template`에 표시된 양식을 작업용 파일로 복사합니다. 지원 시나리오마다 `reviewer`에 검수자 식별자를 적고, 각 후보의 정답 일치 여부와 문장 품질을 검토한 뒤 `verified`를 `true`로 바꿉니다. 한 시나리오의 기존 초안 전체에 판정을 남기고, 서로 다른 최종 참조를 2–3개 확보해야 합니다.

| `decision` | `original_text` | `text` |
| --- | --- | --- |
| `keep` | 양식에 저장된 원문 | 같은 문장 |
| `edit` | 양식에 저장된 원문 | 수정한 문장 |
| `reject` | 양식에 저장된 원문 | `null` |
| `add` | `null` | 해당 시나리오에 빠진 문체로 저자가 작성한 문장 |

기존 초안과 검수 양식의 `original_text`는 일치해야 합니다. `add`는 기존 초안에 없는 문체 키를 사용합니다. 수정·탈락·추가 이유는 `note`에 기록할 수 있습니다. 미지원 사례는 검수 양식에서 제외되고 최종본에는 참조 없이 남습니다.

```bash
# author_reviewed.jsonl은 작성한 검수 파일의 경로로 바꿉니다.
python3 main.py --split pilot30 --run-dir data/runs/pilot30_seed2027_v4 \
  --author-review author_reviewed.jsonl

python3 validate_dataset.py data/runs/pilot30_seed2027_v4/final.jsonl --stage final
```

최종본은 `draft_references`와 `author_review`에 원문과 판단 이력을 보존합니다. 리포트에서 최종 참조 수, 수정·탈락·추가 수, 수정률, 평균 길이를 확인할 수 있습니다. 수정률은 수정한 초안 수를 편집 전 초안 수로 나눕니다.

## 실험 모델 독립 채점

1. 실험 모델 출력을 UTF-8 JSONL로 저장합니다. 지원 여부는 실험 모델의 실제 반환을 그대로 기록합니다.

   ```jsonl
   {"id":"pilot30_ax2_001_a","text":"대사의 원문"}
   {"id":"full240_ax4_unsup_001","status":"unsupported"}
   ```

   위 두 행은 각각 발화와 미지원 반환의 형식 예시입니다. 실제 파일에는 평가 대상 분할의 ID를 사용합니다.

2. `--predictions`로 작업 파일을 만듭니다. 작업 파일에는 발화, 화자·청자와 후보 개체 표기를 담은 추출용 프롬프트가 들어갑니다. 이 파일을 독립 추출기 또는 사람 검수자에게 전달합니다.

   ```bash
   python3 main.py --split pilot30 --run-dir data/runs/pilot30_seed2027_v4 \
     --predictions outputs.jsonl
   ```

3. 역추출 결과를 `{"id":"시나리오 ID","extraction":{...}}` 형식으로 저장합니다. `extraction`에는 `speech_act`, `speech_level`, `events`, `discourse`, `extra_facts`가 필요합니다. 각 사건은 `event_id`, `frame`, `roles`를 담고, 외부 추출 결과의 `roles`는 `{"Theme":"약초"}`처럼 역할명과 표기를 연결한 객체입니다. 미지원 반환은 원래의 `{"id":"…","status":"unsupported"}`를 사용합니다.

   ```bash
   python3 main.py --split pilot30 --run-dir data/runs/pilot30_seed2027_v4 \
     --extractions independent_extractions.jsonl \
     --extractor '독립 추출기 이름과 버전'
   ```

통합 명령은 현재 합성 산출물의 시나리오 ID와 입력 파일의 ID가 일치하는지 검사합니다. 전체 분할의 그래프와 출력만으로 독립 채점을 실행하려면 아래의 개별 명령을 사용할 수 있습니다. 채점은 사건 수·축별 역할 바인딩 일치율, 역할 교차율, 누락률, 환각률, 화행·화체 일치율을 집계합니다. `exact_rate`는 모든 검사에 통과한 지원 시나리오 비율이며, `unsupported_detection_rate`는 미지원 시나리오를 미지원으로 반환한 비율입니다. 대조쌍 양쪽의 역할 바인딩 일치 건수는 `contrast_pairs`에서 확인합니다.

## 단계별 실행과 테스트

```bash
# 그래프만 생성·검증
python3 step1_generate_graphs.py --split pilot30 --seed 2027
python3 validate_dataset.py data/pilot30_graphs.jsonl

# 그래프를 입력해 합성 범위를 확인한 뒤 5개만 합성
python3 step2_3_openai_synthesize.py --input data/pilot30_graphs.jsonl --limit 5 --dry-run
python3 step2_3_openai_synthesize.py --input data/pilot30_graphs.jsonl --limit 5
python3 validate_dataset.py data/pilot30_v3_synthesized.jsonl --stage synthesis --partial

# 현재 저장된 초안의 품질 지표를 API 호출 없이 다시 집계
python3 evaluate_synthetic_quality.py data/runs/pilot30_seed2027_v4/synthesized.jsonl

# 전체 분할의 검수 양식을 별도로 만들거나 작성한 양식으로 최종본 생성
python3 review_dataset.py prepare --input data/runs/pilot30_seed2027_v4/synthesized.jsonl \
  --output author_review_to_fill.jsonl
python3 review_dataset.py finalize --input data/runs/pilot30_seed2027_v4/synthesized.jsonl \
  --review author_reviewed.jsonl --output pilot30_final.jsonl

# 합성과 별개로 기존 전체 그래프에 대한 독립 추출 작업 준비 및 점수 계산
python3 score_outputs.py prepare --input data/runs/pilot30_seed2027_v4/graphs.jsonl \
  --predictions outputs.jsonl --output extraction_jobs.jsonl
python3 score_outputs.py score --input data/runs/pilot30_seed2027_v4/graphs.jsonl \
  --predictions independent_extractions.jsonl \
  --extractor '독립 추출기 이름과 버전' --output scores.json

# API를 모의 호출하는 회귀 테스트
python3 -m unittest -v
```

## 코드별 역할

| 파일 | 역할 |
| --- | --- |
| [main.py](main.py) | 실행 순서, 체크포인트 폴더, 통합 리포트 관리 |
| [schema.py](schema.py), [step1_generate_graphs.py](step1_generate_graphs.py) | 스키마와 어휘 정의, 결정론적 그래프·대조쌍 생성 |
| [step2_3_openai_synthesize.py](step2_3_openai_synthesize.py), [openai_responses.py](openai_responses.py) | 생성·역추출 프롬프트와 스키마, API 요청, 재생성, 체크포인트 집계 |
| [semantic.py](semantic.py), [validate_dataset.py](validate_dataset.py) | 사건·역할·담화 비교와 그래프·대조쌍·초안·최종본 검증 |
| [review_dataset.py](review_dataset.py) | 저자 검수 양식, 최종본, 검수 통계 |
| [evaluate_synthetic_quality.py](evaluate_synthetic_quality.py), [score_outputs.py](score_outputs.py) | 참조 품질 진단과 실험 모델 독립 채점 |
| [dataset_io.py](dataset_io.py), [test_pipeline.py](test_pipeline.py) | UTF-8 JSONL 입출력과 원자적 저장, 파이프라인 회귀 테스트 |
