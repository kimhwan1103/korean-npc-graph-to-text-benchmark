# 프로토콜 리뷰에서 확인한 결함의 재발을 검증한다. API 응답은 테스트 안에서 흉내 낸다.
"""Regressions from the protocol review. All synthesis calls are mocked."""

import contextlib
import io
import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import main as pipeline
import openai_responses as responses
import step1_generate_graphs as generator
import step2_3_openai_synthesize as synth
from dataset_io import read_jsonl, write_jsonl
from review_dataset import finalize, prepare, review_stats
from schema import AXIS2, AXIS3, DIAG_DISCOURSE, SPEECH_LEVELS, STYLE_KEYS, derive_gold
from score_outputs import score
from semantic import comparison_errors
from validate_dataset import validate_records


def extraction(record):
    return {**deepcopy(record["gold_tuples"]), "extra_facts": []}


def api_extraction(extracted):
    return {**extracted, "events": [
        {**event, "roles": [{"role": role, "value": value} for role, value in event["roles"].items()]}
        for event in extracted["events"]
    ]}


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.input = self.generate("pilot30", self.root / "pilot_graphs.jsonl")
        self.pilot = read_jsonl(self.input)

    def generate(self, split, path):
        args = ["generate", "--split", split, "--out", str(path)]
        with patch("sys.argv", args), contextlib.redirect_stdout(io.StringIO()):
            generator.main()
        return path

    def test_split_coverage_and_reproducibility(self):
        full = read_jsonl(self.generate("full240", self.root / "full.jsonl"))
        self.assertEqual(validate_records(self.pilot), ("pilot30", 10))
        self.assertEqual(validate_records(full), ("full240", 40))
        self.assertFalse({rec["id"] for rec in self.pilot} & {rec["id"] for rec in full})
        again = self.generate("pilot30", self.root / "again.jsonl")
        self.assertEqual(self.input.read_bytes(), again.read_bytes())

    def test_validator_rejects_fake_pairs_and_dangling_roles(self):
        # 사건 순서만 뒤집으면 지정 역할의 교환이 없는 대조쌍이 된다.
        a, b = deepcopy(self.pilot[:2])
        b["input_graph"]["events"] = list(reversed(a["input_graph"]["events"]))
        b["gold_tuples"] = derive_gold(b["input_graph"], b["utterance_goal"])
        with self.assertRaisesRegex(ValueError, "transpose"):
            validate_records([a, b], partial=True)
        a["input_graph"]["events"][0]["roles"]["Recipient"] = "unknown"
        with self.assertRaisesRegex(ValueError, "Dangling entity"):
            validate_records([a, b], partial=True)
        with self.assertRaisesRegex(ValueError, "variants"):
            validate_records(self.pilot[:1], partial=True)

    def test_roundtrip_and_generation_use_correct_inputs(self):
        rec = deepcopy(next(rec for rec in self.pilot if rec["axis"] == AXIS3))
        addressee = "모험가"
        good = extraction(rec)
        # 생략된 행동 주체는 허용하고, 명시적으로 잘못된 주체와 뒤집힌 인과는 잡아야 한다.
        del good["events"][1]["roles"]["Agent"]
        self.assertTrue(synth.verify_roundtrip(rec["gold_tuples"], good, addressee)[0])
        bad = deepcopy(good)
        bad["events"][1]["roles"]["Agent"] = "다른 인물"
        self.assertIn("role_value_mismatch", comparison_errors(rec["gold_tuples"], bad, addressee))
        bad = deepcopy(good)
        reason = bad["discourse"][0]
        reason["cause"], reason["effect"] = reason["effect"], reason["cause"]
        self.assertEqual(comparison_errors(rec["gold_tuples"], bad, addressee), ["discourse_mismatch"])
        bad["speech_level"] = "haera" if good["speech_level"] == "haeyo" else "haeyo"
        self.assertIn("speech_level_mismatch", comparison_errors(rec["gold_tuples"], bad, addressee))
        self.assertEqual(comparison_errors(rec["gold_tuples"], {**good, "events": None}), ["invalid_extraction"])

        previous = synth.format_scenario_prompt(rec)
        rec["gold_tuples"]["events"][0]["roles"]["Required_Entity"] = "골드변경"
        self.assertEqual(previous, synth.format_scenario_prompt(rec))
        entity = rec["input_graph"]["events"][0]["roles"]["Required_Entity"]
        next(ent for ent in rec["input_graph"]["entities"] if ent["id"] == entity)["surface"] = "그래프변경"
        self.assertIn("그래프변경", synth.format_scenario_prompt(rec))

    def test_checkpoint_resume_and_cumulative_stats(self):
        out = self.root / "output.jsonl"
        stats = self.root / "output_stats.json"
        args = ["synth", "--input", str(self.input), "--output", str(out), "--limit", "5", "--max-retries", "0"]
        calls = {"generation": 0, "extraction": 0}
        answers = {}

        def api(key, model, system, prompt, **request):
            usage = {"prompt_tokens": 1, "completion_tokens": 1, "model": model}
            if system == synth.GEN_SYSTEM_PROMPT:
                calls["generation"] += 1
                section = prompt.split("[실제 생성 과제]\n", 1)[1]
                task, _ = json.JSONDecoder().raw_decode(section)
                texts = {style: f"{calls['generation']}-{style}" for style in STYLE_KEYS}
                for sentence in texts.values():
                    answers[sentence] = {**task, "extra_facts": []}
                return texts, usage
            calls["extraction"] += 1
            if calls["extraction"] == 2:
                # 후보 한 건의 판정만 저장된 상태에서 API가 끊기는 상황을 재현한다.
                raise RuntimeError("simulated interruption")
            return api_extraction(answers[json.loads(prompt)["utterance"]]), usage

        with patch.dict("os.environ", {"OPENAI_API_KEY": "stub"}), patch("sys.argv", args), \
                patch.object(synth, "call_openai_json", api), contextlib.redirect_stdout(io.StringIO()), \
                contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                synth.main()
            rows = read_jsonl(out)
            self.assertEqual(validate_records(rows, "synthesis", partial=True), ("pilot30", 2))
            self.assertEqual(len(rows[0]["references"]), 1)
            synth.main()
            self.assertEqual(len(read_jsonl(out)), 5)
            baseline = json.loads(stats.read_text())
            self.assertEqual(baseline["first_try_passed"], 15)
            self.assertEqual(baseline["total_candidates_checked"], 15)
            self.assertEqual(calls, {"generation": 5, "extraction": 16})
            synth.main()
            self.assertEqual(json.loads(stats.read_text()), baseline)
            self.assertEqual(calls, {"generation": 5, "extraction": 16})

            # 동일한 시나리오 ID라도 입력의 시드가 달라지면 재개를 거부해야 한다.
            changed = deepcopy(self.pilot)
            for record in changed:
                record["seed"] += 1
            write_jsonl(self.input, changed)
            with self.assertRaises(SystemExit):
                synth.main()
            self.assertEqual(json.loads(stats.read_text()), baseline)

    def test_incomplete_styles_resume_with_remaining_budget(self):
        record = self.pilot[0]
        rounds = []
        calls = [
            dict(zip(STYLE_KEYS, ("첫 발화", "첫 발화", ""))),
            dict(zip(STYLE_KEYS[1:], ("두 번째 발화", "세 번째 발화"))),
        ]
        args = SimpleNamespace(model="test-gen", rt_model="test-rt", temperature=0.7, max_retries=0,
                               reasoning_effort="medium", rt_reasoning_effort="medium",
                               max_output_tokens=4096, rt_max_output_tokens=6144)

        def api(key, model, system, prompt, **request):
            usage = {"prompt_tokens": 1, "completion_tokens": 1, "model": model}
            return (calls.pop(0), usage) if system == synth.GEN_SYSTEM_PROMPT else (api_extraction(extraction(record)), usage)

        with patch.object(synth, "call_openai_json", api):
            synth.process_record(record, rounds, args, "stub", lambda: None)
            self.assertEqual(len(synth.accepted_references(rounds)), 1)
            self.assertEqual([item["reason"] for item in synth.iter_candidates(rounds)],
                             ["ok", "duplicate_reference", "empty_or_invalid_output"])
            args.max_retries = 2
            synth.process_record(record, rounds, args, "stub", lambda: None)
        self.assertEqual(len(rounds), 2)
        self.assertEqual(len(synth.accepted_references(rounds)), 3)

    def test_author_audit_and_independent_scoring(self):
        drafts = [{**rec, "references": [], "author_verified": False,
                   "roundtrip_log": {"status": "retry_limit_reached", "passed_count": 0, "attempts": []}}
                  for rec in self.pilot]
        reviews = list(prepare(drafts))
        with self.assertRaisesRegex(ValueError, "reviewer"):
            finalize(drafts, reviews)
        for review in reviews:
            review.update(reviewer="test", verified=True, references=[
                {"style": style, "original_text": None, "decision": "add", "text": f"테스트 발화 {i}"}
                for i, style in enumerate(STYLE_KEYS[:2])
            ])
        published = finalize(drafts, reviews)
        self.assertEqual(review_stats(published)["final_references"], 60)

        outputs = [{"id": rec["id"], "extraction": extraction(rec)} for rec in self.pilot]
        self.assertEqual(self.pilot[0]["axis"], AXIS2)
        # 같은 대조쌍의 반대쪽 정답을 넣어 역할 교차를 만들고, 다른 행의 사건 하나는 뺀다.
        outputs[0]["extraction"] = extraction(self.pilot[1])
        outputs[2]["extraction"]["events"].pop()
        report = score(self.pilot, outputs)
        self.assertEqual(report["overall"]["errors"], {"role_swap": 1, "missing_event": 1})
        self.assertEqual(report["contrast_pairs"]["both_bindings_exact"], 8)

    def test_main_runs_and_reports_from_generation_through_optional_stages(self):
        run_dir = self.root / "run"
        args = ["main", "--run-dir", str(run_dir), "--limit", "5"]
        sample = synth.select_records(self.pilot, 5)
        remaining = iter(sample)
        answers = {}

        def api(key, model, system, prompt, **request):
            usage = {"prompt_tokens": 1, "completion_tokens": 1, "model": model}
            if system == synth.GEN_SYSTEM_PROMPT:
                record = next(remaining)
                texts = {style: f"{record['id']} {style} 대사" for style in STYLE_KEYS}
                for text in texts.values():
                    answers[text] = extraction(record)
                return texts, usage
            return api_extraction(answers[json.loads(prompt)["utterance"]]), usage

        with patch("sys.argv", args), patch.dict("os.environ", {"OPENAI_API_KEY": "stub"}), \
                patch.object(synth, "call_openai_json", side_effect=api) as http, \
                contextlib.redirect_stdout(io.StringIO()):
            pipeline.main()
        self.assertEqual(http.call_count, 20)

        report_path = run_dir / "report.json"
        report = json.loads(report_path.read_text())
        self.assertEqual(report["scope"]["graph_scenarios"], 30)
        self.assertEqual(report["scope"]["evaluated_scenarios"], 5)
        self.assertTrue(report["scope"]["partial"])
        self.assertEqual(report["scope"]["drafts_with_at_least_two_references"], 5)
        self.assertEqual(report["draft_quality"]["counts"]["total_verified_references"], 15)
        self.assertEqual(report["synthesis"]["roundtrip_first_try_pass_rate"], 1.0)
        self.assertEqual(report["synthesis"]["gen_model"], "gpt-6.2-sol")

        # 검수한 양식을 전달하고 저장된 합성 결과로 최종본과 독립 채점을 만든다.
        template_path = run_dir / report["artifacts"]["author_review_template"]
        reviews = read_jsonl(template_path)
        for review in reviews:
            review.update(reviewer="test", verified=True)
            for entry in review["references"]:
                entry["decision"] = "keep"
        reviews[0]["references"][-1].update(decision="reject", text=None)
        review_file = self.root / "reviews.jsonl"
        extraction_file = self.root / "independent.jsonl"
        predictions_file = self.root / "predictions.jsonl"
        write_jsonl(review_file, reviews)
        write_jsonl(extraction_file, [{"id": record["id"], "extraction": extraction(record)} for record in sample])
        write_jsonl(predictions_file, [{"id": record["id"], "text": "실험 모델 대사"} for record in sample])
        args += ["--author-review", str(review_file), "--extractions", str(extraction_file),
                 "--extractor", "test extractor", "--predictions", str(predictions_file)]
        with patch("sys.argv", args), patch.dict("os.environ", {}, clear=True), \
                patch.object(synth, "load_dotenv_if_needed") as credentials, \
                patch.object(synth, "call_openai_json") as http, contextlib.redirect_stdout(io.StringIO()):
            pipeline.main()
        credentials.assert_not_called()
        http.assert_not_called()

        report = json.loads(report_path.read_text())
        self.assertEqual(report["author_review"]["status"], "verified")
        self.assertEqual(report["final_quality"]["counts"]["total_verified_references"], 14)
        self.assertEqual(report["final_quality"]["1_semantic_fidelity"]["roundtrip_final_yield_pct"], 100.0)
        self.assertEqual(report["independent_scores"]["overall"]["binding_exact_rate"], 1.0)
        self.assertEqual(len(read_jsonl(run_dir / report["artifacts"]["extraction_jobs"])), 5)

    def test_main_dry_run_creates_no_outputs_and_uses_no_api(self):
        run_dir = self.root / "dry_run"
        args = ["main", "--run-dir", str(run_dir), "--limit", "5", "--dry-run"]
        with patch("sys.argv", args), patch.object(synth, "call_openai_json") as http, \
                contextlib.redirect_stdout(io.StringIO()) as stdout:
            pipeline.main()
        self.assertIn("생성 1차 5회 | 역추출 1차 15회", stdout.getvalue())
        self.assertIn("gpt-6.2-sol", stdout.getvalue())
        self.assertFalse(run_dir.exists())
        http.assert_not_called()

    def test_responses_api_requires_structure_and_omits_temperature_for_sol(self):
        expected = api_extraction(extraction(self.pilot[0]))
        body = {
            "id": "resp_test", "model": "gpt-6.2-sol", "status": "completed",
            "output": [{"type": "reasoning"}, {
                "type": "message", "content": [{"type": "output_text", "text": json.dumps(expected)}],
            }],
            "usage": {"input_tokens": 100, "output_tokens": 80, "output_tokens_details": {"reasoning_tokens": 30}},
        }
        response = io.BytesIO(json.dumps(body).encode())
        response.headers = {"x-request-id": "req_test"}
        settings = responses.request_settings("gpt-6.2-sol", "medium", 0.7, 4096)
        with patch.object(responses.urllib.request, "urlopen", return_value=response) as urlopen:
            value, usage = responses.call_openai_json(
                "stub", "gpt-6.2-sol", synth.RT_API_SYSTEM_PROMPT, "대사",
                schema_name="roundtrip_extraction", schema=synth.RT_SCHEMA, settings=settings,
            )
        request = urlopen.call_args.args[0]
        payload = json.loads(request.data)
        self.assertEqual(request.full_url, responses.RESPONSES_URL)
        self.assertEqual(payload["reasoning"], {"effort": "medium"})
        self.assertNotIn("temperature", payload)
        self.assertEqual(responses.request_settings("gpt-4o", "medium", 0.7, 4096)["temperature"], 0.7)
        self.assertIs(payload["text"]["format"]["strict"], True)
        self.assertIn("discourse", payload["text"]["format"]["schema"]["required"])
        self.assertEqual(synth.decode_extraction(value), extraction(self.pilot[0]))
        self.assertEqual(usage["reasoning_tokens"], 30)
        self.assertEqual(usage["request_id"], "req_test")

    def test_responses_api_distinguishes_incomplete_output_and_refusal(self):
        cases = [
            ({"status": "incomplete", "incomplete_details": {"reason": "max_output_tokens"}}, "api_incomplete"),
            ({"status": "completed", "output": [{"type": "message", "content": [{"type": "refusal"}]}]}, "api_refusal"),
        ]
        for body, error in cases:
            with self.subTest(error=error):
                response = io.BytesIO(json.dumps(body).encode())
                response.headers = {}
                with patch.object(responses.urllib.request, "urlopen", return_value=response):
                    value, usage = responses.call_openai_json(
                        "stub", "gpt-6.2-sol", "지시", "요청",
                        schema_name="reference_dialogue", schema=synth.generation_schema(STYLE_KEYS),
                        settings=responses.request_settings("gpt-6.2-sol", "medium", 0.7, 4096),
                    )
                self.assertIsNone(value)
                self.assertEqual(usage["api_error"], error)

    def test_compound_errors_preserve_role_swap_signal(self):
        # 3사건 병렬에서 1사건을 누락하면서 나머지 2사건의 역할을 뒤바꾸면 둘 다 잡혀야 한다.
        rec3 = deepcopy(next(r for r in self.pilot if r["axis"] == AXIS2 and len(r["gold_tuples"]["events"]) == 3))
        gold = rec3["gold_tuples"]
        obs = extraction(rec3)
        ev0, ev1 = obs["events"][0]["roles"], obs["events"][1]["roles"]
        swap_role = next(k for k in ev0 if k != "Agent" and ev0[k] != ev1[k])
        ev0[swap_role], ev1[swap_role] = ev1[swap_role], ev0[swap_role]
        obs["events"].pop()  # missing_event + role_swap
        errs = comparison_errors(gold, obs, "모험가")
        self.assertIn("missing_event", errs)
        self.assertIn("role_swap", errs)

        # 단일 사건에서 Agent와 Recipient 방향만 뒤집어도 role_swap으로 탐지되어야 한다.
        single_gold = {
            "speech_act": "Inform", "speech_level": "haera", "discourse": [],
            "events": [{"event_id": "e1", "frame": "Deliver",
                        "roles": {"Agent": "기사", "Theme": "약초", "Recipient": "상인"}}],
        }
        flipped = deepcopy(single_gold)
        flipped["events"][0]["roles"]["Agent"] = "상인"
        flipped["events"][0]["roles"]["Recipient"] = "기사"
        flipped["extra_facts"] = []
        self.assertIn("role_swap", comparison_errors(single_gold, flipped, "모험가"))

        # 존재하지 않는 개체로의 단순 오기입은 role_swap이 아니라 role_value_mismatch여야 한다.
        bogus = deepcopy(single_gold)
        bogus["events"][0]["roles"]["Recipient"] = "도적"
        bogus["extra_facts"] = []
        errs_bogus = comparison_errors(single_gold, bogus, "모험가")
        self.assertIn("role_value_mismatch", errs_bogus)
        self.assertNotIn("role_swap", errs_bogus)

    def test_cross_event_swap_requires_exchange_and_keeps_other_value_errors(self):
        rec = self.pilot[0]
        gold = rec["gold_tuples"]
        observed = extraction(rec)
        first, second = (event["roles"] for event in observed["events"])
        first["Recipient"] = second["Recipient"]
        second["Recipient"] = "다른 인물"
        self.assertEqual(comparison_errors(gold, observed, "모험가"), ["role_value_mismatch"])

        second["Recipient"] = gold["events"][0]["roles"]["Recipient"]
        second["Agent"] = "다른 인물"
        observed["events"].reverse()
        for i, event in enumerate(observed["events"]):
            event["event_id"] = f"x{i}"
        self.assertEqual(comparison_errors(gold, observed, "모험가"), ["role_swap", "role_value_mismatch"])

    def test_role_omission_survives_a_hallucinated_event(self):
        rec = self.pilot[0]
        observed = extraction(rec)
        del observed["events"][1]["roles"]["Recipient"]
        for extra in (
            {"event_id": "x3", "frame": "Danger", "roles": {"Location": "동굴", "Threat": "도적"}},
            {"event_id": "x3", "frame": "Deliver",
             "roles": {"Agent": "모험가", "Theme": "다른 물건", "Recipient": "다른 인물"}},
        ):
            with self.subTest(frame=extra["frame"]):
                report = score([rec], [{"id": rec["id"], "extraction": {
                    **observed, "events": [*observed["events"], extra],
                }}])["overall"]
                self.assertEqual(report["errors"], {"hallucinated_event": 1, "missing_role": 1})
                self.assertEqual(report["omission_rate"], 1.0)
                self.assertEqual(report["hallucination_rate"], 1.0)

    def test_swap_does_not_consume_an_extra_copy_of_a_role_value(self):
        rec = next(row for row in self.pilot if len(row["gold_tuples"]["events"]) == 3
                   and row["gold_tuples"]["events"][0]["frame"] == "Deliver"
                   and row["axis"] == AXIS2)
        observed = extraction(rec)
        first, second, third = (event["roles"] for event in observed["events"])
        first["Recipient"], second["Recipient"] = second["Recipient"], first["Recipient"]
        third["Recipient"] = first["Recipient"]
        self.assertEqual(comparison_errors(rec["gold_tuples"], observed, "모험가"),
                         ["role_swap", "role_value_mismatch"])

    def test_diag40_generation_validation_and_both_exact_scoring(self):
        diag = read_jsonl(self.generate("diag40", self.root / "diag40.jsonl"))
        self.assertEqual(len(diag), 40)
        self.assertEqual(validate_records(diag), ("diag40", 20))

        outputs = [{"id": rec["id"], "extraction": extraction(rec)} for rec in diag]
        rep_ok = score(diag, outputs)
        self.assertEqual(rep_ok["contrast_pairs"]["both_exact"], 20)
        disc_idx = next(i for i, rec in enumerate(diag) if rec["axis"] == DIAG_DISCOURSE and rec["pair_variant"] == "b")
        for relation in ({"type": "Coord", "args": ["e1", "e2"]}, {"type": "Reason", "cause": "e2", "effect": "e1"}):
            with self.subTest(relation=relation):
                outputs[disc_idx]["extraction"]["discourse"] = [relation]
                rep_bad = score(diag, outputs)
                self.assertEqual(rep_bad["contrast_pairs"]["both_bindings_exact"], 20)
                self.assertEqual(rep_bad["contrast_pairs"]["both_exact"], 19)

    def test_diag40_discourse_frame_pairs_are_held_out_and_cover_all_levels(self):
        seen_frame_pairs = set()
        for split in ("pilot30", "full240"):
            for record in generator.generate_records(split, 4047):
                frames = {event["id"]: event["frame"] for event in record["input_graph"]["events"]}
                for relation in record["utterance_goal"]["discourse"]:
                    if relation["type"] == "Reason" and relation["cause"] in frames and relation["effect"] in frames:
                        seen_frame_pairs.add((frames[relation["cause"]], frames[relation["effect"]]))

        levels_by_frames = {}
        for record in generator.generate_records("diag40", 2027):
            if record["axis"] == DIAG_DISCOURSE and record["pair_variant"] == "b":
                frames = tuple(event["frame"] for event in record["gold_tuples"]["events"])
                levels_by_frames.setdefault(frames, set()).add(record["utterance_goal"]["speech_level"])

        self.assertEqual(len(levels_by_frames), 3)
        self.assertFalse(seen_frame_pairs.intersection(levels_by_frames))
        for frames, levels in levels_by_frames.items():
            with self.subTest(frames=frames):
                self.assertEqual(levels, set(SPEECH_LEVELS))

    def test_diag40_rejects_identical_cause_and_effect_events(self):
        pair = [rec for rec in generator.generate_records("diag40", 2027) if rec["axis"] == DIAG_DISCOURSE][:2]
        for record in pair:
            graph = record["input_graph"]
            first, second = graph["events"]
            graph["events"][1] = {**deepcopy(first), "id": second["id"]}
            record["gold_tuples"] = derive_gold(graph, record["utterance_goal"])
        with self.assertRaisesRegex(ValueError, "requires distinct events"):
            validate_records(pair, partial=True)

    def test_score_counts_only_complete_pairs_with_distinct_variants(self):
        a, b = self.pilot[:2]
        duplicate_a = {**a, "id": f"{a['id']}_duplicate"}
        for records, expected_pairs in (([a], 0), ([a, duplicate_a], 0), ([a, b, duplicate_a], 0), ([b, a], 1)):
            with self.subTest(ids=[rec["id"] for rec in records]):
                outputs = [{"id": rec["id"], "extraction": extraction(rec)} for rec in records]
                report = score(records, outputs)
                self.assertEqual(report["overall"]["exact_rate"], 1.0)
                self.assertEqual(report["contrast_pairs"], {
                    "total": expected_pairs, "both_bindings_exact": expected_pairs, "both_exact": expected_pairs,
                })


if __name__ == "__main__":
    unittest.main()
