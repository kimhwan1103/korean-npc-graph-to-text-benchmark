# Korean RPG Event-to-Utterance Benchmark

[한국어](README.md) · English

This pipeline builds a benchmark for measuring how accurately Korean RPG dialogue expresses specified events, semantic roles, speech acts, speech levels, and discourse relations. It constructs input graphs, synthesizes reference utterances with OpenAI models, and validates them through round-trip extraction and author review. An independent extractor can recover structured predictions from an experimental model's utterances for scoring against the gold graph. [Dataset protocol](../DATASET_PROTOCOL_KO_20261123.md) · [BFGR algorithm specification](../ALGORITHM_SPEC_KO_20261123.md)

## Quick start

Run the commands from `dataset_pipeline/`. The pipeline uses Python 3.9 or later and the standard library. Synthesis uses file locking on macOS and Linux. Graph generation, local validation, and report recomputation work without an API key.

```bash
python3 --version
python3 main.py --help
python3 main.py --split pilot30 --limit 5 --dry-run
```

`--dry-run` prints the graph validation result, the number of selected scenarios, model settings, estimated first-attempt API calls, and an example prompt. It skips API calls and file writes. Synthesis makes billable API requests. The pipeline reads the shell's `OPENAI_API_KEY` first, then looks for it in a local `.env` in the working directory or script directory. Git ignores `.env`.

```dotenv
OPENAI_API_KEY=your_api_key_here
```

### Resume a saved run

Use these commands when the run checkpoint is available locally. After a fresh clone, see [Start a new run](#start-a-new-run).

```bash
# Reuse the completed Pilot-30 checkpoint, report, and review template
python3 main.py --split pilot30 --run-dir data/runs/pilot30_seed2027_v4

# Resume Full-240 from its checkpoint
python3 main.py --split full240 --run-dir data/runs/full240_seed2027_v4
```

The code sets `PIPELINE_VERSION` to `3`, so `--run-dir` defaults to `data/runs/<split>_seed<seed>_v3/`. The `_v4` directories shown above hold runs created with the current prompts. Their checkpoints record synthesis version `3` and graph schema version `2`. The existing `pilot30_seed2027_v3` checkpoint contains five scenarios processed with earlier prompts. Opening it with the current prompts produces a configuration mismatch error.

### Start a new run

```bash
# Synthesize up to five scenarios in a separate directory
python3 main.py --split pilot30 --limit 5 \
  --run-dir data/runs/pilot30_smoke_seed2027

# Continue through the rest of the split from the same checkpoint
python3 main.py --split pilot30 \
  --run-dir data/runs/pilot30_smoke_seed2027
```

`--limit` caps the number of selected scenarios. The selector treats both variants of a contrast pair as a single group and moves on when a group exceeds the remaining capacity. With the current Pilot split, `--limit 5` selects two Axis 2 contrast pairs and one Axis 3 scenario. The pipeline still generates and validates the full graph split. In `report.json`, `scope.evaluated_scenarios` counts all scenarios accumulated in that run directory's checkpoint.

| Run scope | Supported scenarios | Initial generation requests | Estimated initial extraction requests |
| --- | ---: | ---: | ---: |
| Pilot sample of up to 5 | 5 | 5 | 15 |
| Pilot-30 | 30 | 30 | 90 |
| Full-240 | 220 | 220 | 660 |

Each generation request bundles the required styles for one scenario. Extraction runs once per candidate. Empty and duplicate candidates are rejected before extraction. Regeneration and HTTP retries can change the actual request totals. Unsupported boundary cases are included in the evaluation split and skip dialogue synthesis.

## Pipeline workflow

| Stage | What it does | Artifact |
| --- | --- | --- |
| 1. Graph generation | Uses the split, seed, and schema version to generate deterministic events, utterance goals, and gold annotations. Validates contrast pairs, role references, discourse trees, and per-axis counts. | `graphs.jsonl` |
| 2. Reference synthesis | Gives the generation model the input graph and utterance goal. Style A is concise, Style B is colloquial, and Style C varies clause placement or constituent order. | Per-style candidates in the checkpoint |
| 3. Round-trip validation | Extracts events, roles, speech acts, speech levels, and discourse from each candidate and compares them with the gold annotations. Keeps passing references and makes up to two additional attempts for unresolved styles by default. | `synthesized.jsonl`, `synthesis_stats.json` |
| 4. Author review | Records which automatically accepted references the reviewer keeps, edits, or rejects. The reviewer can write a reference for any missing style. | `author_review_<hash>.jsonl` and, after review submission, `final.jsonl` |
| Evaluation | Reports quantitative metrics for drafts and reviewed references. It also scores experimental model outputs when independent extractions are supplied. | `report.json` |

Generation and extraction use Structured Outputs in the Responses API. The generation schema requires the styles requested in that call. The extraction schema requires `speech_act`, `speech_level`, `events`, `discourse`, and `extra_facts`. The pipeline converts API role arrays into JSON objects mapping role names to entity surface forms before saving the results.

The extraction model receives the utterance, the speaker and addressee, and candidate entity surface forms. In requests and warnings, the comparator interprets an omitted `Deliver` or `Fetch` agent as the addressee and compares explicit agents as given. `Coord` arguments can be reordered. `Reason` must preserve the cause, effect, and nesting. The comparator checks events, role assignments, speech act, speech level, and extra facts in addition to data types.

## Dataset splits and graph schema

| Evaluation axis | Pilot-30 | Full-240 | Design |
| --- | ---: | ---: | --- |
| Axis 1 · Single events | 0 | 60 | Five frames × four speech acts × three speech levels. Every speech level includes entities with no final consonant, a final ㄹ, and other final consonants. |
| Axis 2 · Role swaps | 20 | 80 | Each contrast pair swaps the designated role values between exactly two events. For scenarios with 2, 3, and 4 events, Pilot has 4, 3, and 3 pairs, and Full has 14, 13, and 13 pairs. |
| Axis 3 · Complex discourse | 10 | 60 | Includes cases where a state motivates an action and causal relations with nested coordination. |
| Axis 4 · Boundary cases | 0 | 40 | The 20 supported cases have 5, 6, or 8 events, with 7, 7, and 6 cases respectively. The 20 unsupported cases cover four categories of five cases each, including unsupported frames, missing required roles, excessive discourse depth, and conflicting speech acts. |
| Total | 30 | 240 | Pilot contains 30 supported scenarios. Full contains 220 supported and 20 unsupported scenarios. |

Each line of a UTF-8 JSONL file represents one scenario.

| Field | Contents |
| --- | --- |
| `id`, `split`, `seed`, `schema_version` | Scenario ID, split, generation seed, and graph schema version |
| `axis`, `subtype`, `pair_id`, `pair_variant` | Evaluation axis, subtype, Axis 2 contrast pair ID, and variant `a` or `b` |
| `input_graph.entities` | Entity ID, `surface` form, entity type, animacy, and final consonant class |
| `input_graph.events` | Event ID, frame, and `roles` mapping role names to entity IDs |
| `utterance_goal` | Speaker and addressee IDs, `speech_act`, `speech_level`, target event IDs, and discourse relations |
| `gold_tuples` | Gold events with entity IDs resolved to surface forms, plus the gold speech act, speech level, and discourse relations |
| `expect_supported`, `unsupported_reason` | Whether the scenario is supported and the reason when it is unsupported |

The speech acts are `Inform`, `Request`, `Question`, and `Warn`. Korean speech levels are `haeyo` (polite), `haera` (plain), and `hasipsio` (formal polite). Discourse uses `Coord` to group events or nested relations and `Reason` to connect a cause and an effect. The supported depth is 2.

| Frame | Required roles | Optional roles |
| --- | --- | --- |
| `Deliver` | `Agent`, `Theme`, `Recipient` | `Source` |
| `Fetch` | `Agent`, `Theme`, `Source` | |
| `Need` | `Experiencer`, `Required_Entity` | |
| `Located_At` | `Entity`, `Location` | |
| `Danger` | `Location`, `Threat` | `Target` |

## Run options and artifacts

| `main.py` option | Default | Purpose |
| --- | --- | --- |
| `--split`, `--seed` | `pilot30`, `2027` | Graph split and reproducible generation seed |
| `--run-dir` | `data/runs/<split>_seed<seed>_v3/` | Directory for graphs, checkpoints, and reports |
| `--limit`, `--dry-run` | Full split, live run | Cap the selected scenarios or preview the configuration and run scope |
| `--model`, `--rt-model` | `gpt-6.2-sol`, `gpt-6.1-terra` | Reference generation model and round-trip extraction model |
| `--reasoning-effort`, `--rt-reasoning-effort` | Both `medium` | Reasoning effort for each model. Accepted values are `low`, `medium`, and `high`. |
| `--max-output-tokens`, `--rt-max-output-tokens` | `4096`, `6144` | Per-response token caps for generation and extraction |
| `--temperature` | `0.7` for non-reasoning generation models | Sampling temperature for models such as `gpt-4o`. The default GPT-6.2 Sol uses reasoning effort. |
| `--max-retries` | `2` | Number of regeneration attempts for each style after the initial attempt. Accepted values are `0`, `1`, and `2`. |
| `--author-review` | Omitted | Completed review JSONL used to create the final dataset and compute its quality metrics |
| `--predictions` | Omitted | Experimental model output JSONL used to prepare independent extraction jobs |
| `--extractions`, `--extractor` | Omitted | Independent extraction JSONL and extractor name and version. Supply both to add model scores to the report. |

| File in the run directory | Contents |
| --- | --- |
| `graphs.jsonl` | Input graphs, utterance goals, and gold annotations for the full split |
| `synthesized_state.json` | Checkpoint. Its `manifest` stores the input hash, models, prompts, and schemas, and `rows` stores generation, extraction, and validation history. |
| `synthesized.jsonl` | Passing `references` per scenario and per-candidate `roundtrip_log.attempts` |
| `synthesis_stats.json` | Resolved response model names, request settings, acceptance and rejection counts, and input, output, reasoning, and cached token usage |
| `report.json` | Run `scope`, `synthesis` statistics, `draft_quality`, and `author_review` status. The corresponding options add `final_quality` and `independent_scores`. |
| `author_review_<hash>.jsonl` | Review template for the current draft. Find the exact filename in `artifacts.author_review_template` in `report.json`. |
| `final.jsonl`, `extraction_jobs.jsonl` | Author-reviewed dataset and job file for independent extraction |

### Checkpoint and resume behavior

- Reuse the same split, seed, models, prompts, response settings, and `--run-dir` to reuse recorded decisions. The pipeline looks up the API key when synthesis work remains.
- Remove `--limit` to extend a sample run in the same directory. You can also raise `--max-retries` from `0` to `1` or `2` in that directory.
- Use another `--run-dir` when changing the model, prompt, response settings, or graph. Previous checkpoints document the conditions and results of those runs.
- While synthesis runs, progress goes to `synthesized_state.json`. The pipeline writes `synthesized.jsonl` and `synthesis_stats.json` when it exits the synthesis loop. It writes `report.json` after validation and report processing finish.

## Interpreting results and troubleshooting

### What `0/3` means

A `0/3` result means zero of the scenario's three reference styles passed automatic validation. The pipeline can generate a candidate and then reject it for missing roles, speech act or speech level mismatches, discourse discrepancies, or invalid response format. To diagnose a scenario, open `roundtrip_log.attempts` in `synthesized.jsonl` and compare the candidate `text`, `extracted` structure, primary `reason`, and complete `errors` list. During an active run, inspect the checkpoint for the same information.

| Status or rejection reason | Meaning |
| --- | --- |
| `complete` | All three styles passed automatic validation. Check `author_verified` for review completion. |
| `retry_limit_reached` | The run exhausted the regeneration budget with fewer than three passing references. Two surviving references can meet the final dataset requirement after author review. |
| `pending` | The scenario still has validation or regeneration work. |
| `unsupported` | Graph generation marked this scenario as a boundary case outside the supported conditions. |
| `speech_act_mismatch`, `speech_level_mismatch` | The extracted speech act or speech level differs from the gold annotation. Compare the extracted label with the candidate's sentence endings. |
| `missing_event`, `missing_role`, `role_swap` | An event or role was omitted, or role values were swapped across events. Inspect whether the utterance expresses the intended events and roles. |
| `hallucinated_event`, `hallucinated_role`, `extra_fact` | Extraction recovered an extra event, role, or fact that the supported frames cannot encode. |
| `discourse_mismatch`, `invalid_extraction` | The extracted discourse structure differs from the gold annotation, or the response format is invalid. |
| `api_incomplete`, `api_refusal` | The API response was incomplete or the model refused. Find response details in `usage` or in the attempt's `generation_usage`. |

`rejection_reasons` counts one primary reason per rejected candidate. `errors` retains all detected issues for candidates with a recorded comparison. Compare the original utterance and gold annotation to distinguish an extraction error from an omission in the generated dialogue.

### Report metrics

| Metric | Definition |
| --- | --- |
| `first_try_pass_rate` | First-attempt passing references divided by the number of first-attempt style slots recorded after generation |
| `roundtrip_first_try_pass_rate` | First-attempt passing references divided by the number of first-attempt candidates with a recorded extraction call |
| `roundtrip_final_yield_pct` | Passing automatic references ÷ (supported scenarios × 3) × 100. Includes retries. Final dataset reports use the accepted automatic drafts as the numerator. |
| `drafts_with_at_least_two_references` | Supported scenarios with at least two automatically accepted references |
| Entity surface recall and lexicon intrusion | Surface checks for gold entity names and mentions of lexicon entities absent from the input roles |
| Particle, speech level, and speech act metrics | Rule-based checks for particle allomorphs after known entities, sentence endings, and surface markers of questions and warnings |
| `intra_scenario_diversity_mean`, Distinct-1/2 | Character trigram dissimilarity among references for the same scenario, plus the proportions of unique whitespace-delimited words and adjacent word pairs across references |
| Contrast pair non-collapse rate | Among pairs with references for both variants, the percentage whose reference sets contain no identical utterances. Surface divergence compares the first reference from each variant. |

Fields ending in `_pct` use the 0–100 scale. The rates in `synthesis` and the independent scoring results use the 0–1 scale. The quality script records `0.0` for some percentages with a zero denominator. Read those values alongside the reference count and the number of evaluated contrast pairs.

### Saved Pilot results

These results come from local reports inspected on November 24, 2026. Consider the different prompts and run scopes when comparing the rates. Each report is saved as `report.json` in the run directory shown below.

| Run directory | Scenarios evaluated | First-attempt round-trip passes | Automatically accepted references | Scenarios with at least two references |
| --- | ---: | ---: | ---: | ---: |
| `pilot30_seed2027_v3` | 5 | 12/15 (80.00%) | 12/15 | 5/5 |
| `pilot30_seed2027_v4` | 30 | 73/90 (81.11%) | 82/90 | 27/30 |

The 30-scenario run had 27 scenarios with three references, one with a single reference, and two without references. The primary rejection reasons were `extra_fact` (21) and `speech_act_mismatch` (17). These figures describe the automatically validated draft. The run report records its author review status as `pending`.

### Common issues

| Issue | What to check |
| --- | --- |
| `OPENAI_API_KEY is required` | The run has API work remaining. Set the key in the shell environment or the local `.env`. |
| Checkpoint input or synthesis configuration differs | Check the models, prompts, and token caps in `synthesis_stats.json` or the checkpoint `manifest`. Use the original settings to resume or a separate directory for a changed configuration. |
| `Another synthesis run is using ...` | A process holds the run directory's lock. Resume after that process finishes. |
| Repeated `api_incomplete` | Check whether `usage.api_error_detail` is `max_output_tokens`. Try a different token cap for the relevant stage in a separate run directory. |
| Draft validation fails for a partial run | For a partial run, use `validate_dataset.py --stage synthesis --partial`. Include both variants of each selected contrast pair. |
| Review or independent result IDs do not match | Check the run `scope` in `report.json` and the actual IDs in the run directory's `synthesized.jsonl`. |

## Author review and final dataset

After synthesis, copy the template named in `artifacts.author_review_template` in `report.json` to a working file. For each supported scenario, enter a reviewer identifier in `reviewer`. Verify each candidate against the gold annotation and check its language quality before setting `verified` to `true`. Record a decision for every existing draft and retain two to three distinct final references per scenario.

| `decision` | `original_text` | `text` |
| --- | --- | --- |
| `keep` | Original wording from the template | The same utterance |
| `edit` | Original wording from the template | The revised utterance |
| `reject` | Original wording from the template | `null` |
| `add` | `null` | A new utterance written by the reviewer for a missing style |

`original_text` in the review file must match the draft. For `add`, use a style key with no existing draft for that scenario. Record the reason for an edit, rejection, or addition in `note` if helpful. The review template contains supported scenarios. The final dataset retains unsupported scenarios with empty reference lists.

```bash
# Replace author_reviewed.jsonl with the path to the completed review file.
python3 main.py --split pilot30 --run-dir data/runs/pilot30_seed2027_v4 \
  --author-review author_reviewed.jsonl

python3 validate_dataset.py data/runs/pilot30_seed2027_v4/final.jsonl --stage final
```

The final dataset retains original references and review decisions in `draft_references` and `author_review`. The report includes final reference counts, edit, rejection, and addition counts, the edit rate, and mean length. The edit rate divides the number of edited drafts by the number of drafts before review.

## Independent scoring of experimental model outputs

1. Save the experimental model's actual outputs as UTF-8 JSONL. Record the utterance or the model's explicit unsupported response.

   ```jsonl
   {"id":"pilot30_ax2_001_a","text":"약초를 치유사에게 전해 주세요."}
   {"id":"full240_ax4_unsup_001","status":"unsupported"}
   ```

   These rows illustrate the two output formats. Use the IDs of the split you are evaluating in the actual file.

2. Use `--predictions` to prepare the extraction job file. It contains prompts with the utterance, speaker, addressee, and candidate entity names. Give it to an independent extractor or human annotator.

   ```bash
   python3 main.py --split pilot30 --run-dir data/runs/pilot30_seed2027_v4 \
     --predictions outputs.jsonl
   ```

3. Save the extraction results as lines of the form `{"id":"scenario ID","extraction":{...}}`. Each `extraction` requires `speech_act`, `speech_level`, `events`, `discourse`, and `extra_facts`. Each event contains `event_id`, `frame`, and `roles`. In externally extracted results, `roles` is an object mapping roles to surface forms, such as `{"Theme":"약초"}`. Preserve an unsupported response as `{"id":"…","status":"unsupported"}`.

   ```bash
   python3 main.py --split pilot30 --run-dir data/runs/pilot30_seed2027_v4 \
     --extractions independent_extractions.jsonl \
     --extractor 'independent extractor name and version'
   ```

The integrated command checks that the input files contain exactly the scenario IDs in the current synthesis output. Use the standalone commands below to score a full graph split with existing outputs. Scores are grouped by axis and event count and include exact role binding, role swap, omission, hallucination, speech act match, and speech level match rates. `exact_rate` is the proportion of supported scenarios that pass every check. `unsupported_detection_rate` is the proportion of unsupported scenarios for which the model returns unsupported. See `contrast_pairs` for the number of pairs with exact role bindings in both variants.

## Standalone commands and tests

```bash
# Generate and validate graphs only
python3 step1_generate_graphs.py --split pilot30 --seed 2027
python3 validate_dataset.py data/pilot30_graphs.jsonl

# Preview synthesis and then run it for up to five scenarios
python3 step2_3_openai_synthesize.py --input data/pilot30_graphs.jsonl --limit 5 --dry-run
python3 step2_3_openai_synthesize.py --input data/pilot30_graphs.jsonl --limit 5
python3 validate_dataset.py data/pilot30_v3_synthesized.jsonl --stage synthesis --partial

# Recompute quality metrics for the saved draft without API calls
python3 evaluate_synthetic_quality.py data/runs/pilot30_seed2027_v4/synthesized.jsonl

# Prepare a review file for the full split or finalize a completed review
python3 review_dataset.py prepare --input data/runs/pilot30_seed2027_v4/synthesized.jsonl \
  --output author_review_to_fill.jsonl
python3 review_dataset.py finalize --input data/runs/pilot30_seed2027_v4/synthesized.jsonl \
  --review author_reviewed.jsonl --output pilot30_final.jsonl

# Prepare extraction jobs and score an existing graph split
python3 score_outputs.py prepare --input data/runs/pilot30_seed2027_v4/graphs.jsonl \
  --predictions outputs.jsonl --output extraction_jobs.jsonl
python3 score_outputs.py score --input data/runs/pilot30_seed2027_v4/graphs.jsonl \
  --predictions independent_extractions.jsonl \
  --extractor 'independent extractor name and version' --output scores.json

# Run regression tests with mocked API calls
python3 -m unittest -v
```

## Source files

| File | Responsibility |
| --- | --- |
| [main.py](main.py) | Pipeline orchestration, run directories, and the integrated report |
| [schema.py](schema.py), [step1_generate_graphs.py](step1_generate_graphs.py) | Graph schema, lexicon, deterministic graphs, and contrast pairs |
| [step2_3_openai_synthesize.py](step2_3_openai_synthesize.py), [openai_responses.py](openai_responses.py) | Generation and extraction prompts and schemas, API transport, regeneration, and checkpoint summaries |
| [semantic.py](semantic.py), [validate_dataset.py](validate_dataset.py) | Event, role, and discourse comparison, plus validation of graphs, contrast pairs, drafts, and final data |
| [review_dataset.py](review_dataset.py) | Author review templates, final datasets, and review statistics |
| [evaluate_synthetic_quality.py](evaluate_synthetic_quality.py), [score_outputs.py](score_outputs.py) | Reference quality diagnostics and independent model scoring |
| [dataset_io.py](dataset_io.py), [test_pipeline.py](test_pipeline.py) | UTF-8 JSONL I/O, atomic writes, and pipeline regression tests |
