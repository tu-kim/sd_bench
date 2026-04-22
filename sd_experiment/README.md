# SD Trade-off Experimental Harness

Measures how **batch size (B)** and **tree configuration (depth D, branching
K, tree size T)** interact in vLLM speculative decoding (SD) under a
coding-agent-style long-context workload (SWE-Bench Lite).

> **Phase status:** Phase A complete (API verified against vLLM **0.17.1**,
> smoke test written, API notes frozen in `VLLM_API_NOTES.md`).  Phases B–E
> not yet built — see spec at repo root.

## Target stack (non-negotiable)

- `vllm==0.17.1` (pinned in `requirements.txt`; API-notes audit is done
  against this exact version).
- Python 3.10+.
- CUDA 12.x GPU.
- Target model: `meta-llama/Llama-3.1-8B-Instruct`.
- Draft model: `yuhuili/EAGLE3-LLaMA3.1-Instruct-8B` (method `eagle3`).

## Repository layout

```
sd_experiment/
├── README.md                         (this file)
├── VLLM_API_NOTES.md                 ← Phase A gate: 0.17.1 API audit
├── requirements.txt                  ← pins vllm==0.17.1
├── configs/experiment_grid.yaml
├── tests/
│   ├── smoke_test.py                 ← Phase A gate: must pass on GPU box
│   └── smoke_test_output.txt         ← captured transcript (GPU required)
├── workload/                         (Phase B)
├── metrics/                          (Phase B)
├── runner/                           (Phase C)
├── scripts/                          (Phase D)
└── results/                          (auto-generated)
```

## Phase A — verify vLLM 0.17.1 API

```bash
pip install -r requirements.txt
python tests/smoke_test.py 2>&1 | tee tests/smoke_test_output.txt
```

`VLLM_API_NOTES.md` is the ground-truth record of the API surface the rest
of the harness depends on.  Any change to vLLM's version **must** start by
re-running the audit there.

### Key findings (short form — details in `VLLM_API_NOTES.md`)

1. SD config lives **only** in `EngineArgs.speculative_config` (a dict).
   All 0.6.x per-arg fields (`speculative_model`, `num_speculative_tokens`
   on `EngineArgs`, `speculative_draft_tensor_parallel_size`) are gone.
2. Tree branching is expressed via `speculative_token_tree` — a string
   representation of `list[tuple[int, ...]]`.  There is **no `eagle_topk`
   field**; the harness translates CLI `-K` into the tree string.
3. `LLMEngine.step()` returns `list[RequestOutput]` only.  Per-step
   aggregate SD stats flow through a custom `StatLoggerBase` registered via
   `from_engine_args(..., stat_loggers=[...])`.  The key fields are
   `SchedulerStats.spec_decoding_stats.{num_drafts, num_draft_tokens,
   num_accepted_tokens, num_accepted_tokens_per_pos}` and
   `IterationStats.{num_generation_tokens, prompt_token_stats.total}`.
4. A step is **decode-only** iff `IterationStats.prompt_token_stats.total == 0`.
   The first 3 decode steps per run are discarded as warmup.
5. No public draft/verify time split in 0.17.1 — those `StepMetric` fields
   stay `None`; `total_latency_ms` is wall-time around `engine.step()`.

### Phase A artifacts

- `VLLM_API_NOTES.md` — API audit.
- `tests/smoke_test.py` — exercises every 0.17.1 path the harness uses.
- `tests/smoke_test_output.txt` — captured transcript (currently the
  `--help` + env capture from a no-GPU inspection host; **must be
  re-captured on the CUDA box** before Phase B).

## Phases B–E (not yet implemented)

- **Phase B:** `workload/` + `metrics/` + one end-to-end single-config run.
- **Phase C:** `runner/engine_runner.py` with step-level metric collection.
- **Phase D:** `scripts/run_grid.sh` + `scripts/analyze_results.py`.
- **Phase E:** polish and reproducibility pass from a clean checkout.

Workflow gate per spec: stop and review at the end of each phase.
