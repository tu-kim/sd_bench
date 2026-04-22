# SD Trade-off Experimental Harness

Measures how **batch size (B)** and **tree configuration (depth D, branching
K, tree size T)** interact in vLLM speculative decoding (SD) under a
coding-agent-style long-context workload (SWE-Bench Lite).  Focused on
the **memory/compute–constrained Pareto frontier**: for a given verify
work budget `B × (T+1)` or KV-cache budget, which (B, T) wins?

> **Phase status:** A, B, C, D implemented and offline-validated on a
> CPU-only box.  End-to-end execution must happen on a CUDA GPU host
> (Phase A gate: `tests/smoke_test.py` transcript).

## Target stack (non-negotiable)

- `vllm==0.17.1` (pinned in `requirements.txt`).  API is audited at this
  exact version in `VLLM_API_NOTES.md`; do not up/down-grade without
  re-running that audit.
- Python 3.10+; CUDA 12.x GPU.
- Target model: `meta-llama/Llama-3.1-8B-Instruct`.
- Draft model: `yuhuili/EAGLE3-LLaMA3.1-Instruct-8B` (method `eagle3`).

## Repository layout

```
sd_experiment/
├── README.md                         (this file)
├── VLLM_API_NOTES.md                 ← Phase A audit (0.17.1 API surface)
├── requirements.txt                  ← pins vllm==0.17.1
├── configs/experiment_grid.yaml      ← grid definition + minimal subset
├── workload/
│   ├── prompt_builder.py             ← Llama-3.1 chat-literal agent prompt
│   └── swe_bench_loader.py           ← HF / local / mock 3-tier loader
├── runner/
│   ├── tree.py                       ← (D, K, T) → speculative_token_tree
│   └── engine_runner.py              ← LLMEngine step loop + metric capture
├── metrics/
│   └── collector.py                  ← StepMetric + RunMetric.summary()
├── scripts/
│   ├── run_grid.sh                   ← phase 1/2/3 driver (MINIMAL=1 subset)
│   └── analyze_results.py            ← pivots + heatmaps + Pareto plots
├── tests/
│   ├── smoke_test.py                 ← Phase A gate (GPU)
│   ├── phase_b_offline_check.py      ← workload + metrics unit tests
│   ├── phase_b_smoke.py              ← single-config GPU smoke
│   ├── phase_c_offline_check.py      ← tree + CLI + filename checks
│   └── phase_d_offline_check.py      ← synthetic-grid end-to-end
└── results/                          ← auto-generated
```

## End-to-end usage (GPU box)

```bash
cd sd_experiment
pip install -r requirements.txt

# 0. Phase A gate — verify vLLM 0.17.1 API wires up.
python tests/smoke_test.py 2>&1 | tee tests/smoke_test_output.txt

# 1. Bucket SWE-Bench Lite prompts by context length.
python -m workload.swe_bench_loader \
    --tokenizer meta-llama/Llama-3.1-8B-Instruct \
    --buckets 2048 8192 32768 \
    --max-samples 64 \
    --output-dir data/swebench_bucketed

# 2. Minimal grid (2 B × 2 T = 4 configs) to sanity-check the pipeline.
MINIMAL=1 bash scripts/run_grid.sh
#   -> results/{timestamp}/phase1/run_sd_b{B}_d{D}_k{K}_t{T}_*_summary.json

# 3. Full grid (phase 1 + phase 2 + phase 3).  Takes hours.
bash scripts/run_grid.sh

# 4. Build pivots, heatmaps, Pareto plots.
python scripts/analyze_results.py --results-dir results/{timestamp}
#   -> results/{timestamp}/analysis/*.csv, *.png
```

## Pareto plots produced by `analyze_results.py`

- `pareto_throughput_vs_ptl.png` — effective per-token latency (x) vs
  throughput (y); color encodes SD on/off.  The canonical "is SD worth
  it at this batch size?" view.
- `pareto_kv_vs_throughput.png` — avg KV-cache usage (x) vs throughput
  (y); color encodes tree size T, marker size encodes batch B.
  Memory-frontier view for B vs T under a KV budget.
- `pareto_verifywork_vs_throughput.png` — `B × (T+1)` (x, log-scale) vs
  throughput (y); dotted vertical lines mark iso-cost bins (32, 64,
  128, 256, 512).  Same vertical line = same verify-work budget:
  directly compares big-B-small-T against small-B-big-T.
- `acceptance_vs_tree.png` — mean acceptance length vs T, one line per
  batch size.  Shows diminishing returns from deeper/wider trees.

## Key constraints (anchored to VLLM_API_NOTES.md)

1. vLLM 0.17.1 SD is configured **only** via `EngineArgs.speculative_config`
   (a dict).  No `speculative_model` / `num_speculative_tokens` / etc.
   at the top level.
2. Tree branching is expressed only via `speculative_token_tree`.  There
   is no `eagle_topk` field; the harness translates CLI `-K` into the
   tree string via `runner/tree.py::build_tree_string`.
3. Per-step SD stats reach user code through a custom `StatLoggerBase`
   registered at `from_engine_args(..., stat_loggers=[...])`; vLLM's
   `engine.step()` itself returns only `list[RequestOutput]`.
4. A step is **decode-only** iff `IterationStats.prompt_token_stats.total == 0`.
   The first 3 decode steps per run are discarded as warmup.
5. Draft/verify timing is **not** exposed publicly in 0.17.1.
   `StepMetric.draft_latency_ms` and `verify_latency_ms` stay `None`;
   `total_latency_ms` is wall-time around `engine.step()`.
6. Greedy sampling (`temperature=0.0`) + `seed=0` → acceptance rate is
   reproducible for a given (prompt, target, drafter, tree).

## Offline validators (CPU host)

All three pass on a no-GPU sandbox and are run in CI:

```
python tests/phase_b_offline_check.py    # workload + metrics
python tests/phase_c_offline_check.py    # runner tree + CLI
python tests/phase_d_offline_check.py    # end-to-end analysis
```
