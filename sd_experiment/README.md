# SD Trade-off Experimental Harness

Measures how **batch size (B)** and **tree configuration (depth D,
branching K, tree size T)** interact in vLLM speculative decoding (SD)
under a coding-agent-style long-context workload (SWE-Bench Lite).  The
analysis is explicitly framed around the **memory/compute-constrained
Pareto frontier**: given a verify-work budget `B × (T+1)` or a KV-cache
budget, which `(B, T)` combination wins?

All code is pinned to **vLLM 0.17.1**.  The 0.17.1 API surface is audited
in [`VLLM_API_NOTES.md`](./VLLM_API_NOTES.md); re-run that audit before
bumping the version.

## Quickstart

```bash
cd sd_experiment
make install          # pip install -r requirements.txt  (pulls vllm==0.17.1)
make check            # run every offline validator; no GPU needed

# On a CUDA GPU box:
python tests/smoke_test.py | tee tests/smoke_test_output.txt
make bucket           # tokenize SWE-Bench Lite -> data/swebench_bucketed/
make grid-minimal     # 4 SD configs for pipeline smoke (~minutes)
make analyze          # pivots + heatmaps + 3 Pareto plots

# Full sweep (~hours):
make grid
make analyze RESULTS_DIR=results/<timestamp>
```

`make help` lists every variable you can override.

## Target stack (non-negotiable)

- `vllm==0.17.1` (pinned in `requirements.txt`).
- Python 3.10+; CUDA 12.x GPU.
- Target model: `meta-llama/Llama-3.1-8B-Instruct`.
- Draft model: `yuhuili/EAGLE3-LLaMA3.1-Instruct-8B` (method `eagle3`).

## Repository layout

```
sd_experiment/
├── README.md
├── VLLM_API_NOTES.md                 Phase A audit (0.17.1 API surface)
├── Makefile                          make {install,check,bucket,grid,analyze}
├── requirements.txt                  pins vllm==0.17.1
├── configs/experiment_grid.yaml      grid definition + MINIMAL subset
├── workload/
│   ├── prompt_builder.py             Llama-3.1 chat-literal agent prompts
│   └── swe_bench_loader.py           HF / local / mock 3-tier loader + CLI
├── runner/
│   ├── tree.py                       (D, K, T) → speculative_token_tree
│   └── engine_runner.py              LLMEngine step loop + metric capture
├── metrics/
│   └── collector.py                  StepMetric + RunMetric.summary()
├── scripts/
│   ├── check.sh                      run every offline validator
│   ├── run_grid.sh                   phase 1/2/3 driver (MINIMAL=1 subset)
│   └── analyze_results.py            pivots + heatmaps + Pareto plots
├── tests/
│   ├── smoke_test.py                 Phase A gate (GPU)
│   ├── phase_b_smoke.py              single-config GPU smoke
│   ├── phase_b_offline_check.py      workload + metrics unit checks
│   ├── phase_c_offline_check.py      tree + CLI + filename checks
│   └── phase_d_offline_check.py      end-to-end analysis checks
└── results/                          auto-generated (per-run dirs)
```

## What the analysis produces

Running `make analyze RESULTS_DIR=results/<timestamp>` writes to
`results/<timestamp>/analysis/`:

CSVs:
- `summary.csv`, `describe.txt`
- Pivot tables (batch × tree size) for `avg_step_latency_ms`,
  `throughput_tokens_per_sec`, and `avg_accepted_per_req_per_step`.

Plots:
- **`pareto_throughput_vs_ptl.png`** — effective per-token latency vs.
  throughput, SD on/off split by colour.  The canonical "is SD worth it
  at this batch size?" view.
- **`pareto_kv_vs_throughput.png`** — KV-cache utilisation vs.
  throughput, colour = tree size T, marker size = batch B.
  *Memory-frontier view for B vs T under a KV budget.*
- **`pareto_verifywork_vs_throughput.png`** — `B × (T+1)` (log x-axis)
  vs. throughput, with iso-cost guide-lines at `W ∈ {32, 64, 128, 256,
  512}`.  Same vertical line = same verify-work budget, so you can
  read off directly whether to grow B or T at a fixed compute target.
- **`acceptance_vs_tree.png`** — mean acceptance length vs. T, one line
  per batch size.  Surfaces diminishing returns on deeper/wider trees.
- Heatmaps of the three pivot tables.

## vLLM 0.17.1 specifics (the ones that bite)

1. SD is configured **only** via `EngineArgs.speculative_config` (a
   dict).  `speculative_model` / top-level `num_speculative_tokens` /
   `speculative_draft_tensor_parallel_size` are gone.
2. Tree branching is expressed only via `speculative_token_tree`.
   **There is no `eagle_topk` field.**  The harness translates CLI `-K`
   into the tree string via `runner/tree.py::build_tree_string`.
3. Per-step SD stats reach user code through a custom `StatLoggerBase`
   registered at `from_engine_args(..., stat_loggers=[...])`.
   `engine.step()` itself returns only `list[RequestOutput]`.
4. A step is **decode-only** iff
   `IterationStats.prompt_token_stats.total == 0`.  The first 3 decode
   steps per run are discarded as warmup (tunable via
   `--warmup-decode-steps`).
5. Draft / verify timing is **not** publicly exposed in 0.17.1.
   `StepMetric.draft_latency_ms` / `verify_latency_ms` remain `None`;
   `total_latency_ms` is wall-time around the full `engine.step()`.
6. Greedy sampling (`temperature=0.0`) + `seed=0` → acceptance rate is
   reproducible for a given `(prompt, target, drafter, tree)`.

## Reproducing a single configuration

```bash
python -m runner.engine_runner \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --draft-model yuhuili/EAGLE3-LLaMA3.1-Instruct-8B \
    --prompts-file data/swebench_bucketed/bucket_8192.jsonl \
    --num-samples 4 --batch-size 4 \
    --num-speculative-tokens 3 --eagle-topk 2 --num-draft-tokens 6 \
    --max-tokens 64 --max-model-len 8192 \
    --output-dir results/ad_hoc
```

Output filename encodes the config:
`run_sd_b{B}_d{D}_k{K}_t{T}_ctx{ctx}_mt{max_tokens}_{detail,summary}.json`.
The `--no-sd` flag switches to `run_nosd_b{B}_ctx{...}_mt{...}_...`.

## Offline validation (no GPU)

`make check` runs:
- `tests/phase_b_offline_check.py` — workload prompts, bucketing,
  MetricCollector arithmetic (9-step fabricated run verified bit-exact:
  throughput=400 tok/s, effective-per-token-latency=2.5 ms, etc.),
  save/load round-trip.
- `tests/phase_c_offline_check.py` — `build_tree_string` node counts
  and BFS order, `validate_tree_params` round-trip + rejection, CLI
  flag coverage, filename stems, prompt-file round-trip, config-echo
  JSON serialisation.
- `tests/phase_d_offline_check.py` — synthesise a 4 SD + 2 baseline
  grid, run `analyze_results.py`, assert all 12 output artefacts exist
  with expected shapes and contents.
- `--help` smoke for every user-facing entry point.

All checks pass on a CPU-only host.
