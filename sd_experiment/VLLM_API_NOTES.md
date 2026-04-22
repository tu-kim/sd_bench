# vLLM 0.17.1 API Notes (for SD Trade-off Harness)

All findings here were obtained by static introspection against a fresh
`pip install vllm==0.17.1` (Python 3.11.15, CPU-only host used for
inspection; target deployment is CUDA 12.x).  Every claim below is
cross-referenced to the exact source file installed under
`site-packages/vllm/` so that it is easy to re-verify on a GPU box.

## 1. Engine entry point

`vllm.LLMEngine` in 0.17.1 is re-exported from `vllm.v1.engine.llm_engine`
(the v1 engine; the legacy v0 engine path is gone from the public surface).

- Construction:
  ```python
  from vllm import LLMEngine, SamplingParams
  from vllm.engine.arg_utils import EngineArgs

  engine = LLMEngine.from_engine_args(
      engine_args,                       # EngineArgs
      usage_context=UsageContext.ENGINE_CONTEXT,
      stat_loggers=[MyStatLogger],       # list[StatLoggerFactory] | None
      enable_multiprocessing=False,      # keep in-process so step() is synchronous
  )
  ```
  Signature verified at `vllm/v1/engine/llm_engine.py::LLMEngine.from_engine_args`.

- Core per-step API (all on `LLMEngine`):
  - `add_request(request_id: str, prompt, params: SamplingParams, ...)` — returns
    the request id.  `prompt` accepts `str`, `TextPrompt`, `TokensPrompt`, or a
    pre-tokenized `list[int]`.
  - `step() -> list[RequestOutput | PoolingRequestOutput]` — drives **one
    scheduler/executor iteration**.  Returns only the processed request outputs
    (aggregate per-step stats are NOT returned here; see §3 below for how to
    capture them).
  - `has_unfinished_requests() -> bool`, `get_num_unfinished_requests() -> int`.
  - `abort_request(request_id)`.
  - `get_metrics() -> list[Metric]` — returns the current Prometheus snapshot;
    useful for spot-checks but not for per-step collection.

- Sync vs. async:  pass `enable_multiprocessing=False` (default) so
  `step()` is a blocking in-process call; that is what we time with
  `time.perf_counter()` around the call.

## 2. Speculative decoding configuration

In 0.17.1 there is **exactly one** entry point: `EngineArgs.speculative_config`.
All 0.6.x per-arg fields (`speculative_model`, `num_speculative_tokens`,
`speculative_draft_tensor_parallel_size`, `spec_decoding_acceptance_method`,
…) are **gone** from `EngineArgs`.

```python
engine_args = EngineArgs(
    model="meta-llama/Llama-3.1-8B-Instruct",
    speculative_config={
        "method": "eagle3",                                  # see table below
        "model": "yuhuili/EAGLE3-LLaMA3.1-Instruct-8B",      # HF id of drafter
        "num_speculative_tokens": 5,                         # depth D (chain)
        # Optional: full tree specification.  Overrides num_speculative_tokens
        # when provided.  See §2.2 for the format.
        # "speculative_token_tree": "[(0,), (0,0), (0,0,0)]",
        "draft_tensor_parallel_size": 1,
    },
    ...
)
```

Fields verified at `vllm/config/speculative.py::SpeculativeConfig`:

| Key                            | Type / default            | Notes                                                               |
|--------------------------------|---------------------------|---------------------------------------------------------------------|
| `method`                       | Literal, `None`           | `ngram`, `medusa`, `mlp_speculator`, `draft_model`, `suffix`, **`eagle`, `eagle3`**, `deepseek_mtp`, `mtp`, etc. |
| `model`                        | `str \| None`             | HF id or local path of the draft/EAGLE checkpoint.                  |
| `num_speculative_tokens`       | `int` (>0), `None`        | Depth D of a **linear** speculation chain.                          |
| `speculative_token_tree`       | `str \| None`             | String-repr of a list of tuples (see §2.2). Overrides the chain.    |
| `draft_tensor_parallel_size`   | `int` (>=1), `None`       | TP for the drafter; independent of target TP.                       |
| `quantization`                 | Literal, `None`           | Quant for the drafter.                                              |
| `max_model_len`                | `int` (>=1), `None`       | Drafter context length override.                                    |
| `disable_padded_drafter_batch` | `bool`, default `False`   | Leave `False` (padded batch is the supported path for EAGLE).       |
| `prompt_lookup_max/min`        | `int`, `None`             | Only for `method="ngram"`.                                          |
| `suffix_decoding_*`            | various                   | Only for `method="suffix"`.                                         |

**There is no `eagle_topk` parameter** in 0.17.1 `SpeculativeConfig`.  Tree
branching is expressed entirely through `speculative_token_tree` (§2.2).  The
harness's CLI `--eagle-topk K` is translated into a `speculative_token_tree`
string at engine-build time.

### 2.1  `method` values we care about

- `"eagle3"` — what `yuhuili/EAGLE3-LLaMA3.1-Instruct-8B` uses.  The target
  model must implement the `SupportsEagle3` interface
  (`vllm.model_executor.models.interfaces`); `LlamaForCausalLM` does.  Source:
  `vllm/v1/worker/gpu/model_runner.py:166` and
  `vllm/v1/worker/gpu/spec_decode/eagle/eagle3_utils.py`.
- `"eagle"` — original EAGLE (non-3) drafter.
- `"ngram"` — no draft model; set `prompt_lookup_min/max` instead of `model`.

### 2.2  `speculative_token_tree` format

The string is `ast.literal_eval`'d into a `list[tuple[int, ...]]` (see
`vllm/config/speculative.py:566` and `vllm/v1/spec_decode/eagle.py:252`).
Each tuple is a **path from the root** of the speculation tree, one entry per
depth, valued `0..K-1` (the child index at that depth).  vLLM re-sorts the
list breadth-first after parsing (`sorted(..., key=lambda t: (len(t), t))`).

Examples:

| Shape                        | Tree string                                                                                  |
|------------------------------|----------------------------------------------------------------------------------------------|
| chain D=3 (K=1)              | `"[(0,), (0, 0), (0, 0, 0)]"`                                                                |
| K=2, D=2 (T=6)               | `"[(0,), (1,), (0, 0), (0, 1), (1, 0), (1, 1)]"`                                             |
| K=2, D=3 (T=14)              | chain+all children of depth-1 and depth-2 nodes; 2+4+8=14 tuples.                            |

**Harness mapping** (in `runner/engine_runner.py`):

```python
def build_tree(depth: int, branching: int) -> str:
    from itertools import product
    paths = []
    for d in range(1, depth + 1):
        for p in product(range(branching), repeat=d):
            paths.append(p)
    return str(paths)
```

- `num_speculative_tokens` (CLI `-D`) → `depth`.
- `eagle_topk` (CLI `-K`) → `branching`.
- `num_draft_tokens` (CLI `-T`) → total nodes;
  **we derive `T = sum(K**d for d in 1..D)`** and reject CLI configs where
  this doesn't match the user-supplied `--num-draft-tokens`.  (We log the
  actual T that vLLM uses.)

## 3. Per-step metrics

`LLMEngine.step()` returns only `list[RequestOutput]`.  To capture
per-iteration aggregate stats (including spec-decode acceptance), we register
a **custom stat logger** via `stat_loggers=[...]`.  The call graph is:

```
LLMEngine.step
 └── self.logger_manager.record(scheduler_stats, iteration_stats, mm_cache_stats)
      └── for logger in self.stat_loggers: logger.record(...)
```
(Source: `vllm/v1/engine/llm_engine.py::LLMEngine.step` +
`vllm/v1/metrics/loggers.py::StatLoggerManager.record`.)

### 3.1  `StatLoggerBase` contract (verified)

```python
class StatLoggerBase(ABC):
    def __init__(self, vllm_config: VllmConfig, engine_index: int = 0): ...
    def record(self,
               scheduler_stats: SchedulerStats | None,
               iteration_stats: IterationStats | None,
               mm_cache_stats: MultiModalCacheStats | None = None,
               engine_idx: int = 0): ...
    def log_engine_initialized(self): ...
```

Source: `vllm/v1/metrics/loggers.py::StatLoggerBase`.

### 3.2  Fields we read on each `record()` call

From `SchedulerStats` (`vllm/v1/metrics/stats.py`):
- `num_running_reqs`, `num_waiting_reqs`         — current running batch size.
- `kv_cache_usage`                               — HBM utilisation (float).
- `spec_decoding_stats: SpecDecodingStats | None` — **the one we actually need**.

From `SpecDecodingStats` (`vllm/v1/spec_decode/metrics.py`):
- `num_spec_tokens`                  — depth D (or total tree nodes when a tree is configured).
- `num_drafts`                       — number of drafts emitted this step
                                       (≈ number of requests drafted this step).
- `num_draft_tokens`                 — total draft tokens proposed across the batch.
- `num_accepted_tokens`              — total draft tokens accepted across the batch.
- `num_accepted_tokens_per_pos: list[int]` — length D, accepted count at each
  position of the draft.

From `IterationStats` (`vllm/v1/metrics/stats.py`):
- `num_generation_tokens`            — new generation tokens emitted this step
                                       (across all requests).
- `prompt_token_stats.total`         — new prompt tokens this step
                                       (> 0 iff some request was still prefilling).
- `inter_token_latencies_iter`       — per-request inter-token latencies
                                       observed this step.

### 3.3  Prefill vs. decode classification

vLLM does **not** give `step()` a direct flag.  We classify a step as
**decode-only** iff `iteration_stats.prompt_token_stats.total == 0`.
Warmup rule: discard the **first 3 decode-only steps per run** (CUDA-graph
warmup, kernel autotune).  Source: `IterationStats.update_from_output`
branches on `is_prefilling` (`vllm/v1/metrics/stats.py`); we reconstruct the
same signal post-hoc.

### 3.4  Draft-time vs. verify-time breakdown

**Not exposed separately** in 0.17.1 public stats.  `PerfStats` only tracks
FLOPs / read-bytes / write-bytes (`vllm/v1/metrics/perf.py`) and not a
draft/verify time split.  Accordingly, the harness's `StepMetric` keeps
`draft_latency_ms=None` and `verify_latency_ms=None`; `total_latency_ms` is
measured via `time.perf_counter()` around the single `engine.step()` call and
covers both phases end-to-end.

(If a future version of vLLM, or a privately-patched build, adds timers to
the drafter and verifier, we can wire them in through our custom stat
logger — the collector already passes through `StepMetric` unchanged.)

## 4. `RequestOutput` / `CompletionOutput`

- `RequestOutput(request_id, prompt, prompt_token_ids, prompt_logprobs,
  outputs: list[CompletionOutput], finished, metrics: RequestStateStats | None,
  num_cached_tokens, ...)`.
- `CompletionOutput.token_ids` contains **cumulative** token ids by default
  (`RequestOutputKind.CUMULATIVE`), so per-step newly-generated tokens per
  request = `len(curr.token_ids) - len(prev.token_ids)`.  The harness tracks
  `last_len_by_req_id` locally to compute this delta.
- `RequestStateStats` (`vllm/v1/metrics/stats.py`) has timing info
  (`arrival_time`, `queued_ts`, `scheduled_ts`, `first_token_ts`,
  `last_token_ts`, `first_token_latency`) and `num_generation_tokens`.  No
  per-request spec-decode counters here — those live only in the aggregate
  `SchedulerStats.spec_decoding_stats`.  That means our **per-request
  accepted tokens are derived** from the total-accepted delta distributed
  proportionally to each request's `num_generation_tokens` delta this step,
  which is exactly how vLLM's internal dashboard computes it.

## 5. Things we explicitly are NOT using

These existed in older vLLM releases but are **absent or renamed** in 0.17.1,
so any code that references them will error.  Documented here so they are not
accidentally copy-pasted back in:

- `EngineArgs(speculative_model=...)` — removed.  Use `speculative_config`.
- `EngineArgs(num_speculative_tokens=...)` at the top level — removed.
- `EngineArgs(speculative_draft_tensor_parallel_size=...)` — removed.
- `SamplingParams(spec_decoding_acceptance_method=...)` — removed; not
  user-tunable in 0.17.1.  Greedy verification (temperature=0) gives
  deterministic acceptance, which is what we want for the harness.
- `engine.step()` returning `(request_outputs, spec_metrics)` tuple — never
  existed this way; `step()` always returned just outputs.  Spec metrics
  come through the stat-logger channel described in §3.
- `RequestOutput.num_accepted_tokens` / `.num_generated_tokens` — do not
  exist.  Use `CompletionOutput.token_ids` delta + `SchedulerStats.spec_decoding_stats`.
- `engine.stat_loggers` public attribute — the `logger_manager` is private
  (`LLMEngine.logger_manager`) in 0.17.1; supply loggers via
  `from_engine_args(..., stat_loggers=[...])`.

## 6. Seeding and determinism

- `EngineArgs(seed=0)` seeds the engine's RNG.
- `SamplingParams(seed=<int>, temperature=0.0)` makes sampling greedy and
  deterministic.  With greedy sampling, EAGLE's acceptance rule reduces to
  exact-match on draft token vs. target argmax, so acceptance rate is a
  function of (prompt, target model, drafter, tree) alone — reproducible
  across runs.

## 7. Verification status

- ✅ `import vllm; vllm.__version__ == "0.17.1"` on the inspection host.
- ✅ All signatures/fields above quoted directly from
  `/usr/local/lib/python3.11/dist-packages/vllm/…` on that host.
- ⚠️ `tests/smoke_test.py` requires a CUDA GPU to actually load a model; the
  inspection host has no GPU.  The smoke test itself is written against the
  above API and ships with this branch.  It must be run on the target GPU
  box before Phase B; the expected-output capture (`tests/smoke_test_output.txt`)
  is the gating artefact.
