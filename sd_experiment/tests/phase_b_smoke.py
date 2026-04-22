"""Phase B end-to-end single-config smoke.

Runs **one** configuration (small B, small D, chain SD) against 4 prompts
with max_tokens=32, using `workload/` to build prompts and `metrics/` to
record step-level metrics.  Dumps both `_detail.json` and `_summary.json`
under `results/phase_b_smoke/`.

This is **not** the full-featured grid runner (Phase C) — it is the
minimal proof that workload + metrics + engine wire up correctly on a
single config.

Requires a CUDA device with vLLM 0.17.1 installed.  On a CPU-only host
this script exits early with a clear message; the offline validator
`tests/phase_b_offline_check.py` covers everything that doesn't need a
GPU.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument(
        "--draft-model", default="yuhuili/EAGLE3-LLaMA3.1-Instruct-8B"
    )
    ap.add_argument("--num-speculative-tokens", type=int, default=3)
    ap.add_argument("--num-samples", type=int, default=4)
    ap.add_argument("--max-tokens", type=int, default=32)
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    ap.add_argument(
        "--output-dir",
        default="results/phase_b_smoke",
    )
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print(
            "[phase_b_smoke] torch.cuda.is_available() is False; "
            "this script needs a GPU.  Run tests/phase_b_offline_check.py "
            "instead to validate the non-engine parts."
        )
        return 0

    # Imports deferred so CPU boxes can at least --help this script.
    import vllm
    from vllm import LLMEngine, SamplingParams
    from vllm.engine.arg_utils import EngineArgs
    from vllm.v1.metrics.loggers import StatLoggerBase

    assert vllm.__version__ == "0.17.1", (
        f"Expected vllm 0.17.1, got {vllm.__version__}"
    )

    from transformers import AutoTokenizer
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from workload import load_swebench, build_agent_prompt
    from metrics import MetricCollector

    # --- Build prompts -------------------------------------------------
    samples = load_swebench(max_samples=args.num_samples)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    prompts: list[tuple[str, str]] = []  # (request_id, text)
    for i, s in enumerate(samples[: args.num_samples]):
        text = build_agent_prompt(s)
        prompts.append((f"req-{i}", text))
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        print(f"  req-{i}: {s.get('instance_id')} (prompt_tokens={len(ids)})")

    # --- Custom stat logger (captures per-step SchedulerStats) ---------
    class _Cap(StatLoggerBase):
        last_scheduler_stats: object | None = None
        last_iteration_stats: object | None = None

        def __init__(self, vllm_config, engine_index: int = 0) -> None:
            _Cap.cfg = vllm_config

        def record(
            self,
            scheduler_stats,
            iteration_stats,
            mm_cache_stats=None,
            engine_idx: int = 0,
        ) -> None:
            _Cap.last_scheduler_stats = scheduler_stats
            _Cap.last_iteration_stats = iteration_stats

        def log_engine_initialized(self) -> None:
            pass

    # --- Build engine --------------------------------------------------
    engine_args = EngineArgs(
        model=args.model,
        speculative_config={
            "method": "eagle3",
            "model": args.draft_model,
            "num_speculative_tokens": args.num_speculative_tokens,
            "draft_tensor_parallel_size": 1,
        },
        tensor_parallel_size=1,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        seed=0,
        disable_log_stats=False,
    )
    engine = LLMEngine.from_engine_args(
        engine_args, stat_loggers=[_Cap], enable_multiprocessing=False,
    )

    sampling = SamplingParams(temperature=0.0, max_tokens=args.max_tokens, seed=0)
    for rid, txt in prompts:
        engine.add_request(rid, txt, sampling)

    # --- Collector -----------------------------------------------------
    config = {
        "model": args.model,
        "draft_model": args.draft_model,
        "method": "eagle3",
        "num_speculative_tokens": args.num_speculative_tokens,
        "eagle_topk": 1,
        "num_draft_tokens": args.num_speculative_tokens,
        "batch_size_requested": args.num_samples,
        "max_tokens": args.max_tokens,
        "temperature": 0.0,
        "seed": 0,
    }
    collector = MetricCollector(config, warmup_decode_steps=3, seed=0)

    # --- Step loop -----------------------------------------------------
    step_idx = 0
    while engine.has_unfinished_requests():
        t0 = time.perf_counter()
        outs = engine.step()
        t1 = time.perf_counter()
        collector.record_step(
            step_idx=step_idx,
            total_latency_ms=(t1 - t0) * 1000.0,
            request_outputs=outs,
            scheduler_stats=_Cap.last_scheduler_stats,
            iteration_stats=_Cap.last_iteration_stats,
            sd_config={
                "num_speculative_tokens": args.num_speculative_tokens,
                "eagle_topk": 1,
                "num_draft_tokens": args.num_speculative_tokens,
            },
        )
        step_idx += 1

    run = collector.finalize()

    # --- Dump ----------------------------------------------------------
    out_dir = Path(args.output_dir)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    detail = out_dir / f"phase_b_{stamp}_detail.json"
    summary = out_dir / f"phase_b_{stamp}_summary.json"
    run.save(detail, summary)

    # --- Report --------------------------------------------------------
    s = run.summary()
    print("\n================ PHASE B RUN SUMMARY ================")
    for k in (
        "num_decode_steps_post_warmup",
        "avg_step_latency_ms",
        "p50_step_latency_ms",
        "p99_step_latency_ms",
        "avg_accepted_per_step",
        "mean_acceptance_length",
        "draft_acceptance_rate",
        "throughput_tokens_per_sec",
        "effective_per_token_latency_ms",
        "avg_batch_size",
        "avg_kv_cache_usage",
        "avg_verify_work",
    ):
        print(f"  {k}: {s.get(k)!r}")
    print(f"detail: {detail}")
    print(f"summary: {summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
