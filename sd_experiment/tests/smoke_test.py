"""Smoke test for vLLM 0.17.1 + EAGLE-3 speculative decoding.

This test exercises every API surface the SD trade-off harness relies on:

  1. `LLMEngine.from_engine_args(...)` with a `speculative_config` dict.
  2. A custom `StatLoggerBase` subclass capturing per-step spec-decode stats.
  3. `engine.add_request(...)` + repeated `engine.step()` until done.
  4. Extraction of per-step:
       - aggregate spec-decode counters (drafts / draft tokens / accepted)
       - per-position acceptance vector
       - per-request new token count
       - prefill vs. decode classification
       - end-to-end step latency (perf_counter)

It prints every accessible SD-relevant field found on the returned objects
so any 0.17.1 schema drift is visible in the captured
`tests/smoke_test_output.txt`.

Run (on a CUDA GPU host with vLLM 0.17.1 installed):

    python tests/smoke_test.py 2>&1 | tee tests/smoke_test_output.txt
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from typing import Any

import vllm
from vllm import LLMEngine, SamplingParams
from vllm.engine.arg_utils import EngineArgs
from vllm.v1.metrics.loggers import StatLoggerBase
from vllm.v1.metrics.stats import IterationStats, SchedulerStats


# ---------------------------------------------------------------------------
# Custom stat logger: captures every SchedulerStats + IterationStats we see.
# ---------------------------------------------------------------------------
class CapturingStatLogger(StatLoggerBase):
    """Stores per-step SchedulerStats + IterationStats for later inspection."""

    _instances: list["CapturingStatLogger"] = []

    def __init__(self, vllm_config, engine_index: int = 0):
        self.vllm_config = vllm_config
        self.engine_index = engine_index
        self.records: list[dict[str, Any]] = []
        CapturingStatLogger._instances.append(self)

    def record(
        self,
        scheduler_stats: SchedulerStats | None,
        iteration_stats: IterationStats | None,
        mm_cache_stats=None,
        engine_idx: int = 0,
    ) -> None:
        rec: dict[str, Any] = {"engine_idx": engine_idx}
        if scheduler_stats is not None:
            rec["num_running_reqs"] = scheduler_stats.num_running_reqs
            rec["num_waiting_reqs"] = scheduler_stats.num_waiting_reqs
            rec["kv_cache_usage"] = scheduler_stats.kv_cache_usage
            sds = scheduler_stats.spec_decoding_stats
            if sds is not None:
                rec["spec"] = {
                    "num_spec_tokens": sds.num_spec_tokens,
                    "num_drafts": sds.num_drafts,
                    "num_draft_tokens": sds.num_draft_tokens,
                    "num_accepted_tokens": sds.num_accepted_tokens,
                    "num_accepted_tokens_per_pos": list(sds.num_accepted_tokens_per_pos),
                }
            else:
                rec["spec"] = None
        if iteration_stats is not None:
            rec["gen_tokens_this_step"] = iteration_stats.num_generation_tokens
            rec["prompt_tokens_this_step"] = iteration_stats.prompt_token_stats.total
            rec["ttft_observed"] = list(iteration_stats.time_to_first_tokens_iter)
            rec["itl_observed"] = list(iteration_stats.inter_token_latencies_iter)
        self.records.append(rec)

    def log_engine_initialized(self) -> None:
        pass


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument(
        "--draft-model",
        default="yuhuili/EAGLE3-LLaMA3.1-Instruct-8B",
        help="Set to empty string to disable SD.",
    )
    ap.add_argument("--num-speculative-tokens", type=int, default=5)
    ap.add_argument("--tensor-parallel-size", type=int, default=1)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--max-tokens", type=int, default=32)
    ap.add_argument("--prompt", default="Write a haiku about speculative decoding.")
    args = ap.parse_args()

    print(f"vllm version: {vllm.__version__}", flush=True)
    assert vllm.__version__ == "0.17.1", (
        f"Expected vLLM 0.17.1, got {vllm.__version__}.  Pin in requirements.txt."
    )

    spec_cfg: dict | None
    if args.draft_model:
        spec_cfg = {
            "method": "eagle3",
            "model": args.draft_model,
            "num_speculative_tokens": args.num_speculative_tokens,
            "draft_tensor_parallel_size": 1,
        }
    else:
        spec_cfg = None

    engine_args = EngineArgs(
        model=args.model,
        speculative_config=spec_cfg,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        enforce_eager=False,
        seed=0,
        disable_log_stats=False,
    )

    print("Building engine...", flush=True)
    engine = LLMEngine.from_engine_args(
        engine_args,
        stat_loggers=[CapturingStatLogger],
        enable_multiprocessing=False,
    )
    print("Engine built.", flush=True)

    sampling = SamplingParams(
        temperature=0.0,
        max_tokens=args.max_tokens,
        seed=0,
    )

    engine.add_request("req-0", args.prompt, sampling)

    step_idx = 0
    step_rows: list[dict[str, Any]] = []
    last_len: dict[str, int] = {}

    while engine.has_unfinished_requests():
        t0 = time.perf_counter()
        outs = engine.step()
        t1 = time.perf_counter()

        # Extract per-request deltas.
        per_req = []
        for ro in outs:
            cur_len = len(ro.outputs[0].token_ids) if ro.outputs else 0
            delta = cur_len - last_len.get(ro.request_id, 0)
            last_len[ro.request_id] = cur_len
            per_req.append(
                {
                    "request_id": ro.request_id,
                    "new_tokens": delta,
                    "finished": ro.finished,
                    "num_cached_tokens": getattr(ro, "num_cached_tokens", None),
                }
            )

        # Pop the most recent stat-logger record (StatLoggerManager.record is
        # called once per step *per logger*; we registered exactly one).
        logger_recs = CapturingStatLogger._instances[0].records
        latest_rec = logger_recs[-1] if len(logger_recs) > step_idx else None

        row = {
            "step": step_idx,
            "latency_s": t1 - t0,
            "is_decode_step": bool(
                latest_rec is not None
                and latest_rec.get("prompt_tokens_this_step", 0) == 0
            ),
            "per_req": per_req,
            "stats": latest_rec,
        }
        step_rows.append(row)
        step_idx += 1

    # --- Dump everything -----------------------------------------------------
    print("\n================ SMOKE TEST SUMMARY ================", flush=True)
    print(f"num_steps: {len(step_rows)}", flush=True)
    decode_rows = [r for r in step_rows if r["is_decode_step"]]
    prefill_rows = [r for r in step_rows if not r["is_decode_step"]]
    print(f"num_prefill_steps: {len(prefill_rows)}", flush=True)
    print(f"num_decode_steps:  {len(decode_rows)}", flush=True)

    total_accepted = 0
    total_draft = 0
    total_drafts = 0
    for r in decode_rows:
        sp = (r["stats"] or {}).get("spec")
        if sp:
            total_accepted += sp["num_accepted_tokens"]
            total_draft += sp["num_draft_tokens"]
            total_drafts += sp["num_drafts"]
    if total_drafts:
        mean_accept_len = 1 + total_accepted / total_drafts
        accept_rate = total_accepted / total_draft if total_draft else float("nan")
        print(
            f"mean_acceptance_length (incl. bonus): {mean_accept_len:.3f}",
            flush=True,
        )
        print(f"draft_acceptance_rate: {accept_rate:.3f}", flush=True)
    else:
        print("SD disabled or no decode drafts observed.", flush=True)

    # Print the first 3 decode rows verbatim so we can inspect field shapes.
    for r in decode_rows[:3]:
        print(json.dumps(r, indent=2, default=str), flush=True)

    # Final generated text.
    # Re-fetch via a trivial extra step: already finished, so just print last.
    # (We already stored finished=True inside per_req.)
    print("\nGenerated tokens (request 0):", last_len.get("req-0"), flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
