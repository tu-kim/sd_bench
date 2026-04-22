"""Step-by-step vLLM 0.17.1 LLMEngine runner with step-level metric capture.

Entry point for a single experimental configuration.  Phase D's grid
driver (`scripts/run_grid.sh`) invokes this module once per config.

Design choices, all anchored to `VLLM_API_NOTES.md`:

  * We run the engine **in-process** (`enable_multiprocessing=False`) so
    `engine.step()` is a synchronous call we can wall-clock.
  * Per-step aggregate SD stats are captured via a custom
    `StatLoggerBase` subclass pushed into the engine through
    `stat_loggers=[...]`.
  * Tree shape is expressed via `speculative_token_tree`; there is no
    `eagle_topk` field in 0.17.1 (see `runner/tree.py`).
  * Prefill / decode classification reads
    `IterationStats.prompt_token_stats.total`; the first
    `--warmup-decode-steps` decode iterations are discarded by the
    analysis layer (stored but flagged).
  * Greedy sampling (`temperature=0.0`) + seed=0 by default ⇒ acceptance
    rate is reproducible for a given (prompt, target, drafter, tree).

Example:

    python -m runner.engine_runner \\
        --model meta-llama/Llama-3.1-8B-Instruct \\
        --draft-model yuhuili/EAGLE3-LLaMA3.1-Instruct-8B \\
        --batch-size 4 \\
        --num-samples 4 \\
        --num-speculative-tokens 3 \\
        --eagle-topk 2 \\
        --num-draft-tokens 6 \\
        --max-tokens 64 \\
        --output-dir results/ad_hoc
"""

from __future__ import annotations

import argparse
import collections
import dataclasses
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

# Allow `python runner/engine_runner.py` from the package root *and*
# `python -m runner.engine_runner`.
_THIS_DIR = Path(__file__).resolve().parent
_PKG_ROOT = _THIS_DIR.parent
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

from runner.tree import build_tree_string, validate_tree_params, TreeSpec


# ---------------------------------------------------------------------------
# Prompt loading helpers
# ---------------------------------------------------------------------------
def _load_prompts(
    prompts_file: str | None,
    num_samples: int,
    *,
    tokenizer_for_mock: str | None,
) -> list[dict[str, Any]]:
    """Return a list of dicts with at least `prompt` and `instance_id`.

    If `prompts_file` is given, it must be a JSONL where each line is a
    dict produced by `workload.swe_bench_loader` (fields: `prompt`,
    `instance_id`, `prompt_token_len`, ...).  Otherwise we fall back to
    `workload.load_swebench(...)` + `workload.build_agent_prompt(...)`.
    """
    rows: list[dict[str, Any]] = []
    if prompts_file:
        with open(prompts_file) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
        if not rows:
            raise ValueError(f"prompts_file {prompts_file!r} is empty")
        return rows[:num_samples]

    # Fallback: synthesize from mock / HF via workload.
    from workload import build_agent_prompt, load_swebench
    samples = load_swebench(max_samples=num_samples)
    for s in samples:
        rows.append({**s, "prompt": build_agent_prompt(s)})
    return rows[:num_samples]


# ---------------------------------------------------------------------------
# Custom stat logger
# ---------------------------------------------------------------------------
def _make_stat_logger_class():
    """Return a fresh StatLoggerBase subclass per run.

    Fresh class so that class-level state (the stats queue) is not
    leaked across runs if someone imports this module from a notebook.
    """
    from vllm.v1.metrics.loggers import StatLoggerBase

    class _StepStatsCapture(StatLoggerBase):
        # Shared deque: one entry per LLMEngine.step() that produced
        # outputs (i.e. at most one per engine.step() call; zero if
        # step() did nothing).
        queue: collections.deque = collections.deque()

        def __init__(self, vllm_config, engine_index: int = 0) -> None:
            self.vllm_config = vllm_config
            self.engine_index = engine_index

        def record(
            self,
            scheduler_stats,
            iteration_stats,
            mm_cache_stats=None,
            engine_idx: int = 0,
        ) -> None:
            _StepStatsCapture.queue.append(
                (scheduler_stats, iteration_stats)
            )

        def log_engine_initialized(self) -> None:
            pass

    return _StepStatsCapture


# ---------------------------------------------------------------------------
# Engine construction
# ---------------------------------------------------------------------------
def _build_engine_args(
    *,
    model: str,
    draft_model: str | None,
    tree: TreeSpec | None,
    tensor_parallel_size: int,
    gpu_memory_utilization: float,
    max_model_len: int,
    max_num_seqs: int,
    seed: int,
):
    from vllm.engine.arg_utils import EngineArgs

    spec_cfg = None
    if draft_model:
        assert tree is not None, "tree spec required when SD is on"
        tree_str, total_nodes = build_tree_string(tree.depth, tree.branching)
        assert total_nodes == tree.total
        spec_cfg = {
            "method": "eagle3",
            "model": draft_model,
            "num_speculative_tokens": tree.depth,
            "speculative_token_tree": tree_str,
            "draft_tensor_parallel_size": 1,
        }

    return EngineArgs(
        model=model,
        speculative_config=spec_cfg,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        # Cap concurrent running requests so --batch-size actually binds.
        max_num_seqs=max_num_seqs,
        seed=seed,
        disable_log_stats=False,
    )


# ---------------------------------------------------------------------------
# Main run
# ---------------------------------------------------------------------------
def run(args: argparse.Namespace) -> Path:
    """Run one configuration end-to-end.  Returns path to the summary JSON."""

    # Imports are local so `--help` / offline tools don't pay for vllm load.
    import vllm
    from vllm import LLMEngine, SamplingParams

    assert vllm.__version__ == "0.17.1", (
        f"harness pinned to vllm 0.17.1; found {vllm.__version__}"
    )

    # Seed everything user-controllable.
    random.seed(args.seed)
    try:
        import numpy as _np
        _np.random.seed(args.seed)
    except ImportError:
        pass

    # Resolve tree spec (ignored in --no-sd).
    tree: TreeSpec | None
    if args.no_sd:
        tree = None
    else:
        tree = validate_tree_params(
            depth=args.num_speculative_tokens,
            branching=args.eagle_topk,
            total=args.num_draft_tokens,
        )

    # Load prompts.
    prompts = _load_prompts(
        args.prompts_file,
        args.num_samples,
        tokenizer_for_mock=args.model,
    )
    if len(prompts) < args.num_samples:
        print(
            f"[engine_runner] only {len(prompts)} prompts available; "
            f"batch_size will be at most that."
        )
        args.num_samples = len(prompts)

    # We want an exact running batch of B, so we submit exactly min(B, N)
    # prompts and leave max_num_seqs = B.  This matches "one batch size
    # per run".  The caller is responsible for choosing --num-samples.
    batch = min(args.batch_size, args.num_samples)
    prompts = prompts[:batch]

    # Prepare stat logger.
    StatLoggerCls = _make_stat_logger_class()

    # Build engine.
    eng_args = _build_engine_args(
        model=args.model,
        draft_model=(None if args.no_sd else args.draft_model),
        tree=tree,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        max_num_seqs=batch,
        seed=args.seed,
    )
    print("[engine_runner] building LLMEngine ...", flush=True)
    engine = LLMEngine.from_engine_args(
        eng_args,
        stat_loggers=[StatLoggerCls],
        enable_multiprocessing=False,
    )
    print("[engine_runner] engine built.", flush=True)

    # Sampling.
    sampling = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        seed=args.seed,
    )

    # Queue all prompts.
    for i, row in enumerate(prompts):
        rid = row.get("request_id") or row.get("instance_id") or f"req-{i}"
        engine.add_request(rid, row["prompt"], sampling)

    # Collector.
    from metrics import MetricCollector

    config_echo = _config_echo(args, tree, prompts)
    collector = MetricCollector(
        config_echo,
        warmup_decode_steps=args.warmup_decode_steps,
        seed=args.seed,
    )

    sd_config = {
        "num_speculative_tokens": tree.depth if tree else None,
        "eagle_topk": tree.branching if tree else None,
        "num_draft_tokens": tree.total if tree else None,
    }

    # Step loop.
    step_idx = 0
    wall_start = time.perf_counter()
    while engine.has_unfinished_requests():
        if args.max_steps and step_idx >= args.max_steps:
            print(
                f"[engine_runner] hit --max-steps={args.max_steps}; aborting.",
                flush=True,
            )
            break
        t0 = time.perf_counter()
        outs = engine.step()
        t1 = time.perf_counter()

        # step() can return [] if the engine was idle this tick; skip.
        if not outs:
            continue

        # One record() per step that produced outputs; pop it.
        sched_stats, iter_stats = (None, None)
        if StatLoggerCls.queue:
            sched_stats, iter_stats = StatLoggerCls.queue.popleft()

        collector.record_step(
            step_idx=step_idx,
            total_latency_ms=(t1 - t0) * 1000.0,
            request_outputs=outs,
            scheduler_stats=sched_stats,
            iteration_stats=iter_stats,
            sd_config=sd_config,
        )
        step_idx += 1

    wall_end = time.perf_counter()
    run_meta = collector.finalize()
    print(
        f"[engine_runner] finished: {step_idx} steps, "
        f"{wall_end - wall_start:.2f}s wall-clock.",
        flush=True,
    )

    # Output filenames encode the config so grid sweeps land in the same dir.
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = _filename_stem(args, tree)
    detail = out_dir / f"{stem}_detail.json"
    summary = out_dir / f"{stem}_summary.json"
    run_meta.save(detail, summary)
    print(f"[engine_runner] detail:  {detail}", flush=True)
    print(f"[engine_runner] summary: {summary}", flush=True)
    return summary


# ---------------------------------------------------------------------------
# Filename / config-echo helpers
# ---------------------------------------------------------------------------
def _filename_stem(args: argparse.Namespace, tree: TreeSpec | None) -> str:
    if tree is None:
        return (
            f"run_nosd_b{args.batch_size}_ctx{args.max_model_len}"
            f"_mt{args.max_tokens}"
        )
    return (
        f"run_sd_b{args.batch_size}_d{tree.depth}_k{tree.branching}"
        f"_t{tree.total}_ctx{args.max_model_len}_mt{args.max_tokens}"
    )


def _config_echo(
    args: argparse.Namespace,
    tree: TreeSpec | None,
    prompts: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "model": args.model,
        "draft_model": None if args.no_sd else args.draft_model,
        "method": None if args.no_sd else "eagle3",
        "num_speculative_tokens": tree.depth if tree else None,
        "eagle_topk": tree.branching if tree else None,
        "num_draft_tokens": tree.total if tree else None,
        "batch_size": args.batch_size,
        "num_samples": args.num_samples,
        "max_tokens": args.max_tokens,
        "max_model_len": args.max_model_len,
        "tensor_parallel_size": args.tensor_parallel_size,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "temperature": args.temperature,
        "seed": args.seed,
        "warmup_decode_steps": args.warmup_decode_steps,
        "max_steps": args.max_steps,
        "prompts_file": args.prompts_file,
        "sd_disabled": args.no_sd,
        "prompt_token_lens": [
            p.get("prompt_token_len") for p in prompts
        ],
        "instance_ids": [
            p.get("instance_id") or p.get("request_id") for p in prompts
        ],
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Run one SD trade-off config on vLLM 0.17.1 LLMEngine.",
    )
    # Model / drafter.
    ap.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument(
        "--draft-model",
        default="yuhuili/EAGLE3-LLaMA3.1-Instruct-8B",
    )
    ap.add_argument("--no-sd", action="store_true",
                    help="Disable speculative decoding (baseline).")

    # Prompts.
    ap.add_argument(
        "--prompts-file",
        default=None,
        help="JSONL of prompts (e.g. one of data/swebench_bucketed/bucket_*.jsonl). "
             "If omitted, falls back to workload.load_swebench() (mock-friendly).",
    )
    ap.add_argument("--num-samples", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=4)

    # Generation.
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max-steps", type=int, default=0,
                    help="Cap on engine.step() calls; 0 disables the cap.")
    ap.add_argument("--warmup-decode-steps", type=int, default=3)

    # SD shape.
    ap.add_argument("--num-speculative-tokens", type=int, default=3,
                    help="Tree depth D.")
    ap.add_argument("--eagle-topk", type=int, default=None,
                    help="Branching K.  If omitted, derived from D and T.")
    ap.add_argument("--num-draft-tokens", type=int, default=None,
                    help="Total tree nodes T.  If omitted, derived from (D, K).")

    # Hardware.
    ap.add_argument("--tensor-parallel-size", type=int, default=1)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    ap.add_argument("--max-model-len", type=int, default=8192)

    # Output.
    ap.add_argument("--output-dir", default="results/ad_hoc")
    ap.add_argument("--seed", type=int, default=0)

    return ap


def main(argv: list[str] | None = None) -> int:
    ap = build_argparser()
    args = ap.parse_args(argv)
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
