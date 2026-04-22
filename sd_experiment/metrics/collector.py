"""Step-level metric collection + run-level aggregation.

Designed around the vLLM 0.17.1 API surface documented in
`VLLM_API_NOTES.md`:

  * `total_latency_ms` is `time.perf_counter()` around one
    `engine.step()` call — covers draft + verify end-to-end.
  * `draft_latency_ms` / `verify_latency_ms` are kept as `Optional[float]`
    so future vLLM versions with a public split can fill them in without
    schema churn.  In 0.17.1 they stay `None`.
  * `kv_cache_usage`, `num_running_reqs`, `num_preempted_cumulative`
    come from `SchedulerStats`.
  * Per-step aggregate SD counters come from
    `SchedulerStats.spec_decoding_stats`.
  * `per_request_accepted` is derived per step by distributing
    `num_accepted_tokens` proportionally to the number of new generation
    tokens each request received this step (vLLM does not expose a
    per-request accepted counter).

The harness's Pareto analyses live in `scripts/analyze_results.py`;
`RunMetric.summary()` exposes the headline numbers that drive them.
"""

from __future__ import annotations

import dataclasses
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class StepMetric:
    """One scheduler iteration's worth of measurements."""

    step_idx: int
    batch_size: int            # concurrently-running requests this step
    total_latency_ms: float    # wall-clock around engine.step()

    # 0.17.1 does not expose a public draft/verify time split; stays None.
    draft_latency_ms: float | None = None
    verify_latency_ms: float | None = None

    # Aggregate SD counters (SchedulerStats.spec_decoding_stats).
    total_draft_tokens: int = 0          # proposed this step across batch
    total_accepted_tokens: int = 0       # accepted this step across batch
    num_drafts: int = 0                  # ~= number of requests drafted
    num_accepted_tokens_per_pos: list[int] = field(default_factory=list)

    # Per-request arrays, same length == batch_size.
    per_request_accepted: list[int] = field(default_factory=list)
    per_request_new_tokens: list[int] = field(default_factory=list)
    per_request_context_len: list[int] = field(default_factory=list)
    per_request_ids: list[str] = field(default_factory=list)

    # Prefill vs. decode flag reconstructed from IterationStats.
    is_decode_step: bool = True

    # Config echo — kept per-step so downstream dataframes can
    # self-describe without joining against the config JSON.
    num_speculative_tokens: int | None = None
    eagle_topk: int | None = None
    num_draft_tokens: int | None = None   # tree size T, if applicable

    # Memory / preemption signals.
    kv_cache_usage: float | None = None
    num_preempted_cumulative: int = 0


@dataclass
class RunMetric:
    """All step metrics for a single run + configuration echo."""

    config: dict[str, Any]
    warmup_decode_steps: int
    seed: int
    wall_time_start: float
    wall_time_end: float | None = None
    steps: list[StepMetric] = field(default_factory=list)

    # Populated by finalize().
    summary_cache: dict[str, Any] | None = None

    # -----------------------------------------------------------------
    # Derived helpers
    # -----------------------------------------------------------------
    def decode_steps(self) -> list[StepMetric]:
        """Decode-only steps *after* warmup."""
        decodes = [s for s in self.steps if s.is_decode_step]
        return decodes[self.warmup_decode_steps:]

    def summary(self) -> dict[str, Any]:
        if self.summary_cache is not None:
            return self.summary_cache

        ds = self.decode_steps()
        if not ds:
            out = {
                "num_decode_steps_post_warmup": 0,
                "warning": "no decode steps recorded (GPU run may have failed)",
                "config": self.config,
            }
            self.summary_cache = out
            return out

        lat = np.array([s.total_latency_ms for s in ds])
        accepted = np.array([s.total_accepted_tokens for s in ds])
        drafted = np.array([s.total_draft_tokens for s in ds])
        num_drafts_per_step = np.array([s.num_drafts for s in ds])
        batch_sizes = np.array([s.batch_size for s in ds])
        kv = np.array(
            [s.kv_cache_usage for s in ds if s.kv_cache_usage is not None]
        )

        # Include bonus token per request with a draft this step.
        effective_tokens_per_step = accepted + num_drafts_per_step
        # For baseline (no SD) there are no drafts; new tokens == batch_size.
        # Fall back to per_request_new_tokens sum in that case.
        if (num_drafts_per_step == 0).all():
            effective_tokens_per_step = np.array(
                [sum(s.per_request_new_tokens) for s in ds]
            )

        total_lat_ms = float(lat.sum())
        total_eff = int(effective_tokens_per_step.sum())
        throughput = (
            total_eff / (total_lat_ms / 1000.0) if total_lat_ms > 0 else 0.0
        )
        eff_per_token_ms = (
            total_lat_ms / total_eff if total_eff > 0 else float("nan")
        )

        # Mean acceptance length per draft (bonus-inclusive, matches vLLM's
        # convention in SpecDecodingLogging.log).
        total_drafts = int(num_drafts_per_step.sum())
        mean_acc_len = (
            1.0 + float(accepted.sum()) / total_drafts if total_drafts else 1.0
        )
        accept_rate = (
            float(accepted.sum()) / float(drafted.sum())
            if drafted.sum() > 0
            else float("nan")
        )

        # Per-position acceptance (element-wise sum then normalise by drafts).
        per_pos = None
        lengths = {len(s.num_accepted_tokens_per_pos) for s in ds}
        if lengths == {0}:
            per_pos = None
        elif len(lengths) == 1 and total_drafts > 0:
            mat = np.array([s.num_accepted_tokens_per_pos for s in ds])
            per_pos = (mat.sum(axis=0) / total_drafts).tolist()

        # "Verify work" proxy: for each step, batch_size * (T + 1) where T
        # is the tree size that step used.  T can differ per step in
        # principle (different branches preempted) — we use the observed
        # num_draft_tokens / batch_size as the effective T.
        verify_work_per_step = []
        for s in ds:
            effective_T = (
                s.total_draft_tokens / max(1, s.batch_size)
                if s.total_draft_tokens
                else 0
            )
            verify_work_per_step.append(s.batch_size * (effective_T + 1))
        verify_work_arr = np.array(verify_work_per_step)

        summary = {
            # Timing.
            "avg_step_latency_ms": float(lat.mean()),
            "p50_step_latency_ms": float(np.percentile(lat, 50)),
            "p99_step_latency_ms": float(np.percentile(lat, 99)),
            "total_decode_latency_ms": total_lat_ms,
            # Acceptance.
            "avg_accepted_per_step": float(accepted.mean()),
            "avg_accepted_per_req_per_step": float(
                (accepted / np.maximum(num_drafts_per_step, 1)).mean()
            ),
            "mean_acceptance_length": mean_acc_len,
            "draft_acceptance_rate": accept_rate,
            "per_position_acceptance": per_pos,
            # Throughput / effective latency.
            "throughput_tokens_per_sec": throughput,
            "effective_per_token_latency_ms": eff_per_token_ms,
            # Memory / compute proxies (B vs T trade-off).
            "avg_batch_size": float(batch_sizes.mean()),
            "avg_kv_cache_usage": float(kv.mean()) if kv.size else None,
            "max_kv_cache_usage": float(kv.max()) if kv.size else None,
            "avg_verify_work": float(verify_work_arr.mean()),
            "total_verify_work": float(verify_work_arr.sum()),
            "work_bucket_hint": self._closest_work_bucket(
                float(verify_work_arr.mean())
            ),
            # Counts.
            "num_decode_steps_post_warmup": len(ds),
            "num_drafts_total": total_drafts,
            "num_draft_tokens_total": int(drafted.sum()),
            "num_accepted_tokens_total": int(accepted.sum()),
            # Config echo.
            "config": self.config,
            "warmup_decode_steps": self.warmup_decode_steps,
            "seed": self.seed,
            "wall_time_seconds": (
                (self.wall_time_end - self.wall_time_start)
                if self.wall_time_end
                else None
            ),
        }
        self.summary_cache = summary
        return summary

    @staticmethod
    def _closest_work_bucket(w: float) -> int:
        """Bin `B*(T+1)` to a coarse iso-cost bucket for analysis grouping."""
        buckets = [8, 16, 32, 64, 128, 256, 512, 1024, 2048]
        best = buckets[0]
        for b in buckets:
            if abs(w - b) < abs(w - best):
                best = b
        return best

    # -----------------------------------------------------------------
    # Serialisation
    # -----------------------------------------------------------------
    def save(self, detail_path: str | Path, summary_path: str | Path) -> None:
        detail_path = Path(detail_path)
        summary_path = Path(summary_path)
        detail_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.parent.mkdir(parents=True, exist_ok=True)

        detail = {
            "summary": self.summary(),
            "steps": [asdict(s) for s in self.steps],
        }
        detail_path.write_text(json.dumps(detail, indent=2, default=str))
        summary_path.write_text(
            json.dumps(self.summary(), indent=2, default=str)
        )


# ---------------------------------------------------------------------------
# Collector
# ---------------------------------------------------------------------------
class MetricCollector:
    """Thin façade over `RunMetric` for step-by-step ingestion.

    The engine runner pushes one `record_step(...)` call per
    `engine.step()`; this class is stateless besides the accumulating
    `RunMetric` and a preemption counter (since `SchedulerStats` reports
    cumulative preemption-like events rather than per-step deltas).
    """

    def __init__(
        self,
        config: dict[str, Any],
        *,
        warmup_decode_steps: int = 3,
        seed: int = 0,
    ) -> None:
        self.run = RunMetric(
            config=dict(config),
            warmup_decode_steps=warmup_decode_steps,
            seed=seed,
            wall_time_start=time.perf_counter(),
        )
        self._last_len_by_req: dict[str, int] = {}
        self._last_preempt_cum: int = 0

    # -----------------------------------------------------------------
    # Ingestion
    # -----------------------------------------------------------------
    def per_request_new_tokens(
        self, request_outputs: list[Any]
    ) -> tuple[list[str], list[int], list[int]]:
        """Returns (ids, new_tokens, cumulative_context_lens)."""
        ids: list[str] = []
        new_tokens: list[int] = []
        ctx_lens: list[int] = []
        for ro in request_outputs:
            cur_len = len(ro.outputs[0].token_ids) if ro.outputs else 0
            delta = cur_len - self._last_len_by_req.get(ro.request_id, 0)
            self._last_len_by_req[ro.request_id] = cur_len
            ids.append(ro.request_id)
            new_tokens.append(delta)
            prompt_len = (
                len(ro.prompt_token_ids)
                if getattr(ro, "prompt_token_ids", None)
                else 0
            )
            ctx_lens.append(prompt_len + cur_len)
        return ids, new_tokens, ctx_lens

    def record_step(
        self,
        *,
        step_idx: int,
        total_latency_ms: float,
        request_outputs: list[Any],
        scheduler_stats: Any | None,
        iteration_stats: Any | None,
        sd_config: dict[str, Any],
        draft_latency_ms: float | None = None,
        verify_latency_ms: float | None = None,
    ) -> StepMetric:
        """Build and append a StepMetric from one engine.step() result."""
        ids, new_toks, ctx_lens = self.per_request_new_tokens(request_outputs)
        batch_size = len([x for x in new_toks if x >= 0])  # all running reqs

        # --- SD aggregate counters -------------------------------------
        total_draft = 0
        total_accepted = 0
        num_drafts = 0
        per_pos: list[int] = []
        kv_usage = None
        preempt_cum = self._last_preempt_cum
        running_reqs = batch_size

        if scheduler_stats is not None:
            kv_usage = getattr(scheduler_stats, "kv_cache_usage", None)
            running_reqs = getattr(
                scheduler_stats, "num_running_reqs", batch_size
            )
            # Count KV-cache eviction / preemption events emitted this step.
            evs = getattr(scheduler_stats, "kv_cache_eviction_events", []) or []
            preempt_cum = self._last_preempt_cum + len(evs)
            self._last_preempt_cum = preempt_cum

            sds = getattr(scheduler_stats, "spec_decoding_stats", None)
            if sds is not None:
                num_drafts = int(getattr(sds, "num_drafts", 0))
                total_draft = int(getattr(sds, "num_draft_tokens", 0))
                total_accepted = int(getattr(sds, "num_accepted_tokens", 0))
                per_pos = list(
                    getattr(sds, "num_accepted_tokens_per_pos", []) or []
                )

        # --- Prefill vs decode classification --------------------------
        is_decode = True
        if iteration_stats is not None:
            prompt_total = getattr(
                getattr(iteration_stats, "prompt_token_stats", None),
                "total",
                0,
            )
            is_decode = prompt_total == 0

        # --- Per-request accepted token derivation ---------------------
        # vLLM exposes only aggregate accepted tokens per step.  Distribute
        # them across requests proportionally to new_tokens.  For the
        # no-SD baseline this simply puts zero in the accepted column.
        total_new = sum(new_toks) or 1
        per_req_accepted: list[int] = []
        if total_accepted > 0:
            remaining = total_accepted
            for i, nt in enumerate(new_toks):
                if i == len(new_toks) - 1:
                    per_req_accepted.append(remaining)
                else:
                    share = int(round(total_accepted * nt / total_new))
                    share = max(0, min(share, remaining))
                    per_req_accepted.append(share)
                    remaining -= share
        else:
            per_req_accepted = [0] * len(new_toks)

        # Resolve effective batch_size: prefer scheduler view.
        batch = running_reqs if running_reqs else batch_size

        sm = StepMetric(
            step_idx=step_idx,
            batch_size=int(batch),
            total_latency_ms=float(total_latency_ms),
            draft_latency_ms=draft_latency_ms,
            verify_latency_ms=verify_latency_ms,
            total_draft_tokens=total_draft,
            total_accepted_tokens=total_accepted,
            num_drafts=num_drafts,
            num_accepted_tokens_per_pos=per_pos,
            per_request_accepted=per_req_accepted,
            per_request_new_tokens=new_toks,
            per_request_context_len=ctx_lens,
            per_request_ids=ids,
            is_decode_step=is_decode,
            num_speculative_tokens=sd_config.get("num_speculative_tokens"),
            eagle_topk=sd_config.get("eagle_topk"),
            num_draft_tokens=sd_config.get("num_draft_tokens"),
            kv_cache_usage=kv_usage,
            num_preempted_cumulative=preempt_cum,
        )
        self.run.steps.append(sm)
        return sm

    # -----------------------------------------------------------------
    # Finalization
    # -----------------------------------------------------------------
    def finalize(self) -> RunMetric:
        self.run.wall_time_end = time.perf_counter()
        _ = self.run.summary()  # warm cache
        return self.run
