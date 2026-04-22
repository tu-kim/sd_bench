"""Offline validator for Phase B (no GPU required).

Checks, in order:
  1. `build_agent_prompt` produces a Llama-3.1 chat-template prompt with
     the expected tokens in the expected order.
  2. `pad_to_bucket` lengthens a prompt monotonically.
  3. `bucket_by_context_length` assigns samples to the smallest bucket.
  4. `MetricCollector.record_step` accepts fake `SchedulerStats` /
     `IterationStats` surrogates and the resulting `RunMetric.summary()`
     computes throughput / acceptance / KV fields correctly.
  5. `RunMetric.save()` round-trips to disk.

Run:  python tests/phase_b_offline_check.py
Exits non-zero if any check fails.
"""

from __future__ import annotations

import json
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

# Make sd_experiment package importable when run from repo root or here.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from workload.prompt_builder import (
    BOS,
    EOT,
    SH,
    EH,
    build_agent_prompt,
    build_chat_prompt,
    pad_to_bucket,
)
from workload.swe_bench_loader import (
    load_swebench,
    bucket_by_context_length,
    _MOCK_SAMPLES,
)
from metrics.collector import MetricCollector, RunMetric


# ---------------------------------------------------------------------------
# Stubs for vLLM objects we'd normally get from the engine.
# ---------------------------------------------------------------------------
class _Comp:
    def __init__(self, token_ids):
        self.token_ids = token_ids


class _Ro:
    def __init__(self, rid, token_ids, prompt_token_ids):
        self.request_id = rid
        self.outputs = [_Comp(token_ids)]
        self.prompt_token_ids = prompt_token_ids
        self.finished = False


@dataclass
class _SpecStats:
    num_spec_tokens: int = 3
    num_drafts: int = 0
    num_draft_tokens: int = 0
    num_accepted_tokens: int = 0
    num_accepted_tokens_per_pos: list = field(default_factory=list)


@dataclass
class _PromptStats:
    total: int = 0


@dataclass
class _SchedulerStats:
    num_running_reqs: int = 0
    num_waiting_reqs: int = 0
    kv_cache_usage: float = 0.0
    spec_decoding_stats: _SpecStats | None = None
    kv_cache_eviction_events: list = field(default_factory=list)


@dataclass
class _IterStats:
    num_generation_tokens: int = 0
    prompt_token_stats: _PromptStats = field(default_factory=_PromptStats)


# ---------------------------------------------------------------------------
# Simple whitespace tokenizer used where a real tokenizer is unavailable.
# ---------------------------------------------------------------------------
class _WsTokenizer:
    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": text.split()}


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------
def check_prompt_structure() -> None:
    sample = _MOCK_SAMPLES[0]
    p = build_agent_prompt(sample)
    assert p.startswith(BOS), "missing <|begin_of_text|>"
    # Order of headers: system, user, assistant.
    i_sys = p.index(f"{SH}system{EH}")
    i_usr = p.index(f"{SH}user{EH}")
    i_ast = p.index(f"{SH}assistant{EH}")
    assert i_sys < i_usr < i_ast, "header order is wrong"
    # Exactly two EOTs (system, user); assistant turn is left open.
    assert p.count(EOT) == 2, f"expected 2 EOTs, got {p.count(EOT)}"
    # Instance id surfaces in the user message.
    assert sample["instance_id"] in p, "instance_id not present in prompt"
    # Reasoning block is prefilled.
    assert p.endswith("<think>\n"), "assistant turn should open with <think>"
    print("[ok] prompt structure")


def check_build_chat_prompt_minimal() -> None:
    p = build_chat_prompt(system="sys", user="usr")
    assert p == (
        f"{BOS}{SH}system{EH}\n\nsys{EOT}{SH}user{EH}\n\nusr{EOT}"
        f"{SH}assistant{EH}\n\n<think>\n"
    )
    print("[ok] build_chat_prompt literal")


def check_pad_monotone() -> None:
    tok = _WsTokenizer()
    sample = _MOCK_SAMPLES[0]
    base = len(tok(build_agent_prompt(sample))["input_ids"])
    text_short, len_short = pad_to_bucket(sample, tok, base + 50)
    text_long, len_long = pad_to_bucket(sample, tok, base + 500)
    assert len_short >= base + 50 or len_short >= base
    assert len_long >= len_short, "padding not monotone"
    print(f"[ok] pad_to_bucket monotone ({base} -> {len_short} -> {len_long})")


def check_bucketing() -> None:
    tok = _WsTokenizer()
    samples = _MOCK_SAMPLES

    def _builder(s):
        return build_agent_prompt(s)

    # Three buckets chosen so at least one sample lands in each.
    lens = [len(tok(_builder(s))["input_ids"]) for s in samples]
    buckets = sorted({min(lens), sum(lens) // len(lens), max(lens) + 10})
    out = bucket_by_context_length(samples, tok, buckets, _builder)
    total = sum(len(v) for v in out.values())
    assert total == len(samples), "bucket_by_context_length dropped samples"
    for b, lst in out.items():
        for s in lst:
            assert s["prompt_token_len"] <= b, f"misplaced: {s['prompt_token_len']} > {b}"
    print(f"[ok] bucketing ({dict((k, len(v)) for k, v in out.items())})")


def check_collector_math() -> None:
    """Fabricate a 6-step run and check summary arithmetic end-to-end.

    Steps:  prefill, prefill, decode(warm), decode(warm), decode(warm),
            decode, decode, decode, decode  (warmup_decode_steps=3)
    Decode counts: drafts=4/req => total_draft=8 per step (B=2),
                   accepted varies so acceptance length is meaningful.
    """
    config = {
        "model": "mock-model",
        "method": "eagle3",
        "num_speculative_tokens": 3,
        "eagle_topk": 1,
        "num_draft_tokens": 3,
    }
    coll = MetricCollector(config, warmup_decode_steps=3, seed=0)

    # Two requests.
    prompt_ids = [0] * 100
    tok_a = []
    tok_b = []

    def run_step(step_idx, new_a, new_b, accepted_total, is_prefill=False):
        tok_a.extend(range(len(tok_a), len(tok_a) + new_a))
        tok_b.extend(range(len(tok_b), len(tok_b) + new_b))
        ros = [
            _Ro("a", list(tok_a), prompt_ids),
            _Ro("b", list(tok_b), prompt_ids),
        ]
        sds = None
        if not is_prefill:
            sds = _SpecStats(
                num_spec_tokens=3,
                num_drafts=2,
                num_draft_tokens=2 * 3,  # B=2, D=3
                num_accepted_tokens=accepted_total,
                num_accepted_tokens_per_pos=[accepted_total, 0, 0],
            )
        sch = _SchedulerStats(
            num_running_reqs=2,
            kv_cache_usage=0.42,
            spec_decoding_stats=sds,
        )
        it = _IterStats(
            num_generation_tokens=new_a + new_b,
            prompt_token_stats=_PromptStats(total=100 if is_prefill else 0),
        )
        coll.record_step(
            step_idx=step_idx,
            total_latency_ms=10.0,
            request_outputs=ros,
            scheduler_stats=sch,
            iteration_stats=it,
            sd_config={
                "num_speculative_tokens": 3,
                "eagle_topk": 1,
                "num_draft_tokens": 3,
            },
        )

    # Two prefill steps (one per request) then decode steps.
    run_step(0, 1, 0, 0, is_prefill=True)
    run_step(1, 0, 1, 0, is_prefill=True)
    # 3 warmup decode steps.
    run_step(2, 2, 2, 2)
    run_step(3, 2, 2, 2)
    run_step(4, 2, 2, 2)
    # 4 measured decode steps: acceptance varies.
    run_step(5, 4, 4, 4)  # perfect
    run_step(6, 3, 3, 2)  # half
    run_step(7, 2, 2, 0)  # none
    run_step(8, 3, 3, 2)  # half

    run = coll.finalize()
    s = run.summary()

    # Decode-post-warmup = 4 steps.
    assert s["num_decode_steps_post_warmup"] == 4, s
    # Total accepted = 4+2+0+2 = 8.
    assert s["num_accepted_tokens_total"] == 8, s
    # Total draft = 4 * 6 = 24.  Rate = 8/24 = 1/3.
    assert s["num_draft_tokens_total"] == 24, s
    assert abs(s["draft_acceptance_rate"] - 8 / 24) < 1e-9
    # Drafts total = 4 * 2 = 8; mean acceptance length = 1 + 8/8 = 2.0.
    assert abs(s["mean_acceptance_length"] - 2.0) < 1e-9
    # Latency total = 4*10 = 40 ms.
    assert abs(s["total_decode_latency_ms"] - 40.0) < 1e-9
    # Effective tokens = accepted + drafts-per-step = 8 + 8 = 16.
    # Throughput = 16 / 0.04 = 400 tokens/s.
    assert abs(s["throughput_tokens_per_sec"] - 400.0) < 1e-6
    # Effective per-token latency = 40 / 16 = 2.5 ms.
    assert abs(s["effective_per_token_latency_ms"] - 2.5) < 1e-9
    # KV usage average = 0.42.
    assert abs(s["avg_kv_cache_usage"] - 0.42) < 1e-9
    # Verify work per step = B*(T+1) = 2*(3+1) = 8 each; avg = 8.
    assert abs(s["avg_verify_work"] - 8.0) < 1e-9
    # Per-position acceptance should be length 3, weighted sum / drafts.
    assert s["per_position_acceptance"] is not None
    assert len(s["per_position_acceptance"]) == 3

    # Save round-trip.
    with tempfile.TemporaryDirectory() as td:
        d = Path(td) / "detail.json"
        sm = Path(td) / "summary.json"
        run.save(d, sm)
        back = json.loads(sm.read_text())
        assert back["num_decode_steps_post_warmup"] == 4
        back_detail = json.loads(d.read_text())
        assert len(back_detail["steps"]) == 9
    print("[ok] collector math + save round-trip")


def check_swebench_mock() -> None:
    rows = load_swebench(max_samples=2, allow_mock=True)
    assert len(rows) == 2, rows
    assert {"instance_id", "repo", "problem_statement"}.issubset(rows[0]), rows[0]
    print(f"[ok] swebench mock load ({len(rows)} rows)")


def main() -> int:
    check_prompt_structure()
    check_build_chat_prompt_minimal()
    check_pad_monotone()
    check_bucketing()
    check_collector_math()
    check_swebench_mock()
    print("\nAll Phase B offline checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
