"""Offline validator for Phase C (no GPU required).

Checks that don't need the engine:
  1. `build_tree_string(D, K)` produces T=sum(K^d) nodes and BFS ordering.
  2. `validate_tree_params` round-trips (D, K, T) and rejects inconsistent
     triples; it also derives the missing leg when only two are given.
  3. `engine_runner.build_argparser()` accepts every CLI flag in the spec.
  4. `_filename_stem(args, tree)` produces the expected grid-friendly
     filenames for both SD and --no-sd runs.
  5. `_load_prompts(prompts_file=...)` round-trips a tiny JSONL.
  6. `_config_echo(...)` serialises cleanly to JSON.

Run:  python tests/phase_c_offline_check.py
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from runner.tree import build_tree_string, validate_tree_params, TreeSpec
from runner.engine_runner import (
    build_argparser,
    _filename_stem,
    _load_prompts,
    _config_echo,
)


# ---------------------------------------------------------------------------
# Tree construction
# ---------------------------------------------------------------------------
def check_build_tree_string() -> None:
    # Chain: D=3, K=1 -> T=3, paths = [(0,), (0,0), (0,0,0)].
    s, t = build_tree_string(3, 1)
    assert t == 3
    assert ast.literal_eval(s) == [(0,), (0, 0), (0, 0, 0)]

    # K=2, D=2 -> T=6; all length-1 then length-2 tuples with values in {0,1}.
    s, t = build_tree_string(2, 2)
    assert t == 6
    paths = ast.literal_eval(s)
    assert set(paths) == {(0,), (1,), (0, 0), (0, 1), (1, 0), (1, 1)}
    # BFS-order: shorter tuples first.
    assert [len(p) for p in paths] == sorted(len(p) for p in paths)

    # K=2, D=3 -> T = 2+4+8 = 14.
    _, t = build_tree_string(3, 2)
    assert t == 14

    # K=3, D=2 -> T = 3+9 = 12.
    _, t = build_tree_string(2, 3)
    assert t == 12

    print("[ok] build_tree_string")


def check_validate_tree_params() -> None:
    # All three consistent.
    ts = validate_tree_params(depth=3, branching=2, total=14)
    assert ts == TreeSpec(3, 2, 14)

    # Derive K from (D, T).
    ts = validate_tree_params(depth=3, branching=None, total=14)
    assert ts == TreeSpec(3, 2, 14)

    # Derive T from (D, K).
    ts = validate_tree_params(depth=3, branching=2, total=None)
    assert ts == TreeSpec(3, 2, 14)

    # Default: chain.
    ts = validate_tree_params(depth=5, branching=None, total=None)
    assert ts == TreeSpec(5, 1, 5)

    # Inconsistent triple -> error.
    try:
        validate_tree_params(depth=3, branching=2, total=12)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for inconsistent triple")

    # No integer K solves (D=3, T=10) -> error.
    try:
        validate_tree_params(depth=3, branching=None, total=10)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for non-integer K")

    print("[ok] validate_tree_params")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def check_cli_flags_present() -> None:
    ap = build_argparser()
    required = {
        "--model",
        "--draft-model",
        "--prompts-file",
        "--num-samples",
        "--batch-size",
        "--max-tokens",
        "--num-speculative-tokens",
        "--num-draft-tokens",
        "--eagle-topk",
        "--tensor-parallel-size",
        "--gpu-memory-utilization",
        "--max-model-len",
        "--no-sd",
        "--output-dir",
        "--temperature",
        "--max-steps",
    }
    seen = set()
    for action in ap._actions:
        for opt in action.option_strings:
            seen.add(opt)
    missing = required - seen
    assert not missing, f"missing CLI flags: {missing}"
    print(f"[ok] CLI has all {len(required)} required flags")


# ---------------------------------------------------------------------------
# Filename convention
# ---------------------------------------------------------------------------
def _mk(**kw):
    d = dict(
        model="M",
        draft_model="D",
        prompts_file=None,
        num_samples=4,
        batch_size=4,
        max_tokens=64,
        num_speculative_tokens=3,
        eagle_topk=2,
        num_draft_tokens=6,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.85,
        max_model_len=8192,
        no_sd=False,
        output_dir="x",
        temperature=0.0,
        max_steps=0,
        warmup_decode_steps=3,
        seed=0,
    )
    d.update(kw)
    return argparse.Namespace(**d)


def check_filename_stem() -> None:
    a = _mk()
    t = TreeSpec(3, 2, 6)
    stem = _filename_stem(a, t)
    assert stem == "run_sd_b4_d3_k2_t6_ctx8192_mt64", stem

    a = _mk(no_sd=True)
    stem = _filename_stem(a, None)
    assert stem == "run_nosd_b4_ctx8192_mt64", stem
    print("[ok] filename stems")


# ---------------------------------------------------------------------------
# Prompt loading
# ---------------------------------------------------------------------------
def check_load_prompts_jsonl() -> None:
    with tempfile.TemporaryDirectory() as td:
        pf = Path(td) / "bucket.jsonl"
        with pf.open("w") as f:
            for i in range(3):
                f.write(json.dumps(
                    {"instance_id": f"inst-{i}", "prompt": f"hello {i}",
                     "prompt_token_len": 10 + i}
                ) + "\n")
        rows = _load_prompts(str(pf), num_samples=2, tokenizer_for_mock=None)
        assert len(rows) == 2
        assert rows[0]["instance_id"] == "inst-0"
        assert rows[0]["prompt"] == "hello 0"
    print("[ok] _load_prompts jsonl round-trip")


def check_load_prompts_fallback_mock() -> None:
    rows = _load_prompts(
        prompts_file=None, num_samples=2, tokenizer_for_mock=None
    )
    assert len(rows) == 2
    # Each row must have a prompt produced by build_agent_prompt.
    assert "<|begin_of_text|>" in rows[0]["prompt"]
    print("[ok] _load_prompts mock fallback")


# ---------------------------------------------------------------------------
# Config echo JSON-serialisable
# ---------------------------------------------------------------------------
def check_config_echo_json() -> None:
    a = _mk()
    tree = TreeSpec(3, 2, 6)
    prompts = [
        {"instance_id": "x", "prompt": "p", "prompt_token_len": 5},
        {"instance_id": "y", "prompt": "q", "prompt_token_len": 9},
    ]
    cfg = _config_echo(a, tree, prompts)
    s = json.dumps(cfg)  # must not raise
    back = json.loads(s)
    assert back["num_speculative_tokens"] == 3
    assert back["eagle_topk"] == 2
    assert back["num_draft_tokens"] == 6
    assert back["sd_disabled"] is False
    assert back["prompt_token_lens"] == [5, 9]
    print("[ok] _config_echo is JSON-serialisable")


def main() -> int:
    check_build_tree_string()
    check_validate_tree_params()
    check_cli_flags_present()
    check_filename_stem()
    check_load_prompts_jsonl()
    check_load_prompts_fallback_mock()
    check_config_echo_json()
    print("\nAll Phase C offline checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
