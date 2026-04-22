"""SWE-Bench Lite loader + context-length bucketing.

The loader works in three modes, in priority order:

  1. Live HF download via `datasets.load_dataset` (normal path on the
     target GPU box).
  2. Local on-disk cache under `--data-dir` (for air-gapped runs).
  3. Embedded `_MOCK_SAMPLES` list (strictly for offline development of
     the harness; also what the unit tests use).

All three paths yield a list of plain dicts so downstream code is
mode-agnostic.
"""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Embedded mock sample set.
# Shaped exactly like princeton-nlp/SWE-bench_Lite rows (relevant fields).
# Used only when datasets can't reach HF and no local cache is provided.
# ---------------------------------------------------------------------------
_MOCK_SAMPLES: list[dict[str, Any]] = [
    {
        "instance_id": "mock__pkg-issue-001",
        "repo": "example-org/example-pkg",
        "base_commit": "0" * 40,
        "problem_statement": (
            "Calling CoreObject.process() twice with the same input "
            "returns different values after a cache eviction.  Expected: "
            "cache hits should always return the stored value.  Actual: "
            "second call recomputes and sometimes returns a stale result."
        ),
        "hints_text": (
            "Look at CoreObject.__init__ and the eviction logic in "
            "core_03.py.  There's a race between _cache mutation and "
            "the eviction timer."
        ),
    },
    {
        "instance_id": "mock__pkg-issue-002",
        "repo": "example-org/example-pkg",
        "base_commit": "1" * 40,
        "problem_statement": (
            "Handler registration ignores the priority kwarg when "
            "register_handler is called inside a context manager."
        ),
        "hints_text": "",
    },
    {
        "instance_id": "mock__pkg-issue-003",
        "repo": "example-org/other-pkg",
        "base_commit": "2" * 40,
        "problem_statement": (
            "After upgrading to v2.0 the parser accepts malformed input "
            "that should raise ValueError.  Reproducer attached."
        ),
        "hints_text": "Probably a regression in tokenizer.py::_lex().",
    },
    {
        "instance_id": "mock__pkg-issue-004",
        "repo": "example-org/other-pkg",
        "base_commit": "3" * 40,
        "problem_statement": (
            "Segfault in the C extension when passed an empty list.  "
            "Likely missing null check in src/ext/_compiled.c."
        ),
        "hints_text": "",
    },
]


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load_swebench(
    split: str = "test",
    dataset_name: str = "princeton-nlp/SWE-bench_Lite",
    *,
    data_dir: str | None = None,
    max_samples: int | None = None,
    allow_mock: bool = True,
) -> list[dict[str, Any]]:
    """Load SWE-Bench Lite samples as a list of dicts.

    On the target GPU box, pass nothing and rely on HF.  In air-gapped
    mode, set `data_dir` to a directory containing either a
    `swebench_lite.json` file (our own dump format) or a HuggingFace
    datasets cache.

    Parameters
    ----------
    split:
        Dataset split.  SWE-Bench Lite ships `test` and `dev`.
    dataset_name:
        HF dataset id.
    data_dir:
        Optional local cache.
    max_samples:
        Truncate to the first N samples after loading.
    allow_mock:
        If HF is unreachable *and* no local cache is found, return the
        embedded mock list.  Set to False to fail loudly instead.
    """
    # (1) Local JSON dump (cheapest, most reproducible).
    if data_dir:
        cached = Path(data_dir) / "swebench_lite.json"
        if cached.exists():
            data = json.loads(cached.read_text())
            rows = data[split] if isinstance(data, dict) else data
            return _truncate(rows, max_samples)

    # (2) HuggingFace datasets (normal GPU-host path).
    try:
        from datasets import load_dataset

        ds = load_dataset(dataset_name, split=split)
        rows = [dict(r) for r in ds]
        return _truncate(rows, max_samples)
    except Exception as e:
        if not allow_mock:
            raise
        # Fall through to mock.
        print(
            f"[swe_bench_loader] HF load failed ({type(e).__name__}: {e!s:.120}); "
            "falling back to embedded mock samples."
        )

    # (3) Mock fallback (offline dev only).
    return _truncate(_MOCK_SAMPLES, max_samples)


def _truncate(rows: list[dict[str, Any]], n: int | None) -> list[dict[str, Any]]:
    if n is None:
        return rows
    return rows[:n]


# ---------------------------------------------------------------------------
# Bucketing
# ---------------------------------------------------------------------------
def bucket_by_context_length(
    samples: list[dict[str, Any]],
    tokenizer,
    buckets: list[int],
    prompt_builder_fn: Callable[[dict[str, Any]], str],
) -> dict[int, list[dict[str, Any]]]:
    """Assign each sample to the smallest bucket where `len <= bucket_size`.

    Samples longer than the largest bucket are dropped and a warning is
    printed.

    The returned dict has one key per bucket (even if empty) and each
    sample is augmented with `prompt_token_len` for downstream use.
    """
    buckets = sorted(set(int(b) for b in buckets))
    out: dict[int, list[dict[str, Any]]] = {b: [] for b in buckets}
    dropped = 0
    for s in samples:
        prompt = prompt_builder_fn(s)
        ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        length = len(ids)
        s = {**s, "prompt_token_len": length, "prompt": prompt}
        placed = False
        for b in buckets:
            if length <= b:
                out[b].append(s)
                placed = True
                break
        if not placed:
            dropped += 1
    if dropped:
        print(
            f"[bucket_by_context_length] dropped {dropped} samples exceeding "
            f"largest bucket ({buckets[-1]} tokens)."
        )
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _cli() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Load SWE-Bench Lite, build agent prompts, bucket by context "
            "length, and dump each bucket to a JSONL file."
        ),
    )
    ap.add_argument(
        "--tokenizer",
        required=True,
        help="HF tokenizer id (e.g. meta-llama/Llama-3.1-8B-Instruct).",
    )
    ap.add_argument(
        "--buckets",
        type=int,
        nargs="+",
        default=[2048, 8192, 32768],
        help="Bucket sizes in tokens.",
    )
    ap.add_argument(
        "--split",
        default="test",
        help="SWE-Bench split (test or dev).",
    )
    ap.add_argument(
        "--dataset",
        default="princeton-nlp/SWE-bench_Lite",
    )
    ap.add_argument(
        "--data-dir",
        default=None,
        help="Local cache dir with swebench_lite.json (air-gapped runs).",
    )
    ap.add_argument(
        "--max-samples",
        type=int,
        default=None,
    )
    ap.add_argument(
        "--output-dir",
        required=True,
    )
    ap.add_argument(
        "--pad-to-bucket",
        action="store_true",
        help="If set, pad each sample's prompt with filler up to its bucket "
             "size (minus a small safety margin).  Useful for Phase 2.",
    )
    ap.add_argument(
        "--pad-margin-tokens",
        type=int,
        default=128,
    )
    ap.add_argument(
        "--n-fake-files",
        type=int,
        default=40,
    )
    args = ap.parse_args()

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    samples = load_swebench(
        split=args.split,
        dataset_name=args.dataset,
        data_dir=args.data_dir,
        max_samples=args.max_samples,
    )
    print(f"Loaded {len(samples)} samples from {args.dataset}:{args.split}.")

    # Local import to avoid a top-level cycle.
    from .prompt_builder import build_agent_prompt, pad_to_bucket as _pad

    def _build(s: dict[str, Any]) -> str:
        return build_agent_prompt(s, n_fake_files=args.n_fake_files)

    bucketed = bucket_by_context_length(samples, tokenizer, args.buckets, _build)

    # Optional padding pass (Phase 2).
    if args.pad_to_bucket:
        padded: dict[int, list[dict[str, Any]]] = {}
        for b, lst in bucketed.items():
            new_lst = []
            for s in lst:
                target = b - args.pad_margin_tokens
                if s["prompt_token_len"] >= target:
                    new_lst.append(s)
                    continue
                new_prompt, new_len = _pad(s, tokenizer, target)
                new_lst.append(
                    {**s, "prompt": new_prompt, "prompt_token_len": new_len}
                )
            padded[b] = new_lst
        bucketed = padded

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "tokenizer": args.tokenizer,
        "dataset": args.dataset,
        "split": args.split,
        "buckets": args.buckets,
        "pad_to_bucket": args.pad_to_bucket,
        "counts": {b: len(lst) for b, lst in bucketed.items()},
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    for b, lst in bucketed.items():
        path = out_dir / f"bucket_{b}.jsonl"
        with path.open("w") as f:
            for s in lst:
                f.write(json.dumps(s) + "\n")
        print(f"wrote {path} ({len(lst)} samples)")

    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
