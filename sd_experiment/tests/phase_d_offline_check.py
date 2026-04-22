"""Offline validator for Phase D.

Synthesizes a minimal grid of fake `*_summary.json` files (4 SD configs
+ 2 baselines) and runs `scripts/analyze_results.py` against them,
verifying that:

  * A DataFrame is built with the expected column set.
  * Pivot tables are written (shape matches batch x tree size).
  * All PNG plots + CSVs exist and are non-empty.
  * Sanity of numerical values derived from synthetic inputs.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def make_summary(
    *, batch_size, D, K, T, avg_lat, tp, accept_len, kv, verify_work,
    sd_on=True, ctx=2048,
) -> dict:
    return {
        "avg_step_latency_ms": avg_lat,
        "p50_step_latency_ms": avg_lat,
        "p99_step_latency_ms": avg_lat * 1.4,
        "total_decode_latency_ms": avg_lat * 10,
        "avg_accepted_per_step": 0 if not sd_on else batch_size * accept_len * 0.6,
        "avg_accepted_per_req_per_step": 0 if not sd_on else accept_len * 0.6,
        "mean_acceptance_length": accept_len if sd_on else 1.0,
        "draft_acceptance_rate": 0.6 if sd_on else float("nan"),
        "per_position_acceptance": [0.7, 0.5, 0.3] if sd_on else None,
        "throughput_tokens_per_sec": tp,
        "effective_per_token_latency_ms": 1000.0 / tp,
        "avg_batch_size": float(batch_size),
        "avg_kv_cache_usage": kv,
        "max_kv_cache_usage": min(1.0, kv + 0.05),
        "avg_verify_work": verify_work,
        "total_verify_work": verify_work * 10,
        "work_bucket_hint": int(verify_work),
        "num_decode_steps_post_warmup": 10,
        "num_drafts_total": 10 * batch_size if sd_on else 0,
        "num_draft_tokens_total": 10 * batch_size * (T or 0),
        "num_accepted_tokens_total": int(10 * batch_size * (T or 0) * 0.6),
        "config": {
            "model": "mock/model",
            "draft_model": "mock/drafter" if sd_on else None,
            "method": "eagle3" if sd_on else None,
            "num_speculative_tokens": D,
            "eagle_topk": K,
            "num_draft_tokens": T,
            "batch_size": batch_size,
            "num_samples": batch_size,
            "max_tokens": 32,
            "max_model_len": ctx,
            "tensor_parallel_size": 1,
            "gpu_memory_utilization": 0.85,
            "temperature": 0.0,
            "seed": 0,
            "warmup_decode_steps": 3,
            "max_steps": 0,
            "sd_disabled": not sd_on,
            "prompts_file": None,
            "prompt_token_lens": [ctx // 2] * batch_size,
            "instance_ids": [f"mock-{i}" for i in range(batch_size)],
        },
        "warmup_decode_steps": 3,
        "seed": 0,
        "wall_time_seconds": 1.23,
    }


def main() -> int:
    here = Path(__file__).resolve().parent
    root = here.parent
    script = root / "scripts" / "analyze_results.py"
    assert script.exists(), f"missing {script}"

    with tempfile.TemporaryDirectory() as td:
        td_p = Path(td)
        ph1 = td_p / "phase1"
        base = td_p / "baseline"
        ph1.mkdir()
        base.mkdir()

        # 2 B (1,4) x 2 T (6,14)  SD configs.
        grid = [
            (1, 2, 2, 6, 20.0, 80.0),
            (1, 3, 2, 14, 24.0, 100.0),
            (4, 2, 2, 6, 30.0, 200.0),
            (4, 3, 2, 14, 40.0, 260.0),
        ]
        for B, D, K, T, lat, tp in grid:
            s = make_summary(
                batch_size=B, D=D, K=K, T=T,
                avg_lat=lat, tp=tp,
                accept_len=1.0 + 0.5 * (T / 10),
                kv=0.12 + 0.08 * B,
                verify_work=B * (T + 1),
            )
            stem = f"run_sd_b{B}_d{D}_k{K}_t{T}_ctx2048_mt32"
            (ph1 / f"{stem}_summary.json").write_text(json.dumps(s))

        # 2 baselines (no-SD) at B=1, 4.
        for B, lat, tp in [(1, 18.0, 60.0), (4, 26.0, 170.0)]:
            s = make_summary(
                batch_size=B, D=None, K=None, T=None,
                avg_lat=lat, tp=tp, accept_len=1.0,
                kv=0.10 + 0.08 * B, verify_work=B * 1,
                sd_on=False,
            )
            stem = f"run_nosd_b{B}_ctx2048_mt32"
            (base / f"{stem}_summary.json").write_text(json.dumps(s))

        # Run the analyzer.
        cmd = [
            sys.executable, str(script),
            "--results-dir", str(td_p),
        ]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            print("STDOUT:", r.stdout)
            print("STDERR:", r.stderr)
            raise SystemExit(
                f"analyze_results.py exited {r.returncode}"
            )

        out = td_p / "analysis"

        # Expected outputs.
        expected_files = [
            "summary.csv",
            "pivot_step_latency.csv",
            "pivot_throughput.csv",
            "pivot_accepted_per_req.csv",
            "heatmap_step_latency.png",
            "heatmap_throughput.png",
            "heatmap_accepted_per_req.png",
            "pareto_throughput_vs_ptl.png",
            "pareto_kv_vs_throughput.png",
            "pareto_verifywork_vs_throughput.png",
            "acceptance_vs_tree.png",
            "describe.txt",
        ]
        # Pivots are small by design (2x2 header + data ≈ 40 bytes); don't
        # use a uniform size threshold — check each file's minimum
        # individually.
        min_bytes = {
            # CSVs: header + data rows.
            "summary.csv": 500,
            "pivot_step_latency.csv": 25,
            "pivot_throughput.csv": 25,
            "pivot_accepted_per_req.csv": 25,
            "describe.txt": 200,
        }
        for fn in expected_files:
            p = out / fn
            assert p.exists(), f"missing {p}"
            threshold = min_bytes.get(fn, 500)  # PNGs comfortably exceed 500.
            sz = p.stat().st_size
            assert sz >= threshold, f"suspiciously small {p}: {sz} bytes"
            print(f"[ok] {fn} ({sz} bytes)")

        # DataFrame structural checks.
        df = pd.read_csv(out / "summary.csv")
        assert len(df) == 6, f"expected 6 rows (4 SD + 2 baseline), got {len(df)}"
        assert "avg_verify_work" in df.columns
        assert "work_bucket" in df.columns
        assert df["sd_enabled"].sum() == 4, df["sd_enabled"].value_counts()
        # Pivots must be 2x2.
        pv = pd.read_csv(out / "pivot_throughput.csv", index_col=0)
        assert pv.shape == (2, 2), f"throughput pivot {pv.shape}"
        # Our largest throughput config (B=4, T=14) -> 260.0.
        # CSV loader may parse the '14' column header as int or str.
        col14 = next(c for c in pv.columns if int(float(c)) == 14)
        assert float(pv.loc[4, col14]) == 260.0, pv
        print("[ok] DataFrame + pivot structural checks")

        # describe.txt should mention key columns.
        desc = (out / "describe.txt").read_text()
        for col in [
            "batch_size",
            "throughput_tokens_per_sec",
            "avg_kv_cache_usage",
        ]:
            assert col in desc, f"describe.txt missing {col}"
        print("[ok] describe.txt contents")

    print("\nAll Phase D offline checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
