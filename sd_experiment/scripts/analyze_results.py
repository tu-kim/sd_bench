"""Analysis of SD trade-off grid results.

Consumes `*_summary.json` written by `runner.engine_runner` under a
results directory tree (typically `results/{timestamp}/{phase}/...`),
builds a single DataFrame, and emits:

  * `analysis/summary.csv`                  — the full DataFrame.
  * `analysis/pivot_step_latency.csv`       — batch x tree size.
  * `analysis/pivot_throughput.csv`
  * `analysis/pivot_accepted_per_req.csv`
  * `analysis/heatmap_step_latency.png`
  * `analysis/heatmap_throughput.png`
  * `analysis/heatmap_accepted_per_req.png`
  * `analysis/pareto_throughput_vs_ptl.png` — per-token latency vs
    throughput, color = sd_enabled (main Pareto).
  * `analysis/pareto_kv_vs_throughput.png`  — KV usage vs throughput,
    color = tree size T, size = batch B (memory-frontier view).
  * `analysis/pareto_verifywork_vs_throughput.png` — iso-cost lines of
    B*(T+1); curves of constant verify-work.
  * `analysis/acceptance_vs_tree.png`       — one line per batch size.
  * `analysis/describe.txt`                 — pandas describe() for a
    core column set.

Usage:
    python scripts/analyze_results.py --results-dir results/20260422_120000
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")  # no display on headless GPU boxes.
import matplotlib.pyplot as plt
import seaborn as sns


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def _flatten(summary: dict[str, Any], path: Path) -> dict[str, Any]:
    """Lift the config echo into top-level columns for easy DataFrame use."""
    cfg = dict(summary.get("config") or {})
    out = dict(summary)
    out.pop("config", None)
    # Prefix the config keys to avoid collisions.
    for k, v in cfg.items():
        out[f"cfg_{k}"] = v
    out["_source_path"] = str(path)
    out["_run_name"] = path.stem.replace("_summary", "")
    # Convenience columns.
    out["sd_enabled"] = not cfg.get("sd_disabled", False)
    out["batch_size"] = cfg.get("batch_size")
    out["num_speculative_tokens"] = cfg.get("num_speculative_tokens")
    out["eagle_topk"] = cfg.get("eagle_topk")
    out["num_draft_tokens"] = cfg.get("num_draft_tokens")
    out["ctx"] = cfg.get("max_model_len")
    # Canonical tree-size column for pivots; NaN for baseline.
    T = cfg.get("num_draft_tokens")
    out["T"] = T if T is not None else np.nan
    return out


def load_summaries(results_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for p in sorted(results_dir.rglob("*_summary.json")):
        try:
            summary = json.loads(p.read_text())
        except Exception as e:
            print(f"[analyze] skipping {p}: {type(e).__name__}: {e}")
            continue
        rows.append(_flatten(summary, p))
    if not rows:
        raise SystemExit(
            f"No *_summary.json found under {results_dir}.  Did the grid run?"
        )
    df = pd.DataFrame(rows)
    # Work-bucket for iso-cost grouping (B*(T+1) binned to closest power-of-2).
    df["work_bucket"] = df.apply(
        lambda r: _closest_bucket(
            float(r.get("avg_verify_work") or 0.0)
        ),
        axis=1,
    )
    return df


def _closest_bucket(w: float) -> int:
    buckets = [8, 16, 32, 64, 128, 256, 512, 1024, 2048]
    best = buckets[0]
    for b in buckets:
        if abs(w - b) < abs(w - best):
            best = b
    return best


# ---------------------------------------------------------------------------
# Pivots + heatmaps
# ---------------------------------------------------------------------------
def _pivot(df: pd.DataFrame, value: str) -> pd.DataFrame:
    sd = df[df["sd_enabled"].astype(bool)]
    if sd.empty:
        return pd.DataFrame()
    return sd.pivot_table(
        index="batch_size",
        columns="num_draft_tokens",
        values=value,
        aggfunc="mean",
    )


def _heatmap(
    pivot: pd.DataFrame, title: str, out_png: Path, fmt: str = ".2f"
) -> None:
    if pivot.empty:
        print(f"[analyze] empty pivot for {title}; skipping heatmap.")
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(pivot, annot=True, fmt=fmt, cmap="viridis", ax=ax)
    ax.set_title(title)
    ax.set_xlabel("num_draft_tokens T")
    ax.set_ylabel("batch_size B")
    fig.tight_layout()
    fig.savefig(out_png, dpi=140)
    plt.close(fig)
    print(f"[analyze] wrote {out_png}")


# ---------------------------------------------------------------------------
# Pareto + line plots
# ---------------------------------------------------------------------------
def pareto_throughput_vs_ptl(df: pd.DataFrame, out_png: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    for sd, sub in df.groupby("sd_enabled"):
        ax.scatter(
            sub["effective_per_token_latency_ms"],
            sub["throughput_tokens_per_sec"],
            label=f"sd={'on' if sd else 'off'}",
            s=60,
            alpha=0.8,
        )
        for _, r in sub.iterrows():
            label = r.get("_run_name") or ""
            ax.annotate(
                label,
                (r["effective_per_token_latency_ms"], r["throughput_tokens_per_sec"]),
                fontsize=6, alpha=0.55, xytext=(3, 3), textcoords="offset points",
            )
    ax.set_xlabel("effective per-token latency (ms)")
    ax.set_ylabel("throughput (tokens/s)")
    ax.set_title("Pareto: throughput vs per-token latency (SD on/off)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_png, dpi=140)
    plt.close(fig)
    print(f"[analyze] wrote {out_png}")


def pareto_kv_vs_throughput(df: pd.DataFrame, out_png: Path) -> None:
    """Memory-frontier Pareto.  Color = T, size = B."""
    sd = df[df["sd_enabled"].astype(bool)].dropna(
        subset=["avg_kv_cache_usage", "throughput_tokens_per_sec"]
    )
    if sd.empty:
        print("[analyze] no KV-usage data; skipping kv_vs_throughput.")
        return
    fig, ax = plt.subplots(figsize=(7.5, 5))
    sc = ax.scatter(
        sd["avg_kv_cache_usage"],
        sd["throughput_tokens_per_sec"],
        c=sd["num_draft_tokens"].astype(float),
        s=15 + 5 * sd["batch_size"].astype(float),
        cmap="plasma",
        alpha=0.85,
        edgecolors="black", linewidths=0.3,
    )
    plt.colorbar(sc, ax=ax, label="tree size T")
    ax.set_xlabel("avg KV cache usage (fraction)")
    ax.set_ylabel("throughput (tokens/s)")
    ax.set_title("Memory-frontier: throughput vs KV utilisation\n(size ∝ batch B)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_png, dpi=140)
    plt.close(fig)
    print(f"[analyze] wrote {out_png}")


def pareto_verifywork_vs_throughput(df: pd.DataFrame, out_png: Path) -> None:
    """Iso-cost plot: B*(T+1) on x, throughput on y, with iso lines."""
    sd = df[df["sd_enabled"].astype(bool)].dropna(
        subset=["avg_verify_work", "throughput_tokens_per_sec"]
    )
    if sd.empty:
        print("[analyze] no verify-work data; skipping.")
        return
    fig, ax = plt.subplots(figsize=(7.5, 5))
    sc = ax.scatter(
        sd["avg_verify_work"],
        sd["throughput_tokens_per_sec"],
        c=sd["batch_size"].astype(float),
        s=15 + 3 * sd["num_draft_tokens"].astype(float),
        cmap="viridis",
        alpha=0.85,
        edgecolors="black", linewidths=0.3,
    )
    plt.colorbar(sc, ax=ax, label="batch_size B")
    for w in (32, 64, 128, 256, 512):
        ax.axvline(w, color="grey", linestyle=":", alpha=0.35)
        ax.text(w, ax.get_ylim()[1] * 0.98, f"W={w}", color="grey",
                fontsize=7, ha="right", va="top", rotation=90)
    ax.set_xlabel("verify_work = B * (T + 1)")
    ax.set_ylabel("throughput (tokens/s)")
    ax.set_title("Compute-frontier: throughput at each verify_work bin\n(size ∝ T)")
    ax.set_xscale("log")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_png, dpi=140)
    plt.close(fig)
    print(f"[analyze] wrote {out_png}")


def acceptance_vs_tree_size(df: pd.DataFrame, out_png: Path) -> None:
    sd = df[df["sd_enabled"].astype(bool)].dropna(
        subset=["num_draft_tokens", "mean_acceptance_length"]
    )
    if sd.empty:
        print("[analyze] no acceptance data; skipping.")
        return
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for b, sub in sd.sort_values("num_draft_tokens").groupby("batch_size"):
        ax.plot(
            sub["num_draft_tokens"],
            sub["mean_acceptance_length"],
            marker="o",
            label=f"B={b}",
        )
    ax.set_xlabel("tree size T (num_draft_tokens)")
    ax.set_ylabel("mean acceptance length (incl. bonus)")
    ax.set_title("Acceptance vs tree size, per batch")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_png, dpi=140)
    plt.close(fig)
    print(f"[analyze] wrote {out_png}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
_KEY_COLS = [
    "batch_size",
    "num_speculative_tokens",
    "eagle_topk",
    "num_draft_tokens",
    "ctx",
    "avg_step_latency_ms",
    "p50_step_latency_ms",
    "p99_step_latency_ms",
    "throughput_tokens_per_sec",
    "effective_per_token_latency_ms",
    "mean_acceptance_length",
    "draft_acceptance_rate",
    "avg_kv_cache_usage",
    "avg_verify_work",
    "work_bucket",
    "num_decode_steps_post_warmup",
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", required=True)
    ap.add_argument(
        "--out-subdir", default="analysis",
        help="subdirectory under --results-dir where outputs are written."
    )
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    out_dir = results_dir / args.out_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_summaries(results_dir)
    print(f"[analyze] loaded {len(df)} runs from {results_dir}")

    # Full dump.
    df.to_csv(out_dir / "summary.csv", index=False)

    # describe().
    present = [c for c in _KEY_COLS if c in df.columns]
    desc_txt = df[present].describe(include="all").to_string()
    (out_dir / "describe.txt").write_text(desc_txt + "\n")
    print("[analyze] describe():")
    print(desc_txt)

    # Pivots.
    pv_lat = _pivot(df, "avg_step_latency_ms")
    pv_tp = _pivot(df, "throughput_tokens_per_sec")
    pv_acc = _pivot(df, "avg_accepted_per_req_per_step")
    pv_lat.to_csv(out_dir / "pivot_step_latency.csv")
    pv_tp.to_csv(out_dir / "pivot_throughput.csv")
    pv_acc.to_csv(out_dir / "pivot_accepted_per_req.csv")

    # Heatmaps.
    _heatmap(pv_lat, "avg step latency (ms)", out_dir / "heatmap_step_latency.png")
    _heatmap(pv_tp, "throughput (tokens/s)", out_dir / "heatmap_throughput.png")
    _heatmap(
        pv_acc, "accepted tokens / request / step",
        out_dir / "heatmap_accepted_per_req.png",
    )

    # Pareto plots.
    pareto_throughput_vs_ptl(df, out_dir / "pareto_throughput_vs_ptl.png")
    pareto_kv_vs_throughput(df, out_dir / "pareto_kv_vs_throughput.png")
    pareto_verifywork_vs_throughput(
        df, out_dir / "pareto_verifywork_vs_throughput.png"
    )
    acceptance_vs_tree_size(df, out_dir / "acceptance_vs_tree.png")

    print(f"\n[analyze] all outputs in {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
