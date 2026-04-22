#!/usr/bin/env bash
#
# Grid search driver for the SD trade-off harness.
#
# Usage:
#     bash scripts/run_grid.sh                       # full grid
#     MINIMAL=1 bash scripts/run_grid.sh             # 4 configs (2 B x 2 T)
#
# Env overrides (all optional):
#     OUTPUT_ROOT   default: results/$(date +%Y%m%d_%H%M%S)
#     MODEL         default: meta-llama/Llama-3.1-8B-Instruct
#     DRAFT_MODEL   default: yuhuili/EAGLE3-LLaMA3.1-Instruct-8B
#     PROMPTS_DIR   default: data/swebench_bucketed   (bucket_{ctx}.jsonl)
#     MAX_TOKENS    default: 128
#     MAX_STEPS     default: 200
#     GPU_MEM_UTIL  default: 0.85
#     MAX_MODEL_LEN default: 65536
#     NUM_SAMPLES   default: 32
#
# Failures do not abort the sweep — each run is wrapped in
# "|| echo '(failed)'" per spec.  Invalid tree configs surface as
# non-zero exit from the runner and are recorded in failures.log.

set -u

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
cd "$ROOT"

OUTPUT_ROOT="${OUTPUT_ROOT:-results/$(date +%Y%m%d_%H%M%S)}"
MODEL="${MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
DRAFT_MODEL="${DRAFT_MODEL:-yuhuili/EAGLE3-LLaMA3.1-Instruct-8B}"
PROMPTS_DIR="${PROMPTS_DIR:-data/swebench_bucketed}"
MAX_TOKENS="${MAX_TOKENS:-128}"
MAX_STEPS="${MAX_STEPS:-200}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.85}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-65536}"
NUM_SAMPLES="${NUM_SAMPLES:-32}"
TP="${TP:-1}"

mkdir -p "$OUTPUT_ROOT"
FAIL_LOG="$OUTPUT_ROOT/failures.log"
: > "$FAIL_LOG"

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$OUTPUT_ROOT/run.log"; }

run_one() {
    local phase="$1"; shift
    local label="$1"; shift
    local out_dir="$OUTPUT_ROOT/$phase"
    mkdir -p "$out_dir"
    log "[$phase/$label] begin"
    if ! python -m runner.engine_runner "$@" --output-dir "$out_dir" \
        >>"$out_dir/$label.stdout.log" 2>>"$out_dir/$label.stderr.log"
    then
        log "[$phase/$label] (failed, see $out_dir/$label.stderr.log)"
        echo "$phase/$label" >> "$FAIL_LOG"
    else
        log "[$phase/$label] ok"
    fi
}

# ---------------------------------------------------------------------------
# Grid definitions (kept in the script for transparency).
# Edit configs/experiment_grid.yaml AND this file if you change specs.
# ---------------------------------------------------------------------------
if [[ "${MINIMAL:-0}" == "1" ]]; then
    PH1_BATCHES=(1 4)
    # (D, K) pairs
    PH1_TREES=("2:2" "3:2")   # T=6, T=14
    PH1_CTX=2048
    DO_PHASE2=0
    DO_PHASE3=0
else
    PH1_BATCHES=(1 4 8 16 32)
    PH1_TREES=("1:4" "2:2" "3:2" "1:16" "4:2" "2:4" "3:1" "5:1")
    PH1_CTX=8192
    DO_PHASE2=1
    DO_PHASE3=1
fi

PH2_BATCH=8
PH2_TREES=("3:2" "4:2")
PH2_CTX=(2048 8192 32768)

PH3_BATCHES=(1 4 8 16 32)
PH3_CTX=8192

resolve_prompts_file() {
    local ctx="$1"
    echo "$PROMPTS_DIR/bucket_${ctx}.jsonl"
}

# ---------------------------------------------------------------------------
# Phase 1: batch x tree
# ---------------------------------------------------------------------------
log "=== phase1: batch x tree (ctx=$PH1_CTX) ==="
for B in "${PH1_BATCHES[@]}"; do
    for t in "${PH1_TREES[@]}"; do
        D="${t%%:*}"
        K="${t##*:}"
        PROMPTS="$(resolve_prompts_file "$PH1_CTX")"
        run_one "phase1" "b${B}_d${D}_k${K}_ctx${PH1_CTX}" \
            --model "$MODEL" \
            --draft-model "$DRAFT_MODEL" \
            --prompts-file "$PROMPTS" \
            --num-samples "$NUM_SAMPLES" \
            --batch-size "$B" \
            --num-speculative-tokens "$D" \
            --eagle-topk "$K" \
            --max-tokens "$MAX_TOKENS" \
            --max-model-len "$MAX_MODEL_LEN" \
            --max-steps "$MAX_STEPS" \
            --tensor-parallel-size "$TP" \
            --gpu-memory-utilization "$GPU_MEM_UTIL" \
            --seed 0
    done
done

# ---------------------------------------------------------------------------
# Phase 2: context-length sensitivity
# ---------------------------------------------------------------------------
if [[ "$DO_PHASE2" == "1" ]]; then
    log "=== phase2: ctx sensitivity (B=$PH2_BATCH) ==="
    for ctx in "${PH2_CTX[@]}"; do
        for t in "${PH2_TREES[@]}"; do
            D="${t%%:*}"
            K="${t##*:}"
            PROMPTS="$(resolve_prompts_file "$ctx")"
            run_one "phase2" "b${PH2_BATCH}_d${D}_k${K}_ctx${ctx}" \
                --model "$MODEL" \
                --draft-model "$DRAFT_MODEL" \
                --prompts-file "$PROMPTS" \
                --num-samples "$NUM_SAMPLES" \
                --batch-size "$PH2_BATCH" \
                --num-speculative-tokens "$D" \
                --eagle-topk "$K" \
                --max-tokens "$MAX_TOKENS" \
                --max-model-len "$MAX_MODEL_LEN" \
                --max-steps "$MAX_STEPS" \
                --tensor-parallel-size "$TP" \
                --gpu-memory-utilization "$GPU_MEM_UTIL" \
                --seed 0
        done
    done
fi

# ---------------------------------------------------------------------------
# Phase 3: no-SD baseline
# ---------------------------------------------------------------------------
if [[ "$DO_PHASE3" == "1" ]]; then
    log "=== phase3: no-SD baseline (ctx=$PH3_CTX) ==="
    for B in "${PH3_BATCHES[@]}"; do
        PROMPTS="$(resolve_prompts_file "$PH3_CTX")"
        run_one "baseline" "b${B}_nosd_ctx${PH3_CTX}" \
            --model "$MODEL" \
            --no-sd \
            --prompts-file "$PROMPTS" \
            --num-samples "$NUM_SAMPLES" \
            --batch-size "$B" \
            --max-tokens "$MAX_TOKENS" \
            --max-model-len "$MAX_MODEL_LEN" \
            --max-steps "$MAX_STEPS" \
            --tensor-parallel-size "$TP" \
            --gpu-memory-utilization "$GPU_MEM_UTIL" \
            --seed 0
    done
fi

log "grid done.  output: $OUTPUT_ROOT"
log "failures: $(wc -l < "$FAIL_LOG") (see $FAIL_LOG)"
