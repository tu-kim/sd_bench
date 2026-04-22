# SD trade-off harness — convenience targets.
#
# Default variables (override on the command line, e.g. `make grid MINIMAL=1`).
MODEL         ?= meta-llama/Llama-3.1-8B-Instruct
DRAFT_MODEL   ?= yuhuili/EAGLE3-LLaMA3.1-Instruct-8B
BUCKETS       ?= 2048 8192 32768
MAX_SAMPLES   ?= 64
BUCKET_DIR    ?= data/swebench_bucketed
RESULTS_DIR   ?= $(shell ls -dt results/2* 2>/dev/null | head -1)
MINIMAL       ?=

.PHONY: help install check bucket grid grid-minimal analyze clean

help:
	@echo "Targets:"
	@echo "  install        pip install -r requirements.txt (vllm==0.17.1)"
	@echo "  check          run every offline validator (no GPU)"
	@echo "  bucket         tokenize SWE-Bench Lite into bucketed JSONL"
	@echo "  grid           run the full grid (requires GPU)"
	@echo "  grid-minimal   run MINIMAL=1 grid (4 configs; requires GPU)"
	@echo "  analyze        run analyze_results.py on the newest results dir"
	@echo "                 (override via RESULTS_DIR=results/<ts>)"
	@echo "  clean          remove analysis outputs and empty results dirs"
	@echo ""
	@echo "Variables: MODEL, DRAFT_MODEL, BUCKETS, MAX_SAMPLES, BUCKET_DIR,"
	@echo "           RESULTS_DIR, MINIMAL"

install:
	pip install -r requirements.txt

check:
	bash scripts/check.sh

bucket:
	python -m workload.swe_bench_loader \
		--tokenizer $(MODEL) \
		--buckets $(BUCKETS) \
		--max-samples $(MAX_SAMPLES) \
		--output-dir $(BUCKET_DIR)

grid:
	bash scripts/run_grid.sh

grid-minimal:
	MINIMAL=1 bash scripts/run_grid.sh

analyze:
	@if [ -z "$(RESULTS_DIR)" ]; then \
		echo "No results dir found.  Pass RESULTS_DIR=results/<timestamp>"; \
		exit 1; \
	fi
	python scripts/analyze_results.py --results-dir $(RESULTS_DIR)
	@echo "Wrote analysis to $(RESULTS_DIR)/analysis/"

clean:
	find results -maxdepth 2 -name analysis -type d -exec rm -rf {} + 2>/dev/null || true
	find results -maxdepth 1 -type d -empty -delete 2>/dev/null || true
