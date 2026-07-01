# specops-dsto reproduction Makefile.
# All experiments are reproducible on commodity hardware; runtime and memory are
# reported for each benchmark.
PY ?= python
export KMP_DUPLICATE_LIB_OK = TRUE
export OMP_NUM_THREADS = 1

.PHONY: all demo full tables figures audit test reproduce clean

all: demo

demo:
	$(PY) scripts/run.py --config configs/smoke.yaml --out results
	$(MAKE) tables figures

full:
	$(PY) scripts/run.py --config configs/full.yaml --out results
	$(MAKE) tables figures

tables:
	$(PY) scripts/make_tables.py

figures:
	$(PY) scripts/make_figures.py

audit:
	$(PY) scripts/audit_claims.py

test:
	$(PY) -m pytest

# Full-scale reproduction + submission sync via the installed entry point.
reproduce:
	dsto-reproduce --config configs/full.yaml

clean:
	rm -rf build results/*.tex figures/*.pdf figures/*.png
