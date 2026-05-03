# Unified reproduction targets for the Five Axes of Fragility audit.
# Every target in this file is idempotent: running it twice reproduces
# the same CSV outputs bit-for-bit (modulo numpy/BLAS non-determinism).

PYTHON        ?= /Users/ihorkendiukhov/biodyn-work/.venv/bin/python
OUT           ?= outputs
DATA_ROOT     ?= /Users/ihorkendiukhov/biodyn-work/single_cell_mechinterp/data/raw
SEED          ?= 20260218

.PHONY: help install smoke test clean \
        axis1 axis2 axis3 axis4 axis5 \
        wp1 wp2 wp3 wp4 wp5 wp6 wp7 wp8 wp9 wp10 wp11 wp12 wp13 wp14 \
        audit

help:
	@echo "Targets:"
	@echo "  install   editable install of the fragility package"
	@echo "  smoke     run tiny synthetic smoke test (CI target)"
	@echo "  test      pytest"
	@echo "  axis1..5  run individual axis pipelines with default config"
	@echo "  wp1..14   run individual work packages from the revision plan"
	@echo "  audit     full reproduction (all WPs)"

install:
	$(PYTHON) -m pip install -e .

smoke:
	$(PYTHON) -m fragility.tests.smoke --out $(OUT)/smoke --seed $(SEED)

test:
	$(PYTHON) -m pytest tests/ -v

axis1:
	$(PYTHON) -m fragility axis1 --config configs/axis1_default.yaml --out $(OUT)/axis1

axis2:
	$(PYTHON) -m fragility axis2 --config configs/axis2_default.yaml --out $(OUT)/axis2

axis3:
	$(PYTHON) -m fragility axis3 --config configs/axis3_default.yaml --out $(OUT)/axis3

axis4:
	$(PYTHON) -m fragility axis4 --config configs/axis4_default.yaml --out $(OUT)/axis4

axis5:
	$(PYTHON) -m fragility axis5 --config configs/axis5_default.yaml --out $(OUT)/axis5

wp1:
	$(PYTHON) -m fragility wp1 --config configs/wp1_method_diversity.yaml --out $(OUT)/wp1

wp2:
	$(PYTHON) -m fragility wp2 --config configs/wp2_mvcc_multi.yaml --out $(OUT)/wp2

wp3:
	$(PYTHON) -m fragility wp3 --config configs/wp3_rss_redesign.yaml --out $(OUT)/wp3

wp4:
	$(PYTHON) -m fragility wp4 --config configs/wp4_reliability_grid.yaml --out $(OUT)/wp4

wp6:
	$(PYTHON) -m fragility wp6 --config configs/wp6_integration.yaml --out $(OUT)/wp6

wp7:
	$(PYTHON) -m fragility wp7 --config configs/wp7_compliant_cohort.yaml --out $(OUT)/wp7

wp8:
	$(PYTHON) -m fragility wp8 --config configs/wp8_external_datasets.yaml --out $(OUT)/wp8

wp10:
	$(PYTHON) -m fragility wp10 --config configs/wp10_topk_scan.yaml --out $(OUT)/wp10

wp12:
	$(PYTHON) -m fragility wp12 --config configs/wp12_normalization.yaml --out $(OUT)/wp12

wp13:
	$(PYTHON) -m fragility wp13 --config configs/wp13_gene_dim.yaml --out $(OUT)/wp13

wp14:
	$(PYTHON) -m fragility wp14 --config configs/wp14_triple_null.yaml --out $(OUT)/wp14

audit: axis1 axis2 axis3 axis4 axis5 wp1 wp2 wp3 wp4 wp6 wp7 wp8 wp10 wp12 wp13 wp14
	@echo "All axes and work packages reproduced under $(OUT)/"

clean:
	rm -rf $(OUT)
