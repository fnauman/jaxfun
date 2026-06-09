.PHONY: validate-all validate-% parity-cheap report

validate-all:
	./validate_gpu.sh all

validate-%:
	./validate_gpu.sh $*

CHEAP_PARITY_IDS := \
	pcf_hydro_laminar_v1 \
	channel_poiseuille_hydro_v1 \
	pcf_mhd_conducting_v1 \
	pcf_mri_shearbox_v1 \
	taylor_couette_hydro_v1 \
	taylor_couette_mhd_conducting_v1 \
	taylor_couette_mhd_insulating_v1

parity-cheap:
	cd .. && stamp=$$(date -u +%Y%m%dT%H%M%SZ); \
	for id in $(CHEAP_PARITY_IDS); do \
	  .venv/bin/python production/run_problem.py \
	    --config production/examples/$$id.json \
	    --out runs/$$id/$$stamp \
	    --compare-golden \
	    --device auto || exit $$?; \
	done; \
	.venv/bin/python -m production.report --runs-root runs --out runs/_report
	@echo "pipe_hagen_poiseuille_v1 and pipe_womersley_v1 are skipped: pipe hydro is parity_pending until the axis-regular radial basis lands"

report:
	cd .. && .venv/bin/python -m production.report --runs-root runs --out runs/_report
