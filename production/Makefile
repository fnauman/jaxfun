.PHONY: validate-all validate-% parity-cheap parity-dns parity-dns-pcf parity-dns-tc report

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

PCF_DNS_PARITY_IDS := \
	pcf_hydro_primitive_dns_v1 \
	pcf_mri_primitive_dns_v1

TC_DNS_PARITY_IDS := \
	taylor_couette_hydro_dns_v1 \
	taylor_couette_mhd_dns_v1

DNS_PARITY_IDS := $(PCF_DNS_PARITY_IDS) $(TC_DNS_PARITY_IDS)

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

parity-dns:
	cd .. && stamp=$$(date -u +%Y%m%dT%H%M%SZ); \
	for id in $(DNS_PARITY_IDS); do \
	  timeout 1800s .venv/bin/python production/run_problem.py \
	    --config production/examples/$$id.json \
	    --out runs/$$id/$$stamp \
	    --compare-golden \
	    --device auto || exit $$?; \
	done; \
	.venv/bin/python -m production.report --runs-root runs --out runs/_report

parity-dns-pcf:
	cd .. && stamp=$$(date -u +%Y%m%dT%H%M%SZ); \
	for id in $(PCF_DNS_PARITY_IDS); do \
	  timeout 1800s .venv/bin/python production/run_problem.py \
	    --config production/examples/$$id.json \
	    --out runs/$$id/$$stamp \
	    --compare-golden \
	    --device auto || exit $$?; \
	done; \
	.venv/bin/python -m production.report --runs-root runs --out runs/_report

parity-dns-tc:
	cd .. && stamp=$$(date -u +%Y%m%dT%H%M%SZ); \
	for id in $(TC_DNS_PARITY_IDS); do \
	  timeout 1800s .venv/bin/python production/run_problem.py \
	    --config production/examples/$$id.json \
	    --out runs/$$id/$$stamp \
	    --compare-golden \
	    --device auto || exit $$?; \
	done; \
	.venv/bin/python -m production.report --runs-root runs --out runs/_report

report:
	cd .. && .venv/bin/python -m production.report --runs-root runs --out runs/_report
