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
	./validate_gpu.sh cheap

parity-dns:
	./validate_gpu.sh dns

parity-dns-pcf:
	./validate_gpu.sh dns-pcf

parity-dns-tc:
	./validate_gpu.sh dns-tc

report:
	cd .. && .venv/bin/python -m production.report --runs-root runs --out runs/_report
