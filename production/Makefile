.PHONY: validate-all validate-% parity-cheap report

validate-all:
	./validate_gpu.sh all

validate-%:
	./validate_gpu.sh $*

parity-cheap:
	@echo "cheap parity execution is pending solver wiring in production/run_problem.py"
	@echo "vendored golden validation: use ../.venv/bin/python -m pytest -q ../tests/production"

report:
	cd .. && .venv/bin/python -m production.report --runs-root runs --out runs/_report
