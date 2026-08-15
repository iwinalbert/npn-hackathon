# NPN_HACKATHON — product build & run targets
# The research layer is never touched by any target here.

BACKEND := 06_BACKEND
FRONTEND := 07_FRONTEND
PY := python

.PHONY: help build-db api test test-backend verify-integrity clean-db

help:
	@echo "NPN_HACKATHON — product targets"
	@echo ""
	@echo "  make build-db          Build 06_BACKEND/data/product.duckdb from frozen artefacts"
	@echo "  make api               Run the API at http://localhost:8000 (docs at /docs)"
	@echo "  make test              Run the backend test suite"
	@echo "  make verify-integrity  Prove no protected research artefact changed"
	@echo "  make clean-db          Delete the product database (rebuildable)"

build-db:
	$(PY) $(BACKEND)/scripts/build_product_db.py

api:
	cd $(BACKEND) && $(PY) -m uvicorn app.main:app --reload --port 8000

test test-backend:
	cd $(BACKEND) && $(PY) -m pytest

verify-integrity:
	$(PY) scripts/08_organization/61_integrity_manifest.py after
	$(PY) scripts/08_organization/61_integrity_manifest.py compare

clean-db:
	rm -f $(BACKEND)/data/product.duckdb $(BACKEND)/data/product.duckdb.wal
