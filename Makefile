.PHONY: preflight env build run cpu nvcr health logs shell push smoke clean

APP_PORT ?= 8000
TAG := $(shell git rev-parse --short HEAD 2>/dev/null)

preflight:
	./scripts/preflight.sh

env:
	./scripts/parse-secrets.sh

build: preflight
	@if [ -z "$(TAG)" ]; then echo "ERROR: git HEAD not found — run git init and commit first" >&2; exit 1; fi
ifndef ALLOW_DIRTY
	@git diff-index --quiet HEAD -- || (echo "ERROR: dirty tree — commit or set ALLOW_DIRTY=1" >&2; exit 1)
endif
	docker build --build-arg BASE_IMAGE=$${BASE_IMAGE:-nvidia/cuda:12.4.1-runtime-ubuntu22.04} \
		-t verda-local:$(TAG) .

run: preflight
	docker compose up --build -d
	$(MAKE) health

cpu: preflight
	docker compose -f docker-compose.yml -f docker-compose.cpu.yml up --build -d
	@docker compose -f docker-compose.yml -f docker-compose.cpu.yml config | grep -q 'nvidia' \
		&& (echo "CPU override failed — GPU block still present" >&2; exit 1) || true

nvcr:
	docker compose -f docker-compose.yml -f docker-compose.nvcr.yml up --build -d

health:
	@echo "Waiting for /health..."
	@for i in $$(seq 1 30); do \
		code=$$(curl -fsS -o /dev/null -w '%{http_code}' http://localhost:$(APP_PORT)/health 2>/dev/null) && \
		echo "Health: $$code" && exit 0; \
		sleep 1; \
	done; \
	echo "ERROR: /health did not return 200 within 30s" >&2; exit 1

logs:
	docker compose logs -f --tail=200

shell:
	docker compose exec verda-local bash || docker compose exec verda-local sh

push:
	@if [ -z "$(TAG)" ]; then echo "ERROR: TAG is empty" >&2; exit 1; fi
	@echo "$(TAG)" | grep -qE '^[0-9a-f]{7,}$$' || (echo "ERROR: TAG '$(TAG)' is not a git short SHA" >&2; exit 1)
	docker push verda-local:$(TAG)

smoke:
	./scripts/smoke.sh

clean:
	docker compose down -v --remove-orphans
	find data/ -mindepth 1 -not -name '.placeholder' -delete
