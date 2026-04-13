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
		-f infrastructure/local-emulator/docker/Dockerfile -t verda-local:$(TAG) infrastructure/local-emulator

run: preflight
	./infrastructure/local-emulator/start.sh up
	$(MAKE) health

cpu: preflight
	./infrastructure/local-emulator/start.sh cpu
	@docker compose -f infrastructure/local-emulator/compose/docker-compose.yml -f infrastructure/local-emulator/compose/docker-compose.cpu.yml config | grep -q 'nvidia' \
		&& (echo "CPU override failed — GPU block still present" >&2; exit 1) || true

nvcr:
	./infrastructure/local-emulator/start.sh nvcr

health:
	@./infrastructure/local-emulator/start.sh health

logs:
	./infrastructure/local-emulator/start.sh logs

shell:
	./infrastructure/local-emulator/start.sh shell

push:
	@if [ -z "$(TAG)" ]; then echo "ERROR: TAG is empty" >&2; exit 1; fi
	@echo "$(TAG)" | grep -qE '^[0-9a-f]{7,}$$' || (echo "ERROR: TAG '$(TAG)' is not a git short SHA" >&2; exit 1)
	docker push verda-local:$(TAG)

smoke:
	./scripts/smoke.sh

clean:
	./infrastructure/local-emulator/start.sh down
	find data/ -mindepth 1 -not -name '.placeholder' -delete
