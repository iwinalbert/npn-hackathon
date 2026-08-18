
PY ?= python
TASKS := $(PY) tasks.py

.PHONY: help build-db api stop-api test ui ui-install ui-build ui-test \
        preflight smoke docker-config docker-build docker-up docker-down \
        docker-logs docker-ps verify-all verify-integrity openapi clean-db

help:
	@$(TASKS) help

build-db:      ; $(TASKS) build-db
api:           ; $(TASKS) api
stop-api:      ; $(TASKS) stop-api
ui:            ; $(TASKS) ui
ui-install:    ; $(TASKS) ui-install
ui-build:      ; $(TASKS) ui-build
openapi:       ; $(TASKS) openapi
clean-db:      ; $(TASKS) clean-db

test:            ; $(TASKS) test
ui-test:         ; $(TASKS) ui-test
verify-all:      ; $(TASKS) verify-all
verify-integrity: ; $(TASKS) verify-integrity

ARGS ?=

preflight:     ; $(TASKS) preflight $(ARGS)
smoke:         ; $(TASKS) smoke $(ARGS)
docker-config: ; $(TASKS) docker-config $(ARGS)
docker-build:  ; $(TASKS) docker-build $(ARGS)
docker-up:     ; $(TASKS) docker-up $(ARGS)
docker-down:   ; $(TASKS) docker-down $(ARGS)
docker-logs:   ; $(TASKS) docker-logs $(ARGS)
docker-ps:     ; $(TASKS) docker-ps $(ARGS)
