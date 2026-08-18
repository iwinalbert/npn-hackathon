# Retail Demand Forecasting -- make targets
#
# THIS FILE IS A THIN WRAPPER. Every target delegates to `tasks.py`, which is
# the single source of truth for what the commands are and what they do.
#
# It is written this way on purpose. The previous version duplicated the logic
# and named directories (06_BACKEND, 07_FRONTEND) that were renamed to
# backend and frontend -- so every target in it had been
# silently broken since that reorganisation. Delegating means a rename can only
# break one file, and that file is the one people actually run.
#
# `make` is not installed on the Windows machine this project is demonstrated
# on, so this is the SECONDARY entry point. The primary one works everywhere:
#
#     python tasks.py help
#
# Run `python tasks.py help` for the authoritative list; anything there works
# whether or not it has a target below.

PY ?= python
TASKS := $(PY) tasks.py

.PHONY: help build-db api stop-api test ui ui-install ui-build ui-test \
        preflight smoke docker-config docker-build docker-up docker-down \
        docker-logs docker-ps verify-all verify-integrity openapi clean-db

help:
	@$(TASKS) help

# --- product -----------------------------------------------------------------
build-db:      ; $(TASKS) build-db
api:           ; $(TASKS) api
stop-api:      ; $(TASKS) stop-api
ui:            ; $(TASKS) ui
ui-install:    ; $(TASKS) ui-install
ui-build:      ; $(TASKS) ui-build
openapi:       ; $(TASKS) openapi
clean-db:      ; $(TASKS) clean-db

# --- checks ------------------------------------------------------------------
test:            ; $(TASKS) test
ui-test:         ; $(TASKS) ui-test
verify-all:      ; $(TASKS) verify-all
verify-integrity: ; $(TASKS) verify-integrity

# --- devops ------------------------------------------------------------------
# ARGS passes flags through, since make consumes anything that looks like one:
#     make docker-up ARGS=--prod
#     make smoke ARGS="--prod --wait 60"
ARGS ?=

preflight:     ; $(TASKS) preflight $(ARGS)
smoke:         ; $(TASKS) smoke $(ARGS)
docker-config: ; $(TASKS) docker-config $(ARGS)
docker-build:  ; $(TASKS) docker-build $(ARGS)
docker-up:     ; $(TASKS) docker-up $(ARGS)
docker-down:   ; $(TASKS) docker-down $(ARGS)
docker-logs:   ; $(TASKS) docker-logs $(ARGS)
docker-ps:     ; $(TASKS) docker-ps $(ARGS)
