.DEFAULT_GOAL := help

SHELL         := /bin/sh
PYTHON        ?= python3
PIP           ?= pip3

# ──────────────────────────────────────────────────────────────────────────────
# Help
# ──────────────────────────────────────────────────────────────────────────────
.PHONY: help
help:
	@printf "QNN Framework for Stochastic Optimization — available targets\n\n"
	@printf "  format            Format with Ruff\n"
	@printf "  check             Run all linters, typechecks, and tests across the system\n"
	@printf "  test              Run tests with pytest\n"
	@printf "  lint              Lint with Ruff\n"

# ──────────────────────────────────────────────────────────────────────────────
# Top-level targets
# ──────────────────────────────────────────────────────────────────────────────
.PHONY: format check test lint

format:
	ruff format src tests
	ruff check --fix src tests

lint:
	ruff check src tests
	mypy src

test:
	pytest tests/

check: lint test
