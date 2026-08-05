# Development
fmt:
	@poetry run isort openai_usage tests scripts
	@poetry run black openai_usage tests scripts
	@poetry run ruff check --fix openai_usage tests scripts

lint:
	poetry run ruff check openai_usage tests scripts
	poetry run black --check openai_usage tests scripts
	poetry run isort --check-only openai_usage tests scripts

pyright:
	poetry run pyright

check: lint pyright pytest

install:
	poetry install --all-extras --all-groups

update:
	poetry update

fetch-models:
	PYTHONPATH=. poetry run python scripts/fetch_models_to_local.py

# Docs
mkdocs:
	poetry run mkdocs serve -a 0.0.0.0:8000

docs-build:
	poetry run mkdocs build --strict

# Tests
pytest:
	poetry run python -m pytest

.PHONY: fmt lint pyright check install update fetch-models mkdocs docs-build pytest
