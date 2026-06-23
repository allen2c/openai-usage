# Development
fmt:
	@isort openai_usage tests scripts
	@black openai_usage tests scripts
	@ruff check --fix openai_usage tests scripts

install:
	poetry install --all-extras --all-groups

update:
	poetry update

fetch-models:
	PYTHONPATH=. python scripts/fetch_models_to_local.py

# Docs
mkdocs:
	mkdocs serve -a 0.0.0.0:8000

# Tests
pytest:
	python -m pytest
