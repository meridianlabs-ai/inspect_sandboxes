.PHONY: check
check:
	uv run ruff check --fix
	uv run ruff format
	uv run pyright

.PHONY: test
test:
	uv run pytest -m "not integration"

.PHONY: test-integration
test-integration:
	uv run pytest -m integration

.PHONY: test-all
test-all:
	uv run pytest

.PHONY: cov
cov:
	uv run pytest -m "not integration" --cov=inspect_sandboxes --cov-report=html --cov-branch

.PHONY: install
install:
	uv sync
	pre-commit install

.PHONY: clean
clean:
	rm -rf .pytest_cache htmlcov .coverage *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} +
