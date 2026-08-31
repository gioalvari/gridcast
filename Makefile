.PHONY: install check format test demo data weather eda benchmark probabilistic dashboard api docker-build full clean

install:
	uv sync --all-extras
	uv pip install -e .

check:
	uv run ruff check src/ tests/
	uv run ruff format --check src/ tests/
	uv run mypy src/

format:
	uv run ruff check --fix src/ tests/
	uv run ruff format src/ tests/

test:
	uv run pytest

demo:
	uv run gridcast demo --output-dir artifacts/demo

data:
	uv run gridcast data download

weather:
	uv run gridcast data weather

eda:
	uv run gridcast eda

benchmark:
	uv run gridcast benchmark

probabilistic:
	uv run gridcast probabilistic

dashboard:
	uv run streamlit run src/gridcast/dashboard.py

api:
	uv run uvicorn gridcast.api:app --host 127.0.0.1 --port 8000 --reload

docker-build:
	docker build -t gridcast:latest .

full: check test

clean:
	rm -rf artifacts .coverage .mypy_cache .pytest_cache .ruff_cache
