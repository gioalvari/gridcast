.PHONY: install check format test demo data weather entsoe day-ahead eda benchmark probabilistic comparison timesfm timesfm-lock timesfm3 timesfm3-lock performance dashboard api docker-build full clean

install:
	uv sync --all-extras
	uv pip install -e .

check:
	uv run ruff check src/ tests/ scripts/
	uv run ruff format --check src/ tests/ scripts/
	uv run mypy src/

format:
	uv run ruff check --fix src/ tests/ scripts/
	uv run ruff format src/ tests/ scripts/

test:
	uv run pytest

demo:
	uv run gridcast demo --output-dir artifacts/demo

data:
	uv run gridcast data download

weather:
	uv run gridcast data weather

entsoe:
	uv run gridcast data entsoe --start 2024-01-01 --end 2025-01-01

day-ahead:
	uv run gridcast day-ahead contract --delivery-date 2026-08-30

eda:
	uv run gridcast eda

benchmark:
	uv run gridcast benchmark

probabilistic:
	uv run gridcast probabilistic

comparison:
	uv run gridcast comparison

timesfm:
	@test "$$(uname -s)-$$(uname -m)" = "Darwin-arm64" || \
		{ printf '%s\n' 'TimesFM lock supports Apple silicon only.' >&2; exit 1; }
	CUDA_VISIBLE_DEVICES="" uv run --isolated --locked --python 3.12.12 \
		--with-requirements scripts/timesfm-requirements.txt \
		scripts/run_timesfm.py

timesfm-lock:
	@test "$$(uname -s)-$$(uname -m)" = "Darwin-arm64" || \
		{ printf '%s\n' 'TimesFM lock supports Apple silicon only.' >&2; exit 1; }
	MACOSX_DEPLOYMENT_TARGET=14.0 uv pip compile \
		scripts/timesfm-requirements.in \
		--python 3.12.12 \
		--python-platform aarch64-apple-darwin \
		--generate-hashes \
		--output-file scripts/timesfm-requirements.txt \
		--custom-compile-command 'make timesfm-lock'

timesfm3:
	@test "$$(uname -s)-$$(uname -m)" = "Darwin-arm64" || \
		{ printf '%s\n' 'TimesFM 3 lock supports Apple silicon only.' >&2; exit 1; }
	CUDA_VISIBLE_DEVICES="" uv run --isolated --locked --python 3.12.12 \
		--with-requirements scripts/timesfm3-requirements.txt \
		scripts/run_timesfm3.py

timesfm3-lock:
	@test "$$(uname -s)-$$(uname -m)" = "Darwin-arm64" || \
		{ printf '%s\n' 'TimesFM 3 lock supports Apple silicon only.' >&2; exit 1; }
	MACOSX_DEPLOYMENT_TARGET=14.0 uv pip compile \
		scripts/timesfm3-requirements.in \
		--python 3.12.12 \
		--python-platform aarch64-apple-darwin \
		--generate-hashes \
		--output-file scripts/timesfm3-requirements.txt \
		--custom-compile-command 'make timesfm3-lock'

performance:
	uv run gridcast performance

dashboard:
	uv run streamlit run src/gridcast/dashboard.py

api:
	uv run uvicorn gridcast.api:app --host 127.0.0.1 --port 8000 --reload

docker-build:
	docker build -t gridcast:latest .

full: check test

clean:
	rm -rf artifacts .coverage .mypy_cache .pytest_cache .ruff_cache
