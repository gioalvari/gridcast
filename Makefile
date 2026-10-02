.PHONY: install install-extended check format test demo data weather entsoe day-ahead eda benchmark benchmark-extended probabilistic comparison timesfm timesfm-lock timesfm3 timesfm3-lock performance dashboard api serve model-package openapi docker-build docker-build-serving stack-up stack-down loadtest-smoke loadtest loadtest-stress rollout rollout-bad-canary chaos full clean

install:
	uv sync --extra dev
	uv pip install -e .

install-extended:
	uv sync --extra dev --extra extended
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

benchmark-extended:
	uv run --extra extended gridcast benchmark \
		--extended-models \
		--output-dir artifacts/benchmark-extended

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

serve:
	GRIDCAST_MODEL_URI=$${GRIDCAST_MODEL_URI:-file://$(PWD)/artifacts/models/0.1.0} uv run --extra serving uvicorn --factory gridcast.serving.app:create_app --host 127.0.0.1 --port 8080

VERSION ?= 0.1.0
model-package:
	uv run gridcast model package --synthetic --version $(VERSION) --output-dir artifacts/models/$(VERSION)

openapi:
	uv run python scripts/export_openapi.py

docker-build:
	docker build -t gridcast:latest .

docker-build-serving:
	docker build --target serving -t gridcast-serving:latest .

stack-up: docker-build-serving
	docker compose -f deploy/compose.yaml up -d --wait

stack-down:
	docker compose -f deploy/compose.yaml down -v

loadtest-smoke:
	mkdir -p load/results
	docker compose -f deploy/compose.yaml run --rm -e RESULT_NAME=smoke k6 run /load/smoke.js

loadtest:
	mkdir -p load/results
	docker compose -f deploy/compose.yaml run --rm -e RESULT_NAME=baseline k6 run /load/baseline.js

loadtest-stress:
	mkdir -p load/results
	docker compose -f deploy/compose.yaml run --rm -e RESULT_NAME=stress k6 run /load/stress.js

rollout:
	@docker compose -f deploy/compose.yaml run --rm -e RATE=20 -e DURATION=60s -e RESULT_NAME=rollout-good k6 run /load/baseline.js & load_pid=$$!; \
	uv run gridcast rollout --prometheus-url http://localhost:9090 --envoy-admin-url http://localhost:9901 --policy deploy/rollout-policy.json --report artifacts/rollout/good.json; result=$$?; wait $$load_pid; exit $$result

rollout-bad-canary:
	CANARY_FAULT_ERROR_RATE=1 docker compose -f deploy/compose.yaml up -d --no-deps --force-recreate predict-canary
	@docker compose -f deploy/compose.yaml run --rm -e RATE=20 -e DURATION=30s -e RESULT_NAME=rollout-bad k6 run /load/baseline.js & load_pid=$$!; \
	uv run gridcast rollout --prometheus-url http://localhost:9090 --envoy-admin-url http://localhost:9901 --policy deploy/rollout-policy.json --report artifacts/rollout/bad.json; result=$$?; wait $$load_pid; test $$result -eq 2 || exit 1; \
	uv run python -c "import json, sys; check = json.load(open('artifacts/rollout/bad.json'))['failing_check']; print('rollback reason:', check); sys.exit(check.startswith('insufficient_requests'))"

chaos:
	sh load/chaos.sh

full: check test

clean:
	rm -rf artifacts .coverage .mypy_cache .pytest_cache .ruff_cache
