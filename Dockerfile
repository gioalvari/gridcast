FROM python:3.11-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:0.9.27 /uv /uvx /bin/

WORKDIR /app

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

COPY pyproject.toml uv.lock README.md LICENSE ./
COPY src ./src

RUN uv sync --frozen --no-dev

FROM python:3.11-slim AS runtime

RUN groupadd --system gridcast \
    && useradd --system --gid gridcast --create-home gridcast

WORKDIR /app

COPY --from=builder --chown=gridcast:gridcast /app/.venv /app/.venv

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1

USER gridcast

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2)"

CMD ["uvicorn", "gridcast.api:app", "--host", "0.0.0.0", "--port", "8000"]
