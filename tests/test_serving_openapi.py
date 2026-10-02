import json
from pathlib import Path

from gridcast.serving.app import create_app
from gridcast.serving.settings import ServingSettings


def test_openapi_matches_documented_snapshot() -> None:
    settings = ServingSettings.from_env(
        {"GRIDCAST_MODEL_URI": "file:///tmp/gridcast-model"}
    )
    documented = json.loads(
        Path("docs/serving/openapi.json").read_text(encoding="utf-8")
    )
    assert create_app(settings).openapi() == documented
