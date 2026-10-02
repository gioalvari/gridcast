"""Export the stable production-serving OpenAPI contract."""

import json
from pathlib import Path

from gridcast.serving.app import create_app
from gridcast.serving.settings import ServingSettings


def main() -> None:
    """Write sorted, indented serving OpenAPI JSON to the documentation tree."""
    settings = ServingSettings.from_env(
        {"GRIDCAST_MODEL_URI": "file:///tmp/gridcast-model"}
    )
    output = Path("docs/serving/openapi.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(create_app(settings).openapi(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
