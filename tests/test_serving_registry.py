from pathlib import Path

import pytest

from gridcast.serving.registry import publish_bundle, resolve_model_uri


class MemoryS3:
    """Tiny paginated in-memory S3 double matching registry operations."""

    def __init__(self) -> None:
        self.objects: dict[tuple[str, str], bytes] = {}

    def list_objects_v2(self, **kwargs: object) -> dict[str, object]:
        bucket = str(kwargs["Bucket"])
        prefix = str(kwargs["Prefix"])
        keys = sorted(
            key
            for name, key in self.objects
            if name == bucket and key.startswith(prefix)
        )
        token = kwargs.get("ContinuationToken")
        start = int(str(token)) if token else 0
        page = keys[start : start + 2]
        next_start = start + len(page)
        return {
            "Contents": [{"Key": key} for key in page],
            "IsTruncated": next_start < len(keys),
            "NextContinuationToken": str(next_start)
            if next_start < len(keys)
            else None,
        }

    def upload_file(self, source: str, bucket: str, key: str) -> None:
        self.objects[(bucket, key)] = Path(source).read_bytes()

    def download_file(self, bucket: str, key: str, target: str) -> None:
        Path(target).write_bytes(self.objects[(bucket, key)])


class FailingS3(MemoryS3):
    """S3 double that fails a transfer after listing an object."""

    def download_file(self, bucket: str, key: str, target: str) -> None:
        raise RuntimeError("download failed")


def test_local_registry_paths_and_immutability(
    bundle_directory: Path, tmp_path: Path
) -> None:
    assert (
        resolve_model_uri(str(bundle_directory), tmp_path) == bundle_directory.resolve()
    )
    assert (
        resolve_model_uri(bundle_directory.as_uri(), tmp_path)
        == bundle_directory.resolve()
    )
    destination = tmp_path / "published"
    publish_bundle(bundle_directory, destination.as_uri())
    assert (destination / "manifest.json").exists()
    with pytest.raises(FileExistsError):
        publish_bundle(bundle_directory, destination.as_uri())
    with pytest.raises(FileNotFoundError):
        publish_bundle(tmp_path / "absent", destination.as_uri())


def test_s3_registry_paginated_resolve_publish_and_errors(
    bundle_directory: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = MemoryS3()
    uri = "s3://models/releases/0.1.0"
    publish_bundle(bundle_directory, uri, client=client)
    assert len(client.objects) == 5
    resolved = resolve_model_uri(uri, tmp_path, client=client)
    assert (resolved / "manifest.json").exists()
    assert resolve_model_uri(uri, tmp_path, client=client) == resolved
    with pytest.raises(FileExistsError):
        publish_bundle(bundle_directory, uri, client=client)
    with pytest.raises(FileNotFoundError):
        resolve_model_uri("s3://models/missing/1.0.0", tmp_path, client=client)
    with pytest.raises(ValueError, match="model URI"):
        resolve_model_uri("https://example.com/model", tmp_path)
    with pytest.raises(ValueError, match="publish URI"):
        publish_bundle(bundle_directory, "https://example.com/model")
    monkeypatch.setitem(__import__("sys").modules, "boto3", None)
    with pytest.raises(ImportError, match="S3 registry requires"):
        resolve_model_uri("s3://models/new/1.0.0", tmp_path)


def test_registry_client_contract_errors(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="list_objects_v2"):
        resolve_model_uri("s3://models/one/1.0.0", tmp_path, client=object())


def test_registry_cleans_failed_download_and_bad_pagination(tmp_path: Path) -> None:
    client = FailingS3()
    client.objects[("models", "releases/1.0.0/manifest.json")] = b"{}"
    with pytest.raises(RuntimeError, match="download failed"):
        resolve_model_uri("s3://models/releases/1.0.0", tmp_path, client=client)
    assert not (tmp_path / "1.0.0").exists()

    class BadPagination:
        def list_objects_v2(self, **_: object) -> dict[str, object]:
            return {"Contents": [{"Key": "releases/2.0.0/a"}], "IsTruncated": True}

    with pytest.raises(ValueError, match="continuation"):
        resolve_model_uri(
            "s3://models/releases/2.0.0", tmp_path, client=BadPagination()
        )
