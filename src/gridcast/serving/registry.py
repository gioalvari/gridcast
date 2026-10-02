"""Immutable local and S3 model bundle registry operations."""

import shutil
from pathlib import Path
from typing import Any, cast
from urllib.parse import urlparse


def resolve_model_uri(
    uri: str, cache_dir: Path, *, client: object | None = None
) -> Path:
    """Resolve a local URI or download one immutable S3 bundle into the cache."""
    parsed = urlparse(uri)
    if parsed.scheme in ("", "file"):
        return Path(parsed.path if parsed.scheme else uri).resolve()
    if parsed.scheme != "s3" or not parsed.netloc:
        raise ValueError("model URI must be a local path, file:// URI, or s3:// URI")
    s3 = client or _boto3_client()
    prefix = parsed.path.lstrip("/").rstrip("/")
    version = Path(prefix).name
    destination = cache_dir / version
    if destination.exists():
        return destination
    objects = _list_objects(s3, parsed.netloc, f"{prefix}/")
    if not objects:
        raise FileNotFoundError(f"no bundle objects found under {uri}")
    destination.mkdir(parents=True, exist_ok=False)
    try:
        for item in objects:
            key = item["Key"]
            relative = Path(key).relative_to(prefix)
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            _call(s3, "download_file", parsed.netloc, key, str(target))
    except Exception:
        shutil.rmtree(destination, ignore_errors=True)
        raise
    return destination


def publish_bundle(bundle_dir: Path, uri: str, *, client: object | None = None) -> None:
    """Publish a bundle once, refusing to replace an existing model version."""
    if not bundle_dir.is_dir():
        raise FileNotFoundError(f"bundle directory does not exist: {bundle_dir}")
    parsed = urlparse(uri)
    if parsed.scheme in ("", "file"):
        destination = Path(parsed.path if parsed.scheme else uri)
        if destination.exists():
            raise FileExistsError(
                f"refusing to overwrite existing bundle: {destination}"
            )
        shutil.copytree(bundle_dir, destination)
        return
    if parsed.scheme != "s3" or not parsed.netloc:
        raise ValueError("publish URI must be a local path, file:// URI, or s3:// URI")
    s3 = client or _boto3_client()
    prefix = parsed.path.lstrip("/").rstrip("/")
    if _list_objects(s3, parsed.netloc, f"{prefix}/"):
        raise FileExistsError(f"refusing to overwrite existing S3 bundle: {uri}")
    for source in bundle_dir.rglob("*"):
        if source.is_file():
            key = f"{prefix}/{source.relative_to(bundle_dir).as_posix()}"
            _call(s3, "upload_file", str(source), parsed.netloc, key)


def _boto3_client() -> object:
    try:
        import boto3  # type: ignore[import-not-found]
    except ImportError as error:
        raise ImportError(
            "S3 registry requires GridCast installed with `--extra serving`"
        ) from error
    return boto3.client("s3")


def _list_objects(client: object, bucket: str, prefix: str) -> list[dict[str, Any]]:
    """List every object below an S3 prefix, following continuation tokens."""
    objects: list[dict[str, Any]] = []
    token: str | None = None
    while True:
        kwargs: dict[str, object] = {"Bucket": bucket, "Prefix": prefix}
        if token is not None:
            kwargs["ContinuationToken"] = token
        response = cast(dict[str, Any], _call(client, "list_objects_v2", **kwargs))
        objects.extend(cast(list[dict[str, Any]], response.get("Contents", [])))
        if not response.get("IsTruncated", False):
            return objects
        token = response.get("NextContinuationToken")
        if not isinstance(token, str) or not token:
            raise ValueError(
                "S3 list response is truncated without a continuation token"
            )


def _call(client: object, name: str, *args: object, **kwargs: object) -> object:
    method = getattr(client, name, None)
    if not callable(method):
        raise TypeError(f"S3 client does not implement {name}")
    return method(*args, **kwargs)
