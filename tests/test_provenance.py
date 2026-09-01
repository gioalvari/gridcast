import hashlib
import json
import subprocess
from pathlib import Path

import pandas as pd
import pytest

from gridcast.provenance import (
    build_experiment_manifest,
    dataframe_sha256,
    file_sha256,
    git_worktree_state,
    write_manifest,
)


def test_dataframe_hash_is_deterministic_and_value_sensitive() -> None:
    first = pd.DataFrame({"value": [1, 2], "label": ["a", "b"]})
    second = first.copy()
    changed = first.assign(value=[1, 3])

    assert dataframe_sha256(first) == dataframe_sha256(second)
    assert dataframe_sha256(first) != dataframe_sha256(changed)


def test_file_hash_and_manifest_are_auditable(tmp_path: Path) -> None:
    file = tmp_path / "input.txt"
    file.write_text("gridcast", encoding="utf-8")
    data = pd.DataFrame({"value": [1.0, 2.0]})

    manifest = build_experiment_manifest(
        "test-experiment",
        {"seed": 42},
        {"load": data},
        features=["lag_168h"],
        boundaries={"holdout_start": "2024-01-01T00:00:00"},
    )
    output = tmp_path / "manifest.json"
    write_manifest(manifest, output)

    assert file_sha256(file) is not None
    assert file_sha256(tmp_path / "missing") is None
    parsed = json.loads(output.read_text())
    assert parsed["schema_version"] == 1
    assert "git_dirty" in parsed
    assert "git_worktree_sha256" in parsed
    assert parsed["dataset_sha256"]["load"] == dataframe_sha256(data)
    assert parsed["features"] == ["lag_168h"]


def test_file_hash_follows_symlink_for_artifact_content(tmp_path: Path) -> None:
    target = tmp_path / "outside.txt"
    target.write_text("first secret", encoding="utf-8")
    link = tmp_path / "link"
    link.symlink_to(target)

    initial = hashlib.sha256(b"first secret").hexdigest()
    assert file_sha256(link) == initial
    target.write_text("changed secret", encoding="utf-8")
    assert file_sha256(link) != initial
    assert file_sha256(tmp_path) is None


def test_git_worktree_digest_covers_clean_tracked_and_untracked_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    subprocess.run(["git", "init", "--quiet", tmp_path], check=True)
    tracked = tmp_path / "tracked.txt"
    tracked.write_text("initial", encoding="utf-8")
    subprocess.run(["git", "add", "tracked.txt"], cwd=tmp_path, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=GridCast",
            "-c",
            "user.email=gridcast@example.invalid",
            "commit",
            "--quiet",
            "-m",
            "Initial",
        ],
        cwd=tmp_path,
        check=True,
    )
    monkeypatch.chdir(tmp_path)

    assert git_worktree_state() == (False, None)
    tracked.write_text("unstaged", encoding="utf-8")
    dirty, unstaged_digest = git_worktree_state()
    assert dirty is True and unstaged_digest is not None

    subprocess.run(["git", "add", "tracked.txt"], check=True)
    _, staged_digest = git_worktree_state()
    assert staged_digest is not None and staged_digest != unstaged_digest

    (tmp_path / "untracked.txt").write_text("new", encoding="utf-8")
    _, untracked_digest = git_worktree_state()
    assert untracked_digest is not None and untracked_digest != staged_digest


def test_git_worktree_digest_does_not_follow_untracked_symlink(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    subprocess.run(["git", "init", "--quiet"], cwd=repository, check=True)
    target = tmp_path / "external.txt"
    target.write_text("first secret", encoding="utf-8")
    (repository / "link").symlink_to(target)
    monkeypatch.chdir(repository)

    dirty, initial_digest = git_worktree_state()
    target.write_text("changed secret", encoding="utf-8")

    assert dirty is True
    assert git_worktree_state() == (True, initial_digest)


@pytest.mark.parametrize("failed_command", ["diff", "ls-files"])
def test_git_worktree_digest_is_unavailable_after_partial_failure(
    monkeypatch: pytest.MonkeyPatch,
    failed_command: str,
) -> None:
    def run(command: list[str], **_: object) -> subprocess.CompletedProcess[bytes]:
        name = command[1]
        if name == "status":
            return subprocess.CompletedProcess(command, 0, stdout=b" M file\0")
        return subprocess.CompletedProcess(
            command,
            1 if name == failed_command else 0,
            stdout=b"diff" if name == "diff" else b"file\0",
        )

    monkeypatch.setattr("gridcast.provenance.subprocess.run", run)

    assert git_worktree_state() == (True, None)
