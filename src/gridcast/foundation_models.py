from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class FoundationModelIdentity:
    """Immutable identity and license for one supported checkpoint."""

    model_name: str
    model_id: str
    model_revision: str
    weights_license: str
    package_version: str
    checkpoint_sha256: str | None = None
    config_sha256: str | None = None


class FoundationArtifact(Protocol):
    """Foundation artifact fields required for identity validation."""

    model_name: str
    model_id: str
    model_revision: str
    weights_license: str


def validate_foundation_identity(
    artifact: FoundationArtifact,
    expected: FoundationModelIdentity,
) -> None:
    """Reject artifacts that do not match a supported checkpoint identity."""
    if (
        artifact.model_name,
        artifact.model_id,
        artifact.model_revision,
        artifact.weights_license,
    ) != (
        expected.model_name,
        expected.model_id,
        expected.model_revision,
        expected.weights_license,
    ):
        raise ValueError("foundation artifact identity does not match checkpoint")


TIMESFM_2P5 = FoundationModelIdentity(
    model_name="timesfm_2_5_200m_zero_shot",
    model_id="google/timesfm-2.5-200m-pytorch",
    model_revision="1d952420fba87f3c6dee4f240de0f1a0fbc790e3",
    weights_license="Apache-2.0",
    package_version="2.0.2",
)

TIMESFM_3 = FoundationModelIdentity(
    model_name="timesfm_3_0_zero_shot",
    model_id="google/timesfm-3.0-pytorch",
    model_revision="c71907076f28b1241d1fccc37efd183d0912cd13",
    weights_license="timesfm-non-commercial-license-v1.0",
    package_version="3.0.0",
    checkpoint_sha256=(
        "a7592b0a8432baee54483254e5647856911ce69e09d09a9bb65904b2d98f17da"
    ),
    config_sha256=("ff17bbc07b792c5a904cca265b8468579d736a4fe84981da25eb871b0a125bc6"),
)
