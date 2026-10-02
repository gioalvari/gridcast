# ADR 0001: Native model format, not pickle

## Context
Serving must load an untrusted deployment artifact without executing arbitrary
Python object deserialization.

## Decision
Ship native LightGBM artifacts with a manifest and SHA-256 verification. The
loader in `src/gridcast/serving/bundle.py` validates the manifest before model
construction. Pickle/joblib artifacts are not a supported registry format.

## Consequences
The package is inspectable, cross-process friendly, and avoids pickle RCE. New
model families need an explicit safe loader and manifest schema rather than a
convenient Python-object shortcut.

## Alternatives considered
Pickle/joblib were rejected for code-execution risk. ONNX was deferred: it adds
conversion/runtime complexity without covering the native LightGBM path today.
