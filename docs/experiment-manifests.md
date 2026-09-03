# Experiment manifests

GridCast benchmark commands emit `experiment_manifest.json` beside their result
artifacts. The manifest makes each result traceable to:

- the current Git commit SHA;
- dirty-worktree state and a digest of tracked and untracked changes;
- UTC generation time;
- Python and platform versions;
- the `uv.lock` SHA-256 digest;
- deterministic SHA-256 digests of ordered load and weather dataframes;
- resolved model and backtest configuration;
- exact feature names;
- validation and historical holdout boundaries.

Generate manifests with:

```bash
make benchmark
make probabilistic
make timesfm
make timesfm3
```

Dataset files remain ignored and are not redistributed. Their deterministic
digests let two local runs confirm identical normalized inputs without
publishing those inputs.

Foundation-model manifests additionally record immutable checkpoint revisions,
weights digests, CPU runtime environments, and dedicated dependency-lock
digests. TimesFM 3 also verifies and records its checkpoint configuration digest.
The manifest records the checked-out commit and worktree state. A dirty worktree
can still produce a manifest, so publication-quality runs should use a clean
committed revision.
