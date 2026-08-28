# Final result export

`training_summary.json` records the validated dataset composition,
source-specific splits, class counts, and fail-fast checks. The remote path in
the source summary was reduced to the CVAT artifact filename.

`artifact_sha256.txt` records hashes for the sanitized manifest, summary, and
the untracked source bundle used to produce them.

The supplied bundle did not contain the completed run's `results.csv` or final
model checksums. Before publishing trained-model results, add:

- `results.csv`;
- `metrics_summary.json`;
- selected non-sensitive plots;
- `best.pt` and `last.pt` SHA256 values.

Do not add model binaries or raw console logs to ordinary Git history.
