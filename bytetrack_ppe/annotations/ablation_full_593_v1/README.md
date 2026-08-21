# Ablation ground truth

- `gt_tracks.csv`: one visible person box per frame. `gt_person_id` must stay stable for the same real person.
- `gt_events.csv`: one real PPE event interval. Use `NO_HELMET` or `NO_VEST`; `label` should be `VIOLATION` or `COMPLIANT`.
- `gt_coverage.csv`: declares which frame ranges were annotated exhaustively. This prevents an empty file from being mistaken for 'no violations'.

Recommended values:

- `visibility`: `VISIBLE`, `PARTIAL`, or `OCCLUDED`.
- PPE state: `COMPLIANT`, `VIOLATION`, or `UNKNOWN`.
- Boolean fields: `0` or `1`.
