# Detector checkpoint handoff

Model weights are intentionally excluded from Git.  Before running inference,
place the released checkpoint at:

```text
weights/candidates/SEQ-C-N2-best.pt
```

Checkpoint used for the verified Phase 2 release:

| Field | Value |
|---|---|
| Filename | `SEQ-C-N2-best.pt` |
| Size | 136,724,083 bytes (about 130.39 MiB) |
| SHA-256 | `931088AB16DCD832AC139B74809D67A1395311FC63C08BF2E3138EF41135BB70` |
| Runtime classes | `person`, `head`, `helmet`, `vest`, `glass` |

Run `verify_install.py` after copying the file.  A different hash means the
experiment is using a different model and its results must be reported as a
separate run.
