# Detector checkpoint handoff

Model weights are excluded from Git. The current backend requires:

```text
weights/candidates/CHVG4-best.pt
```

Required metadata:

| Field | Required value |
|---|---|
| Filename | `CHVG4-best.pt` |
| Runtime class order | `person`, `head`, `helmet`, `vest` |
| Number of classes | 4 |
| Training dataset | Validated `data_4class.yaml` |
| SHA-256 | Record after the team selects the final checkpoint |

Run `verify_install.py` after copying the file. The verifier reads checkpoint
metadata and rejects the old five-class model. Record filename, size, SHA-256,
training run name, W&B run URL and test metrics here when the final model is
approved.

The former `SEQ-C-N2-best.pt` belongs to the historical Phase 2 five-class
ablation. Keep it outside the release path and do not rename it to
`CHVG4-best.pt`.
