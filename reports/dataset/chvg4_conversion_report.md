# CHVG4 dataset conversion report

## Result

**Validation status: PASS**

The repository supports the requested direct CHVG 8-class to 4-class mapping.
The local 8-class export contained only `train` (1,699 images) even though its
YAML declared `val` and `test`, so it was not used to invent new splits. The
actual conversion used the verified CHVG5 handoff previously derived from that
export. Its fixed 1,358/170/170 split was copied unchanged and class 4 `glass`
was removed.

## Target schema

| ID | Class |
|---:|---|
| 0 | person |
| 1 | head |
| 2 | helmet |
| 3 | vest |

Direct mapping supported by the converter:

| CHVG8 ID | Source | CHVG4 ID | Target action |
|---:|---|---:|---|
| 0 | blue | 2 | helmet |
| 1 | glass | — | drop complete row |
| 2 | head | 1 | head |
| 3 | person | 0 | person |
| 4 | red | 2 | helmet |
| 5 | vest | 3 | vest |
| 6 | white | 2 | helmet |
| 7 | yellow | 2 | helmet |

## Verified statistics

| Split | Source images | Target images | Labels checked |
|---|---:|---:|---:|
| train | 1,358 | 1,358 | 1,358 |
| val | 170 | 170 | 170 |
| test | 170 | 170 | 170 |
| **Total** | **1,698** | **1,698** | **1,698** |

| Class | Source CHVG5 boxes | Target CHVG4 boxes | Change |
|---|---:|---:|---:|
| person | 4,674 | 4,674 | 0 |
| head | 730 | 730 | 0 |
| helmet | 3,531 | 3,531 | 0 |
| vest | 2,137 | 2,137 | 0 |
| glass | 532 | 0 | −532 |
| **Total** | **11,604** | **11,072** | **−532** |

## Invariants checked

- Source directories were read-only; output was created in a separate folder.
- Image relative paths and split membership are identical.
- SHA-256 of every copied image matches its source.
- Every target class ID is in `0, 1, 2, 3`.
- Every YOLO label row has five valid tokens and normalized coordinates.
- Coordinate tokens match the source exactly and in the same row order.
- `11,072 = 11,604 − 532`, so the only removed boxes are `glass`.
- Target helmet count equals the source helmet aggregate.

The machine-readable manifest and full validation report are generated beside
the local dataset in `data/processed/chvg4/`; images and labels are excluded
from Git.
