# Dataset card: Phase-11 no-SARD baseline

## Purpose

The dataset supports four-class construction PPE object detection with the
canonical schema `person`, `head`, `helmet`, and `vest`.

## Post-adjudication composition

| Derived source | Train | Validation | Test | Total | Role |
|---|---:|---:|---:|---:|---|
| CHVG4 | 1,358 | 170 | 170 | 1,698 | Core construction PPE |
| SHEL4 | 2,400 | 300 | 300 | 3,000 | Scale, crowd, head, helmet coverage |
| SH17 | 54 | 7 | 7 | 68 | Corrected industrial/PPE diversity |
| Pictor-yolo | 62 | 7 | 9 | 78 | Human-verified no-person negatives |
| **Total** | **3,874** | **484** | **486** | **4,844** | |

SARD contributes zero images to this baseline.

## Annotation semantics

- `person`: visible person/worker body.
- `head`: directly visible bare or unhelmeted head.
- `helmet`: worn hard hat or helmet.
- `vest`: worn safety/high-visibility vest.

Original SH17 generic head boxes were discarded because they were incompatible
with the canonical bare-head meaning. Pictor negative labels are deliberately
empty after independent full-resolution confirmation that no people are visible.

## Splitting policy

The adjudicated records were split approximately 80/10/10 using deterministic
seed 42, with source representation and duplicate/group leakage controls. Image
and label pairs remain together. The final manifest, not the upstream source
split, is authoritative.

## Distribution

This repository does not distribute the images or labels. Before distributing a
reconstructed dataset, review every upstream license, attribution obligation,
and any restrictions that apply to derived annotations.

## Known limitations

- SARD/CCTV imagery is absent.
- SH17 is a very small selected subset.
- Pictor contributes negative backgrounds only.
- Site/camera and geographic coverage may be uneven.
- Small, distant, heavily occluded workers and uncommon PPE remain challenging.
- Aggregate performance can hide weak per-class `head` performance.
