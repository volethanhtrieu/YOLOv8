# YOLOv8
Workplace Safety Monitoring of Personal Protective Equipment (PPE) Non-Compliance Using the You Only Look Once Version 8 (YOLOv8) Object Detector in Python
# PPE Non-Compliance Detection Using YOLOv8l

## Scope

The system detects construction workers and PPE items using YOLOv8l.

Datasets:

- CHVG
- SHEL5K

Final target classes:

- person
- head
- helmet
- vest
- glass

NoHelmet, NoVest and NoGlass are inferred using detection association and tracking rules.

## Repository structure

- configs: dataset and class mapping configurations
- data: local dataset directories and manifests
- scripts/data: dataset preparation scripts
- scripts/train: training scripts
- src/tracking: person tracking
- src/compliance: PPE violation rules
- src/app: application and interface
- reports: generated data reports
- docs: report and technical documentation

## Experiments

- [Phase 11 no-SARD baseline](experiments/phase11-no-sard-baseline/README.md):
  reproducible four-class training package using the adjudicated CHVG4, SHEL4,
  SH17, and Pictor-yolo selections. SARD is intentionally excluded.
