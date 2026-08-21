# Project Structure

## Luồng chính

```text
dashboard.py
  └─ HTTP → app.py
       ├─ VideoJobManager → run_pipeline_safe.py
       │    ├─ run_tiled_ppe_pipeline_v3.py
       │    └─ event_engine_v2.py
       ├─ JobPreviewService
       ├─ EventStore
       ├─ EvidenceService
       ├─ ClipService
       └─ HumanReviewStore
```

## Module đang dùng trong release

| File | Vai trò |
|---|---|
| `app.py` | Flask application factory và toàn bộ REST API. |
| `dashboard.py` | Streamlit UI: System, Process Video, Events, Review Queue và review form. |
| `job_manager.py` | Single-worker queue, tiến độ, cancel, job metadata và publish. |
| `run_pipeline_safe.py` | Runner không phá dữ liệu published; tạo run folder riêng và backup khi publish. |
| `run_tiled_ppe_pipeline_v3.py` | YOLO tiled inference, seam-aware merge, ByteTrack và PPE association. |
| `event_engine_v2.py` | Temporal state machine cho event PPE. |
| `event_store.py` | Đọc published event/tracking output và cung cấp query/statistics. |
| `job_preview.py` | Đọc output của completed job trước khi publish. |
| `evidence_service.py` | Tạo/cached evidence image Before/Open/After. |
| `clip_service.py` | Tạo/cached event clip H.264 bằng FFmpeg. |
| `human_review_store.py` | Lưu review riêng trong JSON và giữ history. |
| `configs/bytetrack_ppe.yaml` | Ngưỡng ByteTrack đã chốt. |
| `weights/candidates/SEQ-C-N2-best.pt` | Detector mặc định của runner. |

## Script release

| File | Vai trò |
|---|---|
| `bootstrap_windows.ps1` | Tạo `.venv`, cài dependency và verify. |
| `run_backend.ps1` | Chạy Waitress mặc định hoặc Flask debug. |
| `run_dashboard.ps1` | Chạy Streamlit bằng project `.venv`. |
| `verify_install.py` | Kiểm tra import, model/config, FFmpeg, CUDA và API read-only. |
| `requirements-lock.txt` | Snapshot từ `.venv` đã PASS. |
| `requirements-release.txt` | Snapshot cộng Waitress. |

## Script ablation

| File | Vai trò |
|---|---|
| `run_ablation.ps1` | Chạy YOLO một lần, replay cùng detection cache cho ByteTrack và tracking-off, rồi tạo report. |
| `evaluate_ablation.py` | Kiểm tra detection parity và xuất proxy metrics cho ba tầng A/B/C. |
| `create_ablation_gt_template.py` | Tạo template ground truth cho track, event và annotation coverage. |
| `export_cvat_mot_preannotations.py` | Xuất predicted ByteTrack tracks sang MOT 1.1 để review nhanh trong CVAT. |
| `merge_cvat_backup_with_mot_preannotations.py` | Ghép track thủ công từ CVAT backup vào pre-annotation và cảnh báo bước nhảy bất thường. |
| `test_ablation_v1.py` | Unit test identity tracking-off, cache parity và internal gap proxy. |
| `ABLATION_PROTOCOL.md` | Hướng dẫn chạy và giới hạn diễn giải metric. |

## Báo cáo Phase 2

```text
reports/phase2/
  PHASE2_BYTETRACK_REPORT.md
  ablation_full_593_table.csv
```

Hai file này lưu kết quả đã kiểm tra của full ablation 593 frame. Các output thô,
video và detection cache vẫn nằm trong `outputs/` và không đưa lên Git.

## Thư mục runtime

```text
inputs/uploads/
  Video do API nhận. Tên file được thêm UUID và secure_filename.

outputs/jobs/<job_id>/
  job.json
  pipeline.log

outputs/runs/<job_id>/
  tracking/
    tiled_ppe_association.mp4
    detections.csv
    track_ppe_rows.csv
    frame_metrics.csv
    summary.json
  events/
    events.json
    ppe_temporal_states.csv
    summary.json
  preview/
    video/ và clip cache

outputs/tiled_ppe_pipeline_v2/
  Published tracking dataset mà dashboard đọc.

outputs/event_engine_v2/
  Published Event Engine dataset mà dashboard đọc.

outputs/reviews/reviews.json
  Human decisions theo scope `published` hoặc `job:<job_id>`.

outputs/event_evidence/
  Evidence image/clip cache cho published events.

outputs/ablation/<experiment_name>/
  metrics.json, comparison.csv và report.md. Đây là output thí nghiệm,
  không được publish sang dashboard production.
```

## Legacy, chẩn đoán và thí nghiệm

Các snapshot đã bị thay thế như `run_tiled_ppe_pipeline.py`, `_v2.py`,
`event_engine_v1.py`, `dashboard_before_stats_fix.py`, `inject_test_*` và test
API/review phiên bản cũ được giữ local nhưng bị `.gitignore` loại khỏi release
commit. Các script `compare_*`, `diagnose_tracking.py`, `track_metrics.py` và
`test_tiling_*` vẫn được giữ như công cụ nghiên cứu có thể tái sử dụng.

Không xóa snapshot local trước khi nhóm xác nhận đã lưu provenance và không còn
cần đối chiếu.
