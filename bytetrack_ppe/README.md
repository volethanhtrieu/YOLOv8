# ByteTrack PPE Safety Monitoring

Ứng dụng local xử lý video công trường theo pipeline:

```text
Video upload
→ YOLO tiled inference + seam-aware merge
→ ByteTrack theo dõi person
→ Association ghép helmet / bare_head / vest / glass vào person
→ Event Engine V2
→ isolated job preview / cancel / publish
→ Flask REST API
→ Streamlit dashboard
→ Human Review + Review Queue + final_disposition
```

## Trạng thái đã xác nhận

- Human Review V3 đã PASS smoke test.
- AI status và human decision được lưu riêng để giữ audit trail.
- Job mới chạy trong `outputs/runs/<job_id>` và không thay dữ liệu published cho đến khi operator bấm Publish.
- Evidence/clip ưu tiên video annotated của run. Nếu artifact annotated bị ngắn hơn `processed_frames`, backend tự chuyển sang video nguồn của run; các run mới ghi đường dẫn này trong `summary.json`.
- Model mặc định: `weights/candidates/SEQ-C-N2-best.pt`.
- ByteTrack config: `configs/bytetrack_ppe.yaml`.
- Lock hiện tại được xuất từ môi trường đã PASS: Python 3.14.6, Ultralytics 8.4.104, Flask 3.1.3 và Streamlit 1.62.0.
- Torch trong môi trường đã dùng để tạo lock là CPU build. Máy muốn chạy GPU phải cài PyTorch/CUDA tương thích riêng và kiểm tra lại bằng `verify_install.py`.

## 1. Chuẩn bị máy Windows mới

Yêu cầu:

- Windows 10/11 64-bit.
- Python 3.14 64-bit và `py.exe` hoặc `python.exe` có trong PATH.
- Dung lượng trống đủ cho `.venv`, model, video upload và output.
- Các file `.pt` được bàn giao riêng trong `weights/candidates/`; Git không lưu các weight lớn.

Không copy `.venv` từ máy cũ. Mở PowerShell tại thư mục `bytetrack_ppe` và chạy:

```powershell
.\bootstrap_windows.ps1
```

Script sẽ:

1. Tạo `.venv` mới ngay trong project.
2. Cài đúng `requirements-lock.txt` và Waitress.
3. Chạy `verify_install.py`.

Nếu PowerShell chặn script, có thể chạy riêng cho phiên hiện tại:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
```

## 2. Chạy ứng dụng

Terminal 1 — backend:

```powershell
.\run_backend.ps1
```

Mặc định backend dùng Waitress tại `http://127.0.0.1:5000`. Chỉ khi debug local mới dùng Flask development server:

```powershell
.\run_backend.ps1 -Mode flask
```

Terminal 2 — dashboard:

```powershell
.\run_dashboard.ps1
```

Mở `http://127.0.0.1:8501`. Sidebar phải trỏ API URL tới `http://127.0.0.1:5000`.

Kiểm tra backend nhanh:

```powershell
Invoke-RestMethod http://127.0.0.1:5000/api/health
```

## 3. Quy trình operator

1. Vào tab **Process Video** và upload video.
2. Đặt `max_frames=0` để xử lý toàn bộ hoặc dùng số nhỏ cho smoke test.
3. Theo dõi trạng thái `QUEUED → RUNNING → COMPLETED`; có thể Cancel khi cần.
4. Xem processed video, run events, evidence và clip trong isolated preview.
5. Chỉ bấm **Publish this run** khi output đúng. Publish sẽ thay published dataset sau khi backup bản cũ.
6. Vào **Review Queue** hoặc **Events** để đưa ra quyết định:
   - `CONFIRMED_VIOLATION`
   - `FALSE_ALARM`
   - `NEEDS_REVIEW`
7. AI event gốc không bị sửa. Dashboard tạo `final_disposition` từ human decision ở lớp overlay.

## 4. Smoke tests

Chạy sau khi bootstrap, không cần mở Flask thật:

```powershell
.\.venv\Scripts\python.exe .\verify_install.py
.\.venv\Scripts\python.exe .\test_api.py
.\.venv\Scripts\python.exe .\test_job_api_v3.py
.\.venv\Scripts\python.exe .\test_human_review_v3.py
.\.venv\Scripts\python.exe .\test_ablation_v1.py
```

Kết quả Human Review đúng phải kết thúc bằng:

```text
ALL HUMAN REVIEW V3 SMOKE TESTS PASSED
```

## 5. Dữ liệu và model không đưa lên Git

`.gitignore` loại trừ:

- `.venv/`
- `inputs/`, `videos/`, `outputs/`
- model `.pt`, `.onnx`, `.engine`
- cache và log

Khi bàn giao sang máy khác, copy riêng:

```text
weights/candidates/SEQ-C-N2-best.pt
inputs hoặc video test cần dùng
published outputs nếu muốn giữ event/review hiện tại
```

Thông tin filename, dung lượng và SHA-256 của checkpoint đã kiểm thử nằm trong
`weights/candidates/README.md`.

## 6. Tài liệu kỹ thuật

- `API_CONTRACT.md`: endpoint, input, output và mã lỗi.
- `PROJECT_STRUCTURE.md`: vai trò từng module và thư mục runtime.
- `ABLATION_PROTOCOL.md`: quy trình Detection-only vs ByteTrack vs Event Engine dùng cùng detection cache.
- `reports/phase2/PHASE2_BYTETRACK_REPORT.md`: báo cáo Phase 2 và cách diễn giải kết quả.
- `reports/phase2/ablation_full_593_table.csv`: bảng ablation 593 frame sẵn để đưa vào báo cáo.
- `requirements-lock.txt`: snapshot đúng từ `.venv` đã PASS.
- `requirements-release.txt`: lock trên cộng thêm Waitress cho Windows.

## 7. Giới hạn hiện tại

- Backend là local single-worker queue: chỉ một job video chạy tại một thời điểm.
- Không có authentication; chỉ bind `127.0.0.1`, không public Internet trực tiếp.
- Upload Streamlit được đặt tối đa 2 GB trong `.streamlit/config.toml`; video lớn cần đủ RAM/disk.
- `no_vest` là cảnh báo dựa trên thiếu bằng chứng vest và có thể sai khi thân người bị che; Human Review là bắt buộc với sự kiện nghi ngờ.
- Không xóa hoặc di chuyển video nguồn của một run đã publish nếu muốn tiếp tục mở evidence/clip khi video annotated của run không đầy đủ.
- Đánh giá production vẫn cần event-level recall, false alarms/giờ, duplicate alerts, latency và FPS trên video test được gán ground truth.

## 8. Checklist trước khi chạy

Từ repository root, vào đúng thư mục ứng dụng:

```powershell
Set-Location .\bytetrack_ppe
```

Kiểm tra các file bắt buộc:

```powershell
Test-Path .\.venv\Scripts\python.exe
Test-Path .\weights\candidates\SEQ-C-N2-best.pt
Test-Path .\configs\bytetrack_ppe.yaml
```

Cả ba lệnh phải trả `True`. Checkpoint của release đã xác minh:

```text
Filename : SEQ-C-N2-best.pt
Size     : 136,724,083 bytes
SHA-256  : 931088AB16DCD832AC139B74809D67A1395311FC63C08BF2E3138EF41135BB70
Classes  : person, head, helmet, vest, glass
```

Kiểm tra hash:

```powershell
Get-FileHash `
  -Algorithm SHA256 `
  -LiteralPath ".\weights\candidates\SEQ-C-N2-best.pt"
```

Hash khác nghĩa là đang dùng checkpoint khác; không so sánh trực tiếp kết quả
với bảng Phase 2 nếu chưa chạy lại experiment.

## 9. Hướng dẫn dashboard chi tiết

### 9.1. Khởi động

Mở hai cửa sổ PowerShell tại `bytetrack_ppe`.

Terminal backend:

```powershell
.\run_backend.ps1
```

Backend mặc định dùng Waitress tại `http://127.0.0.1:5000`. Chỉ dùng Flask
development server khi debug:

```powershell
.\run_backend.ps1 -Mode flask
```

Terminal dashboard:

```powershell
.\run_dashboard.ps1
```

Mở `http://127.0.0.1:8501`. API URL trong sidebar phải là
`http://127.0.0.1:5000`.

Nếu cần đổi port:

```powershell
.\run_backend.ps1 -Port 5050
.\run_dashboard.ps1 -Port 8502
```

Khi đổi port backend, sửa API URL của dashboard thành
`http://127.0.0.1:5050`.

### 9.2. Process Video

1. Mở tab **Process Video**.
2. Upload video `.mp4`, `.mov`, `.avi`, `.mkv` hoặc `.m4v`.
3. Đặt `max_frames`:
   - `0`: xử lý toàn bộ;
   - số dương: chỉ xử lý số frame đó để smoke test.
4. Tạo job.
5. Theo dõi trạng thái:

```text
QUEUED → RUNNING → COMPLETED
```

Job cũng có thể chuyển sang `FAILED` hoặc `CANCELLED`. Backend dùng single-worker
queue nên chỉ một job inference chạy tại một thời điểm.

### 9.3. Preview và publish

Khi job `COMPLETED`:

1. Xem processed video.
2. Kiểm tra số frame đã xử lý và số predicted track.
3. Mở danh sách event.
4. Kiểm tra evidence Before/Open/After và event clip.
5. Chỉ chọn **Publish this run** khi output hợp lý.

Preview là isolated output trong `outputs/runs/<job_id>`; nó chưa thay dữ liệu
published. Publish sẽ backup dataset cũ rồi mới thay hai published folders.

### 9.4. Review event

Trong **Review Queue** hoặc **Events**, chọn:

- `CONFIRMED_VIOLATION`;
- `FALSE_ALARM`;
- `NEEDS_REVIEW`.

AI status gốc không bị sửa. Mapping cuối:

| Human state | `final_disposition` |
|---|---|
| Chưa review | `PENDING_REVIEW` |
| `CONFIRMED_VIOLATION` | `CONFIRMED_VIOLATION` |
| `FALSE_ALARM` | `FALSE_ALARM` |
| `NEEDS_REVIEW` | `NEEDS_REVIEW` |

Quyết định trước được lưu trong `history` để audit.

## 10. Chạy pipeline bằng CLI

CLI phù hợp khi không cần dashboard hoặc muốn tạo experiment có tên cố định.
`run_pipeline_safe.py` luôn tạo run folder riêng và mặc định không publish.

### 10.1. Smoke test 60 frame

```powershell
.\.venv\Scripts\python.exe .\run_pipeline_safe.py `
  --video ".\videos\test.mp4" `
  --max-frames 60 `
  --run-name "smoke_60f"
```

### 10.2. Chạy toàn bộ video

```powershell
.\.venv\Scripts\python.exe .\run_pipeline_safe.py `
  --video ".\videos\test.mp4" `
  --max-frames 0 `
  --run-name "full_video_v1"
```

`--max-frames 0` nghĩa là toàn video. Nếu bỏ `--run-name`, runner tạo tên theo
timestamp. Runner từ chối ghi đè một run đã tồn tại; hãy chọn tên mới.

### 10.3. Chạy và publish ngay

Chỉ dùng khi đã kiểm tra input và thật sự muốn thay dataset published:

```powershell
.\.venv\Scripts\python.exe .\run_pipeline_safe.py `
  --video ".\videos\test.mp4" `
  --max-frames 0 `
  --run-name "release_video_v1" `
  --publish
```

Runner tạo backup trong `outputs/published_backups/<timestamp>` trước khi thay
published output.

### 10.4. Dùng model hoặc tracker khác

```powershell
.\.venv\Scripts\python.exe .\run_pipeline_safe.py `
  --video ".\videos\test.mp4" `
  --model ".\weights\candidates\another-best.pt" `
  --tracker ".\configs\bytetrack_ppe.yaml" `
  --run-name "model_comparison_v1"
```

### 10.5. Replay detection cache

```powershell
.\.venv\Scripts\python.exe .\run_pipeline_safe.py `
  --video ".\videos\test.mp4" `
  --detections-cache ".\outputs\runs\source_run\tracking\detections.csv" `
  --tracking-mode bytetrack `
  --run-name "cache_replay_v1"
```

Runner kiểm tra metadata, video/model/config và SHA-256 cache. Cache không tương
thích sẽ bị từ chối.

### 10.6. Tracking off

```powershell
.\.venv\Scripts\python.exe .\run_pipeline_safe.py `
  --video ".\videos\test.mp4" `
  --detections-cache ".\outputs\runs\source_run\tracking\detections.csv" `
  --tracking-mode off `
  --run-name "tracking_off_v1"
```

Chế độ `off` cấp một ID mới cho mỗi person detection ở mỗi frame. Đây chỉ là
baseline ablation, không phải production mode và không thể `--publish`.

## 11. Cấu trúc và ý nghĩa đầu ra

### 11.1. Job metadata

```text
outputs/jobs/<job_id>/
├── job.json
└── pipeline.log
```

- `job.json`: status, progress, input, run paths, publish state, timestamp và
  lỗi nếu có.
- `pipeline.log`: stdout/stderr của pipeline để chẩn đoán job failed.

### 11.2. Một run hoàn chỉnh

```text
outputs/runs/<run_name>/
├── manifest.json
├── tracking/
│   ├── tiled_ppe_association.mp4
│   ├── detections.csv
│   ├── track_ppe_rows.csv
│   ├── frame_metrics.csv
│   ├── progress.json
│   └── summary.json
└── events/
    ├── events.csv
    ├── events.json
    ├── ppe_temporal_states.csv
    └── summary.json
```

| File | Nội dung |
|---|---|
| `manifest.json` | Trạng thái toàn pipeline, tham số, elapsed time và publish state. |
| `tiled_ppe_association.mp4` | Video đã vẽ person/PPE/track ID. |
| `detections.csv` | Merged YOLO detections dùng cho replay ablation. |
| `track_ppe_rows.csv` | Person track observations và PPE association. |
| `frame_metrics.csv` | Detection/tracking statistics theo frame. |
| `progress.json` | Tiến độ live để dashboard đọc. |
| `tracking/summary.json` | Model, video, cấu hình, class, runtime và tổng tracking. |
| `events.json` | Event lifecycle dùng cho API/dashboard. |
| `ppe_temporal_states.csv` | Temporal evidence theo track/frame. |
| `events/summary.json` | Tổng confirmed/suspected events. |

Các cột chính của `track_ppe_rows.csv`:

```text
frame_index,timestamp_s,track_id,person_conf,x1,y1,x2,y2,
head_conf,helmet_conf,vest_conf,glass_conf
```

`unique_track_ids` là số identity ByteTrack dự đoán, không phải số người thật
nếu chưa có ground truth.

### 11.3. Published data

```text
outputs/tiled_ppe_pipeline_v2/
outputs/event_engine_v2/
outputs/reviews/reviews.json
outputs/event_evidence/
outputs/published_backups/<timestamp>/
```

Flask/Streamlit đọc hai published folders cố định. Human review được lưu riêng
trong `reviews.json`. Evidence/clip được cache trong `event_evidence`.

Không xóa hoặc di chuyển video nguồn của run đã publish nếu vẫn cần mở evidence
hoặc clip.

## 12. Kiểm thử code

Các smoke test dùng Flask test client, không cần mở backend/dashboard:

```powershell
.\.venv\Scripts\python.exe .\verify_install.py
.\.venv\Scripts\python.exe .\test_api.py
.\.venv\Scripts\python.exe .\test_job_api_v3.py
.\.venv\Scripts\python.exe .\test_human_review_v3.py
.\.venv\Scripts\python.exe .\test_ablation_v1.py
```

Kết quả thành công:

```text
Status : PASS
ALL API V3 SMOKE TESTS PASSED
ALL JOB MANAGER V3 SMOKE TESTS PASSED
ALL HUMAN REVIEW V3 SMOKE TESTS PASSED
ALL ABLATION V1 UNIT TESTS PASSED
```

Kiểm tra cú pháp toàn bộ Python:

```powershell
.\.venv\Scripts\python.exe -m compileall -q .
```

`test_tiling_hard_frames.py` và `test_horizontal_tiling_hard_frames.py` là test
diagnostic nặng hơn, cần model và test assets tương ứng; chúng không thuộc bộ
smoke test release hằng ngày.

## 13. Chạy ablation

Ablation so sánh ba tầng:

```text
A. Detection + Association, tracking off
B. Cùng detection cache + ByteTrack + Association
C. B + Event Engine V2
```

YOLO chạy đúng một lần. A và B replay cùng `detections.csv`, nên khác biệt không
đến từ việc detector chạy ra kết quả khác nhau.

### 13.1. Smoke test 50 frame

```powershell
.\run_ablation.ps1 `
  -Video ".\videos\test.mp4" `
  -MaxFrames 50 `
  -ExperimentName "ablation_50f_v1"
```

Chỉ chạy full video khi terminal có:

```text
Detection parity: PASS
ABLATION COMPLETE
No ablation run was published.
```

### 13.2. Full video

```powershell
.\run_ablation.ps1 `
  -Video ".\videos\test.mp4" `
  -MaxFrames 0 `
  -ExperimentName "ablation_full_593_v1"
```

Tên experiment phải mới. Script không ghi đè output cũ.

### 13.3. Output ablation

```text
outputs/ablation/<experiment_name>/
├── metrics.json
├── comparison.csv
└── report.md
```

Kết quả full ablation đã xác minh:

| Metric | Kết quả |
|---|---:|
| Frame | 593 |
| Detection parity | PASS |
| Merged detection rows | 39,953 |
| Detection-only identities | 15,880 identity một-frame |
| ByteTrack predicted IDs | 226 |
| Temporal evidence rows | 66 |
| Opened events | 1 suspected |
| Candidate-to-event reduction | 98.48% |

Không có ground truth thì không báo cáo IDF1, MOTA, HOTA, true ID Switch,
tracking fragmentation, event recall hoặc false alarms/hour.

## 14. API thường dùng

Base URL mặc định: `http://127.0.0.1:5000`.

```powershell
# Health
Invoke-RestMethod http://127.0.0.1:5000/api/health

# Published events
Invoke-RestMethod http://127.0.0.1:5000/api/events

# Stats
Invoke-RestMethod http://127.0.0.1:5000/api/stats

# Jobs
Invoke-RestMethod http://127.0.0.1:5000/api/jobs

# Review queue
Invoke-RestMethod http://127.0.0.1:5000/api/review-queue
```

Endpoint, query, body và mã lỗi đầy đủ nằm trong `API_CONTRACT.md`.

## 15. Xử lý lỗi thường gặp

### `.venv` không tồn tại

```powershell
.\bootstrap_windows.ps1
```

### Thiếu checkpoint

Đảm bảo có `weights/candidates/SEQ-C-N2-best.pt`, sau đó chạy
`verify_install.py` và đối chiếu SHA-256.

### CUDA không khả dụng

CPU vẫn chạy được. Full tiled inference 593 frame trên máy kiểm thử đạt khoảng
0.149 FPS và mất khoảng 3,987.8 giây; không phải realtime.

### Port đã được sử dụng

```powershell
.\run_backend.ps1 -Port 5050
.\run_dashboard.ps1 -Port 8502
```

Sau đó sửa dashboard API URL thành `http://127.0.0.1:5050`.

### Run folder đã tồn tại

Chọn `--run-name` hoặc `ExperimentName` mới. Không xóa output cũ nếu chưa backup.

### Dashboard không thấy job

1. Kiểm tra backend còn chạy.
2. Kiểm tra API URL trong sidebar.
3. Nhấn Refresh.
4. Đọc `outputs/jobs/<job_id>/pipeline.log` nếu job failed.

### Không có event dù có detection

Event Engine cần identity tồn tại đủ lâu và đủ temporal evidence.
`tracking-mode=off` thường không đủ điều kiện mở event.

### Cảnh báo no-vest sai

Nếu thân bị che, absence của `vest` không đáng tin. Xem evidence/clip và chọn
`FALSE_ALARM` hoặc `NEEDS_REVIEW`; không sửa AI event gốc.

### Windows khóa `progress.json`

Pipeline đã retry khi Streamlit hoặc antivirus giữ file tạm thời. Nếu vẫn có
cảnh báo, đóng chương trình đang đọc output; lỗi telemetry không nên làm dừng
inference.

## 16. Chuẩn bị commit và bàn giao

Không đưa lên Git:

- `.venv`, cache và log;
- `inputs`, `videos`, `outputs`;
- model `.pt`, `.onnx`, `.engine`;
- ZIP/manifest CVAT;
- published review data thực tế;
- local recovery snapshots và implementation v1/v2.

Trước commit:

```powershell
git status --short
git diff --cached --check
```

Khi bàn giao sang máy khác, copy riêng checkpoint, video cần xử lý và published
outputs nếu muốn giữ event/review hiện tại. Không đưa video công trường hoặc dữ
liệu cá nhân lên repository nếu chưa được phép.
