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
