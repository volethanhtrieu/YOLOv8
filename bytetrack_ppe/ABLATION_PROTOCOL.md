# Quy trình ablation ByteTrack

## Mục tiêu

So sánh ba tầng trên cùng video và cùng merged YOLO detections:

```text
A. Detection + Association (tracking off)
B. Detection + ByteTrack + Association
C. Detection + ByteTrack + Association + Event Engine V2
```

Pipeline chạy YOLO đúng một lần để tạo `detections.csv`. Hai nhánh A và B replay cùng file này. Vì vậy khác biệt giữa chúng không đến từ việc YOLO dự đoán khác nhau.

Các run ablation luôn nằm trong `outputs/runs/` và không thay published dashboard. Backend từ chối `tracking-mode=off --publish`.

## 1. Smoke test trước

Đóng các job inference khác, mở PowerShell tại thư mục `bytetrack_ppe` và chạy:

```powershell
.\run_ablation.ps1 `
  -Video ".\videos\test.mp4" `
  -MaxFrames 50 `
  -ExperimentName "ablation_50f_v1"
```

Script tự thực hiện:

1. Chạy YOLO một lần và lưu detection cache.
2. Replay cache qua ByteTrack.
3. Replay cache với tracking off.
4. Kiểm tra detection parity và tạo báo cáo.

Kết quả nằm tại:

```text
outputs/ablation/ablation_50f_v1/
├── metrics.json
├── comparison.csv
└── report.md
```

Chỉ tiếp tục full video khi terminal có dòng:

```text
Detection parity: PASS
ABLATION COMPLETE
No ablation run was published.
```

## 2. Chạy full 593 frame

Dùng tên experiment mới:

```powershell
.\run_ablation.ps1 `
  -Video ".\videos\test.mp4" `
  -MaxFrames 0 `
  -ExperimentName "ablation_full_593_v1"
```

Trên máy CPU hiện tại, bước YOLO có thể mất khoảng 75–90 phút. Hai bước replay không chạy YOLO nên nhanh hơn nhiều. Không thêm `--publish`.

## 3. Ý nghĩa ba cấu hình

### A — Detection only

Mỗi person detection có một identity duy nhất chỉ tồn tại trong frame đó. Đây không phải `track_id` của người thật.

Giá trị đo được:

- Số person detection theo frame.
- PPE association theo frame.
- Số candidate alert nếu xử lý từng frame độc lập.
- Count jitter giữa các frame.

### B — ByteTrack

ByteTrack gom các person detections qua thời gian thành persistent predicted IDs.

Giá trị đo được khi chưa có ground truth:

- Unique predicted IDs.
- Track length/span.
- Short-track ratio.
- Khoảng mất detection và phục hồi cùng ID.
- Tỷ lệ detector observation được ByteTrack xuất thành active track.

Các chỉ số này chỉ là internal proxy. Không được gọi chúng là ID switch hoặc fragmentation thật.

### C — Event Engine

Event Engine dùng lịch sử theo `track_id`, visibility guard và temporal threshold để biến nhiều evidence frame thành ít event hơn.

Giá trị đo được:

- Bare-head evidence rows.
- Vest-absence evidence rows.
- Số event được mở.
- Candidate-to-event reduction.
- Trạng thái `CONFIRMED` hoặc `SUSPECTED`.

## 4. Cách đọc report

`report.md` luôn ghi rõ:

- Hai nhánh có dùng cùng detections hay không.
- Các metric nào chỉ là proxy.
- Metric nào chưa thể tính vì chưa có annotation.

Không có ground truth thì không báo cáo các giá trị sau:

- Person precision/recall.
- Unique-person count error.
- ID switch.
- Tracking fragmentation thật.
- IDF1, MOTA, HOTA.
- Event precision/recall.
- Missed event.
- False alarms/giờ.

## 5. Tạo ground-truth template

```powershell
.\.venv\Scripts\python.exe .\create_ablation_gt_template.py `
  --output-dir ".\annotations\ablation_full_593_v1"
```

Các file được tạo:

```text
gt_tracks.csv
gt_events.csv
gt_coverage.csv
README.md
```

`gt_coverage.csv` bắt buộc phải ghi rõ phạm vi đã annotate. File event rỗng không tự động có nghĩa là video không có vi phạm.

## 6. Câu mô tả dùng trong báo cáo

> Nhóm thực hiện ablation bằng cách chạy YOLO tiled inference một lần và lưu merged detections vào cache. Cấu hình không tracking và cấu hình ByteTrack cùng replay chính xác cache này, nhờ đó biến độc lập duy nhất giữa hai nhánh là cơ chế duy trì identity. Event Engine được đánh giá ở tầng thứ ba bằng số candidate evidence được hợp nhất thành event. Các chỉ số ID switch, fragmentation và event recall chỉ được báo cáo khi có ground truth tương ứng.

## 7. File kỹ thuật

- `run_ablation.ps1`: chạy tự động A/B/C.
- `run_tiled_ppe_pipeline_v3.py`: hỗ trợ `--tracking-mode` và `--detections-cache`.
- `evaluate_ablation.py`: kiểm tra parity và tạo bảng proxy.
- `create_ablation_gt_template.py`: tạo schema annotation.
- `test_ablation_v1.py`: unit test ID, cache và gap proxy.
