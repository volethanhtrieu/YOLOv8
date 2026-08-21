# BÁO CÁO PHASE 2 — BYTETRACK, ASSOCIATION VÀ EVENT ENGINE

## 1. Mục tiêu

Phase 2 chuyển hệ thống từ phát hiện vật thể trên từng ảnh sang xử lý video có định danh theo thời gian. Phần việc ByteTrack không huấn luyện lại YOLO mà nhận các bounding box `person` đã được YOLO phát hiện, duy trì `track_id`, cung cấp identity ổn định cho Association Module và Event Engine.

Luồng triển khai thực tế:

```text
Video
→ YOLO tiled inference + seam-aware merge
→ ByteTrack cho lớp person
→ Association ghép PPE vào person track
→ Event Engine V2 tích lũy bằng chứng theo thời gian
→ Flask API + Streamlit dashboard
→ Human Review và final_disposition
```

Về bản chất, hệ thống kết hợp cả **detection** và **tracking**:

- YOLO thực hiện detection người và PPE trên từng frame.
- ByteTrack thực hiện multi-object tracking cho người.
- Association Module gắn `helmet`, `head`, `vest`, `glass` vào từng person track.
- Event Engine biến nhiều bằng chứng theo frame thành sự kiện có vòng đời.

## 2. Thành phần đã triển khai

### 2.1. Detection và tiling

Checkpoint sử dụng khi chạy ablation là `SEQ-C-N2-best.pt`. Runtime của model cung cấp 5 lớp:

| ID | Lớp | Ý nghĩa trong pipeline |
|---:|---|---|
| 0 | `person` | Người, đầu vào cho ByteTrack. |
| 1 | `head` | Bare head/head không có hardhat; là bằng chứng dương cho tình huống không đội mũ. |
| 2 | `helmet` | Mũ bảo hộ. |
| 3 | `vest` | Áo phản quang. |
| 4 | `glass` | Kính bảo hộ. |

Pipeline dùng `imgsz=960`, chia ảnh thành lưới `2 × 2`, overlap `0.2` và merge detection ở đường nối tile. Cấu hình này ưu tiên vật thể PPE nhỏ; không dùng input 416 × 416 trong thí nghiệm Phase 2.

### 2.2. ByteTrack

ByteTrack chỉ chạy trên detection lớp `person`. Cấu hình đã dùng:

| Tham số | Giá trị |
|---|---:|
| `track_high_thresh` | 0.4 |
| `track_low_thresh` | 0.1 |
| `new_track_thresh` | 0.7 |
| `track_buffer` | 60 frame |
| `match_thresh` | 0.9 |
| `fuse_score` | `true` |

Mỗi dòng tracking lưu `frame_index`, thời gian, `track_id`, confidence, bounding box và kết quả Association PPE. Chế độ `tracking-mode=off` cấp một identity mới cho mỗi person detection ở mỗi frame và chỉ được dùng cho ablation; backend từ chối publish run này.

### 2.3. Association và Event Engine

Association Module tìm PPE phù hợp trong vùng người. Event Engine V2 giữ trạng thái theo `track_id`, áp dụng visibility guard và temporal threshold trước khi mở event.

Quy tắc diễn giải quan trọng:

- `head` là bằng chứng trực tiếp cho bare head; không kết luận không đội mũ chỉ vì thiếu detection `helmet`.
- Dataset runtime không có lớp âm `no_vest`. Vì vậy thiếu `vest` chỉ tạo cảnh báo nghi ngờ khi vùng thân đủ quan sát và có bằng chứng theo thời gian.
- AI status được lưu riêng với quyết định của người review để giữ audit trail.

### 2.4. Backend và dashboard

Hệ thống dùng cả hai framework, không còn mâu thuẫn Flask/Streamlit:

- Flask cung cấp REST API, job manager, event/evidence/clip và review endpoints.
- Streamlit là giao diện operator cho upload, tiến độ, preview, publish, event log và review queue.

Mỗi job chạy trong thư mục riêng. Kết quả chỉ thay dữ liệu dashboard sau khi operator chọn Publish. Run ablation không được publish.

## 3. Thiết kế ablation

Ba tầng được so sánh trên cùng video:

| Cấu hình | Nội dung |
|---|---|
| A — Detection only | YOLO detection + Association; mỗi detection người có ID chỉ tồn tại một frame. |
| B — ByteTrack | Cùng detection cache + ByteTrack + Association. |
| C — Event Engine | Output B + tích lũy temporal evidence thành event. |

YOLO chỉ chạy một lần để tạo `detections.csv`. Cấu hình A và B replay chính xác cache đó. Vì vậy biến độc lập giữa A và B là cơ chế tracking, không phải sự ngẫu nhiên của detector.

Thông tin thí nghiệm:

| Thuộc tính | Giá trị |
|---|---|
| Video | `videos/test.mp4` |
| Số frame | 593 |
| FPS nguồn | 29.9700 |
| Thời lượng | khoảng 19.79 giây |
| Merged detection rows | 39,953 |
| Detection parity | PASS |
| Runtime | PyTorch 2.13.0 CPU; CUDA không khả dụng |

## 4. Kết quả ablation

Bảng chi tiết có tại `reports/phase2/ablation_full_593_table.csv`.

| Metric | A: Detection only | B: ByteTrack | C: Event Engine |
|---|---:|---:|---:|
| Merged detection rows dùng chung | 39,953 | 39,953 | 39,953 |
| Person observation rows | 15,880 | 9,609 | 9,609 |
| Unique predicted IDs | 15,880 identity một-frame | 226 predicted track IDs | 226 predicted track IDs |
| Mean observations/ID | 1.00 | 42.52 | 42.52 |
| Frame candidate rows | 424 | 384 | 66 temporal evidence rows |
| Opened events | Không áp dụng | Không áp dụng | 1 |
| Confirmed events | Không áp dụng | Không áp dụng | 0 |
| Suspected events | Không áp dụng | Không áp dụng | 1 |
| Cache-replay processing FPS | 11.42 | 12.75 | Không áp dụng |

Event Engine giảm 66 dòng temporal evidence xuống 1 event, tương đương candidate-to-event reduction **98.48%**.

### 4.1. Chẩn đoán nội bộ ByteTrack

| Proxy | Kết quả |
|---|---:|
| Predicted IDs có tối đa 5 observations | 49/226, tương đương 21.68% |
| Predicted IDs có internal gap | 187/226, tương đương 82.74% |
| Recovered gap episodes | 1,513 |
| Internal gap dài nhất | 61 frame |
| Mean observations/ID | 42.52 |
| Median observations/ID | 22 |
| Mean predicted track span | 86.15 frame |

Các giá trị trên là **internal behavior proxy**, không phải ID Switch hoặc fragmentation ground-truth. Ví dụ, một internal gap có thể xuất phát từ detector bỏ sót, vật thể bị che, track không được xuất khi chưa đủ điều kiện, hoặc identity thực sự bị đứt.

### 4.2. Hiệu năng

- Lượt inference YOLO tiled đầy đủ trên CPU mất khoảng 3,987.8 giây, tương đương khoảng **0.149 FPS**.
- FPS 11.42 và 12.75 trong bảng là tốc độ **replay detection cache**, không bao gồm YOLO inference và không đại diện cho tốc độ end-to-end realtime.
- Replay ByteTrack tạo `track_ppe_rows.csv` và `events.csv` giống hệt lượt nguồn theo SHA-256, chứng minh tính tái lập trên cùng cache.

Không nên kết luận ByteTrack nhanh hơn tracking-off từ chênh lệch FPS replay nhỏ này; thời gian còn chịu ảnh hưởng của ghi video, Association và I/O.

## 5. Kiểm thử hệ thống

Các kiểm thử đã chạy sau full ablation:

| Nhóm kiểm thử | Kết quả |
|---|---|
| Release verification | PASS |
| API V3 smoke tests | PASS |
| Job Manager V3 smoke tests | PASS |
| Human Review V3 smoke tests | PASS |
| Ablation V1 unit tests | PASS |

Runtime phát hiện CUDA không khả dụng nhưng cho phép chạy CPU với cảnh báo tốc độ.

## 6. Kết quả đạt được trong Phase 2

1. Tích hợp ByteTrack vào pipeline YOLO tiled inference.
2. Duy trì `track_id` cho person và truyền identity sang Association/Event Engine.
3. Thêm chế độ tracking bật/tắt phục vụ ablation.
4. Tạo detection cache có kiểm tra metadata và SHA-256 để hai nhánh dùng cùng đầu vào.
5. Chặn publish đối với run tracking-off.
6. Hoàn thiện isolated job, cancel, preview và publish an toàn.
7. Hoàn thiện Event Engine V2, evidence, clip và Human Review V3.
8. Vá lỗi khóa `progress.json` trên Windows bằng retry để lỗi telemetry không dừng inference.
9. Chạy full ablation 593 frame và xác minh kết quả replay có tính tái lập.

## 7. Giới hạn và cách trình bày trung thực

Thí nghiệm chưa có ground truth tracking/event đầy đủ. Vì vậy chưa báo cáo:

- person precision/recall;
- unique-person count error;
- ID Switch và fragmentation thật;
- IDF1, MOTA, HOTA;
- event precision/recall;
- missed event, duplicate alert và false alarms/hour.

Số 226 chỉ là **predicted track IDs**, không phải 226 người thật. Tỷ lệ giảm candidate cũng không tự động đồng nghĩa tăng độ chính xác; nó chỉ cho thấy Event Engine hợp nhất nhiều bằng chứng theo frame thành ít event hơn.

Video demo nên trình bày tối thiểu các tình huống: người đi bình thường, bị che bởi trụ, ra khỏi/đi vào khung hình, hai người đi gần nhau và detector bỏ sót người. Khi detector không tạo person box trong thời gian dài, ByteTrack không thể đảm bảo phục hồi người đó; đây là giới hạn đầu vào detection.

## 8. Kết luận

Phase 2 đã hoàn thành luồng kỹ thuật từ detection theo frame sang tracking và event-level monitoring. Giá trị chính không nằm ở việc fine-tune YOLO mà ở khả năng duy trì identity, liên kết PPE theo người, tích lũy temporal evidence, cô lập job trước khi publish và hỗ trợ human review. Ablation hiện tại chứng minh pipeline hoạt động có kiểm soát và tái lập; đánh giá độ chính xác tracking chuẩn là công việc tiếp theo khi nhóm có ground truth.
