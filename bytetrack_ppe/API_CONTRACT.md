# Flask API Contract — Human Review V3

Base URL local: `http://127.0.0.1:5000`

Response JSON lỗi có ít nhất field `error`; một số lỗi có thêm `detail`. Evidence và clip trả binary thay vì JSON.

## Khái niệm scope

- `published`: dataset đang được dashboard công bố.
- `job:<job_id>`: output isolated của một job đã hoàn tất.

Human review được lưu riêng theo `(scope, event_id)` và không sửa Event Engine JSON gốc.

## Event overlay

Mỗi event trả về từ event API được bổ sung:

```json
{
  "human_decision": "FALSE_ALARM",
  "human_reviewer": "test_operator",
  "human_review_note": "...",
  "human_review_updated_at": "2026-08-20T...+00:00",
  "human_review_state": "FALSE_ALARM",
  "final_disposition": "FALSE_ALARM",
  "review_priority": 3
}
```

Nếu chưa review:

```text
human_decision = null
human_review_state = UNREVIEWED
final_disposition = PENDING_REVIEW
```

## System và published data

### `GET /api/health`

Kiểm tra source files, evidence, annotated video và FFmpeg. Trả `200` khi store sẵn sàng, nếu không có thể trả `503` cùng chi tiết source status.

### `GET /api/stats`

Trả thống kê detector/event store cùng:

- `human_review.by_decision`
- `human_review.by_final_disposition`
- `human_review.pending_review`
- `human_review.published_event_count`
- `human_review_store`

### `GET /api/events`

Query tùy chọn:

| Query | Kiểu | Ý nghĩa |
|---|---:|---|
| `status` | string | Lọc AI status. |
| `ppe_type` | string | Lọc loại PPE. |
| `event_type` | string | Lọc event type. |
| `state` | string | Lọc lifecycle state. |
| `track_id` | int | Lọc track. |
| `limit` | int | Mặc định 100. |
| `offset` | int | Mặc định 0. |

Trả page JSON có `items`; mọi item đã có human review overlay.

### `GET /api/events/<event_id>`

Trả một published event. `404 event_not_found` nếu không có.

### `GET /api/events/<event_id>/evidence`

Query:

- `phase`: `before`, `open`, `after`; mặc định `open`.
- `view`: `crop` hoặc view mà EvidenceService hỗ trợ; mặc định `crop`.

Trả `image/jpeg`. Header `X-Evidence-Frame` chứa source frame.

### `GET /api/events/<event_id>/clip`

Trả `video/mp4` H.264. Clip được cache theo event và mtime video nguồn.

### `GET /api/tracks/<track_id>`

Trả timeline của một published track. `404 track_not_found` nếu không có.

## Video jobs

### `GET /api/jobs?limit=50`

Trả `{ "items": [...] }`, mới nhất trước. `limit` phải là integer.

### `GET /api/jobs/<job_id>`

Trả job metadata: status, progress, input, run paths, publish state, error và timestamps.

### `POST /api/jobs`

Content type: `multipart/form-data`.

| Field | Bắt buộc | Ý nghĩa |
|---|---|---|
| `video` | Có | `.mp4`, `.mov`, `.avi`, `.mkv`, `.m4v`. |
| `max_frames` | Không | Integer không âm; `0` = toàn video. |

Thành công: `202` và job ở trạng thái `QUEUED`. Chỉ một job được chạy tại một thời điểm; job thứ hai có thể nhận `409 job_creation_failed`.

Backend giới hạn request 5 GB. Streamlit release giới hạn upload 2 GB.

### `POST /api/jobs/<job_id>/cancel`

Yêu cầu cancel job đang chạy. Trả `404` nếu không tồn tại, `409 cancel_failed` nếu trạng thái không cho phép.

### `POST /api/jobs/<job_id>/publish`

Chỉ publish output hợp lệ của completed job. Runner backup published folders trước khi thay. Trả `409 publish_failed` nếu job chưa sẵn sàng.

## Isolated preview

Các endpoint sau không publish dữ liệu:

- `GET /api/jobs/<job_id>/preview/video` → `video/mp4`
- `GET /api/jobs/<job_id>/preview/events`
- `GET /api/jobs/<job_id>/preview/events/<event_id>`
- `GET /api/jobs/<job_id>/preview/events/<event_id>/evidence?phase=open&view=crop` → `image/jpeg`
- `GET /api/jobs/<job_id>/preview/events/<event_id>/clip` → `video/mp4`
- `GET /api/jobs/<job_id>/preview/tracks/<track_id>`

Job phải có status `COMPLETED`; nếu không thường trả `409 preview_unavailable` hoặc lỗi tương ứng.

## Review Queue

### `GET /api/review-queue`

Query:

| Query | Giá trị |
|---|---|
| `scope` | `published` hoặc `job:<job_id>`; mặc định `published`. |
| `limit` | 1–500; mặc định 100. |

Queue chỉ chứa event có `human_review_state` là `UNREVIEWED` hoặc `NEEDS_REVIEW`, sắp theo `review_priority` rồi `start_s`.

## Human Review

Decision hợp lệ:

- `CONFIRMED_VIOLATION`
- `FALSE_ALARM`
- `NEEDS_REVIEW`

### `GET /api/reviews?scope=<scope>`

Trả danh sách review và store stats. Không truyền `scope` sẽ trả tất cả.

### `GET /api/reviews/<event_id>?scope=published`

Trả `{ "review": null, ... }` với event chưa review; không xem đây là lỗi.

### `POST /api/reviews`

Content type: `application/json`.

```json
{
  "scope": "published",
  "event_id": "suspected_no_vest_T176_F149",
  "decision": "FALSE_ALARM",
  "reviewer": "test_operator",
  "note": "Vest status cannot be verified because the torso is occluded."
}
```

Validation:

- `400 event_id_required`
- `400 invalid_decision`
- `400 invalid_review`
- `404 event_not_found_in_scope`

Review mới thay quyết định hiện tại và đưa bản trước vào `history` để audit.

## Mã HTTP chính

| Code | Ý nghĩa |
|---:|---|
| 200 | GET/POST đồng bộ thành công. |
| 202 | Job đã được nhận và chạy nền. |
| 400 | Input/query/decision không hợp lệ. |
| 404 | Job, event, track hoặc file preview không tồn tại. |
| 409 | Trạng thái job/preview/publish không cho phép. |
| 413 | Upload vượt giới hạn backend. |
| 500 | Không tạo được evidence/clip hoặc lỗi nội bộ. |
| 503 | Published store/health chưa sẵn sàng. |
