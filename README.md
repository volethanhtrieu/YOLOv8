# PPE Ablation Backend — YOLOv8x, ByteTrack và Event Engine

Backend xử lý video để phát hiện vi phạm trang bị bảo hộ (PPE). Profile đầy đủ
`D_full_system` sử dụng pipeline:

```text
YOLOv8x → ByteTrack → Association Head/Torso ROI
         → Event Engine 2 giây → SQLite + ảnh bằng chứng + log CSV
```

Hướng dẫn này dành cho Windows và có hai cách cài đặt:

- Cách 1: dùng Anaconda/Miniconda.
- Cách 2: không dùng Anaconda, sử dụng Python `venv`.

## 1. Chuẩn bị model và video

Đặt model tại:

```text
weights\S-N0-coco-best.pt
```

Đặt video cần kiểm tra tại:

```text
video\test.mp4
```

Cấu trúc tối thiểu trước khi chạy:

```text
ppe_ablation_backend\
├── backend\
├── video\
│   └── test.mp4
├── weights\
│   └── S-N0-coco-best.pt
├── app.py
├── check_model.py
├── config.yaml
├── requirements.txt
└── run_video.py
```

Mở **Anaconda Prompt** hoặc **Command Prompt**, sau đó vào đúng thư mục dự án:

```cmd
cd /d "PPE_Ablation_Backend_YOLOv8x\ppe_ablation_backend"     # sửa thành địa chỉ
```

Kiểm tra đang đứng đúng thư mục:

```cmd
dir
```

Phải nhìn thấy `run_video.py`, `config.yaml`, thư mục `backend`, `video` và
`weights`.

> Không dùng lệnh `cd weights\S-N0-coco-best.pt` vì `.pt` là file model, không
> phải thư mục.

## 2. Cách 1 — Cài đặt bằng Anaconda

### 2.1. Tạo môi trường

Chỉ cần tạo môi trường một lần:

```cmd
conda create -n ppe_backend python=3.10 -y
```

Kích hoạt môi trường:

```cmd
conda activate ppe_backend
```

Khi thành công, đầu dòng lệnh có dạng:

```text
(ppe_backend) D:\...\ppe_ablation_backend>
```

### 2.2. Cài thư viện

```cmd
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Nếu chạy CPU, chuyển đến [mục 4](#4-cấu-hình-model-hiện-tại).

Nếu máy có GPU NVIDIA, tiếp tục cài PyTorch CUDA:

```cmd
python -m pip uninstall torch torchvision torchaudio -y
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

Đóng và mở lại Anaconda Prompt, sau đó quay lại dự án và kích hoạt môi trường:

```cmd
cd /d "PPE_Ablation_Backend_YOLOv8x\ppe_ablation_backend"   # sửa thành địa chỉ
conda activate ppe_backend
```

## 3. Cách 2 — Cài đặt không dùng Anaconda

Máy cần cài Python 3.10 hoặc 3.11 và có thể chạy lệnh `python` trong Command
Prompt.

### 3.1. Tạo môi trường ảo

Đứng tại thư mục dự án rồi chạy:

```cmd
python -m venv .venv
```

Kích hoạt môi trường trong Command Prompt:

```cmd
.venv\Scripts\activate.bat
```

Khi thành công, đầu dòng lệnh có dạng:

```text
(.venv) D:\...\ppe_ablation_backend>
```

Nếu dùng PowerShell, kích hoạt bằng:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 3.2. Cài thư viện

```cmd
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Nếu máy có GPU NVIDIA và muốn chạy bằng GPU:

```cmd
python -m pip uninstall torch torchvision torchaudio -y
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

Sau khi cài lại PyTorch, đóng cửa sổ lệnh, mở lại Command Prompt, vào dự án rồi
kích hoạt lại môi trường:

```cmd
cd /d "PPE_Ablation_Backend_YOLOv8x\ppe_ablation_backend"     # sửa thành địa chỉ nơi lưu
.venv\Scripts\activate.bat
```

## 4. Cấu hình model hiện tại

### Trường hợp A — Model hiện tại có 3 class

Model `S-N0-coco-best.pt` hiện tại có:

```text
{0: 'person', 1: 'head', 2: 'helmet'}
```

Model này chưa có class `vest`, vì vậy cấu hình tạm thời trong `config.yaml`
phải là:

```yaml
model:
  path: weights/S-N0-coco-best.pt
  imgsz: 416
  confidence: 0.25
  iou: 0.50
  device: "0"

classes:
  person: [person]
  head: [head]
  helmet: [helmet]
  vest: []

event:
  enabled: true
  mode: consecutive
  violation_seconds: 2.0
  recovery_seconds: 0.5
  voting_window_seconds: 2.0
  voting_ratio: 0.70
  min_voting_samples: 5
  required_ppe: [helmet]
```

Nếu chỉ chạy CPU, đổi:

```yaml
device: cpu
```

Với model 3 class, `python check_model.py` sẽ báo thiếu `vest`. Đây là kết quả
đúng của bước kiểm tra model; hệ thống vẫn có thể chạy tạm để phát hiện
`no_helmet` khi cấu hình như trên.

### Trường hợp B — Sau này có model đủ 4 class

Model hoàn chỉnh cần có:

```text
person, head, helmet, vest
```

Khi đó sửa lại `config.yaml`:

```yaml
classes:
  person: [person]
  head: [head]
  helmet: [helmet]
  vest: [vest]

event:
  required_ppe: [helmet, vest]
```

Kiểm tra model:

```cmd
python check_model.py
```

Kết quả hợp lệ:

```text
OK: model có đúng 4 class person, head, helmet, vest.
```

## 5. Kiểm tra GPU

Kiểm tra Windows nhận card NVIDIA:

```cmd
nvidia-smi
```

Kiểm tra PyTorch nhận CUDA:

```cmd
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('PyTorch CUDA:', torch.version.cuda); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'Khong nhan GPU')"
```

Kết quả chạy GPU phải có dạng:

```text
CUDA available: True
PyTorch CUDA: 11.8
GPU: NVIDIA GeForce GTX 1650 Ti
```

Đồng thời kiểm tra trong `config.yaml`:

```yaml
model:
  device: "0"
```

Trong lúc chương trình xử lý video, mở một cửa sổ lệnh khác và chạy:

```cmd
nvidia-smi -l 1
```

Nếu GPU đang được sử dụng, `GPU-Util`, `Memory-Usage` sẽ tăng và thường xuất
hiện tiến trình `python.exe`. Nhấn `Ctrl + C` để dừng màn hình theo dõi.

## 6. Chạy hệ thống đầy đủ trên video

Đảm bảo môi trường đang được kích hoạt và đang đứng tại thư mục
`ppe_ablation_backend`.

Chạy profile đầy đủ, xuất video và log CSV:

```cmd
python run_video.py --source "video\test.mp4" --output "outputs\D_full_system.mp4" --log-output "logs\violations.csv" --camera-id video-01 --profile D_full_system
```

Trong khi chạy, cửa sổ lệnh có thể không in phần trăm tiến độ. Không đóng cửa
sổ và không nhấn `Ctrl + C`. Chờ đến khi xuất hiện các dòng:

```text
Saved: ...\outputs\D_full_system.mp4
Violation log: ...\logs\violations.csv
Frames: ... | Average processing FPS: ...
Violations in this run: ...
```

Log CSV chỉ được hoàn tất sau khi chương trình xử lý hết video.

Mỗi lần chạy lại nên dùng tên output, tên log và `camera-id` khác để dễ phân
biệt kết quả:

```cmd
python run_video.py --source "video\test.mp4" --output "outputs\D_full_system_run2.mp4" --log-output "logs\violations_run2.csv" --camera-id video-02 --profile D_full_system
```

## 7. Xem kết quả

Mở video kết quả:

```cmd
start "" "outputs\D_full_system.mp4"
```

Mở log vi phạm bằng Excel:

```cmd
start "" "logs\violations.csv"
```

Các kết quả được lưu tại:

| Kết quả | Vị trí |
|---|---|
| Video đã vẽ bounding box | `outputs\D_full_system.mp4` |
| Danh sách vi phạm CSV | `logs\violations.csv` |
| Ảnh bằng chứng | `evidence\` |
| Database sự kiện | `data\detections.db` |

Các cột chính trong log CSV:

| Cột | Ý nghĩa |
|---|---|
| `camera_id` | Mã video/camera của lần chạy |
| `track_id` | ID do ByteTrack gán cho người |
| `violation_type` | `no_helmet` hoặc `no_vest` |
| `status` | `active` hoặc `resolved` |
| `started_at` | Thời điểm bắt đầu sự kiện |
| `ended_at` | Thời điểm kết thúc sự kiện |
| `confidence` | Độ tin cậy liên quan đến sự kiện |
| `evidence_path` | Đường dẫn ảnh bằng chứng |

## 8. Chạy các profile ablation

| Profile | Thành phần được bật |
|---|---|
| `A_yolo` | YOLO thuần |
| `B_tracking` | YOLO + ByteTrack |
| `C_association` | Tracking + Association |
| `D_full_system` | Tracking + Association + Event Engine |

Ví dụ chạy từng profile:

```cmd
python run_video.py --source "video\test.mp4" --output "outputs\A_yolo.mp4" --log-output "logs\A_yolo.csv" --camera-id ablation-A --profile A_yolo
python run_video.py --source "video\test.mp4" --output "outputs\B_tracking.mp4" --log-output "logs\B_tracking.csv" --camera-id ablation-B --profile B_tracking
python run_video.py --source "video\test.mp4" --output "outputs\C_association.mp4" --log-output "logs\C_association.csv" --camera-id ablation-C --profile C_association
python run_video.py --source "video\test.mp4" --output "outputs\D_full_system.mp4" --log-output "logs\D_full_system.csv" --camera-id ablation-D --profile D_full_system
```

Profile A, B và C tắt Event Engine nên có thể không tạo danh sách vi phạm. Log
sự kiện quan trọng nhất nằm ở profile `D_full_system`.

## 9. Chạy Flask API

Khởi động backend:

```cmd
python app.py --profile D_full_system
```

Mặc định server chạy tại:

```text
http://127.0.0.1:5000
```

Các endpoint chính:

| Method | Endpoint | Chức năng |
|---|---|---|
| `GET` | `/health` | Kiểm tra server |
| `POST` | `/api/start` | Bắt đầu video hoặc camera |
| `POST` | `/api/stop` | Dừng video hoặc camera |
| `GET` | `/video_feed` | Xem MJPEG stream |
| `GET` | `/api/stats` | Xem thống kê |
| `GET` | `/api/events` | Xem sự kiện dạng JSON |
| `GET` | `/api/events.csv` | Tải lịch sử sự kiện CSV |

Nhấn `Ctrl + C` trong cửa sổ đang chạy server để dừng backend.
