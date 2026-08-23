# Tracking và Association — Variant C

Hướng dẫn này dành cho thành viên trong nhóm muốn **tải code từ GitHub về máy và chạy Variant C** mà không cần train lại model.

Variant C thực hiện:

```text
Video
  ↓
YOLO detection
  ├── person ──> ByteTrack ──> track_id
  └── head / helmet / vest
                    ↓
             PPE Association
                    ↓
     Annotated video + JSONL
```

> **Không train model. Không có Event Engine.**
>
> Chỉ cần có checkpoint `.pt` và một video đầu vào là có thể chạy.

---

## 1. Variant C làm gì?

Variant C gồm 3 thành phần chính:

1. **YOLO**
   - Detect các class:
     - `person`
     - `head`
     - `helmet`
     - `vest`

2. **ByteTrack**
   - Chỉ track class `person`.
   - Gán một `track_id` cho từng người.
   - Cố giữ cùng ID khi người di chuyển qua nhiều frame.

3. **Association**
   - Ghép PPE vào đúng người.
   - `helmet` được ghép vào vùng đầu.
   - `vest` được ghép vào vùng thân.
   - Một PPE không được gán cho nhiều người cùng lúc.

Output hiển thị dạng:

```text
ID 1 H:1 V:1
```

Trong đó:

```text
H = Helmet
V = Vest
1 = Có
0 = Không có / không detect được
```

Ví dụ:

```text
ID 1 H:1 V:1
```

nghĩa là người `ID 1`:

- có helmet;
- có vest;

---

# 2. Cấu trúc project

Sau khi clone branch Variant C, cấu trúc cần quan tâm là:

```text
YOLOv8/
│
├── models/
│   ├── C-N0-coco-best.pt
│   └── C-N0-scratch-best.pt
│
├── assets/
│   └── [đặt video test ở đây]
│
├── scripts/
│   └── run_variant_c.py
│
├── src/
│   └── variant_c/
│       ├── __init__.py
│       ├── association.py
│       └── pipeline.py
│
├── tests/
│   └── test_association.py
│
├── outputs/
│   └── [video + JSONL được tạo sau khi chạy]
│
├── README_VARIANT_C.md
└── .gitignore
```

`outputs/` là folder sinh tự động khi chạy.

Video test cá nhân không cần push lên GitHub. Người chạy có thể tự đưa video của mình vào `assets/`.

---

# 3. Clone đúng branch Variant C

Mở PowerShell hoặc Terminal.

Chạy:

```powershell
cd "$env:USERPROFILE\Desktop"
```

Clone trực tiếp branch:

```powershell
git clone -b feature/ppe-association --single-branch https://github.com/volethanhtrieu/YOLOv8.git
```

Vào project:

```powershell
cd YOLOv8
```

Kiểm tra branch:

```powershell
git branch --show-current
```

Phải thấy:

```text
feature/ppe-association
```

Nếu thấy đúng như trên thì đang dùng đúng code **Tracking và Association**.

---

# 4. Tạo môi trường Python

Khuyến nghị:

```text
Python 3.11
```

Kiểm tra:

```powershell
python --version
```

Nếu máy có Python 3.11, tạo virtual environment:

```powershell
python -m venv .venv
```

## Windows PowerShell

Nếu PowerShell cho phép chạy script:

```powershell
.\.venv\Scripts\Activate.ps1
```

Nếu gặp thông báo Execution Policy, chạy:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
```

sau đó:

```powershell
.\.venv\Scripts\Activate.ps1
```

Khi thành công, terminal sẽ có:

```text
(.venv)
```

ở đầu dòng.

Ví dụ:

```text
(.venv) PS C:\Users\...\YOLOv8>
```

---

# 5. Cài thư viện

Cập nhật pip:

```powershell
python -m pip install --upgrade pip
```

Cài các package chính:

```powershell
pip install ultralytics opencv-python lap
```

Cài PyTorch:

```powershell
pip install torch torchvision torchaudio
```

> Nếu muốn chạy bằng NVIDIA GPU, nên cài bản PyTorch có CUDA phù hợp với máy theo hướng dẫn chính thức của PyTorch.
>
> Sau khi cài, luôn kiểm tra CUDA trước khi chạy.

Kiểm tra toàn bộ:

```powershell
python -c "import torch, cv2, ultralytics; print('Torch:', torch.__version__); print('OpenCV:', cv2.__version__); print('CUDA:', torch.cuda.is_available())"
```

Nếu dùng NVIDIA GPU, mong đợi:

```text
CUDA: True
```

Kiểm tra tên GPU:

```powershell
python -c "import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO CUDA GPU')"
```

---

# 6. Chuẩn bị model `.pt`

Variant C đã được phát triển với hai checkpoint:

```text
C-N0-coco-best.pt
C-N0-scratch-best.pt
```

Đặt model vào:

```text
models/
```

Ví dụ:

```text
models/C-N0-coco-best.pt
```

Kiểm tra:

```powershell
Get-ChildItem models
```

Phải thấy file `.pt`.

Có thể kiểm tra model đọc được hay không:

```powershell
python -c "from ultralytics import YOLO; m=YOLO('models/C-N0-coco-best.pt'); print(m.names)"
```

Kết quả cần có đúng 4 class:

```text
{0: 'person', 1: 'head', 2: 'helmet', 3: 'vest'}
```

Nếu thiếu class, checkpoint đó không phù hợp với Variant C.

---

# 7. Mount / thêm video test

Không cần sửa code khi muốn dùng clip khác.

Cách dễ nhất là copy video vào:

```text
assets/
```

Ví dụ:

```text
assets/my_video.mp4
```

## Cách 1 — Kéo thả bằng File Explorer

1. Mở project.
2. Mở folder `assets`.
3. Copy video vào đó.
4. Ghi nhớ tên video.

Ví dụ:

```text
assets/construction_test.mp4
```

## Cách 2 — Copy bằng PowerShell

Ví dụ video đang ở Downloads:

```powershell
Copy-Item "$env:USERPROFILE\Downloads\construction_test.mp4" "assets\construction_test.mp4"
```

Kiểm tra:

```powershell
Get-Item assets\construction_test.mp4
```

## Cách 3 — Không copy, dùng đường dẫn tuyệt đối

Có thể chạy trực tiếp video ở chỗ khác:

```powershell
--source "C:\Users\User\Downloads\construction_test.mp4"
```

Tuy nhiên, để dễ sử dụng và tránh lỗi path, khuyến nghị đặt video vào `assets/`.

---

# 8. Test code trước khi chạy video

Từ root project:

```powershell
$env:PYTHONPATH="."
```

Test Association:

```powershell
python tests\test_association.py
```

Kết quả đúng:

```text
Association test PASSED
```

Kiểm tra syntax toàn bộ code:

```powershell
python -m compileall -q src scripts tests
```

Nếu terminal không báo lỗi thì code compile thành công.

---

# 9. Chạy Variant C

Ví dụ có:

```text
Model:
models/C-N0-coco-best.pt

Video:
assets/construction_test.mp4
```

Chạy:

```powershell
$env:PYTHONPATH="."

python -m scripts.run_variant_c `
  --model models\C-N0-coco-best.pt `
  --source assets\construction_test.mp4 `
  --output outputs\construction_test_result.mp4 `
  --json outputs\construction_test_result.jsonl `
  --conf 0.25
```

Nếu chạy thành công, cuối terminal sẽ hiện:

```text
Variant C complete
Video: ...\outputs\construction_test_result.mp4
JSONL: ...\outputs\construction_test_result.jsonl
```

---

# 10. Xem kết quả

Mở folder output:

```powershell
explorer outputs
```

Sẽ có:

```text
construction_test_result.mp4
construction_test_result.jsonl
```

## Video

File `.mp4` chứa:

- bounding box của từng người;
- `track_id`;
- trạng thái PPE.

Ví dụ:

```text
ID 1 H:1 V:1
ID 2 H:0 V:1
```

## JSONL

Xem vài frame đầu:

```powershell
Get-Content outputs\construction_test_result.jsonl -TotalCount 5
```

Mỗi frame sẽ chứa thông tin dạng:

```json
{
  "frame": 1,
  "people": [
    {
      "track_id": 1,
      "person_bbox": [100, 100, 400, 700],
      "person_conf": 0.91,
      "has_head": true,
      "has_helmet": true,
      "has_vest": true
    }
  ]
}
```

---

# 11. Muốn chạy clip khác thì làm gì?

Không cần thay đổi bất kỳ file Python nào.

Ví dụ clip mới:

```text
assets/video02.mp4
```

Chỉ đổi `--source`, `--output` và `--json`:

```powershell
python -m scripts.run_variant_c `
  --model models\C-N0-coco-best.pt `
  --source assets\video02.mp4 `
  --output outputs\video02_result.mp4 `
  --json outputs\video02_result.jsonl `
  --conf 0.25
```

Clip thứ ba:

```powershell
python -m scripts.run_variant_c `
  --model models\C-N0-coco-best.pt `
  --source assets\video03.mp4 `
  --output outputs\video03_result.mp4 `
  --json outputs\video03_result.jsonl `
  --conf 0.25
```

Quy tắc rất đơn giản:

```text
clip mới
   ↓
đổi --source
   ↓
đổi tên --output
   ↓
đổi tên --json
   ↓
RUN
```

---

# 12. Muốn đổi model thì làm gì?

Không sửa code.

Chỉ đổi:

```text
--model
```

Ví dụ dùng model scratch:

```powershell
python -m scripts.run_variant_c `
  --model models\C-N0-scratch-best.pt `
  --source assets\construction_test.mp4 `
  --output outputs\scratch_result.mp4 `
  --json outputs\scratch_result.jsonl `
  --conf 0.25
```

---

# 13. Confidence threshold

Mặc định khi chạy:

```text
--conf 0.25
```

Trong backend hiện tại:

- PPE sử dụng threshold từ `--conf`;
- person sử dụng threshold riêng cao hơn để hạn chế false positive;
- ByteTrack chỉ track `person`.

Không nên tự ý tăng confidence quá cao vì có thể bỏ sót helmet/vest nhỏ hoặc bị che.

Nếu chỉ muốn thử nghiệm:

```powershell
--conf 0.30
```

hoặc:

```powershell
--conf 0.40
```

Nhưng nên giữ `0.25` khi muốn tái tạo kết quả test ban đầu của Variant C.

---

# 14. Cách Association hoạt động

Association không đơn giản lấy PPE gần người nhất.

Backend chia person thành các vùng.

## Helmet

Ưu tiên vùng đầu:

```text
┌──────────────────┐
│    HEAD REGION   │  ← helmet
├──────────────────┤
│                  │
│      PERSON      │
│                  │
└──────────────────┘
```

Nếu model detect được `head`, helmet sẽ ưu tiên bbox head.

## Vest

Vest chỉ được tìm trong vùng torso:

```text
┌──────────────────┐
│       HEAD       │
├──────────────────┤
│                  │
│      TORSO       │  ← vest
│                  │
├──────────────────┤
│                  │
└──────────────────┘
```

Association score sử dụng:

```text
bbox containment
+ spatial proximity
+ detection confidence
```

Mục tiêu là giảm việc:

```text
helmet của người A
        ↓
bị gán nhầm
        ↓
người B
```

---

# 15. Tracking hoạt động như thế nào?

Variant C sử dụng:

```text
ByteTrack
```

ByteTrack chỉ track:

```text
person
```

PPE không cần `track_id` riêng.

Pipeline:

```text
person
  ↓
ByteTrack
  ↓
ID 1
ID 2
ID 3

head / helmet / vest
  ↓
PPE detection
  ↓
Association
  ↓
ghép vào ID tương ứng
```

Việc này giúp tracking ổn định hơn so với việc đưa tất cả PPE class vào tracker.

---

# 16. Một số hiện tượng bình thường

## 16.1 Một người mất helmet trong vài frame

Có thể xảy ra khi:

- helmet bị che;
- người quay đầu;
- ảnh mờ;
- confidence giảm;
- vật khác che khuất.

Nếu YOLO không detect được helmet ở frame đó:

```text
H:0
```

Variant C không tự giữ `H:1` từ frame trước vì temporal voting / Event Engine không thuộc phạm vi Variant C.

## 16.2 Track ID có thể mất tạm thời

Khi người bị che mạnh, ByteTrack có thể tạm mất track.

Nếu tracking phục hồi tốt, ID có thể được giữ lại.

Nếu mất dấu quá lâu, tracker có thể tạo ID mới.

## 16.3 Detector có thể nhận nhầm vật thể

YOLO không hoàn hảo.

Một object có thể bị nhận nhầm là `person`.

Backend hiện dùng:

- person confidence riêng;
- lọc person bbox bất thường;

để giảm false-positive trước khi đưa vào Association.

---

# 17. Troubleshooting

## Lỗi: `ModuleNotFoundError: No module named 'cv2'`

Cài:

```powershell
pip install opencv-python
```

Kiểm tra:

```powershell
python -c "import cv2; print(cv2.__version__)"
```

---

## Lỗi: `No module named 'src'`

Đảm bảo terminal đang ở root project:

```powershell
cd YOLOv8
```

Set:

```powershell
$env:PYTHONPATH="."
```

Sau đó chạy bằng module:

```powershell
python -m scripts.run_variant_c ...
```

Không nên chạy:

```powershell
python scripts\run_variant_c.py
```

nếu Python path chưa được cấu hình.

---

## Lỗi: `Cannot open video`

Kiểm tra file:

```powershell
Get-Item assets\construction_test.mp4
```

Nếu không tồn tại thì `--source` đang sai.

---

## Lỗi: model không tồn tại

Kiểm tra:

```powershell
Get-ChildItem models
```

Sau đó dùng đúng tên:

```powershell
--model models\C-N0-coco-best.pt
```

---

## Lỗi: `Model missing required classes`

Checkpoint không có đủ:

```text
person
head
helmet
vest
```

Dùng đúng model C-N0 của nhóm.

---

## CUDA = False

Kiểm tra:

```powershell
python -c "import torch; print(torch.cuda.is_available())"
```

Nếu:

```text
False
```

thì PyTorch hiện tại không truy cập được CUDA GPU.

Kiểm tra NVIDIA:

```powershell
nvidia-smi
```

Nếu máy có NVIDIA GPU nhưng CUDA vẫn `False`, cài lại PyTorch CUDA phù hợp.

---

## PowerShell hỏi `untrusted publisher`

Đây là cảnh báo shell integration của VS Code, không phải lỗi Variant C.

Có thể chọn:

```text
D
```

và tiếp tục sử dụng terminal.

Nếu cần activate environment trong phiên hiện tại:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\.venv\Scripts\Activate.ps1
```

---

# 18. Xóa output cũ

Muốn chạy lại từ đầu và xóa toàn bộ kết quả cũ:

```powershell
Remove-Item "outputs\*" -Recurse -Force -ErrorAction SilentlyContinue
```

Kiểm tra:

```powershell
Get-ChildItem outputs
```

Nếu không hiện file nào thì folder đã sạch.

---

# 19. Workflow ngắn nhất

Nếu máy đã setup xong, mỗi lần chạy chỉ cần:

```powershell
cd YOLOv8

.\.venv\Scripts\Activate.ps1

$env:PYTHONPATH="."

python -m scripts.run_variant_c `
  --model models\C-N0-coco-best.pt `
  --source assets\my_video.mp4 `
  --output outputs\my_video_result.mp4 `
  --json outputs\my_video_result.jsonl `
  --conf 0.25
```

Xong.

---

# 20. Checklist trước khi hỏi lỗi

Nếu code không chạy, kiểm tra lần lượt:

```text
[ ] Đang ở đúng branch feature/ppe-association?
[ ] Đang đứng ở root project?
[ ] .venv đã activate?
[ ] import cv2 được?
[ ] import ultralytics được?
[ ] torch.cuda.is_available() = True?
[ ] model .pt tồn tại?
[ ] model có đủ 5 class?
[ ] video source tồn tại?
[ ] đã set PYTHONPATH="."?
```

Kiểm tra nhanh:

```powershell
git branch --show-current

python -c "import cv2, torch, ultralytics; print('IMPORT OK'); print('CUDA:', torch.cuda.is_available())"

Get-ChildItem models

Get-ChildItem assets
```

---

# 21. File nào không cần push lên GitHub?

Không cần push:

```text
outputs/
__pycache__/
.venv/
```

Video test cá nhân trong:

```text
assets/
```

cũng không cần push nếu thành viên khác đã có source video riêng.

Nếu model checkpoint được nhóm chia sẻ qua Google Drive thì cũng có thể để ngoài GitHub và mount vào `models/` sau khi clone.

Điều quan trọng để tái sử dụng code là:

```text
scripts/run_variant_c.py
src/variant_c/
tests/
README_VARIANT_C.md
```

cùng với checkpoint thích hợp.

---

# 22. Tóm tắt cho người chỉ muốn chạy

Nếu không muốn đọc toàn bộ README:

### Bước 1

Clone:

```powershell
git clone -b feature/ppe-association --single-branch https://github.com/volethanhtrieu/YOLOv8.git
cd YOLOv8
```

### Bước 2

Setup:

```powershell
python -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\.venv\Scripts\Activate.ps1

pip install --upgrade pip
pip install torch torchvision torchaudio ultralytics opencv-python lap
```

### Bước 3

Đặt:

```text
model -> models/C-N0-coco-best.pt
video -> assets/my_video.mp4
```

### Bước 4

Run:

```powershell
$env:PYTHONPATH="."

python -m scripts.run_variant_c `
  --model models\C-N0-coco-best.pt `
  --source assets\my_video.mp4 `
  --output outputs\result.mp4 `
  --json outputs\result.jsonl `
  --conf 0.25
```

### Bước 5

Mở:

```powershell
explorer outputs
```

Hoàn tất.
