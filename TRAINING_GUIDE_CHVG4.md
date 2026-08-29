# Hướng dẫn train và bàn giao model CHVG4

Tài liệu này dành cho người nhận code để chuẩn bị dataset, fine-tune YOLOv8,
theo dõi thí nghiệm bằng Weights & Biases (W&B), chọn checkpoint và bàn giao
weight cho backend ByteTrack.

Toàn bộ lệnh PowerShell bên dưới được chạy từ thư mục gốc của repository.

## 1. Mục tiêu và schema bắt buộc

Model cuối cùng phải có đúng bốn class theo đúng thứ tự:

```text
0 person
1 head
2 helmet
3 vest
```

Backend không chấp nhận:

- model 3-class thiếu `vest`;
- model 5-class còn `glass`;
- model có đủ tên nhưng sai class ID hoặc sai thứ tự.

Luồng làm việc:

```text
Dataset nguồn
→ convert sang CHVG4
→ validation PASS
→ smoke train
→ train chính thức + W&B
→ đánh giá best.pt
→ đổi tên CHVG4-best.pt
→ verify backend
→ chạy video test
→ bàn giao weight + metadata
```

## 2. Các file người train cần biết

| File/thư mục | Vai trò |
|---|---|
| `scripts/data/convert_chvg_to_4class.py` | Chuyển dataset 8-class hoặc handoff 5-class sang 4-class. |
| `scripts/data/validate_chvg_4class.py` | So sánh độc lập dataset nguồn và đích. |
| `scripts/train/train_chvg4.py` | Fine-tune YOLOv8, log W&B và lưu weight theo giai đoạn. |
| `configs/data_4class.yaml` | YAML dùng khi dataset nằm ở `data/processed/chvg4`. |
| `data/processed/chvg4/` | Dataset local sau conversion; không commit lên Git. |
| `runs/chvg4/` | Kết quả training local; không commit lên Git. |
| `bytetrack_ppe/weights/candidates/` | Nơi đặt checkpoint đã chọn để backend sử dụng. |
| `reports/dataset/` | Báo cáo thống kê không chứa ảnh, có thể commit. |

## 3. Chuẩn bị máy train

Khuyến nghị:

- Windows 10/11 64-bit;
- Python 3.14 64-bit theo môi trường release hiện tại;
- NVIDIA GPU và PyTorch CUDA tương thích;
- Git;
- đủ dung lượng cho dataset, `.venv`, training output và checkpoint;
- tài khoản W&B nếu cần theo dõi online.

Clone repository và chuyển vào đúng branch:

```powershell
git clone <repository-url>
Set-Location .\YOLOv8
git switch feature/ppe-association
```

Nếu repository đã có sẵn:

```powershell
git status --short
git branch --show-current
```

Không tiếp tục nếu đang có thay đổi cá nhân chưa được lưu hoặc chưa hiểu nguồn
gốc của chúng.

## 4. Tạo môi trường Python

Không copy `.venv` từ máy khác. Tạo môi trường bằng script release:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
Set-Location .\bytetrack_ppe
.\bootstrap_windows.ps1
Set-Location ..
```

Nếu checkpoint `CHVG4-best.pt` chưa tồn tại, `verify_install.py` có thể báo FAIL
ở bước model. Đây là trạng thái bình thường trước khi train.

Cài thêm W&B vào cùng môi trường:

```powershell
.\bytetrack_ppe\.venv\Scripts\python.exe -m pip install -r .\requirements-training.txt
```

### Kiểm tra GPU

```powershell
.\bytetrack_ppe\.venv\Scripts\python.exe -c "import torch; print('torch=', torch.__version__); print('cuda=', torch.cuda.is_available()); print('gpu=', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

Kết quả mong đợi:

```text
cuda= True
gpu= <tên GPU NVIDIA>
```

Nếu `cuda=False`, code vẫn chạy nhưng YOLOv8l sẽ rất chậm. Cài lại PyTorch bản
CUDA phù hợp với driver/GPU trước khi train chính thức.

## 5. Chuẩn bị dataset

### 5.1. Mapping từ CHVG 8-class

| Source ID | Source class | Target |
|---:|---|---|
| 0 | blue | 2 helmet |
| 1 | glass | DROP |
| 2 | head | 1 head |
| 3 | person | 0 person |
| 4 | red | 2 helmet |
| 5 | vest | 3 vest |
| 6 | white | 2 helmet |
| 7 | yellow | 2 helmet |

Converter chỉ đổi class ID hoặc xóa cả dòng `glass`. Bốn token bbox
`x_center y_center width height` phải được giữ nguyên.

### 5.2. Chọn đúng dataset nguồn

Converter hỗ trợ hai schema:

1. CHVG gốc 8-class có đủ `train`, `val`, `test`.
2. `CHVG5_DATASET_HANDOFF` có schema `person, head, helmet, vest, glass` và ba
   split cố định; converter bỏ class `glass` nhưng không chia lại dữ liệu.

Bản 8-class từng được kiểm tra trên máy phát triển chỉ có 1.699 ảnh trong
`train`, trong khi YAML khai báo `val/test` không tồn tại. Không được tự ý chia
lại bộ này. Dùng bản handoff có ba split hoặc xin lại bản 8-class đầy đủ.

## 6. Convert dataset sang 4-class

### Trường hợp dùng CHVG5 handoff đã có ba split

```powershell
.\bytetrack_ppe\.venv\Scripts\python.exe `
  .\scripts\data\convert_chvg_to_4class.py `
  --source-yaml "C:\path\to\CHVG5_DATASET_HANDOFF\data\processed\chvg5\chvg5.yaml" `
  --output ".\data\processed\chvg4"
```

### Trường hợp dùng CHVG 8-class đầy đủ

```powershell
.\bytetrack_ppe\.venv\Scripts\python.exe `
  .\scripts\data\convert_chvg_to_4class.py `
  --source-yaml "C:\path\to\CHVG8\data.yaml" `
  --output ".\data\processed\chvg4"
```

Quy tắc an toàn:

- converter không sửa source;
- output không được tồn tại trước khi chạy;
- thiếu split, sai class order, label malformed hoặc orphan label sẽ làm lệnh
  dừng;
- chỉ khi validation PASS thì thư mục staging mới được đổi thành output chính.

Nếu `data/processed/chvg4` đã tồn tại, không xóa ngay. Đọc
`validation_report.md`, xác định provenance và backup nếu cần. Mỗi thí nghiệm
nên dùng dataset có nguồn gốc rõ ràng.

## 7. Kiểm tra output dataset

Output đúng:

```text
data/processed/chvg4/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
├── labels/
│   ├── train/
│   ├── val/
│   └── test/
├── data_4class.yaml
├── conversion_manifest.json
├── validation_report.json
└── validation_report.md
```

Mở báo cáo:

```powershell
Get-Content .\data\processed\chvg4\validation_report.md
```

Chạy validator độc lập lần nữa:

```powershell
.\bytetrack_ppe\.venv\Scripts\python.exe `
  .\scripts\data\validate_chvg_4class.py `
  --source-yaml "C:\path\to\source\chvg5.yaml" `
  --target-yaml ".\data\processed\chvg4\data_4class.yaml" `
  --report-dir ".\data\processed\chvg4\recheck"
```

Chỉ train khi thấy:

```text
CHVG 4-class validation: PASS
```

Validator kiểm tra:

- đủ ba split và giữ nguyên số ảnh;
- đường dẫn ảnh giữa source/target giống nhau;
- SHA-256 từng ảnh giống nhau;
- class ID đích chỉ có 0, 1, 2, 3;
- không có label malformed;
- token tọa độ bbox không đổi;
- target box = source box − glass box;
- helmet đích bằng tổng các class màu mũ nguồn.

Kết quả đã xác nhận của handoff hiện tại:

| Metric | Giá trị |
|---|---:|
| Train images | 1.358 |
| Validation images | 170 |
| Test images | 170 |
| Source boxes | 11.604 |
| Dropped glass boxes | 532 |
| Target boxes | 11.072 |

## 8. Thiết lập W&B

Đăng nhập một lần trên máy train:

```powershell
.\bytetrack_ppe\.venv\Scripts\wandb.exe login
```

Không ghi API key vào README, source code, `.env` được commit hoặc ảnh chụp màn
hình. Nếu nhóm dùng W&B organization/team, lấy đúng entity từ người quản lý.

Script training sẽ:

- log metric validation, train loss và learning rate sau mỗi epoch;
- ghi cấu hình dataset/model/imgsz/epochs/batch/seed;
- lưu `best.pt`, `last.pt` và `epoch*.pt` thành W&B model artifact khi train kết
  thúc;
- vẫn giữ toàn bộ output cục bộ trong `runs/chvg4/`.

## 9. Smoke train trước

Smoke train dùng để kiểm tra dataset, CUDA, W&B và quyền ghi output. Nó không
phải kết quả báo cáo cuối.

```powershell
.\bytetrack_ppe\.venv\Scripts\python.exe `
  .\scripts\train\train_chvg4.py `
  --data ".\configs\data_4class.yaml" `
  --model yolov8l.pt `
  --imgsz 640 `
  --epochs 2 `
  --batch -1 `
  --device 0 `
  --workers 4 `
  --save-period 1 `
  --name smoke_yolov8l_640
```

Smoke train đúng phải:

- nhận đủ 4 class;
- đọc được train/val;
- CUDA không báo out-of-memory;
- W&B có run `smoke_yolov8l_640`;
- tạo `runs/chvg4/smoke_yolov8l_640/weights/best.pt`;
- không sửa dataset.

Mỗi lần chạy phải dùng `--name` mới vì script đặt `exist_ok=False` để tránh ghi
đè thí nghiệm cũ.

## 10. Train baseline chính thức

Lệnh khuyến nghị:

```powershell
.\bytetrack_ppe\.venv\Scripts\python.exe `
  .\scripts\train\train_chvg4.py `
  --data ".\configs\data_4class.yaml" `
  --model yolov8l.pt `
  --imgsz 640 `
  --epochs 100 `
  --batch -1 `
  --device 0 `
  --workers 4 `
  --seed 42 `
  --patience 30 `
  --save-period 10 `
  --wandb-project chvg4-ppe `
  --name yolov8l_640_baseline
```

Giải thích tham số:

| Tham số | Ý nghĩa |
|---|---|
| `--model` | Weight khởi tạo; `yolov8l.pt` là fine-tune từ pretrained COCO. |
| `--imgsz 640` | Input 640×640 để giữ chi tiết vật thể nhỏ tốt hơn 416. |
| `--epochs 100` | Số epoch tối đa. Early stopping có thể dừng sớm. |
| `--batch -1` | Ultralytics tự chọn batch phù hợp VRAM. |
| `--device 0` | Dùng GPU CUDA số 0. |
| `--workers 4` | Số worker đọc dữ liệu; giảm nếu Windows bị lỗi multiprocessing. |
| `--seed 42` | Cố định seed để dễ tái lập. |
| `--patience 30` | Dừng nếu metric không cải thiện trong 30 epoch. |
| `--save-period 10` | Lưu checkpoint epoch 10, 20, 30... |
| `--name` | Tên folder local và W&B run; phải duy nhất. |

Nếu không dùng W&B:

```powershell
# Thêm vào lệnh training
--no-wandb
```

## 11. Chọn cấu hình theo VRAM

Ưu tiên giữ `imgsz=640`. Khi thiếu VRAM:

1. Giữ `--batch -1` để tự chọn.
2. Nếu vẫn lỗi, dùng batch cụ thể nhỏ hơn như `--batch 4` hoặc `--batch 2`.
3. Chỉ giảm model từ `yolov8l.pt` xuống `yolov8m.pt` khi GPU không đáp ứng.
4. Không âm thầm giảm `imgsz` xuống 416; nếu phải giảm thì ghi thành thí nghiệm
   riêng vì kính/mũ/head nhỏ có thể mất recall.

Ví dụ GPU nhỏ:

```powershell
.\bytetrack_ppe\.venv\Scripts\python.exe `
  .\scripts\train\train_chvg4.py `
  --data ".\configs\data_4class.yaml" `
  --model yolov8m.pt `
  --imgsz 640 `
  --epochs 100 `
  --batch 4 `
  --device 0 `
  --name yolov8m_640_batch4
```

Không dùng cùng `--name` cho hai cấu hình khác nhau.

## 12. Theo dõi metric trong W&B

Không chọn model chỉ dựa trên mAP50 tổng. Cần xem đồng thời:

- `mAP50-95` tổng và từng class;
- precision/recall từng class;
- recall của `person`, vì ByteTrack cần person detection liên tục;
- recall của `head` và `helmet`, vì hai vật thể này thường nhỏ;
- recall/false positive của `vest`, vì Event Engine dùng absence của vest;
- train/validation loss để phát hiện overfitting;
- confusion matrix và các prediction sample;
- thời gian mỗi epoch và VRAM.

Dấu hiệu cần kiểm tra:

- train loss giảm nhưng validation metric xấu dần: có thể overfit;
- mAP tổng tốt nhưng `head` recall thấp: model chưa phù hợp Event Engine;
- `person` bỏ sót ở vùng xa hoặc gần trụ: cần xem lại data coverage/tile inference;
- `vest` false positive nhiều: no-vest event có thể bị che mất;
- metric dao động mạnh: dataset nhỏ hoặc batch quá nhỏ.

## 13. Output training

Một run thường có cấu trúc:

```text
runs/chvg4/<run_name>/
├── args.yaml
├── results.csv
├── results.png
├── confusion_matrix.png
├── PR_curve.png
├── F1_curve.png
├── labels.jpg
├── train_batch*.jpg
├── val_batch*_pred.jpg
└── weights/
    ├── best.pt
    ├── last.pt
    └── epoch*.pt
```

- `best.pt`: checkpoint có fitness validation tốt nhất theo Ultralytics.
- `last.pt`: trạng thái epoch cuối, phù hợp cho mục đích tiếp tục/debug.
- `epoch*.pt`: mốc trung gian do `--save-period` tạo.
- `results.csv`: metric theo epoch, dùng để audit ngoài W&B.

Không commit `runs/`, `wandb/` hoặc `.pt` vào Git.

## 14. Kiểm tra checkpoint trước khi bàn giao

Đọc class metadata:

```powershell
.\bytetrack_ppe\.venv\Scripts\python.exe -c "from ultralytics import YOLO; m=YOLO(r'runs\chvg4\yolov8l_640_baseline\weights\best.pt'); print(m.names)"
```

Kết quả bắt buộc:

```text
{0: 'person', 1: 'head', 2: 'helmet', 3: 'vest'}
```

Tính SHA-256:

```powershell
Get-FileHash `
  -Algorithm SHA256 `
  -LiteralPath ".\runs\chvg4\yolov8l_640_baseline\weights\best.pt"
```

Ghi lại:

- W&B project/run URL;
- Git commit dùng để train;
- dataset validation report;
- model nền;
- imgsz, epochs, batch, seed;
- best epoch;
- mAP50, mAP50-95 và recall từng class;
- filename, dung lượng, SHA-256.

## 15. Đưa checkpoint vào backend

Chỉ sau khi model được duyệt, copy nó vào release path:

```powershell
Copy-Item `
  -LiteralPath ".\runs\chvg4\yolov8l_640_baseline\weights\best.pt" `
  -Destination ".\bytetrack_ppe\weights\candidates\CHVG4-best.pt"
```

Weight nằm ngoài Git nên việc copy này không tạo file commit.

Chạy verifier:

```powershell
Set-Location .\bytetrack_ppe
.\.venv\Scripts\python.exe .\verify_install.py
Set-Location ..
```

Verifier chỉ PASS khi checkpoint tồn tại và class order đúng chính xác bốn class.

## 16. Smoke test checkpoint trên video

Chạy 60 frame trong isolated run, không publish:

```powershell
Set-Location .\bytetrack_ppe
.\.venv\Scripts\python.exe .\run_pipeline_safe.py `
  --video "C:\path\to\test.mp4" `
  --model ".\weights\candidates\CHVG4-best.pt" `
  --max-frames 60 `
  --tracking-mode bytetrack `
  --conf 0.10 `
  --iou 0.70 `
  --ppe-assoc-conf 0.20 `
  --tile-rows 1 `
  --tile-cols 1 `
  --device cpu `
  --display-mode clean `
  --run-name chvg4_smoke_60f
Set-Location ..
```

Kiểm tra:

- video annotated mở được;
- person detection không bị bỏ sót liên tục;
- track ID tương đối ổn định;
- association chỉ có head/helmet/vest;
- `track_ppe_rows.csv` không có `glass_conf`;
- `frame_metrics.csv` không có `glass_detections`;
- `summary.json` không có association count `glass`;
- Event Engine hoàn tất mà không lỗi.

Output nằm tại:

```text
bytetrack_ppe/outputs/runs/chvg4_smoke_60f/
```

Không dùng `--publish` ở smoke test.

## 17. Full video và ablation

Sau khi smoke test đạt yêu cầu, chạy toàn video:

```powershell
Set-Location .\bytetrack_ppe
.\.venv\Scripts\python.exe .\run_pipeline_safe.py `
  --video "C:\path\to\test.mp4" `
  --model ".\weights\candidates\CHVG4-best.pt" `
  --max-frames 0 `
  --tracking-mode bytetrack `
  --conf 0.10 `
  --iou 0.70 `
  --ppe-assoc-conf 0.20 `
  --tile-rows 1 `
  --tile-cols 1 `
  --device cpu `
  --display-mode clean `
  --run-name chvg4_full_video
Set-Location ..
```

Chỉ publish sau khi đã xem video, event, evidence và output CSV/JSON. Ablation
Detection-only/ByteTrack/Event Engine phải dùng cùng detection cache để so sánh
công bằng; xem `bytetrack_ppe/ABLATION_PROTOCOL.md`.

Kết quả Phase 2 dùng checkpoint 5-class là số liệu lịch sử. Không được trình bày
nó như kết quả của model 4-class mới.

## 18. Checklist bàn giao

Người train bàn giao riêng các file sau:

- `CHVG4-best.pt`;
- SHA-256 của weight;
- W&B run URL;
- `results.csv`;
- `args.yaml`;
- confusion matrix và PR/F1 curves;
- metric tổng và metric từng class;
- validation report của dataset;
- ghi chú GPU, CUDA, thời gian train và lỗi/điều chỉnh nếu có.

Checklist cuối:

- [ ] Dataset validation là PASS.
- [ ] Dataset có đúng train/val/test đã khóa.
- [ ] Model có đúng 4 class và đúng thứ tự ID.
- [ ] W&B hoặc local results có đủ metric theo epoch.
- [ ] `best.pt`, `last.pt`, checkpoint định kỳ đã được lưu.
- [ ] SHA-256 đã ghi nhận.
- [ ] `verify_install.py` PASS.
- [ ] Smoke video 60 frame PASS.
- [ ] Full video đã được xem thủ công.
- [ ] Ablation mới không trộn với số liệu Phase 2 5-class.
- [ ] Không đưa dataset, video, `.pt`, `runs/`, `wandb/` hoặc secret lên Git.

## 19. Lỗi thường gặp

### `Missing validation report`

Đang train trên dataset chưa qua converter/validator, hoặc `path` trong YAML sai.
Chạy lại phần 6–7 và không tự tạo report giả.

### `Dataset must use exactly nc=4`

Đang dùng YAML 8-class/5-class cũ hoặc class order sai. Dùng
`configs/data_4class.yaml` hoặc YAML được converter sinh ra.

### CUDA out of memory

Giảm batch trước; sau đó cân nhắc YOLOv8m. Giữ `imgsz=640` nếu có thể.

### W&B không đăng nhập

Chạy lại `wandb.exe login`, kiểm tra Internet/proxy hoặc dùng `--no-wandb` cho
smoke test. Không gửi API key qua Git.

### Run folder đã tồn tại

Đổi `--name`. Script cố ý không ghi đè một thí nghiệm cũ.

### `verify_install.py` báo thiếu model

Copy checkpoint đã duyệt thành đúng tên
`bytetrack_ppe/weights/candidates/CHVG4-best.pt`.

### `verify_install.py` báo sai schema

Không đổi tên model 3-class/5-class để né lỗi. Train hoặc xin đúng checkpoint
4-class.
