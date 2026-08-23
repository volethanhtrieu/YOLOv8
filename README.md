# CHVG4 PPE Safety Monitoring

Hệ thống phát hiện vi phạm PPE trong video công trường:

```text
YOLOv8 detection (person, head, helmet, vest)
→ ByteTrack chỉ theo dõi person
→ Association gắn PPE vào từng person track
→ Event Engine tổng hợp bằng chứng theo thời gian
→ Flask API + Streamlit dashboard + Human Review
```

Schema runtime hiện tại có đúng 4 class: `person`, `head`, `helmet`, `vest`.
Các màu mũ của CHVG được merge thành `helmet`; class `glass` bị loại khỏi
dataset, checkpoint, association, CSV, API và dashboard.

## Bắt đầu ở đâu

- Hướng dẫn đầy đủ từ dataset đến chạy ứng dụng: [`bytetrack_ppe/README.md`](bytetrack_ppe/README.md)
- Hướng dẫn riêng cho người train: [`TRAINING_GUIDE_CHVG4.md`](TRAINING_GUIDE_CHVG4.md)
- Converter/validator: `scripts/data/`
- Cấu hình dataset 4-class có thể commit: `configs/data_4class.yaml`
- Fine-tune + W&B: `scripts/train/train_chvg4.py`
- Backend chính: `bytetrack_ppe/run_tiled_ppe_pipeline_v3.py`
- Dataset sinh local: `data/processed/chvg4/` (không commit)
- Model sinh local: `runs/chvg4/` và `bytetrack_ppe/weights/candidates/` (không commit)

## Trạng thái hiện tại

- Dataset 4-class đã validation PASS trên 1.698 ảnh với split 1.358/170/170.
- Unit test dataset/backend và smoke test API/Job/Human Review đã PASS.
- Inference thật đang chờ checkpoint `CHVG4-best.pt` được fine-tune từ dataset
  4-class; checkpoint 5-class Phase 2 không tương thích với backend mới.
