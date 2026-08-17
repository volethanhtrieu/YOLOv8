# Variant C - Tracking + PPE Association

## 1. Objective

Variant C implements the backend pipeline for:

**YOLO Detection -> ByteTrack -> PPE-to-Person Association**

The purpose of this variant is to evaluate whether detected PPE items can be correctly associated with the corresponding tracked person.

This module does **not** train a YOLO model and does **not** include an Event Engine.

---

## 2. Variant Configuration

| Component | Variant C |
|---|---|
| YOLO Detection | Yes |
| ByteTrack | Yes |
| PPE Association | Yes |
| Event Engine | No |

---

## 3. Model

Variant C uses pretrained YOLO checkpoints provided by the team:

- `C-N0-coco-best.pt`
- `C-N0-scratch-best.pt`

Model classes:

```text
0: person
1: head
2: helmet
3: vest
4: glass