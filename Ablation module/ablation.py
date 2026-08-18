import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import cv2
from ultralytics import YOLO


# ABLATION 
USE_TRACKING = True          # True: Dùng ByteTrack | False: YOLO thuần
USE_ASSOCIATION = True       # True: Ghép PPE vào Người | False: Không ghép
USE_EVENT_ENGINE = True      # True: Dùng độ trễ thời gian | False: Báo ngay lập tức
TIME_THRESHOLD = 1.0         # Ngưỡng thời gian (giây): 0.5, 1.0, 2.0...


# CẤU HÌNH ĐƯỜNG DẪN VÀ THÔNG SỐ
VIDEO_PATH = "video02.mp4"    # ĐỔI TÊN CHO KHỚP VỚI VIDEO
MODEL_PATH = "S-N0-coco-best.pt"
FPS = 30 
BUFFER_SIZE = int(TIME_THRESHOLD * FPS)


# 0 là Người (Person), 1 là Mũ (Helmet), 2 là Áo (Vest)
PERSON_CLASS = 0
HELMET_CLASS = 2

# Khởi tạo mô hình
model = YOLO(MODEL_PATH)
cap = cv2.VideoCapture(VIDEO_PATH)

history_buffer = {} # Bộ nhớ của Event Engine

def associate_ppe(persons, helmets):
    """Ghép Mũ vào đúng Người dựa trên khoảng cách từ đáy Mũ đến viền trên (Vai) của Người"""
    status = {p['id']: False for p in persons} 
    
    for h in helmets:
        hx1, hy1, hx2, hy2 = h['box']
        hc_x = (hx1 + hx2) / 2 # Tọa độ X tâm của mũ
        hc_y = (hy1 + hy2) / 2 # Tọa độ Y tâm của mũ
        
        best_person_id = -1
        min_distance = float('inf')
        
        for p in persons:
            p_id = p['id']
            px1, py1, px2, py2 = p['box']
            pc_x = (px1 + px2) / 2 # Trục xương sống của người
            
            w_person = px2 - px1
            h_person = py2 - py1
            
            # Tâm mũ phải nằm trong khoảng chiều rộng của người (có nới rộng 30% phòng sai số)
            cond_x = (px1 - w_person * 0.3) <= hc_x <= (px2 + w_person * 0.3)
            
            # Tâm mũ phải nằm ở khu vực nửa trên ngực, hoặc trôi nổi trên đầu (không quá cao)
            cond_y = (py1 - h_person * 0.8) <= hc_y <= (py1 + h_person * 0.3)
            
            if cond_x and cond_y:
                # Mũ có nằm thẳng với trục xương sống ? (dx)
                dx = abs(hc_x - pc_x)
                
                # Đáy mũ có nằm sát với đường viền của người ? (dy)
                dy = abs(hy2 - py1) 
                
                # Tổng khoảng cách càng nhỏ -> Khả năng mũ của người này càng cao
                distance = dx + dy
                
                if distance < min_distance:
                    min_distance = distance
                    best_person_id = p_id
                    
        # Gán mũ 
        if best_person_id != -1:
            status[best_person_id] = True
            
    return status

# XỬ LÝ VIDEO
print("Bắt đầu chạy nhận diện...")
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Đã chạy hết video!")
        break

    frame = cv2.resize(frame, (1024, 768))

    # Có Tracking hoặc Không Tracking
    if USE_TRACKING:
        results = model.track(frame, persist=True, tracker="bytetrack.yaml", verbose=False)
    else:
        results = model(frame, verbose=False)
        
    boxes = results[0].boxes
    persons = []
    helmets = []
    
    # Bóc tách dữ liệu Bounding Box
    if boxes is not None:
        for idx, box in enumerate(boxes):
            cls_id = int(box.cls[0])
            coords = box.xyxy[0].cpu().numpy()
            
            # Lấy ID (nếu không có tracking thì gán tạm ID = -1)
            track_id = int(box.id[0]) if (box.id is not None and USE_TRACKING) else f"no_id_{idx}"
            
            if cls_id == PERSON_CLASS:
                persons.append({'id': track_id, 'box': coords})
            elif cls_id == HELMET_CLASS:
                helmets.append({'box': coords})

    # ASSOCIATION
    if USE_ASSOCIATION and USE_TRACKING:
        person_status = associate_ppe(persons, helmets)
    else:
        person_status = {p['id']: False for p in persons} # Mặc định coi như lỗi

    # EVENT ENGINE 
    alerts = [] 
    
    if USE_EVENT_ENGINE and USE_TRACKING:
        for p_id, has_helmet in person_status.items():
            if p_id not in history_buffer:
                history_buffer[p_id] = []
                
            history_buffer[p_id].append(has_helmet)
            
            # Cắt bớt lịch sử cho đúng kích thước bộ nhớ
            if len(history_buffer[p_id]) > BUFFER_SIZE:
                history_buffer[p_id].pop(0)
                
            # Nếu toàn bộ n frame gần nhất đều KHÔNG CÓ MŨ -> Báo động
            if len(history_buffer[p_id]) == BUFFER_SIZE:
                if not any(history_buffer[p_id]):
                    alerts.append(p_id)
    else:
        # tắt Event Engine, không thấy mũ 1 frame là báo động ngay
        for p_id, has_helmet in person_status.items():
            if not has_helmet:
                alerts.append(p_id)

    # KẾT QUẢ TRỰC QUAN
    annotated_frame = results[0].plot() 
    
    # In danh sách ID vi phạm lên góc trái màn hình
    if alerts:
        cv2.putText(annotated_frame, f"VI PHAM (Khong Mu): ID {alerts}", (30, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

    # Thu nhỏ video
    cv2.imshow("He Thong Giam Sat PPE - Ablation", annotated_frame)
    
    # Bấm phím 'q' để thoát video giữa chừng
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


"""

HƯỚNG DẪN CHẠY CODE VÀ LẤY SỐ LIỆU ABLATION STUDY

1. Chuẩn bị trước khi chạy:
   - Mở phần mềm Anaconda Prompt.
   - Gõ lệnh kích hoạt môi trường ảo: conda activate ml (hoặc tên môi trường của bạn).
   - Dùng lệnh cd di chuyển đến thư mục chứa file này (VD: cd D:\DoAn_PPE).
   - Đảm bảo file AI (S-N0-coco-best.pt) và file video cùng nằm ở thư mục này.

2. Cấu hình Ablation Study (Thay đổi ở phần # ABLATION dòng 8-11):
   - Thay đổi các biến USE_TRACKING, USE_ASSOCIATION, USE_EVENT_ENGINE 
     thành True hoặc False để kiểm tra tác dụng của từng module.

3. Chạy chương trình:
   - Tại cửa sổ Anaconda Prompt, gõ lệnh: python ablation.py
   - Khi video đang chạy, nhấn phím 'q' trên bàn phím để tắt nhanh.
"""