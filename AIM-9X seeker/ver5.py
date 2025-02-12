import cv2
import numpy as np
import time

# 로그 파일 초기화
log_file = open("log_file.txt", "w", encoding="utf-8")

def write_log(message):
    timestamp = time.strftime("[%Y-%m-%d %H:%M:%S]")
    log_file.write(f"{timestamp} {message}\n")
    log_file.flush()

cap = cv2.VideoCapture(0)

# 추적기 두 개를 병렬로 생성 (KCF와 CSRT)
tracker_kcf = cv2.legacy.TrackerKCF_create()
tracker_csrt = cv2.legacy.TrackerCSRT_create()

# 상태 변수들
active_tracking = False
roi = None
selection = False
target_images = []
template_matching_state = False
template_matching_attempts = 0
lock_on_counter = 0

# 칼만 필터 초기화 (위치와 속도 포함)
kalman = cv2.KalmanFilter(6, 3)
kalman.measurementMatrix = np.eye(3, 6, dtype=np.float32)
kalman.transitionMatrix = np.array([
    [1, 0, 0, 1, 0, 0],
    [0, 1, 0, 0, 1, 0],
    [0, 0, 1, 0, 0, 1],
    [0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 1]
], dtype=np.float32)

# 디버깅 변수
fps_counter = 0
start_time = time.time()
wing_angle = {"top": 0, "bottom": 0, "left": 0, "right": 0}
missile_position = [250, 250]

def calculate_wing_angle(target_x, target_y, frame_center_x, frame_center_y):
    """타겟과 화면 중심의 오프셋을 기준으로 미사일 날개의 각도를 조정"""
    offset_x = target_x - frame_center_x
    offset_y = target_y - frame_center_y

    # 각도 계산 및 조정
    wing_adjustment_x = offset_x * 0.05
    wing_adjustment_y = offset_y * 0.05

    wing_angle["top"] = wing_adjustment_x
    wing_angle["bottom"] = -wing_adjustment_x
    wing_angle["left"] = -wing_adjustment_y
    wing_angle["right"] = wing_adjustment_y

    # 미사일 위치 업데이트
    missile_position[0] += int(wing_adjustment_x)
    missile_position[1] += int(wing_adjustment_y)

    # 화면 경계를 벗어나지 않도록 제한
    missile_position[0] = max(0, min(missile_position[0], frame.shape[1]))
    missile_position[1] = max(0, min(missile_position[1], frame.shape[0]))

def select_roi(event, x, y, flags, param):
    global active_tracking, roi, selection, template_matching_state, lock_on_counter

    if event == cv2.EVENT_LBUTTONDOWN:
        roi = (x, y, x, y)
        selection = True
    elif event == cv2.EVENT_MOUSEMOVE and selection:
        roi = (roi[0], roi[1], x, y)
    elif event == cv2.EVENT_LBUTTONUP:
        selection = False
        if roi is not None:
            (x1, y1, x2, y2) = roi
            if abs(x2 - x1) < 20 or abs(y2 - y1) < 20:
                write_log("[ERROR] ROI 크기가 너무 작습니다.")
                print("[ERROR] ROI 크기가 너무 작습니다.")
                return

            roi_rect = (x1, y1, abs(x2 - x1), abs(y2 - y1))

            # 두 추적기 모두 초기화
            tracker_kcf.init(frame, roi_rect)
            tracker_csrt.init(frame, roi_rect)

            write_log(f"ROI 선택됨: {roi_rect}")
            active_tracking = True
            template_matching_state = False
            lock_on_counter = 1

            initial_target = frame[y1:y2, x1:x2].copy()

            # 이미지가 비어있는지 확인
            if initial_target.size == 0:
                write_log("[ERROR] 비어 있는 이미지가 선택되었습니다.")
                print("[ERROR] 비어 있는 이미지가 선택되었습니다.")
                return

            target_images.clear()
            target_images.extend(create_rotated_templates(initial_target))
            target_images.extend(create_scaled_templates(initial_target))
            roi = None

def create_rotated_templates(image):
    """이미지를 회전하여 여러 각도의 템플릿 생성"""
    if image.size == 0:
        write_log("[ERROR] 회전 템플릿 생성 중 비어 있는 이미지 발견")
        return []

    templates = []
    for angle in range(0, 360, 45):
        matrix = cv2.getRotationMatrix2D((image.shape[1] // 2, image.shape[0] // 2), angle, 1.0)
        rotated = cv2.warpAffine(image, matrix, (image.shape[1], image.shape[0]))
        templates.append(rotated)
    return templates

def create_scaled_templates(image):
    """이미지를 다른 크기로 조정하여 여러 템플릿 생성"""
    if image.size == 0:
        write_log("[ERROR] 크기 조정 템플릿 생성 중 비어 있는 이미지 발견")
        return []

    scales = [0.8, 1.0, 1.2]
    return [cv2.resize(image, (0, 0), fx=s, fy=s) for s in scales]

def draw_wing(angles):
    """미사일 시뮬레이션 날개 각도 그리기"""
    canvas = np.zeros((500, 500, 3), dtype=np.uint8)
    center = tuple(missile_position)

    cv2.circle(canvas, center, 30, (0, 255, 0), -1)

    wing_position = {
        "top": (center[0], center[1] - 100),
        "bottom": (center[0], center[1] + 100),
        "left": (center[0] - 100, center[1]),
        "right": (center[0] + 100, center[1])
    }

    for position, angle in angles.items():
        end_point = rotate_point(center, wing_position[position], angle)
        cv2.line(canvas, center, end_point, (255, 255, 255), 3)
        cv2.putText(canvas, f"{position}: {angle:.1f} deg", (end_point[0] - 50, end_point[1] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return canvas

def rotate_point(origin, point, angle):
    angle_rad = np.radians(angle)
    ox, oy = origin
    px, py = point

    qx = ox + int(np.cos(angle_rad) * (px - ox) - np.sin(angle_rad) * (py - oy))
    qy = oy + int(np.sin(angle_rad) * (px - ox) + np.cos(angle_rad) * (py - oy))

    return qx, qy

def draw_debug_info():
    """FPS 및 추적 상태 디버깅 정보 표시"""
    global fps_counter

    fps_counter += 1
    elapsed_time = time.time() - start_time
    fps = fps_counter / elapsed_time

    tracking_status = "Tracking Active" if active_tracking else "Tracking Lost"
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(frame, tracking_status, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    cv2.putText(frame, f"Template Match Attempts: {template_matching_attempts}", (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

cv2.namedWindow("camera tracking window")
cv2.setMouseCallback("camera tracking window", select_roi)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_center_x = frame.shape[1] // 2
    frame_center_y = frame.shape[0] // 2

    success_kcf, bbox_kcf = tracker_kcf.update(frame)
    success_csrt, bbox_csrt = tracker_csrt.update(frame)

    if success_kcf or success_csrt:
        bbox = bbox_kcf if success_kcf else bbox_csrt
        x, y, w, h = [int(v) for v in bbox]

        predicted = kalman.predict()
        measurement = np.array([[np.float32(x + w // 2)], [np.float32(y + h // 2)], [0]], dtype=np.float32)
        kalman.correct(measurement)

        calculate_wing_angle(x + w // 2, y + h // 2, frame_center_x, frame_center_y)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    else:
        write_log("추적 실패: 타겟 재탐색 중")
    
    draw_debug_info()
    sim_canvas = draw_wing(wing_angle)

    cv2.imshow("camera tracking window", frame)
    cv2.imshow("missile window", sim_canvas)

    key = cv2.waitKey(16) & 0xFF
    if key == ord('q'):
        break
    
    elif key == ord('x'):
        tracker = cv2.legacy.TrackerCSRT_create()
        active_tracking = False
        template_matching_state = False


cap.release()
log_file.close()
cv2.destroyAllWindows()
