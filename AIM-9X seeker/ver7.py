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

# 추적기 두 개를 병렬로 생성 (CSRT와 MOSSE)
tracker_csrt = None
tracker_mosse = None

# 상태 변수들
active_tracking = False
roi = None
selection = False

# 미사일 관련 변수들
wing_angle = {"top": 0, "bottom": 0, "left": 0, "right": 0}
missile_position = [250, 250]

# 칼만 필터 초기화 (위치와 크기 포함)
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

fps_counter = 0
start_time = time.time()

def reset_trackers():
    """추적기 및 상태 초기화"""
    global tracker_csrt, tracker_mosse, active_tracking
    tracker_csrt = cv2.legacy.TrackerCSRT_create()
    tracker_mosse = cv2.legacy.TrackerMOSSE_create()
    active_tracking = False
    write_log("추적기 초기화 완료")

def calculate_wing_angle(target_x, target_y, frame_center_x, frame_center_y):
    """타겟과 화면 중심의 오프셋을 기준으로 미사일 날개의 각도를 조정"""
    offset_x = target_x - frame_center_x
    offset_y = target_y - frame_center_y

    # 각도 계산 및 조정 (조정 폭 증가)
    wing_adjustment_x = offset_x * 0.1
    wing_adjustment_y = offset_y * 0.1

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

    # 로그 기록
    write_log(f"미사일 위치: {missile_position}, 날개 각도: {wing_angle}")

def draw_wing(angles):
    """미사일과 날개를 시뮬레이션 화면에 그리기"""
    canvas = np.zeros((500, 500, 3), dtype=np.uint8)
    center = tuple(missile_position)

    # 미사일 본체 그리기
    cv2.circle(canvas, center, 30, (0, 255, 0), -1)

    # 날개 위치 설정
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
    """점 회전 함수"""
    angle_rad = np.radians(angle)
    ox, oy = origin
    px, py = point

    qx = ox + int(np.cos(angle_rad) * (px - ox) - np.sin(angle_rad) * (py - oy))
    qy = oy + int(np.sin(angle_rad) * (px - ox) + np.cos(angle_rad) * (py - oy))

    return qx, qy

def select_roi(event, x, y, flags, param):
    global active_tracking, roi, selection

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
                return

            roi_rect = (x1, y1, abs(x2 - x1), abs(y2 - y1))
            reset_trackers()
            tracker_csrt.init(frame, roi_rect)
            tracker_mosse.init(frame, roi_rect)
            write_log(f"ROI 선택됨: {roi_rect}")
            active_tracking = True

def draw_debug_info():
    global fps_counter

    fps_counter += 1
    elapsed_time = time.time() - start_time
    fps = fps_counter / elapsed_time

    tracking_status = "Tracking Active" if active_tracking else "Tracking Lost"
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(frame, tracking_status, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

cv2.namedWindow("camera tracking window")
cv2.setMouseCallback("camera tracking window", select_roi)

reset_trackers()  # 초기 추적기 설정

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_center_x = frame.shape[1] // 2
    frame_center_y = frame.shape[0] // 2

    if active_tracking:
        success_csrt, bbox_csrt = tracker_csrt.update(frame)
        success_mosse, bbox_mosse = tracker_mosse.update(frame)

        if success_csrt or success_mosse:
            bbox = bbox_csrt if success_csrt else bbox_mosse
            x, y, w, h = [int(v) for v in bbox]

            predicted = kalman.predict()
            measurement = np.array([[np.float32(x + w // 2)], [np.float32(y + h // 2)], [np.float32(w * h)]], dtype=np.float32)
            kalman.correct(measurement)

            calculate_wing_angle(x + w // 2, y + h // 2, frame_center_x, frame_center_y)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        else:
            write_log("추적 실패: 타겟 재탐색 중")
            active_tracking = False

    draw_debug_info()
    sim_canvas = draw_wing(wing_angle)

    cv2.imshow("camera tracking window", frame)
    cv2.imshow("missile window", sim_canvas)

    key = cv2.waitKey(16) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('x'):
        write_log("사용자가 락온 해제")
        reset_trackers()

cap.release()
log_file.close()
cv2.destroyAllWindows()
