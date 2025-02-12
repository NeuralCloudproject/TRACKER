import cv2
import numpy as np

cap = cv2.VideoCapture(0)

tracker = cv2.legacy.TrackerCSRT_create()

active_tracking = False
roi = None
selection = False
target_image = None
template_matching_state = False
template_matching_attempts = 0

wing_angle = {
    "top": 0,
    "bottom": 0,
    "left": 0,
    "right": 0
}

missile_position = [250, 250]

def select_roi(event, x, y, flags, param):
    global active_tracking, roi, selection, target_image, template_matching_state
    
    if event == cv2.EVENT_LBUTTONDOWN:
        roi = (x, y, x, y)
        selection = True
    elif event == cv2.EVENT_MOUSEMOVE and selection:
        roi = (roi[0], roi[1], x, y)
    elif event == cv2.EVENT_LBUTTONUP:
        selection = False
        if roi is not None:
            (x1, y1, x2, y2) = roi
            tracker.init(frame, (x1, y1, x2 - x1, y2 - y1))
            active_tracking = True
            template_matching_state = False
            
            target_image = frame[y1:y2, x1:x2].copy()
            roi = None

def calculate_wing_angle(target_x, target_y, frame_center_x, frame_center_y):
    offset_x = target_x - frame_center_x
    offset_y = target_y - frame_center_y

    wing_adjustment_x = offset_x * 0.05
    wing_adjustment_y = offset_y * 0.05

    wing_angle["top"] = wing_adjustment_x
    wing_angle["bottom"] = -wing_adjustment_x
    wing_angle["left"] = -wing_adjustment_y
    wing_angle["right"] = wing_adjustment_y

    missile_position[0] += int(wing_adjustment_x)
    missile_position[1] += int(wing_adjustment_y)

    missile_position[0] = max(0, min(missile_position[0], frame.shape[1]))
    missile_position[1] = max(0, min(missile_position[1], frame.shape[0]))

def draw_wing(angles):
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

def retracking():
    global active_tracking, template_matching_state, template_matching_attempts

    if target_image is not None:
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_template = cv2.cvtColor(target_image, cv2.COLOR_BGR2GRAY)

        result = cv2.matchTemplate(gray_frame, gray_template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)

        if max_val > 0.85:
            h, w = gray_template.shape[:2]
            tracker.init(frame, (max_loc[0], max_loc[1], w, h))
            active_tracking = True
            template_matching_state = False
            template_matching_attempts = 0
        else:
            template_matching_attempts += 1

cv2.namedWindow("camera tracking window")
cv2.setMouseCallback("camera tracking window", select_roi)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_center_x = frame.shape[1] // 2
    frame_center_y = frame.shape[0] // 2

    success = False
    if active_tracking:
        success, bbox = tracker.update(frame)

    if success:
        x, y, w, h = [int(v) for v in bbox]

        calculate_wing_angle(x + w // 2, y + h // 2, frame_center_x, frame_center_y)

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    else:
        if not template_matching_state:
            template_matching_state = True
        retracking()

    if template_matching_state:
        cv2.putText(frame, f"Template Matching count: attempt {template_matching_attempts}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    cv2.imshow("camera tracking window", frame)
    sim_canvas = draw_wing(wing_angle)
    cv2.imshow("missile window", sim_canvas)

    key = cv2.waitKey(16) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('x'):
        tracker = cv2.legacy.TrackerCSRT_create()
        active_tracking = False
        template_matching_state = False

cap.release()
cv2.destroyAllWindows()
