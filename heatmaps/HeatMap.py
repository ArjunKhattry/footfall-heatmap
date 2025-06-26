import cv2
import numpy as np
from ultralytics import YOLO
import os
from datetime import datetime
import json

# --- Load Configuration ---
CONFIG_PATH = "alert_config.json"
if not os.path.exists(CONFIG_PATH):
    default_config = {
        "alert_threshold": 1,
        "alert_zones": {
            "0": [[300, 140], [280, 260]],
            "1": [[240, 180], [360, 300]]
        }
    }
    with open(CONFIG_PATH, "w") as f:
        json.dump(default_config, f, indent=4)

with open(CONFIG_PATH, "r") as f:
    config = json.load(f)

ALERT_THRESHOLD = config["alert_threshold"]
alert_zones = {int(k): v for k, v in config["alert_zones"].items()}

# --- Configurable Thresholds ---
HEAT_INCREMENT = 25
FIXED_SIZE = (640, 480)
CONFIDENCE_THRESHOLD = 0.4
DUPLICATE_DISTANCE_THRESHOLD = 80
MIN_MOVEMENT_THRESHOLD = 45

# Load YOLOv8 model
MODEL_PATH = os.path.join(os.path.dirname(__file__), "yolov8n.pt")
model = YOLO(MODEL_PATH)

# Load predefined background images for each camera
bg_image_paths = ["/home/cbginnovation/Desktop/cam1.jpg", "/home/cbginnovation/Desktop/cam0.png"]
background_images = []
frame_widths = []

for i, path in enumerate(bg_image_paths):
    if not os.path.exists(path):
        print(f"Error: Background image not found at {path}")
        exit(1)
    bg = cv2.imread(path)
    bg = cv2.resize(bg, FIXED_SIZE)
    background_images.append(bg.copy())
    frame_widths.append(bg.shape[1])

combined_background = np.hstack(background_images)
h, w = combined_background.shape[:2]

# Initialize cameras
cams = [cv2.VideoCapture(0), cv2.VideoCapture(2)]
for i, cam in enumerate(cams):
    if not cam.isOpened():
        print(f"Error: Camera {i} not opened.")
        exit(1)
    else:
        print(f"✅ Camera {i} opened.")

heatmap = np.zeros((h, w), dtype=np.float32)
persistent_heatmap = np.zeros_like(heatmap)
position_duration = {}
last_positions = {}
frame_person_count = 0
previous_logged_count = -1

# --- Setup JSON Logging ---
os.makedirs("heatmaps", exist_ok=True)
timestamp_now = datetime.now().strftime('%Y%m%d_%H%M%S')

person_json_path = f"heatmaps/person_count_log_{timestamp_now}.json"
blob_json_path = f"heatmaps/blob_coordinates_{timestamp_now}.json"

with open(person_json_path, "w") as f:
    json.dump([], f, indent=4)

with open(blob_json_path, "w") as f:
    json.dump([], f, indent=4)

def append_json(filepath, data_entry):
    if os.path.exists(filepath):
        with open(filepath, "r") as f:
            data = json.load(f)
    else:
        data = []
    data.append(data_entry)
    with open(filepath, "w") as f:
        json.dump(data, f, indent=4)

def deduplicate_points(points, threshold):
    unique = []
    for pt in points:
        if all(np.hypot(pt[0]-u[0], pt[1]-u[1]) > threshold for u in unique):
            unique.append(pt)
    return unique

# --- Frame Processing ---
def process_frame(frame, offset_x, cam_index):
    global last_positions
    frame = cv2.resize(frame, FIXED_SIZE)
    results = model(frame, verbose=False)[0]
    current_detections = []
    zone_count = 0

    zone = alert_zones.get(cam_index)
    for idx, box in enumerate(results.boxes):
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        if cls_id == 0 and conf >= CONFIDENCE_THRESHOLD:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            if (x2 - x1) < 15 or (y2 - y1) < 15:
                continue

            cx = (x1 + x2) // 2 + offset_x
            cy = y2
            cy = min(max(cy, 0), persistent_heatmap.shape[0] - 1)
            cx = min(max(cx, 0), persistent_heatmap.shape[1] - 1)

            key = (cam_index, idx)
            prev_pos = last_positions.get(key)

            if prev_pos:
                dist = np.hypot(cx - prev_pos[0], cy - prev_pos[1])
            else:
                dist = float('inf')

            if prev_pos is None or dist >= MIN_MOVEMENT_THRESHOLD:
                position_duration[key] = 1
            else:
                position_duration[key] = position_duration.get(key, 0) + 1

            last_positions[key] = (cx, cy)
            intensity = HEAT_INCREMENT * position_duration[key]
            persistent_heatmap[cy, cx] = min(persistent_heatmap[cy, cx] + intensity, 255)

            append_json(blob_json_path, {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "camera_id": cam_index,
                "x": cx,
                "y": cy
            })

            if zone:
                (xA, yA), (xB, yB) = zone
                if xA + offset_x <= cx <= xB + offset_x and yA <= cy <= yB:
                    zone_count += 1

            current_detections.append((cx, cy))
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(frame, (cx - offset_x, cy), 4, (0, 0, 255), -1)

    if zone:
        (xA, yA), (xB, yB) = zone
        cv2.rectangle(frame, (xA, yA), (xB, yB), (0, 0, 255), 2)

    return frame, current_detections, zone_count

# --- Main Loop ---
while True:
    frames = []
    all_detections = []
    total_zone_count = 0

    for i, cam in enumerate(cams):
        ret, frame = cam.read()
        if not ret:
            continue
        processed, detections, zone_count = process_frame(frame, offset_x=sum(frame_widths[:i]), cam_index=i)
        frames.append(processed)
        all_detections.extend(detections)
        total_zone_count += zone_count

    if not frames:
        continue

    unique_detections = deduplicate_points(all_detections, DUPLICATE_DISTANCE_THRESHOLD)
    total_detected = len(unique_detections)

    combined = np.hstack(frames)
    blurred = cv2.GaussianBlur(persistent_heatmap.copy(), (51, 51), 0)
    norm = cv2.normalize(blurred, None, 0, 255, cv2.NORM_MINMAX)
    color = cv2.applyColorMap(norm.astype(np.uint8), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(combined, 0.6, color, 0.4, 0)

    cv2.putText(overlay, f"Live Person Count: {total_detected}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)

    if total_zone_count > ALERT_THRESHOLD:
        cv2.putText(overlay, "ALERT: Overcrowding!", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)
        print("🚨 ALERT: People exceeded limit in marked zone!")

    if total_detected != previous_logged_count:
        append_json(person_json_path, {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "person_count": total_detected
        })
        previous_logged_count = total_detected

    cv2.imshow("Persistent Footfall Heatmap (Press 'q' to Quit)", overlay)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()

# --- Final Output Generation ---
blurred_final = cv2.GaussianBlur(persistent_heatmap.copy(), (51, 51), 0)
normalized_final = cv2.normalize(blurred_final, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
color_map_final = cv2.applyColorMap(normalized_final, cv2.COLORMAP_JET)
final_overlay = cv2.addWeighted(combined_background, 0.6, color_map_final, 0.4, 0)

final_image_path = f"heatmaps/final_heatmap_on_background_{timestamp_now}.png"
cv2.imwrite(final_image_path, final_overlay)
print(f"\n✅ Final heatmap saved at: {final_image_path}")

for cam in cams:
    cam.release()
cv2.destroyAllWindows()
