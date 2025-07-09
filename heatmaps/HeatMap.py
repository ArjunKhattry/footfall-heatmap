import cv2
import numpy as np
from ultralytics import YOLO
import os
from datetime import datetime, timedelta
import json
from PIL import Image, ImageDraw

CONFIG_PATH = "alert_config.json"
if not os.path.exists(CONFIG_PATH):
    default_config = {
        "alert_threshold": 1,
        "alert_zones": {
            "0": [[200, 140], [280, 260]]
        }
    }
    with open(CONFIG_PATH, "w") as f:
        json.dump(default_config, f, indent=4)

with open(CONFIG_PATH, "r") as f:
    config = json.load(f)

ALERT_THRESHOLD = config["alert_threshold"]
alert_zones = {int(k): v for k, v in config["alert_zones"].items()}

HEAT_INCREMENT = 20
FIXED_SIZE = (640, 480)
CONFIDENCE_THRESHOLD = 0.4
DUPLICATE_DISTANCE_THRESHOLD = 80
MIN_MOVEMENT_THRESHOLD = 45
BLOB_THRESHOLD_DURATION = 10
BLOB_LOG_DISTANCE_THRESHOLD = 10

MODEL_PATH = os.path.join(os.path.dirname(__file__), "yolov8n.pt")
model = YOLO(MODEL_PATH)

bg_image_path = "/home/cbginnovation/Desktop/cam0.png"
if not os.path.exists(bg_image_path):
    print(f"Error: Background image not found at {bg_image_path}")
    exit(1)
background_image = cv2.imread(bg_image_path)
background_image = cv2.resize(background_image, FIXED_SIZE)
h, w = background_image.shape[:2]

cam = None
heatmap = np.zeros((h, w), dtype=np.float32)
persistent_heatmap = np.zeros_like(heatmap)
position_duration = {}
last_positions = {}
last_logged_blob_positions = {}
last_logged_blob_positions["last_logged_time"] = {}
previous_logged_count = -1
alert_triggered = False

os.makedirs("heatmaps", exist_ok=True)
person_json_path = "heatmaps/person_count_log.json"
blob_json_path = "heatmaps/blob_coordinates.json"
for path in [person_json_path, blob_json_path]:
    if not os.path.exists(path):
        with open(path, "w") as f:
            json.dump([], f, indent=4)

def append_json(filepath, data_entry):
    with open(filepath, "r") as f:
        data = json.load(f)
    data.append(data_entry)
    with open(filepath, "w") as f:
        json.dump(data, f, indent=4)

def deduplicate_points(points, threshold):
    unique = []
    for pt in points:
        if all(np.hypot(pt[0] - u[0], pt[1] - u[1]) > threshold for u in unique):
            unique.append(pt)
    return unique

def process_frame(frame):
    global last_positions, last_logged_blob_positions
    frame = cv2.resize(frame, FIXED_SIZE)
    results = model(frame, verbose=False)[0]
    current_detections = []
    zone_count = 0

    zone = alert_zones.get(0)
    for idx, box in enumerate(results.boxes):
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        if cls_id == 0 and conf >= CONFIDENCE_THRESHOLD:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            if (x2 - x1) < 15 or (y2 - y1) < 15:
                continue

            cx = (x1 + x2) // 2
            cy = y2
            cx = min(max(cx, 0), w - 1)
            cy = min(max(cy, 0), h - 1)

            key = (0, idx)
            prev_pos = last_positions.get(key)
            dist = np.hypot(cx - prev_pos[0], cy - prev_pos[1]) if prev_pos else float('inf')

            if prev_pos is None or dist >= MIN_MOVEMENT_THRESHOLD:
                position_duration[key] = 1
            else:
                position_duration[key] = position_duration.get(key, 0) + 1

            last_positions[key] = (cx, cy)

            if position_duration[key] >= BLOB_THRESHOLD_DURATION:
                intensity = HEAT_INCREMENT
                persistent_heatmap[cy, cx] = min(persistent_heatmap[cy, cx] + intensity, 255)

                last_logged_pos = last_logged_blob_positions.get(key)
                last_logged_time = last_logged_blob_positions["last_logged_time"].get(key)
                current_time = datetime.now()

                should_log = False

                if last_logged_pos is None:
                    should_log = True
                elif np.hypot(cx - last_logged_pos[0], cy - last_logged_pos[1]) > BLOB_LOG_DISTANCE_THRESHOLD:
                    should_log = True
                elif last_logged_time is None or (current_time - last_logged_time).total_seconds() > 5:
                    should_log = True

                if should_log:
                    append_json(blob_json_path, {
                        "timestamp": current_time.strftime("%Y-%m-%d %H:%M:%S"),
                        "camera_id": 0,
                        "x": cx,
                        "y": cy
                    })
                    last_logged_blob_positions[key] = (cx, cy)
                    last_logged_blob_positions["last_logged_time"][key] = current_time

            if zone:
                (xA, yA), (xB, yB) = zone
                if xA <= cx <= xB and yA <= cy <= yB:
                    zone_count += 1

            current_detections.append((cx, cy))
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)

    if zone:
        (xA, yA), (xB, yB) = zone
        cv2.rectangle(frame, (xA, yA), (xB, yB), (0, 0, 255), 2)

    return frame, current_detections, zone_count

def generate_processed_frames():
    global cam, previous_logged_count, alert_triggered
    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        print("\u274c Webcam not accessible")
        return

    while True:
        ret, frame = cam.read()
        if not ret:
            continue

        processed, detections, zone_count = process_frame(frame)
        unique_detections = deduplicate_points(detections, DUPLICATE_DISTANCE_THRESHOLD)
        total_detected = len(unique_detections)

        blurred = cv2.GaussianBlur(persistent_heatmap.copy(), (51, 51), 0)
        norm = cv2.normalize(blurred, None, 0, 255, cv2.NORM_MINMAX)
        color = cv2.applyColorMap(norm.astype(np.uint8), cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(processed, 0.6, color, 0.4, 0)

        cv2.putText(overlay, f"Live Person Count: {total_detected}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)

        if zone_count > ALERT_THRESHOLD:
            alert_triggered = True
            cv2.putText(overlay, "ALERT: Overcrowding!", (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)
        else:
            alert_triggered = False

        if total_detected != previous_logged_count:
            append_json(person_json_path, {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "person_count": total_detected
            })
            previous_logged_count = total_detected

        ret, buffer = cv2.imencode('.jpg', overlay)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

def generate_transparent_heatmap_report(period="today"):
    blob_path = "heatmaps/blob_coordinates.json"
    output_path = f"heatmaps/transparent_heatmap_report_{period}.png"
    now = datetime.now()

    if period == "today":
        start_time = now.replace(hour=0, minute=0, second=0, microsecond=0)
    elif period == "yesterday":
        start_time = (now - timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
        now = start_time + timedelta(days=1)
    elif period == "weekly":
        start_time = now - timedelta(days=7)
    elif period == "monthly":
        start_time = now - timedelta(days=30)
    else:
        return None

    with open(blob_path, "r") as f:
        blob_data = json.load(f)

    transparent = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(transparent, 'RGBA')

    heat_counter = {}
    for item in blob_data:
        try:
            timestamp = datetime.strptime(item["timestamp"], "%Y-%m-%d %H:%M:%S")
            if start_time <= timestamp <= now:
                x, y = int(item["x"]), int(item["y"])
                key = (x, y)
                heat_counter[key] = heat_counter.get(key, 0) + 1
        except:
            continue

    for (x, y), count in heat_counter.items():
        r = 25
        alpha = min(40 + count * 12, 255)
        for i in range(5):
            draw.ellipse(
                [(x - r + i, y - r + i), (x + r - i, y + r - i)],
                fill=(255, 0, 0, alpha)
            )

    transparent.save(output_path)
    return output_path
