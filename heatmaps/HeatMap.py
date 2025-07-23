import cv2
import numpy as np
from ultralytics import YOLO
import os
from datetime import datetime, timedelta
import json
import atexit
from PIL import Image, ImageDraw
from collections import defaultdict

CONFIG_PATH = "alert_config.json"
CONFIG_STATUS = 0
HEAT_INCREMENT = 50
FIXED_SIZE = (1032, 616)
CONFIDENCE_THRESHOLD = 0.4
DUPLICATE_DISTANCE_THRESHOLD = 80
MIN_MOVEMENT_THRESHOLD = 60
BLOB_THRESHOLD_DURATION = 10
BLOB_LOG_DISTANCE_THRESHOLD = 20
STEP_DISTANCE = 20  # Movement threshold to start new blob

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
last_seen_time = {}
temp_blob_log = []
previous_logged_count = -1
alert_triggered = False
alert_zone_person = 0
last_logged_blob_positions = {}
last_logged_blob_positions["last_logged_time"] = {}

os.makedirs("heatmaps", exist_ok=True)
person_json_path = "heatmaps/person_count_log.json"
blob_json_path = "heatmaps/heatmap_log.json"
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

def update_alertconfig(data):
    global CONFIG_STATUS
    with open(CONFIG_PATH, "w") as f:
        json.dump(data, f, indent=4)
        CONFIG_STATUS = 1

def deduplicate_points(points, threshold):
    unique = []
    for pt in points:
        if all(np.hypot(pt[0] - u[0], pt[1] - u[1]) > threshold for u in unique):
            unique.append(pt)
    return unique

def load_alert_config():
    with open(CONFIG_PATH, "r") as f:
        config = json.load(f)
    return config["alert_threshold"], config["alert_zones"]

ALERT_THRESHOLD, zone = load_alert_config()

def save_blob_log():
    if temp_blob_log:
        with open(blob_json_path, "r") as f:
            existing = json.load(f)
        existing.extend(temp_blob_log)
        with open(blob_json_path, "w") as f:
            json.dump(existing, f, indent=4)
        temp_blob_log.clear()
        
def get_today_max_person_count_by_30min():
    with open(person_json_path, "r") as f:
        data = json.load(f)

    today = datetime.now().date()
    interval_counts = defaultdict(int)

    # Filter only today's data between 9:00 and 21:00
    for entry in data:
        try:
            timestamp = datetime.strptime(entry["timestamp"], "%Y-%m-%d %H:%M:%S")
            if timestamp.date() == today and 9 <= timestamp.hour < 21:
                rounded = timestamp.replace(minute=(timestamp.minute // 30) * 30, second=0, microsecond=0)
                interval_counts[rounded] = max(interval_counts[rounded], entry["person_count"])
        except:
            continue

    # Fill all intervals from 9:00 to 21:00 with default 0
    intervals = []
    start_time = datetime.combine(today, datetime.min.time()).replace(hour=9)
    for i in range(24):  # 12 hours * 2 (30 min blocks)
        slot = start_time + timedelta(minutes=30 * i)
        intervals.append(interval_counts.get(slot, 0))

    return intervals

def get_max_person_count_by_30min():
    with open(person_json_path, "r") as f:
        data = json.load(f)

    time_buckets = {}
    for entry in data:
        try:
            ts = datetime.strptime(entry["timestamp"], "%Y-%m-%d %H:%M:%S")
            bucket_start = ts.replace(minute=(ts.minute // 30) * 30, second=0, microsecond=0)
            bucket_str = bucket_start.strftime("%Y-%m-%d %H:%M:%S")
            time_buckets[bucket_str] = max(time_buckets.get(bucket_str, 0), entry["person_count"])
        except:
            continue
    return time_buckets

atexit.register(save_blob_log)

def process_frame(frame):
    global last_positions, last_logged_blob_positions, ALERT_THRESHOLD, zone, CONFIG_STATUS
    frame = cv2.resize(frame, FIXED_SIZE)
    output = frame.copy()
    results = model(frame, verbose=False)[0]
    current_detections = []
    zone_count = 0
    current_time = datetime.now()
    new_last_positions = {}

    if CONFIG_STATUS == 1:
        ALERT_THRESHOLD, zone = load_alert_config()
        CONFIG_STATUS = 0

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

            # Existing: key = (0, idx)
            key = (0, idx)
            prev_pos = last_positions.get(key)
            current_pos = (cx, cy)

# Calculate movement distance
            moved_distance = np.hypot(cx - prev_pos[0], cy - prev_pos[1]) if prev_pos else float('inf')

# Always update last seen time
            last_seen_time[key] = current_time

# Increment heatmap at new location if moved enough
            if prev_pos is None or moved_distance >= STEP_DISTANCE:
    # Start a new heatmap blob
               position_duration[key] = 1  # Reset duration
               persistent_heatmap[cy, cx] = min(persistent_heatmap[cy, cx] + HEAT_INCREMENT, 255)
               temp_blob_log.append({
                 "timestamp": current_time.strftime("%Y-%m-%d %H:%M:%S"),
                 "camera_id": 0,
                 "x": cx,
                 "y": cy
                 })
            else:
    # If not moved enough, increase intensity gradually
             position_duration[key] += 1
            if position_duration[key] % 10 == 0:  # Every ~10 frames
               persistent_heatmap[cy, cx] = min(persistent_heatmap[cy, cx] + 10, 255)
               temp_blob_log.append({
            "timestamp": current_time.strftime("%Y-%m-%d %H:%M:%S"),
            "camera_id": 0,
            "x": cx,
            "y": cy
        })

            else:
                elapsed = (current_time - last_seen_time.get(key, current_time)).total_seconds()
                position_duration[key] = int(elapsed)

            new_last_positions[key] = (cx, cy)

            if zone:
                (xA, yA), (xB, yB) = zone
                if xA <= cx <= xB and yA <= cy <= yB:
                    zone_count += 1
            
            current_detections.append((cx, cy))
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)

            duration = position_duration.get(key, 0)
            if duration < 5:
                timer_color = (0, 255, 0)
            elif duration < 10:
                timer_color = (0, 255, 255)
            else:
                timer_color = (0, 0, 255)

            ##### corner
            cv2.rectangle(frame, (x1, y2-36), (x1+120, y2-2), (80,80,80), -1)

            ##### putText
            cv2.putText(frame, f" Idle {duration}s", (x1, y2-10), cv2.FONT_HERSHEY_DUPLEX, 0.7, timer_color, 2)
    
    last_positions = new_last_positions
    if zone:
        (xA, yA), (xB, yB) = zone
        cv2.rectangle(frame, (xA, yA), (xB, yB), (0, 0, 255), 2)

    return frame, current_detections, zone_count

def generate_processed_frames():
    global cam, previous_logged_count, alert_triggered, ALERT_THRESHOLD, alert_zone_person
    cam = cv2.VideoCapture(0)
    # cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1000)
    # cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)
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

        alert_zone_person = zone_count

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

def _get_filtered_heatmap_data(period):
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
        return None, None

    save_blob_log()

    with open(blob_json_path, "r") as f:
        blob_data = json.load(f)

    heatmap_data = np.zeros((h, w), dtype=np.float32)

    for item in blob_data:
        try:
            timestamp = datetime.strptime(item["timestamp"], "%Y-%m-%d %H:%M:%S")
            if start_time <= timestamp <= now:
                x, y = int(item["x"]), int(item["y"])
                if 0 <= x < w and 0 <= y < h:
                    heatmap_data[y, x] += HEAT_INCREMENT
        except:
            continue

    return heatmap_data, now

def generate_transparent_heatmap_report(period="today"):
    heatmap_data, _ = _get_filtered_heatmap_data(period)
    if heatmap_data is None:
        return None

    blurred = cv2.GaussianBlur(heatmap_data, (51, 51), 0)
    norm = cv2.normalize(blurred, None, 0, 255, cv2.NORM_MINMAX)
    color_map = cv2.applyColorMap(norm.astype(np.uint8), cv2.COLORMAP_JET)

    heatmap_rgba = np.zeros((h, w, 4), dtype=np.uint8)
    for y in range(h):
        for x in range(w):
            r, g, b = color_map[y, x]
            alpha = norm[y, x] if norm[y, x] > 0 else 0
            heatmap_rgba[y, x] = [r, g, b, alpha]

    final_img = Image.fromarray(heatmap_rgba, mode="RGBA")
    output_path = f"heatmaps/heatmap_report_{period}.png"
    final_img.save(output_path)
    return output_path

def generate_heatmap_on_background(period="today"):
    heatmap_data, _ = _get_filtered_heatmap_data(period)
    if heatmap_data is None:
        return None

    blurred = cv2.GaussianBlur(heatmap_data, (51, 51), 0)
    norm = cv2.normalize(blurred, None, 0, 255, cv2.NORM_MINMAX)
    color_map = cv2.applyColorMap(norm.astype(np.uint8), cv2.COLORMAP_JET)

    bg_copy = background_image.copy()
    overlay = cv2.addWeighted(bg_copy, 0.6, color_map, 0.4, 0)

    output_path = f"heatmaps/heatmap_report_on_background_{period}.png"
    cv2.imwrite(output_path, overlay)
    return output_path
