from flask import Flask, jsonify, send_file, request, render_template, Response
from threading import Thread
import os
import json
from engine import HeatMap
import base64
import cv2
import numpy as np
from flask_cors import CORS
from datetime import datetime, timedelta

app = Flask(__name__)
CORS(app) # Enable CORS for all routes and origins

detection_running = False

@app.route('/ui')
def frontend():
    return render_template("index.html")

@app.route('/')
def home():
    return "✅ API is running. Use /start to begin detection, /heatmap to get image, /person_count for live count, /cumulative_count for total, /alert_status for alert, /heatmap_report to generate filtered report."

@app.route('/start', methods=['GET'])
def start_detection():
    global detection_running
    if detection_running:
        return jsonify({"status": "already running"})

    detection_running = True
    t = Thread(target=HeatMap.generate_processed_frames)
    t.daemon = True
    t.start()
    return jsonify({"status": "started", "message": "Heatmap detection is running in background."})

@app.route('/video_feed')
def video_feed():
    return Response(HeatMap.generate_processed_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/heatmap')
def get_heatmap():
    heatmap_dir = "heatmaps"
    if not os.path.exists(heatmap_dir):
        return jsonify({"error": "Heatmap directory not found"}), 404

    files = [f for f in os.listdir(heatmap_dir) if f.startswith("final_heatmap_on_background") and f.endswith(".png")]
    if not files:
        return jsonify({"error": "No heatmap image available yet"}), 404

    latest_file = max(files, key=lambda x: os.path.getctime(os.path.join(heatmap_dir, x)))
    with open(os.path.join(heatmap_dir, latest_file), "rb") as img_file:
        encoded_string = base64.b64encode(img_file.read()).decode('utf-8')

    return jsonify({"heatmap_base64": encoded_string})

@app.route('/person_count')
def get_person_count():
    filepath = "database/person_count_log.json"
    if not os.path.exists(filepath):
        return jsonify({"error": "person_count_log.json not found"}), 404

    try:
        with open(filepath, "r") as f:
            data = json.load(f)
        if not data:
            return jsonify({"error": "Log file is empty"}), 404
        return jsonify(data[-1])
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/cumulative_count')
def get_cumulative_count():
    filepath = "database/person_count_log.json"
    if not os.path.exists(filepath):
        return jsonify({"error": "person_count_log.json not found"}), 404

    try:
        with open(filepath, "r") as f:
            data = json.load(f)

        max_count = max((entry.get("person_count", 0) for entry in data), default=0)
        return jsonify({
            "cumulative_person_count": max_count,
            "entries_considered": len(data)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/alert_status')
def get_alert_status():
    try:
        status = HeatMap.alert_triggered
        return jsonify({"alert": status, "person_count": HeatMap.alert_zone_person})
    except AttributeError:
        return jsonify({"error": "Alert status not available"}), 500

@app.route('/heatmap_report')
def heatmap_report():
    period = request.args.get("period", "daily")
    try:
        out_path = HeatMap.generate_transparent_heatmap_report(period)
        with open(out_path, "rb") as f:
            encoded_string = base64.b64encode(f.read()).decode('utf-8')
        return jsonify({"heatmap_base64": encoded_string})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/download_report')
def download_report():
    period = request.args.get("period", "today")
    out_path = HeatMap.generate_heatmap_on_background(period)
    
    if not os.path.exists(out_path):
        return jsonify({"error": "Report not available to download."}), 404

    return send_file(out_path, as_attachment=True)

@app.route('/update_alert_config', methods=['POST'])
def update_alert_config():
    try:
        data = request.get_json()
        HeatMap.update_alertconfig(data)
        return jsonify({"message": "Alert configuration updated successfully."})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/max_count_by_30min')
def max_count_by_30min():
    filepath = "heatmaps/person_count_log.json"
    if not os.path.exists(filepath):
        return jsonify({"error": "person_count_log.json not found"}), 404

    try:
        with open(filepath, "r") as f:
            logs = json.load(f)

        interval_counts = {}

        for entry in logs:
            timestamp_str = entry.get("timestamp")
            count = entry.get("person_count", 0)

            try:
                timestamp = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")
            except Exception:
                continue

            # Round timestamp to nearest 30-min block
            rounded = timestamp.replace(minute=(timestamp.minute // 30) * 30, second=0, microsecond=0)
            key = rounded.strftime("%Y-%m-%d %H:%M")

            if key not in interval_counts or count > interval_counts[key]:
                interval_counts[key] = count

        sorted_result = [{"interval_start": k, "max_person_count": v}
                         for k, v in sorted(interval_counts.items())]

        return jsonify(sorted_result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/today_max_count_array')
def today_max_count_array():
    try:
        filepath = "database/person_count_log.json"
        if not os.path.exists(filepath):
            return jsonify({"error": "person_count_log.json not found"}), 404

        with open(filepath, "r") as f:
            data = json.load(f)

        today = datetime.now().date()
        start_time = datetime.combine(today, datetime.min.time()).replace(hour=9)
        end_time = start_time.replace(hour=21)

        # Prepare all 30-min slots
        intervals = [(start_time + timedelta(minutes=30*i)).strftime("%H:%M") for i in range(24)]
        interval_counts = {key: 0 for key in intervals}

        for entry in data:
            try:
                ts = datetime.strptime(entry["timestamp"], "%Y-%m-%d %H:%M:%S")
                if ts.date() == today and start_time <= ts <= end_time:
                    rounded = ts.replace(minute=(ts.minute // 30) * 30, second=0, microsecond=0)
                    key = rounded.strftime("%H:%M")
                    interval_counts[key] = max(interval_counts[key], entry.get("person_count", 0))
            except:
                continue

        return jsonify([interval_counts[slot] for slot in intervals])
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
