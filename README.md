STORE MONITORING SYSTEM

----------------------------------------------------------------------------------------------------------------

Overview : StoreiQ
Key Feature
Tech Stack Overview
YOLO Model Justification
Use Case Scenarios
Market Comparison & Competition
Future Enhancements
Conclusion

----------------------------------------------------------------------------------------------------------------
Overview : StoreiQ

This system presents a real-time footfall analytics system built using computer vision techniques. It detects and analyzes the movement of people within a defined physical space (like a retail store or an office floor) using a single webcam, and generates live person counts, heatmaps, and reports through a web dashboard and REST API.

WHY THIS PROJECT?
Provides a foundation for AI-driven Analytics without needing Expensive Tools.
Helps in Alert Monitoring , Staff Allocation and Space Optimization.
Manual Observation of live video is inefficient and error-prone.

GOAL
Build a system for a Retail Store which monitors the footfall and performs Analytics.


----------------------------------------------------------------------------------------------------------------
Key Feature

FEATURE
Real-time person detection using YOLOv8.
Heatmap generation from blob tracking.
Overcrowding alerting in marked zones.
Flask API with multiple endpoints (/start, /video_feed, /person_count, etc.).
HTML-based user dashboard with live feed, metrics, and report download.
Historical reports by day, week, month.

----------------------------------------------------------------------------------------------------------------
Tech Stack Overview

To clarify the architecture and simplify technical onboarding, the table below summarizes the technologies used in each layer of the system:
Layer
Technology/Tools
Purpose

Frontend
React.js, CSS
User interface, routing, theming

Backend
Python (Flask), OpenCV, Ultralytics YOLOv8
REST API, logic engine

Model
YOLOv8n (lightweight neural net for object detection)
Person identification

Storage
JSON files (wardrobe.json, users.json)
Lightweight, local data storage

Video Processing
Blob tracking, zone marking, heatmap overlays
Analyzing real time video input

----------------------------------------------------------------------------------------------------------------
YOLO Model Justification

YOLO Model
YOLOv8n (nano version) is ideal for edge devices due to its balance between speed and accuracy.
Class ID `0` used to detect only "person" class.
Average FPS achieved: 25-30 on standard laptops.
Accuracy (person detection precision): ~85-90% depending on camera placement and lighting.
Compared to SSD MobileNet or OpenCV HOG detectors, YOLOv8n provides better localization, reduced false positives, and modern training dataset support.


----------------------------------------------------------------------------------------------------------------
Use Case Scenarios

Retail store footfall monitoring.
Smart office crowd analytics.
College/classroom occupancy tracking.
Entry-exit surveillance systems.
Real estate customer interaction analytics.
Event management crowd flow analysis.


----------------------------------------------------------------------------------------------------------------
Market Comparison & Competition

COMPETITORS
RetailNext
V-Count
Cisco Meraki Vision
Camlytics
Density.io Retail Clients


ADVANTAGES OF StoreiQ
Open-source and customizable
On-device processing (no cloud dependency)
Lightweight YOLOv8n model support
Cost-effective: just a laptop + webcam


----------------------------------------------------------------------------------------------------------------
Future Enhancements

Multi-camera support
Real-time heatmap updates on UI
Gender and age classification
Integration with cloud (AWS/GCP dashboards)
Time-based footfall trends and graph generation

----------------------------------------------------------------------------------------------------------------
Conclusion

StoreiQ system is a powerful base for real-world people analytics. With further enhancement and integration into business environments, it can compete with commercial surveillance intelligence tools.










