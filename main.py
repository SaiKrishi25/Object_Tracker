import os
from ultralytics import YOLO
import cv2
import time
import numpy as np
import json
from datetime import datetime
from jinja2 import Environment, FileSystemLoader

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

model = YOLO("yolov8s.pt")
video_path = "G:/Object_monitoring_MacV/input/macv-obj-tracking-video.mp4"
cap = cv2.VideoCapture(video_path)

fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
video_duration = total_frames / fps
frame_time = 1.0 / fps

output_path = "G:/Object_monitoring_MacV/output/processed_output.mp4"
fourcc = cv2.VideoWriter_fourcc(*'avc1')
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

current_tracks = {}
track_history = {}
object_times = {}
object_centroids = {}
next_id = 1
id_colors = {}
fps_list = []

frame_count = 0
start_time = time.time()

def calculate_centroid(box):
    x1, y1, x2, y2 = box
    return (int((x1 + x2) / 2), int((y1 + y2) / 2))

def draw_trail(frame, track_id, color):
    if track_id in track_history and len(track_history[track_id]) > 1:
        points = []
        for box in track_history[track_id]:
            centroid = calculate_centroid(box)
            points.append(centroid)
        for i in range(1, len(points)):
            cv2.line(frame, points[i-1], points[i], color, 2)

print(f"Processing video: {video_path}")
print(f"Stats: {frame_height}x{frame_width} at {fps} FPS")
print(f"Total frames: {total_frames}, Duration: {video_duration:.2f} seconds")

try:
    model.tracker = "botsort.yaml"
    model.conf = 0.5
    model.iou = 0.4
except Exception as e:
    print(f"Warning: Could not set tracker parameters: {e}")
    print("Continuing with default tracker...")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame_count += 1
    current_time = frame_count * frame_time
    
    results = model.track(frame, persist=True, verbose=False)
    current_tracks = {}
    
    if results[0].boxes is not None and hasattr(results[0].boxes, 'id') and results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
        track_ids = results[0].boxes.id.cpu().numpy().astype(int)
        classes = results[0].boxes.cls.cpu().numpy().astype(int)
        confs = results[0].boxes.conf.cpu().numpy()
        
        for box, track_id, cls_id, conf in zip(boxes, track_ids, classes, confs):
            x1, y1, x2, y2 = box
            label = results[0].names[cls_id]
            width = x2 - x1
            height = y2 - y1
            
            centroid = calculate_centroid((x1, y1, x2, y2))
            
            current_tracks[track_id] = {
                'box': (x1, y1, width, height),
                'label': label,
                'last_seen': frame_count,
                'confidence': conf,
                'centroid': centroid
            }
            
            if track_id not in object_times:
                object_times[track_id] = {
                    'start_frame': frame_count,
                    'start_time': current_time,
                    'total_time': 0
                }
                id_colors[track_id] = (
                    int((track_id * 57) % 255),
                    int((track_id * 121) % 255),
                    int((track_id * 233) % 255)
                )
            else:
                object_times[track_id]['total_time'] = current_time - object_times[track_id]['start_time']
            
            if track_id not in track_history:
                track_history[track_id] = []
            
            if len(track_history[track_id]) > 30:
                track_history[track_id].pop(0)
            track_history[track_id].append((x1, y1, x2, y2))
            
            color = id_colors[track_id]
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            cv2.circle(frame, centroid, 4, color, -1)
            draw_trail(frame, track_id, color)
            
            time_str = f"Time: {object_times[track_id]['total_time']:.1f}s"
            cv2.putText(frame, time_str, (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    active_count = len(current_tracks)
    unique_count = len(track_history)
    
    overlay = frame.copy()
    cv2.rectangle(overlay, (frame_width - 300, 10), (frame_width - 10, 80), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    
    cv2.putText(frame, f"Active Objects: {active_count}", (frame_width - 280, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, f"Unique Objects: {unique_count}", (frame_width - 280, 70), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    try:
        resized = cv2.resize(frame, (960, 540))
        cv2.imshow("Object Monitor", resized)
    except Exception as e:
        print(f"Error displaying frame: {e}")
    out.write(frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

class_totals = {}
for track_id, info in current_tracks.items():
    label = info['label']
    class_totals[label] = class_totals.get(label, 0) + 1

env = Environment(loader=FileSystemLoader('templates'))
template = env.get_template('report.html')

video_filename = os.path.basename(output_path)
html_content = template.render(
    video_path=video_filename,
    total_objects=len(track_history),
    unique_objects=len(set(track_id for track_id in track_history.keys())),
    active_objects=len(current_tracks),
    average_fps=fps,
    class_totals=class_totals
)

os.makedirs(os.path.dirname(output_path), exist_ok=True)

html_path = os.path.join(os.path.dirname(output_path), 'index.html')
with open(html_path, 'w') as f:
    f.write(html_content)

print(f"\nVideo FPS: {fps}")
print(f"Total Objects Tracked: {len(track_history)}")
print(f"Active Objects: {len(current_tracks)}")
print(f"Objects by class: {class_totals}")
print(f"Output video saved at: {output_path}")
print(f"HTML report generated at: {html_path}")
print("\nTo view the results:")
print(f"1. Open {html_path} in your web browser")
print("2. If the video doesn't play, try using a different browser (Chrome or Firefox recommended)")