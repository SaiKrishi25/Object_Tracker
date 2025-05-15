import streamlit as st
import cv2
import tempfile
import os
from ultralytics import YOLO
import numpy as np
import time
from datetime import datetime
import pandas as pd
import plotly.express as px
import torch

# Fix OpenMP error
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Create output directory
OUTPUT_DIR = "streamlit_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Set page config
st.set_page_config(
    page_title="Object Tracking Dashboard",
    page_icon="🎥",
    layout="wide"
)

# Initialize session state
if 'processed_video' not in st.session_state:
    st.session_state.processed_video = None
if 'stats' not in st.session_state:
    st.session_state.stats = None
if 'model' not in st.session_state:
    st.session_state.model = None
if 'temp_files' not in st.session_state:
    st.session_state.temp_files = []

def load_model(model_size="small"):
    if st.session_state.model is None:
        with st.spinner("Loading YOLO model..."):
            try:
                # Use the selected model size
                model_path = f"yolov8{model_size[0]}.pt"  # 'n' for nano, 's' for small
                st.session_state.model = YOLO(model_path)
                st.session_state.model.conf = 0.5
                st.session_state.model.iou = 0.4
            except Exception as e:
                st.error(f"Error loading model: {str(e)}")
                return None
    return st.session_state.model

def cleanup_temp_files():
    for temp_file in st.session_state.temp_files:
        try:
            if os.path.exists(temp_file):
                os.unlink(temp_file)
        except Exception as e:
            st.warning(f"Could not delete temporary file {temp_file}: {str(e)}")
    st.session_state.temp_files = []

def calculate_centroid(box):
    x1, y1, x2, y2 = box
    return (int((x1 + x2) / 2), int((y1 + y2) / 2))

def draw_trail(frame, track_id, track_history, color):
    if track_id in track_history and len(track_history[track_id]) > 1:
        points = []
        for box in track_history[track_id]:
            centroid = calculate_centroid(box)
            points.append(centroid)
        
        # Draw the trail
        for i in range(1, len(points)):
            cv2.line(frame, points[i-1], points[i], color, 2)

def process_video(video_file, model, conf_threshold=0.5):
    if model is None:
        st.error("Model not loaded properly. Please try again.")
        return None, None
        
    # Create a temporary file to store the uploaded video
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(video_file.read())
    tfile.close()  # Close the file before using it
    st.session_state.temp_files.append(tfile.name)
    
    # Initialize video capture
    cap = cv2.VideoCapture(tfile.name)
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_duration = total_frames / fps  # Total video duration in seconds
    frame_time = 1.0 / fps  # Time per frame in seconds
    
    # Initialize tracking variables
    current_tracks = {}
    track_history = {}
    object_times = {}
    next_id = 1
    id_colors = {}
    
    # Generate output filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"processed_{timestamp}.mp4"
    output_path = os.path.join(OUTPUT_DIR, output_filename)
    
    # Create output video writer
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    
    # Statistics tracking
    stats = {
        'frame_count': 0,
        'objects_by_class': {},
        'active_objects': [],
        'unique_objects': set(),
        'processing_time': 0,
        'fps': [],
        'object_times': {},
        'video_duration': video_duration
    }
    
    start_time = time.time()
    
    # Process video
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_start_time = time.time()
            stats['frame_count'] += 1
            current_time = (stats['frame_count'] - 1) * frame_time  # Calculate time based on frame number and fps
            current_tracks = {}
            
            # Update progress
            progress = stats['frame_count'] / total_frames
            progress_bar.progress(progress)
            status_text.text(f"Processing frame {stats['frame_count']} of {total_frames}")
            
            # Run YOLO tracking with error handling
            try:
                results = model.track(frame, persist=True, conf=conf_threshold)
            except Exception as e:
                st.error(f"Error during tracking: {str(e)}")
                break
            
            if results[0].boxes is not None and hasattr(results[0].boxes, 'id') and results[0].boxes.id is not None:
                try:
                    boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
                    track_ids = results[0].boxes.id.cpu().numpy().astype(int)
                    classes = results[0].boxes.cls.cpu().numpy().astype(int)
                    confs = results[0].boxes.conf.cpu().numpy()
                except Exception as e:
                    st.error(f"Error processing detection results: {str(e)}")
                    continue
                
                for box, track_id, cls_id, conf in zip(boxes, track_ids, classes, confs):
                    x1, y1, x2, y2 = box
                    label = results[0].names[cls_id]
                    
                    # Update statistics
                    stats['objects_by_class'][label] = stats['objects_by_class'].get(label, 0) + 1
                    stats['unique_objects'].add(track_id)
                    
                    # Track object
                    current_tracks[track_id] = {
                        'box': (x1, y1, x2, y2),
                        'label': label,
                        'confidence': conf
                    }
                    
                    # Update object time tracking using video frame timestamps
                    if track_id not in object_times:
                        object_times[track_id] = {
                            'start_frame': stats['frame_count'],
                            'start_time': current_time,
                            'total_time': 0,
                            'last_seen_time': current_time
                        }
                    else:
                        # Update last seen time
                        object_times[track_id]['last_seen_time'] = current_time
                        # Calculate total time based on video frame timestamps
                        object_times[track_id]['total_time'] = current_time - object_times[track_id]['start_time']
                    
                    # Update track history
                    if track_id not in track_history:
                        track_history[track_id] = []
                    if len(track_history[track_id]) > 30:  # Keep last 30 positions
                        track_history[track_id].pop(0)
                    track_history[track_id].append((x1, y1, x2, y2))
                    
                    # Assign color
                    if track_id not in id_colors:
                        id_colors[track_id] = (
                            int((track_id * 57) % 255),
                            int((track_id * 121) % 255),
                            int((track_id * 233) % 255)
                        )
                    
                    # Draw bounding box
                    color = id_colors[track_id]
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1-10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    
                    # Draw trail
                    draw_trail(frame, track_id, track_history, color)
                    
                    # Add time information
                    time_str = f"Time: {object_times[track_id]['total_time']:.1f}s"
                    cv2.putText(frame, time_str, (x1, y2 + 20),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Update active objects count
            stats['active_objects'].append(len(current_tracks))
            
            # Calculate and store FPS
            frame_time = time.time() - frame_start_time
            current_fps = 1.0 / frame_time if frame_time > 0 else 0
            stats['fps'].append(current_fps)
            
            # Write frame
            out.write(frame)
    
    finally:
        # Cleanup
        cap.release()
        out.release()
    
    # Calculate processing time
    stats['processing_time'] = time.time() - start_time
    stats['average_fps'] = sum(stats['fps']) / len(stats['fps']) if stats['fps'] else 0
    stats['object_times'] = object_times
    
    return output_path, stats

def main():
    st.title("Object Tracking Dashboard")
    
    # Sidebar
    st.sidebar.header("Settings")
    conf_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.05)
    
    # Model selection
    model_size = st.sidebar.selectbox(
        "Select YOLO Model Size",
        ["nano", "small"],
        index=1
    )
    
    # File uploader
    uploaded_file = st.file_uploader("Choose a video file", type=['mp4', 'avi', 'mov'])
    
    if uploaded_file is not None:
        # Load YOLO model with selected size
        model = load_model(model_size)
        
        # Process video
        if st.button("Process Video"):
            try:
                with st.spinner("Processing video"):
                    output_path, stats = process_video(uploaded_file, model, conf_threshold)
                    st.session_state.processed_video = output_path
                    st.session_state.stats = stats
                    
                    # Display processed video
                    st.video(output_path)
                    
                    # Display statistics
                    st.header("Tracking Statistics")
                    
                    # Active Objects Over Time
                    st.subheader("Active Objects Over Time")
                    active_df = pd.DataFrame(stats['active_objects'],
                                          columns=['Active Objects'])
                    fig = px.line(active_df, title='Active Objects per Frame')
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Additional statistics
                    st.subheader("Summary Statistics")
                    col3, col4, col5, col6 = st.columns(4)
                    
                    with col3:
                        st.metric("Total Frames Processed", stats['frame_count'])
                    with col4:
                        st.metric("Unique Objects Tracked", len(stats['unique_objects']))
                    with col5:
                        st.metric("Processing Time", f"{stats['processing_time']:.2f} seconds")
                    with col6:
                        st.metric("Average FPS", f"{stats['average_fps']:.1f}")
                    
                    # Display object tracking times
                    st.subheader("Object Tracking Times")
                    time_data = []
                    for track_id, time_info in stats['object_times'].items():
                        time_data.append({
                            'Track ID': track_id,
                            'Duration (s)': time_info['total_time'],
                            'Start Time (s)': time_info['start_time']
                        })
                    time_df = pd.DataFrame(time_data)
                    st.dataframe(time_df)
                    
                    # Display video duration
                    st.info(f"Video Duration: {stats['video_duration']:.2f} seconds")
            
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
            
            finally:
                # Cleanup temporary files
                cleanup_temp_files()

if __name__ == "__main__":
    main() 