import cv2
import numpy as np
from ultralytics import YOLO
from tqdm import tqdm

# Configuration
OUTPUT_POSFIX = "MOT"
MODEL = "yolo11n"
VIDEO_FILE = "running4.mkv" # Input video file as MP4 or MKV format

# MODELS Directory
MODEL_POSE = f"./models/{MODEL}-pose.pt"
MODEL_OBJ =  f"./models/{MODEL}.pt"

# Default video input and output directories
VIDEO_INPUT = f"videos/input/{VIDEO_FILE}"
OUTPUT_DIR =  "videos/output/"

# Adjusted movement thresholds based on stride frequency (strides per second)
STOPPED_THRESH = 0.4      # Reduced from 0.5
WALKING_THRESH = 1.0      # Reduced from 1.2
TROTTING_THRESH = 1.8     # Reduced from 2.0
RUNNING_THRESH = 5      # Added explicit running threshold

# Adjusted angle thresholds for gait analysis (degrees)
WALKING_KNEE_ANGLE = 100  # Increased from 90
TROTTING_KNEE_ANGLE = 130 # Increased from 120
RUNNING_KNEE_ANGLE = 145  # Increased from 140

# Initialize pose estimation model
model = YOLO(MODEL_POSE)

# Open video
cap = cv2.VideoCapture(VIDEO_INPUT)
if not cap.isOpened():
    raise FileNotFoundError(f"Cannot open video: {VIDEO_INPUT}")

# Video properties
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Prepare output video
output_path = f"{OUTPUT_DIR}/{VIDEO_INPUT.split('/')[-1].replace('.mp4', f'_{OUTPUT_POSFIX}_{MODEL}.mp4').replace('.mkv', f'_{OUTPUT_POSFIX}_{MODEL}.mp4')}"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# Track stride history: {track_id: {'stride_times': [], 'prev_foot_y': float, 'state_history': []}}
stride_data = {}
prev_time = 0

print(f"Processing video: {VIDEO_INPUT}")
print(f"Total frames: {total_frames} | FPS: {fps:.1f} | Resolution: {width}x{height}")
print(f"Output will be saved to: {output_path}")

# Initialize tqdm progress bar
progress_bar = tqdm(total=total_frames, desc="Processing Frames", unit="frame")

def calculate_angle(a, b, c):
    """Calculate angle between three points (b is the vertex) in degrees"""
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)
    return np.degrees(angle)

def calculate_knee_angle(keypoints, side="left"):
    """Calculate knee angle for left or right leg"""
    if side == "left":
        hip = keypoints[11]  # Left hip
        knee = keypoints[13] # Left knee
        ankle = keypoints[15] # Left ankle
    else:
        hip = keypoints[12]  # Right hip
        knee = keypoints[14] # Right knee
        ankle = keypoints[16] # Right ankle
    
    # Check visibility
    if hip[1] == 0 or knee[1] == 0 or ankle[1] == 0:
        return None
    
    return calculate_angle(hip, knee, ankle)

def get_vertical_ankle_position(keypoints, side="left"):
    """Get vertical position of ankle (y-coordinate)"""
    if side == "left":
        return keypoints[15][1]  # Left ankle y
    return keypoints[16][1]      # Right ankle y

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    current_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0  # Current time in seconds

    # Run pose estimation with tracking
    results = model.track(
        frame,
        persist=True,
        verbose=False,
        conf=0.4,
        iou=0.6,
        classes=[0],
        tracker="botsort.yaml"
    )

    # Process detections
    if results[0].boxes is not None and results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        track_ids = results[0].boxes.id.int().cpu().tolist()
        keypoints_list = results[0].keypoints.xy.cpu().numpy()

        for box, track_id, kpts in zip(boxes, track_ids, keypoints_list):
            x1, y1, x2, y2 = box
            bbox_height = y2 - y1

            # Initialize tracker for new ID
            if track_id not in stride_data:
                stride_data[track_id] = {
                    'stride_times': [],
                    'prev_left_y': None,
                    'prev_right_y': None,
                    'state_history': ['Stopped'] * 5,
                    'confirmed_state': 'Stopped',
                    'last_stride_time': current_time
                }
            
            track_info = stride_data[track_id]
            
            # Calculate knee angles
            left_knee_angle = calculate_knee_angle(kpts, "left")
            right_knee_angle = calculate_knee_angle(kpts, "right")
            
            # Get current ankle positions
            left_ankle_y = get_vertical_ankle_position(kpts, "left")
            right_ankle_y = get_vertical_ankle_position(kpts, "right")
            
            # Detect stride events (foot moving upward)
            stride_detected = False
            
            # Check left leg stride
            if track_info['prev_left_y'] is not None and left_ankle_y is not None:
                if left_ankle_y < track_info['prev_left_y'] - (bbox_height * 0.02):
                    if current_time - track_info['last_stride_time'] > 0.1:  # Min stride interval
                        track_info['stride_times'].append(current_time)
                        track_info['last_stride_time'] = current_time
                        stride_detected = True
            
            # Check right leg stride
            if track_info['prev_right_y'] is not None and right_ankle_y is not None:
                if right_ankle_y < track_info['prev_right_y'] - (bbox_height * 0.02):
                    if current_time - track_info['last_stride_time'] > 0.1:  # Min stride interval
                        track_info['stride_times'].append(current_time)
                        track_info['last_stride_time'] = current_time
                        stride_detected = True
            
            # Update previous positions
            track_info['prev_left_y'] = left_ankle_y
            track_info['prev_right_y'] = right_ankle_y
            
            # Calculate stride frequency (strides per second)
            # Remove strides older than 2 seconds
            track_info['stride_times'] = [t for t in track_info['stride_times'] 
                                         if current_time - t <= 2.0]
            
            if len(track_info['stride_times']) >= 2:
                stride_freq = len(track_info['stride_times']) / (
                    track_info['stride_times'][-1] - track_info['stride_times'][0])
            else:
                stride_freq = 0
            
            # NEW: Calculate knee angle range for better classification
            knee_angles = []
            if left_knee_angle: knee_angles.append(left_knee_angle)
            if right_knee_angle: knee_angles.append(right_knee_angle)
            min_knee_angle = min(knee_angles) if knee_angles else None
            max_knee_angle = max(knee_angles) if knee_angles else None
            
            # Determine state based on stride frequency and knee angles
            new_state = "Stopped"
            
            if stride_freq > STOPPED_THRESH:
                # Classify based primarily on stride frequency
                if stride_freq < WALKING_THRESH:
                    new_state = "Walking"
                elif stride_freq < TROTTING_THRESH:
                    new_state = "Trotting"
                elif stride_freq < RUNNING_THRESH:
                    # Use knee angle to differentiate trotting vs running
                    if min_knee_angle and min_knee_angle < RUNNING_KNEE_ANGLE:
                        new_state = "Running"
                    else:
                        new_state = "Trotting"
                else:
                    new_state = "Running"
                    
                # Use knee angle to refine walking vs trotting classification
                if new_state == "Walking" and min_knee_angle and min_knee_angle < TROTTING_KNEE_ANGLE:
                    new_state = "Trotting"
                elif new_state == "Trotting" and max_knee_angle and max_knee_angle > TROTTING_KNEE_ANGLE:
                    new_state = "Walking"
            
            # Update state history and confirm state
            track_info['state_history'].pop(0)
            track_info['state_history'].append(new_state)
            
            # Confirm state if consistent in 3/5 recent frames
            if track_info['state_history'].count(new_state) >= 3:
                track_info['confirmed_state'] = new_state
            
            state = track_info['confirmed_state']
            
            # Set color based on state
            if state == "Stopped":
                color = (0, 255, 0)  # Green
            elif state == "Walking":
                color = (0, 255, 255)  # Yellow
            elif state == "Trotting":
                color = (0, 165, 255)  # Orange
            else:  # Running
                color = (0, 0, 255)  # Red
                
            # Draw bounding box and info
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            cv2.putText(frame, f"ID: {track_id}", (int(x1), int(y1)-65), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.putText(frame, f"State: {state}", (int(x1), int(y1)-45), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.putText(frame, f"Stride: {stride_freq:.1f}/s", (int(x1), int(y1)-25), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # Draw knee angles if available
            angle_text_y = int(y1) - 5
            if left_knee_angle:
                cv2.putText(frame, f"L Knee: {left_knee_angle:.0f}°", 
                            (int(x1), angle_text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)
                angle_text_y += 20
            if right_knee_angle:
                cv2.putText(frame, f"R Knee: {right_knee_angle:.0f}°", 
                            (int(x1), angle_text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)
    
    # Cleanup disappeared tracks
    active_ids = set(track_ids) if results[0].boxes is not None and results[0].boxes.id is not None else set()
    disappeared_ids = set(stride_data.keys()) - active_ids
    for track_id in disappeared_ids:
        # Keep disappeared tracks for 10 frames before removing
        if 'disappeared_count' not in stride_data[track_id]:
            stride_data[track_id]['disappeared_count'] = 0
        stride_data[track_id]['disappeared_count'] += 1
        
        if stride_data[track_id]['disappeared_count'] > 10:
            del stride_data[track_id]
    
    out.write(frame)
    prev_time = current_time
    progress_bar.update(1)  # Update progress bar

cap.release()
out.release()
progress_bar.close()  # Ensure progress bar is closed
print(f"\nProcessing complete. Output saved to: {output_path}")