import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict, deque
from collections import Counter

# Configuration
OUTPUT_POSFIX = "angles"
MODEL = "yolo11n"
VIDEO_FILE = "running4.mkv" # Input video file as MP4 or MKV format

# MODELS Directory
MODEL_POSE = f"./models/{MODEL}-pose.pt"
MODEL_OBJ =  f"./models/{MODEL}.pt"

# Default video input and output directories
VIDEO_INPUT = f"videos/input/{VIDEO_FILE}"
OUTPUT_DIR =  "videos/output/"

# Movement state thresholds based on knee angles (degrees)
TH_STOPPED = 175  # Legs mostly straight
TH_WALKING = 160  # Moderate bend
TH_TROTTING = 145 # More pronounced bend

# State persistence settings
MIN_FRAMES_SAME_STATE = 8  # Minimum frames to maintain a state before switching
STATE_HISTORY_LENGTH = 15  # Number of frames to consider for state smoothing

# Colors for visualization
COLORS = {
    "stopped": (0, 255, 0),      # Green
    "walking": (0, 255, 255),    # Yellow
    "trotting": (0, 165, 255),   # Orange
    "running": (0, 0, 255)       # Red
}

# Keypoint indices (COCO format)
NOSE = 0
LEFT_SHOULDER = 5
RIGHT_SHOULDER = 6
LEFT_HIP = 11
RIGHT_HIP = 12
LEFT_KNEE = 13
RIGHT_KNEE = 14
LEFT_ANKLE = 15
RIGHT_ANKLE = 16

def compute_iou(box1, box2):
    """Calculate Intersection over Union (IoU) between two bounding boxes"""
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    
    # Calculate intersection area
    xi_min = max(x1_min, x2_min)
    yi_min = max(y1_min, y2_min)
    xi_max = min(x1_max, x2_max)
    yi_max = min(y1_max, y2_max)
    
    inter_area = max(xi_max - xi_min, 0) * max(yi_max - yi_min, 0)
    
    # Calculate union area
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = box1_area + box2_area - inter_area
    
    return inter_area / union_area if union_area > 0 else 0

def calculate_angle(a, b, c):
    """Calculate the angle in degrees between three points"""
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    # Calculate vectors
    ba = a - b
    bc = c - b
    
    # Calculate angle between vectors
    dot_product = np.dot(ba, bc)
    norm_ba = np.linalg.norm(ba)
    norm_bc = np.linalg.norm(bc)
    
    # Avoid division by zero
    if norm_ba == 0 or norm_bc == 0:
        return 180
    
    cosine_angle = dot_product / (norm_ba * norm_bc)
    angle_rad = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    angle_deg = np.degrees(angle_rad)
    
    return angle_deg

def get_leg_angles(keypoints, conf_threshold=0.3):
    """Calculate knee angles for both legs"""
    angles = []
    
    # Get keypoints with confidence
    kp = keypoints
    
    # Left leg angle (hip-knee-ankle)
    if (kp[LEFT_HIP][2] > conf_threshold and 
        kp[LEFT_KNEE][2] > conf_threshold and 
        kp[LEFT_ANKLE][2] > conf_threshold):
        
        left_hip = kp[LEFT_HIP][:2]
        left_knee = kp[LEFT_KNEE][:2]
        left_ankle = kp[LEFT_ANKLE][:2]
        
        left_angle = calculate_angle(left_hip, left_knee, left_ankle)
        angles.append(left_angle)
    
    # Right leg angle (hip-knee-ankle)
    if (kp[RIGHT_HIP][2] > conf_threshold and 
        kp[RIGHT_KNEE][2] > conf_threshold and 
        kp[RIGHT_ANKLE][2] > conf_threshold):
        
        right_hip = kp[RIGHT_HIP][:2]
        right_knee = kp[RIGHT_KNEE][:2]
        right_ankle = kp[RIGHT_ANKLE][:2]
        
        right_angle = calculate_angle(right_hip, right_knee, right_ankle)
        angles.append(right_angle)
    
    return angles

def main():
    # Load models
    model_obj = YOLO(MODEL_OBJ)
    model_pose = YOLO(MODEL_POSE)
    
    # Initialize video capture
    cap = cv2.VideoCapture(VIDEO_INPUT)
    if not cap.isOpened():
        print(f"Error opening video: {VIDEO_INPUT}")
        return
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Initialize video writer (MP4 format)
    output_path = f"{OUTPUT_DIR}/{VIDEO_INPUT.split('/')[-1].replace('.mp4', f'_{OUTPUT_POSFIX}_{MODEL}.mp4').replace('.mkv', f'_{OUTPUT_POSFIX}_{MODEL}.mp4')}"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # MP4 codec
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Tracking variables
    track_history = defaultdict(lambda: {
        'state': "stopped",
        'state_history': deque(maxlen=STATE_HISTORY_LENGTH),
        'id': None,
        'last_angles': deque(maxlen=5),
        'last_centroid': None
    })
    
    next_id = 0
    frame_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        print(f"Processing frame {frame_count}/{total_frames}")
        
        # Run object detection
        obj_results = model_obj(frame, conf=0.5, classes=[0], verbose=False)[0]
        person_boxes = []
        if obj_results.boxes:
            for box in obj_results.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = float(box.conf[0])
                person_boxes.append((x1, y1, x2, y2, conf))
        
        # Run pose estimation - use keypoints with confidence scores
        pose_results = model_pose(frame, conf=0.5, classes=[0], verbose=False)[0]
        pose_data = []
        if pose_results.keypoints:
            for kp, box in zip(pose_results.keypoints, pose_results.boxes):
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                # Get full keypoints with confidence (shape: [17, 3])
                keypoints = kp.data[0].cpu().numpy()
                conf = float(box.conf[0])
                pose_data.append((x1, y1, x2, y2, conf, keypoints))
        
        # Match object detections with pose estimations
        matched_data = []
        for p_box in person_boxes:
            best_iou = 0
            best_pose = None
            for pose in pose_data:
                iou = compute_iou(p_box[:4], pose[:4])
                if iou > best_iou:
                    best_iou = iou
                    best_pose = pose
            
            if best_pose and best_iou > 0.3:
                matched_data.append((p_box, best_pose))
        
        # Current frame detections
        current_detections = []
        for idx, (p_box, pose) in enumerate(matched_data):
            x1, y1, x2, y2, conf = p_box
            centroid = ((x1 + x2) // 2, (y1 + y2) // 2)
            current_detections.append({
                'centroid': centroid,
                'p_box': p_box,
                'pose': pose,
                'keypoints': pose[5],
                'height': y2 - y1
            })
        
        # Update tracking and assign IDs
        active_ids = set()
        annotated_detections = []
        
        for detection in current_detections:
            centroid = detection['centroid']
            p_box = detection['p_box']
            pose = detection['pose']
            keypoints = detection['keypoints']
            height = detection['height']
            x1, y1, x2, y2, conf = p_box
            
            # Calculate leg angles
            angles = get_leg_angles(keypoints)
            avg_angle = np.mean(angles) if angles else 180
            
            # Find closest existing track
            min_distance = float('inf')
            best_match_id = None
            
            for track_id, track_info in track_history.items():
                if track_info['last_centroid'] is None:
                    continue
                
                # Get last known position
                last_centroid = track_info['last_centroid']
                
                # Calculate distance
                distance = np.sqrt((centroid[0] - last_centroid[0])**2 + 
                                  (centroid[1] - last_centroid[1])**2)
                
                # Normalize distance by person height
                normalized_distance = distance / height if height > 0 else distance
                
                if normalized_distance < min_distance and normalized_distance < 0.5:
                    min_distance = normalized_distance
                    best_match_id = track_id
            
            state = "stopped"
            track_id = None
            
            # If found a match, update existing track
            if best_match_id is not None:
                track_info = track_history[best_match_id]
                track_info['last_angles'].append(avg_angle)
                track_info['last_centroid'] = centroid
                active_ids.add(best_match_id)
                
                # Calculate movement state based on leg angles
                if len(track_info['last_angles']) >= 3:
                    recent_angles = list(track_info['last_angles'])[-3:]
                    smoothed_angle = np.mean(recent_angles)
                    
                    if smoothed_angle > TH_STOPPED:
                        state = "stopped"
                    elif smoothed_angle > TH_WALKING:
                        state = "walking"
                    elif smoothed_angle > TH_TROTTING:
                        state = "trotting"
                    else:
                        state = "running"
                
                # Update state history
                track_info['state_history'].append(state)
                
                # Apply state smoothing and persistence
                if len(track_info['state_history']) >= MIN_FRAMES_SAME_STATE:
                    # Get most common state in recent history
                    state_counter = Counter(list(track_info['state_history'])[-MIN_FRAMES_SAME_STATE:])
                    most_common_state = state_counter.most_common(1)[0][0]
                    
                    # Only change state if we've seen it consistently
                    if track_info['state'] != most_common_state:
                        if state_counter[most_common_state] >= MIN_FRAMES_SAME_STATE - 2:
                            track_info['state'] = most_common_state
                else:
                    track_info['state'] = state
                
                state = track_info['state']
                track_id = best_match_id
                
            else:
                # New person detected
                track_id = next_id
                next_id += 1
                active_ids.add(track_id)
                
                # Initialize new track
                track_info = track_history[track_id]
                track_info['state'] = "stopped"
                track_info['state_history'] = deque(["stopped"], maxlen=STATE_HISTORY_LENGTH)
                track_info['last_angles'] = deque([avg_angle], maxlen=5)
                track_info['last_centroid'] = centroid
                track_info['id'] = track_id
                
                state = "stopped"
            
            # Store detection for drawing
            annotated_detections.append({
                'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                'state': state,
                'track_id': track_id,
                'angles': angles,
                'keypoints': keypoints
            })
        
        # Remove old tracks (not detected in this frame)
        for track_id in list(track_history.keys()):
            if track_id not in active_ids:
                del track_history[track_id]
        
        # Draw annotations
        for detection in annotated_detections:
            x1 = detection['x1']
            y1 = detection['y1']
            x2 = detection['x2']
            y2 = detection['y2']
            state = detection['state']
            track_id = detection['track_id']
            angles = detection['angles']
            keypoints = detection['keypoints']
            
            color = COLORS[state]
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw state text with ID
            state_text = f"ID:{track_id} {state.upper()}"
            cv2.putText(frame, state_text, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # Draw angle information
            if angles:
                angle_text = f"Angle: {np.mean(angles):.1f}°"
                cv2.putText(frame, angle_text, (x1, y1 - 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
            
            # Draw keypoints and leg connections
            # Draw left leg
            if (keypoints[LEFT_HIP][2] > 0.3 and 
                keypoints[LEFT_KNEE][2] > 0.3):
                cv2.line(frame, 
                         (int(keypoints[LEFT_HIP][0]), int(keypoints[LEFT_HIP][1])),
                         (int(keypoints[LEFT_KNEE][0]), int(keypoints[LEFT_KNEE][1])),
                         color, 2)
            if (keypoints[LEFT_KNEE][2] > 0.3 and 
                keypoints[LEFT_ANKLE][2] > 0.3):
                cv2.line(frame, 
                         (int(keypoints[LEFT_KNEE][0]), int(keypoints[LEFT_KNEE][1])),
                         (int(keypoints[LEFT_ANKLE][0]), int(keypoints[LEFT_ANKLE][1])),
                         color, 2)
            
            # Draw right leg
            if (keypoints[RIGHT_HIP][2] > 0.3 and 
                keypoints[RIGHT_KNEE][2] > 0.3):
                cv2.line(frame, 
                         (int(keypoints[RIGHT_HIP][0]), int(keypoints[RIGHT_HIP][1])),
                         (int(keypoints[RIGHT_KNEE][0]), int(keypoints[RIGHT_KNEE][1])),
                         color, 2)
            if (keypoints[RIGHT_KNEE][2] > 0.3 and 
                keypoints[RIGHT_ANKLE][2] > 0.3):
                cv2.line(frame, 
                         (int(keypoints[RIGHT_KNEE][0]), int(keypoints[RIGHT_KNEE][1])),
                         (int(keypoints[RIGHT_ANKLE][0]), int(keypoints[RIGHT_ANKLE][1])),
                         color, 2)
            
            # Draw keypoints
            for i in [LEFT_HIP, RIGHT_HIP, LEFT_KNEE, RIGHT_KNEE, LEFT_ANKLE, RIGHT_ANKLE]:
                if keypoints[i][2] > 0.3:
                    cv2.circle(frame, 
                              (int(keypoints[i][0]), int(keypoints[i][1])),
                              5, (0, 255, 255), -1)
        
        # Write frame to output
        out.write(frame)
    
    # Cleanup
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Processing complete. Output saved to: {output_path}")

if __name__ == "__main__":
    main()