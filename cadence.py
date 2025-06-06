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

# Cadence thresholds (steps per second)
TH_STOPPED = 0.5
TH_WALKING = 1.2
TH_TROTTING = 2.0

# Step length thresholds (normalized by height)
TH_STEP_SHORT = 0.15
TH_STEP_MEDIUM = 0.25

# State persistence settings
MIN_FRAMES_SAME_STATE = 10
STATE_HISTORY_LENGTH = 20

# Colors for visualization
COLORS = {
    "stopped": (0, 255, 0),      # Green
    "walking": (0, 255, 255),    # Yellow
    "trotting": (0, 165, 255),   # Orange
    "running": (0, 0, 255)       # Red
}

# Keypoint indices (COCO format)
LEFT_ANKLE = 15
RIGHT_ANKLE = 16
LEFT_HIP = 11
RIGHT_HIP = 12

def compute_iou(box1, box2):
    """Calculate Intersection over Union (IoU) between two bounding boxes"""
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    
    xi_min = max(x1_min, x2_min)
    yi_min = max(y1_min, y2_min)
    xi_max = min(x1_max, x2_max)
    yi_max = min(y1_max, y2_max)
    
    inter_area = max(xi_max - xi_min, 0) * max(yi_max - yi_min, 0)
    
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = box1_area + box2_area - inter_area
    
    return inter_area / union_area if union_area > 0 else 0

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
    
    # Initialize video writer with "_cadence" suffix
    output_path = f"{OUTPUT_DIR}/{VIDEO_INPUT.split('/')[-1].replace('.mp4', f'_{OUTPUT_POSFIX}_{MODEL}.mp4').replace('.mkv', f'_{OUTPUT_POSFIX}_{MODEL}.mp4')}"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Tracking variables
    track_history = defaultdict(lambda: {
        'state': "stopped",
        'state_history': deque(maxlen=STATE_HISTORY_LENGTH),
        'id': None,
        'ankle_positions': {
            'left': deque(maxlen=30),
            'right': deque(maxlen=30)
        },
        'step_events': {
            'left': deque(maxlen=20),
            'right': deque(maxlen=20)
        },
        'last_centroid': None,
        'height': None,
        'step_lengths': deque(maxlen=10),
        'prev_hip_center': None,
        'hip_movement': deque(maxlen=10)
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
        
        # Run pose estimation
        pose_results = model_pose(frame, conf=0.5, classes=[0], verbose=False)[0]
        pose_data = []
        if pose_results.keypoints:
            for kp, box in zip(pose_results.keypoints, pose_results.boxes):
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                keypoints = kp.data[0].cpu().numpy()
                conf = float(box.conf[0])
                pose_data.append((x1, y1, x2, y2, conf, keypoints))
        
        # Match detections
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
            height_val = y2 - y1
            
            # Calculate hip center
            hip_center = None
            keypoints = pose[5]
            if keypoints[LEFT_HIP][2] > 0.3 and keypoints[RIGHT_HIP][2] > 0.3:
                hip_center = (
                    (keypoints[LEFT_HIP][0] + keypoints[RIGHT_HIP][0]) / 2,
                    (keypoints[LEFT_HIP][1] + keypoints[RIGHT_HIP][1]) / 2
                )
            
            current_detections.append({
                'centroid': centroid,
                'p_box': p_box,
                'pose': pose,
                'keypoints': keypoints,
                'height': height_val,
                'hip_center': hip_center
            })
        
        # Update tracking
        active_ids = set()
        annotated_detections = []
        
        for detection in current_detections:
            centroid = detection['centroid']
            p_box = detection['p_box']
            pose = detection['pose']
            keypoints = detection['keypoints']
            height_val = detection['height']
            hip_center = detection['hip_center']
            x1, y1, x2, y2, conf = p_box
            
            # Find closest existing track
            min_distance = float('inf')
            best_match_id = None
            
            for track_id, track_info in track_history.items():
                if track_info['last_centroid'] is None:
                    continue
                
                last_centroid = track_info['last_centroid']
                distance = np.sqrt((centroid[0] - last_centroid[0])**2 + 
                                  (centroid[1] - last_centroid[1])**2)
                
                # Normalize distance by height
                normalized_distance = distance / track_info['height'] if track_info['height'] > 0 else distance
                
                if normalized_distance < min_distance and normalized_distance < 0.5:
                    min_distance = normalized_distance
                    best_match_id = track_id
            
            state = "stopped"
            track_id = None
            cadence = 0
            avg_step_length = 0
            hip_movement = 0
            
            # Update existing track
            if best_match_id is not None:
                track_info = track_history[best_match_id]
                track_info['last_centroid'] = centroid
                track_info['height'] = height_val
                active_ids.add(best_match_id)
                track_id = best_match_id
                
                # Calculate hip movement relative to previous position
                if hip_center and track_info['prev_hip_center']:
                    dx = hip_center[0] - track_info['prev_hip_center'][0]
                    dy = hip_center[1] - track_info['prev_hip_center'][1]
                    movement = np.sqrt(dx**2 + dy**2) / height_val
                    track_info['hip_movement'].append(movement)
                    hip_movement = np.mean(track_info['hip_movement']) if track_info['hip_movement'] else 0
                
                # Update hip center
                track_info['prev_hip_center'] = hip_center
                
                # Update ankle positions relative to hip center
                if hip_center:
                    if keypoints[LEFT_ANKLE][2] > 0.3:
                        rel_x = keypoints[LEFT_ANKLE][0] - hip_center[0]
                        rel_y = keypoints[LEFT_ANKLE][1] - hip_center[1]
                        track_info['ankle_positions']['left'].append((rel_x, rel_y))
                    
                    if keypoints[RIGHT_ANKLE][2] > 0.3:
                        rel_x = keypoints[RIGHT_ANKLE][0] - hip_center[0]
                        rel_y = keypoints[RIGHT_ANKLE][1] - hip_center[1]
                        track_info['ankle_positions']['right'].append((rel_x, rel_y))
                
                # Detect step events using relative ankle positions
                for side in ['left', 'right']:
                    positions = list(track_info['ankle_positions'][side])
                    if len(positions) < 5:
                        continue
                    
                    # Get vertical positions relative to hip
                    y_positions = [pos[1] for pos in positions]
                    
                    # Find local minima (foot on ground)
                    min_index = np.argmin(y_positions)
                    max_index = np.argmax(y_positions)
                    
                    # Significant vertical movement indicates step
                    if len(y_positions) > 1:
                        vertical_range = max(y_positions) - min(y_positions)
                        if vertical_range > 0.15:  # Relative to hip position
                            # Check if we have a new step event
                            last_step_frame = track_info['step_events'][side][-1][0] if track_info['step_events'][side] else 0
                            
                            # Only record new step if enough frames have passed
                            if frame_count - last_step_frame > fps / 5:  # Max 5 steps/sec
                                # Foot is on ground when at lowest position
                                if min_index == len(y_positions) - 1:
                                    track_info['step_events'][side].append((frame_count, positions[-1]))
                
                # Calculate cadence (steps per second)
                total_steps = 0
                for side in ['left', 'right']:
                    # Count steps in last 1 second
                    steps = [frame for frame, _ in track_info['step_events'][side] if frame_count - frame <= fps]
                    total_steps += len(steps)
                
                cadence = total_steps / 2  # Steps per second (both legs)
                
                # Calculate step length (horizontal displacement relative to hip)
                if track_info['step_events']['left'] and track_info['step_events']['right']:
                    # Get last step positions
                    last_left = track_info['step_events']['left'][-1][1] if track_info['step_events']['left'] else None
                    last_right = track_info['step_events']['right'][-1][1] if track_info['step_events']['right'] else None
                    
                    if last_left and last_right:
                        # Distance between feet relative to hip
                        step_length = abs(last_left[0] - last_right[0])
                        track_info['step_lengths'].append(step_length)
                
                if track_info['step_lengths']:
                    avg_step_length = np.mean(track_info['step_lengths'])
                else:
                    avg_step_length = 0
                
                # Classify movement state
                if cadence < TH_STOPPED:
                    state = "stopped"
                elif cadence < TH_WALKING:
                    state = "walking"
                elif cadence < TH_TROTTING:
                    state = "trotting"
                else:
                    state = "running"
                
                # Adjust based on step length and hip movement
                if state == "trotting" and avg_step_length > TH_STEP_MEDIUM:
                    state = "running"
                elif state == "walking" and avg_step_length > TH_STEP_MEDIUM:
                    state = "trotting"
                elif state == "walking" and avg_step_length < TH_STEP_SHORT:
                    state = "stopped"
                
                # If hip is moving significantly but cadence is low, it might be camera movement
                if hip_movement > 0.1 and cadence < TH_WALKING:
                    state = "walking"
                
                # Update state history
                track_info['state_history'].append(state)
                
                # Apply state smoothing
                if len(track_info['state_history']) >= MIN_FRAMES_SAME_STATE:
                    state_counter = Counter(list(track_info['state_history'])[-MIN_FRAMES_SAME_STATE:])
                    most_common_state = state_counter.most_common(1)[0][0]
                    
                    if track_info['state'] != most_common_state:
                        if state_counter[most_common_state] >= MIN_FRAMES_SAME_STATE - 2:
                            track_info['state'] = most_common_state
                else:
                    track_info['state'] = state
                
                state = track_info['state']
                
            else:
                # New person detected
                track_id = next_id
                next_id += 1
                active_ids.add(track_id)
                
                # Initialize new track
                track_info = track_history[track_id]
                track_info['state'] = "stopped"
                track_info['state_history'] = deque(["stopped"], maxlen=STATE_HISTORY_LENGTH)
                track_info['last_centroid'] = centroid
                track_info['height'] = height_val
                track_info['id'] = track_id
                track_info['prev_hip_center'] = hip_center
                track_info['hip_movement'] = deque(maxlen=10)
                
                # Initialize ankle positions relative to hip
                track_info['ankle_positions']['left'] = deque(maxlen=30)
                track_info['ankle_positions']['right'] = deque(maxlen=30)
                
                if hip_center:
                    if keypoints[LEFT_ANKLE][2] > 0.3:
                        rel_x = keypoints[LEFT_ANKLE][0] - hip_center[0]
                        rel_y = keypoints[LEFT_ANKLE][1] - hip_center[1]
                        track_info['ankle_positions']['left'].append((rel_x, rel_y))
                    
                    if keypoints[RIGHT_ANKLE][2] > 0.3:
                        rel_x = keypoints[RIGHT_ANKLE][0] - hip_center[0]
                        rel_y = keypoints[RIGHT_ANKLE][1] - hip_center[1]
                        track_info['ankle_positions']['right'].append((rel_x, rel_y))
                
                # Initialize step events
                track_info['step_events']['left'] = deque(maxlen=20)
                track_info['step_events']['right'] = deque(maxlen=20)
                track_info['step_lengths'] = deque(maxlen=10)
                
                state = "stopped"
                track_id = track_id
            
            # Store for annotation
            annotated_detections.append({
                'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                'state': state,
                'track_id': track_id,
                'keypoints': keypoints,
                'cadence': cadence,
                'step_length': avg_step_length,
                'hip_movement': hip_movement
            })
        
        # Remove old tracks
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
            keypoints = detection['keypoints']
            cadence = detection['cadence']
            step_length = detection['step_length']
            hip_movement = detection['hip_movement']
            
            color = COLORS[state]
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw state text with ID
            state_text = f"ID:{track_id} {state.upper()}"
            cv2.putText(frame, state_text, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # Draw cadence information
            cadence_text = f"Cadence: {cadence:.1f} steps/s"
            cv2.putText(frame, cadence_text, (x1, y1 - 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
            
            # Draw step length information
            step_text = f"Step: {step_length:.2f} rel"
            cv2.putText(frame, step_text, (x1, y1 - 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
            
            # Draw hip movement information
            hip_text = f"Hip Move: {hip_movement:.2f}"
            cv2.putText(frame, hip_text, (x1, y1 - 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
            
            # Draw ankle markers
            if keypoints[LEFT_ANKLE][2] > 0.3:
                cv2.circle(frame, 
                          (int(keypoints[LEFT_ANKLE][0]), int(keypoints[LEFT_ANKLE][1])),
                          8, (0, 255, 255), -1)
            
            if keypoints[RIGHT_ANKLE][2] > 0.3:
                cv2.circle(frame, 
                          (int(keypoints[RIGHT_ANKLE][0]), int(keypoints[RIGHT_ANKLE][1])),
                          8, (0, 255, 255), -1)
            
            # Draw hip center
            if keypoints[LEFT_HIP][2] > 0.3 and keypoints[RIGHT_HIP][2] > 0.3:
                hip_center_x = int((keypoints[LEFT_HIP][0] + keypoints[RIGHT_HIP][0]) / 2)
                hip_center_y = int((keypoints[LEFT_HIP][1] + keypoints[RIGHT_HIP][1]) / 2)
                cv2.circle(frame, (hip_center_x, hip_center_y), 6, (255, 0, 255), -1)
        
        # Write frame to output
        out.write(frame)
    
    # Cleanup
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Processing complete. Output saved to: {output_path}")

if __name__ == "__main__":
    main()