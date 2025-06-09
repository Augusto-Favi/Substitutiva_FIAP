import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict, deque
from collections import Counter
from tqdm import tqdm

# Configuration
OUTPUT_POSFIX = "angles"
MODEL = "yolo11n"
VIDEO_FILE = "running4.mkv"  # Input video file

# Model paths
MODEL_POSE = f"./models/{MODEL}-pose.pt"
MODEL_OBJ = f"./models/{MODEL}.pt"

# Video paths
VIDEO_INPUT = f"videos/input/{VIDEO_FILE}"
OUTPUT_DIR = "videos/output/"

# Movement state thresholds based on knee angles (degrees)
TH_STOPPED = 175  # Legs mostly straight
TH_WALKING = 160  # Moderate bend
TH_TROTTING = 145 # More pronounced bend

# State persistence settings
MIN_FRAMES_SAME_STATE = 8  # Minimum frames to maintain a state before switching
STATE_HISTORY_LENGTH = 15  # Number of frames to consider for state smoothing

# Visualization colors
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
    xi_min = max(box1[0], box2[0])
    yi_min = max(box1[1], box2[1])
    xi_max = min(box1[2], box2[2])
    yi_max = min(box1[3], box2[3])
    inter_area = max(0, xi_max - xi_min) * max(0, yi_max - yi_min)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter_area
    return inter_area / union if union > 0 else 0

def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba = a - b
    bc = c - b
    if np.linalg.norm(ba) == 0 or np.linalg.norm(bc) == 0:
        return 180
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    return np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))

def get_leg_angles(kp, conf_threshold=0.3):
    angles = []
    if kp[LEFT_HIP][2] > conf_threshold and kp[LEFT_KNEE][2] > conf_threshold and kp[LEFT_ANKLE][2] > conf_threshold:
        angles.append(calculate_angle(kp[LEFT_HIP][:2], kp[LEFT_KNEE][:2], kp[LEFT_ANKLE][:2]))
    if kp[RIGHT_HIP][2] > conf_threshold and kp[RIGHT_KNEE][2] > conf_threshold and kp[RIGHT_ANKLE][2] > conf_threshold:
        angles.append(calculate_angle(kp[RIGHT_HIP][:2], kp[RIGHT_KNEE][:2], kp[RIGHT_ANKLE][:2]))
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

    with tqdm(total=total_frames, desc="Processing video", unit="frame") as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            pbar.update(1)

            obj_results = model_obj(frame, conf=0.5, classes=[0], verbose=False)[0]
            person_boxes = []
            if obj_results.boxes:
                for box in obj_results.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    conf = float(box.conf[0])
                    person_boxes.append((x1, y1, x2, y2, conf))

            pose_results = model_pose(frame, conf=0.5, classes=[0], verbose=False)[0]
            pose_data = []
            if pose_results.keypoints:
                for kp, box in zip(pose_results.keypoints, pose_results.boxes):
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    keypoints = kp.data[0].cpu().numpy()
                    conf = float(box.conf[0])
                    pose_data.append((x1, y1, x2, y2, conf, keypoints))

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

            current_detections = []
            for p_box, pose in matched_data:
                x1, y1, x2, y2, conf = p_box
                centroid = ((x1 + x2) // 2, (y1 + y2) // 2)
                current_detections.append({
                    'centroid': centroid,
                    'p_box': p_box,
                    'pose': pose,
                    'keypoints': pose[5],
                    'height': y2 - y1
                })

            for detection in current_detections:
                centroid = detection['centroid']
                keypoints = detection['keypoints']
                height = detection['height']
                angles = get_leg_angles(keypoints)
                avg_angle = np.mean(angles) if angles else 180

                best_id = None
                min_dist = float('inf')
                for tid, tinfo in track_history.items():
                    if tinfo['last_centroid'] is None:
                        continue
                    dist = np.linalg.norm(np.array(centroid) - np.array(tinfo['last_centroid'])) / height
                    if dist < min_dist and dist < 0.5:
                        min_dist = dist
                        best_id = tid

                if best_id is not None:
                    tinfo = track_history[best_id]
                    tinfo['last_centroid'] = centroid
                    tinfo['last_angles'].append(avg_angle)
                else:
                    track_history[next_id]['last_centroid'] = centroid
                    track_history[next_id]['last_angles'].append(avg_angle)
                    next_id += 1

            out.write(frame)

    cap.release()
    out.release()
    print(f"Finished. Output saved to: {output_path}")

if __name__ == "__main__":
    main()
