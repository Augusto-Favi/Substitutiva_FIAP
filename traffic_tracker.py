import cv2
import shutil
import subprocess
import numpy as np
from ultralytics import YOLO
from tqdm import tqdm
import os
from collections import defaultdict
import json
from datetime import datetime

# Parametros =================================================================================================================================================

CLASS_IDS = [2, 3, 5, 7]  # COCO class IDs for vehicles (car, motorcycle, bus, truck)
CLASS_NAMES = {2: "Carro", 3: "Motocicleta", 5: "Onibús", 7: "Caminhao"}  # Mapping of class IDs to names
MIN_DIRECTION_FRAMES = 5  # Minimum frames to determine direction
SPEED_SMOOTHING = 5  # Number of frames to average speed over

# Vizualicação =================================================================================================================================================
TRACK_PERSIST = True
SHOW_SPEED = True
SHOW_IDS = True
SHOW_LINES = True
SHOW_BOXES = True
SHOW_COUNTS = True

DIRECTION_COLORS = {
    "up": (0, 255, 255),  # YELLOW
    "down": (255, 0, 0)    # BLUE
}
LINE_COLOR = (0, 0, 255)    # RED
TEXT_COLOR = (0, 255, 0)    # GREEN


# MAIN FUNCTION ==============================================================================================================================================

class VideoAnalyzer():
    def __init__(self, model_name, distance_m, confidence_thresh):
        self.models_folder = "./models/"
        os.makedirs(self.models_folder, exist_ok=True)
        self.model_name = model_name
        self.model = YOLO(os.path.join(self.models_folder, f'{model_name}.pt'))
        
        self.input_video = "videos/input/"
        self.output_dir = "videos/output/"
        self.report_dir = "reports/"
        self.output_suffix = "_LIVE_SPEED"

        self.distance_m = distance_m
        self.confidence_thresh = confidence_thresh
        
        # For JSON reporting
        self.vehicle_records = []
        self.processing_date = datetime.now().isoformat()

    def get_output_suffix(self):
        return self.output_suffix

    def process_video(self, video_file):
        self.vehicle_records = []
        
        # Config directories
        VIDEO_INPUT = os.path.join(self.input_video, video_file)
        os.makedirs(self.output_dir, exist_ok=True)
        output_path = os.path.join(self.output_dir, video_file.replace('.mp4', f'{self.output_suffix}.webm'))
        
        # JSON report path
        json_report_path = os.path.join(self.report_dir, video_file.replace('.mp4', '_report.json'))

        # Initialize video capture
        cap = cv2.VideoCapture(VIDEO_INPUT)
        if not cap.isOpened():
            raise FileNotFoundError(f"Could not open video: {VIDEO_INPUT}")
        
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Calculate line positions (horizontal lines for vertical movement)
        line_top = (height // 2) + 80
        line_bottom = line_top + 80
        
        # Initialize video writer with VP9 codec for web compatibility
        fourcc = cv2.VideoWriter_fourcc(*'VP90')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # If VP9 not available, fall back to H264
        if not out.isOpened():
            print("VP9 codec not available, falling back to H264")
            output_path = output_path.replace('.webm', '.mp4')
            fourcc = cv2.VideoWriter_fourcc(*'avc1')  # H264 codec
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            if not out.isOpened():
                print("H264 codec not available, falling back to MP4V")
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Tracking data storage
        vehicle_info = defaultdict(lambda: {
            'first_cross_time': None,
            'last_cross_time': None,
            'speed': None,
            'counted': False,
            'direction': None,
            'positions': [],
            'speed_history': [],
            'crossed_top': False,
            'crossed_bottom': False,
            'class_id': None,
            'last_confidence': None,
            'record_added': False
        })
        
        # Counters
        down_count = 0
        up_count = 0
        frame_count = 0

       # Process video frames
        for _ in tqdm(range(total_frames), desc="Processing Frames"):
            ret, frame = cap.read()
            if not ret:
                break
            
            current_time = frame_count / fps
            frame_count += 1
            
            # Run object tracking
            results = self.model.track(
                frame,
                persist=TRACK_PERSIST,
                classes=CLASS_IDS,
                conf=self.confidence_thresh,
                verbose=False
            )
            
            # Process detected vehicles
            if results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                track_ids = results[0].boxes.id.int().cpu().tolist()
                confidences = results[0].boxes.conf.cpu().numpy()
                class_ids = results[0].boxes.cls.int().cpu().tolist()
                
                for box, track_id, conf, cls_id in zip(boxes, track_ids, confidences, class_ids):
                    x1, y1, x2, y2 = box
                    cx = int((x1 + x2) / 2)  # Center x-coordinate
                    cy = int((y1 + y2) / 2)  # Center y-coordinate

                    if frame_count < 36 and (line_top <= y1 <= line_bottom):
                        continue
                    
                    # Initialize vehicle info
                    info = vehicle_info[track_id]
                    info['positions'].append((cx, cy))
                    info['last_confidence'] = float(conf)  # Update confidence
                    
                    # Set class ID on first detection
                    if info['class_id'] is None:
                        info['class_id'] = cls_id
                    
                    # Determine direction after sufficient frames
                    if len(info['positions']) >= MIN_DIRECTION_FRAMES:
                        # Calculate movement direction based on y-position history
                        y_positions = [pos[1] for pos in info['positions'][-MIN_DIRECTION_FRAMES:]]
                        y_movement = np.mean(np.diff(y_positions))
                        
                        if y_movement > 0.5:
                            info['direction'] = 'down'  # Moving down (increasing y)
                        elif y_movement < -0.5:
                            info['direction'] = 'up'   # Moving up (decreasing y)
                    
                    # Line crossing detection for DOWN-moving vehicles
                    if info['direction'] == 'down' and not info['counted']:
                        # Top line crossing
                        if not info['crossed_top'] and y1 >= line_top:
                            info['crossed_top'] = True
                            info['first_cross_time'] = current_time
                        
                        # Bottom line crossing after top line
                        if info['crossed_top'] and not info['crossed_bottom'] and y1 >= line_bottom:
                            info['crossed_bottom'] = True
                            info['last_cross_time'] = current_time
                            time_diff = info['last_cross_time'] - info['first_cross_time']
                            
                            if time_diff > 0:
                                speed_mps = self.distance_m / time_diff
                                speed_kph = speed_mps * 3.6
                                
                                # Smooth speed over multiple frames
                                info['speed_history'].append(speed_kph)
                                if len(info['speed_history']) > SPEED_SMOOTHING:
                                    info['speed_history'].pop(0)
                                
                                avg_speed = np.mean(info['speed_history'])
                                info['speed'] = avg_speed
                                info['counted'] = True
                                down_count += 1
                    
                    # FIXED: Line crossing detection for UP-moving vehicles
                    elif info['direction'] == 'up' and not info['counted']:
                        # BOTTOM line crossing FIRST for upward vehicles
                        if not info['crossed_bottom'] and y1 <= line_bottom:
                            info['crossed_bottom'] = True
                            info['first_cross_time'] = current_time
                        
                        # TOP line crossing SECOND for upward vehicles
                        if info['crossed_bottom'] and not info['crossed_top'] and y1 <= line_top:
                            info['crossed_top'] = True
                            info['last_cross_time'] = current_time
                            time_diff = info['last_cross_time'] - info['first_cross_time']
                            
                            if time_diff > 0:
                                speed_mps = self.distance_m / time_diff
                                speed_kph = speed_mps * 3.6
                                
                                # Smooth speed over multiple frames
                                info['speed_history'].append(speed_kph)
                                if len(info['speed_history']) > SPEED_SMOOTHING:
                                    info['speed_history'].pop(0)
                                
                                avg_speed = np.mean(info['speed_history'])
                                info['speed'] = avg_speed
                                info['counted'] = True
                                up_count += 1
                    
                    # Create JSON record when vehicle is counted and not already added
                    if info['counted'] and not info['record_added'] and info['speed'] is not None:
                        # Get vehicle type name
                        vehicle_type = CLASS_NAMES.get(info['class_id'], "unknown")
                        
                        # Create record
                        record = {
                            "unique_id": int(track_id),
                            "timestamps": {
                                "first_cross": float(info['first_cross_time']),
                                "last_cross": float(info['last_cross_time'])
                            },
                            "registered_speed": float(info['speed']),
                            "type_of_vehicle": vehicle_type,
                            "side": info['direction'],
                            "confidence": info['last_confidence'],
                            "model_info": {
                                "distance_M": float(self.distance_m),
                                "model_name": self.model_name,
                                "confidence_threshold": float(self.confidence_thresh),
                                "date": self.processing_date
                            }
                        }
                        self.vehicle_records.append(record)
                        info['record_added'] = True  # Mark as added to prevent duplicates
                    
                    # Visualization
                    box_color = DIRECTION_COLORS.get(info['direction'], (0, 255, 0))
                    
                    if SHOW_BOXES:
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), 
                                    box_color, 2)
                    
                    if SHOW_IDS:
                        cv2.putText(frame, f"ID: {track_id}", 
                                    (int(x1), int(y1) - 10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, TEXT_COLOR, 2)
                    
                    if SHOW_SPEED and info['speed'] is not None:
                        speed_text = f"{info['speed']:.1f} km/h"
                        cv2.putText(frame, speed_text, 
                                    (int(x1), int(y1) - 40 if SHOW_IDS else int(y1) - 10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, TEXT_COLOR, 2)
            
            # Draw measurement lines
            if SHOW_LINES:
                cv2.line(frame, (0, line_top), (width, line_top), LINE_COLOR, 2)
                cv2.line(frame, (0, line_bottom), (width, line_bottom), LINE_COLOR, 2)
                
                # Add line labels
                cv2.putText(frame, "TOP LINE", (width - 150, line_top - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, LINE_COLOR, 2)
                cv2.putText(frame, "BOTTOM LINE", (width - 180, line_bottom - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, LINE_COLOR, 2)
                cv2.putText(frame, f"{self.distance_m}m", (width // 2 - 30, (line_top + line_bottom) // 2), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Display vehicle counts
            if SHOW_COUNTS:
                cv2.rectangle(frame, (width - 250, 10), (width - 10, 130), (40, 40, 40), -1)
                cv2.putText(frame, "VEHICLE COUNTS", (width - 240, 40), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.putText(frame, f"Down: {down_count}", (width - 240, 70), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, DIRECTION_COLORS['down'], 2)
                cv2.putText(frame, f"Up: {up_count}", (width - 240, 100), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, DIRECTION_COLORS['up'], 2)
                cv2.putText(frame, f"Total: {down_count + up_count}", (width - 240, 130), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, TEXT_COLOR, 2)
            
            # Write frame to output
            out.write(frame)
        
        cap.release()
        out.release()
        cv2.destroyAllWindows()

        # After processing, convert to web-friendly format if needed
        if not output_path.endswith('.webm'):
            try:
                webm_path = output_path.replace('.mp4', '.webm')
                # Use FFmpeg for conversion if available
                if shutil.which('ffmpeg'):
                    cmd = [
                        'ffmpeg', '-i', output_path,
                        '-c:v', 'libvpx-vp9', '-b:v', '2M', 
                        '-c:a', 'libopus', '-b:a', '64k',
                        '-f', 'webm', webm_path
                    ]
                    subprocess.run(cmd, check=True)
                    os.remove(output_path)  # Remove original
                    output_path = webm_path
                    print(f"Converted video to web-friendly format: {webm_path}")
            except Exception as e:
                print(f"Video conversion failed: {e}")

        
        # Save JSON report - pass counts as arguments
        self.save_json_report(json_report_path, video_file, down_count, up_count)
        
        print(f"Processing complete. Output saved to: {output_path}")
        print(f"Down vehicles: {down_count}, Up vehicles: {up_count}, Total: {down_count + up_count}")
        
    def save_json_report(self, json_path, video_file, down_count, up_count):
        """Save collected data to JSON report"""
        report = {
            "processing_date": self.processing_date,
            "video_file": video_file,
            "model_info": {
                "model_name": self.model_name,
                "distance_M": float(self.distance_m),
                "confidence_threshold": float(self.confidence_thresh)
            },
            "vehicle_count": {
                "down": down_count,
                "up": up_count,
                "total": down_count + up_count
            },
            "vehicles": self.vehicle_records
        }
        
        with open(json_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"JSON report saved to: {json_path}")