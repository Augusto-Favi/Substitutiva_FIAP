from traffic_tracker import VideoAnalyzer
import os

# AJUSTES ================================================================================================================================================

# Model configuration
MODEL_NAME = "yolo11n"

# Speed calculation parameters
DISTANCE_M = 8  # Real-world distance between lines in meters
CONFIDENCE_THRESH = 0.2  # Minimum detection confidence

# =========================================================================================================================================================

print("Iniciando a análise de vídeo...")
analyser = VideoAnalyzer(
    model_name=MODEL_NAME,
    distance_m=DISTANCE_M,
    confidence_thresh=CONFIDENCE_THRESH,
)

VIDEO_FILE = "traffic3.mp4"

analyser.process_video(VIDEO_FILE)