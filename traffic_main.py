import os
import webbrowser
from traffic_tracker import VideoAnalyzer
from traffic_report_generator import TrafficReportGeneratorWithVideo

# AJUSTES ================================================================================================================================================

# Model configuration
MODEL_NAME = "yolo11n"
VIDEO_FILE = "traffic2"

# Speed calculation parameters
DISTANCE_M = 12  # Distancia em metros entre as linhas
CONFIDENCE_THRESH = 0.2  # Treshold pra veiculos

# PROCESSA VIDEO =========================================================================================================================================================

print("Iniciando a análise de vídeo...")
analyser = VideoAnalyzer(
    model_name=MODEL_NAME,
    distance_m=DISTANCE_M,
    confidence_thresh=CONFIDENCE_THRESH,
)

analyser.process_video(f'{VIDEO_FILE}.mp4')

# GERA RELÁTORIO =========================================================================================================================================================
generator = TrafficReportGeneratorWithVideo(video_path = f"../videos/output/{VIDEO_FILE}{analyser.get_output_suffix()}.webm", json_file_path = f"reports/{VIDEO_FILE}_report.json")
output_file = generator.generate_html_report(output_file = f"reports/{VIDEO_FILE}.html")

print(f"\nRelatório HTML com vídeo gerado com sucesso: {output_file}")
print(f"Total de anomalias encontradas: {len(generator.anomalies)}")
webbrowser.open(f"file://{os.path.abspath(output_file)}")