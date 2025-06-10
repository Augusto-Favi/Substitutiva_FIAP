import json
import datetime
from pathlib import Path

class TrafficReportGeneratorWithVideo:
    def __init__(self, json_file_path, video_path="../videos/output/traffic3_LIVE_SPEED.mp4"):
        with open(json_file_path, 'r') as f:
            self.data = json.load(f)
        self.video_path = video_path
        self.anomalies = self._analyze_anomalies()
    
    def _analyze_anomalies(self):
        anomalies = []
        for vehicle in self.data['vehicles']:
            speed = vehicle['registered_speed']
            if speed > 70:
                severity_level = min(4, max(1, int((speed - 70) // 10) + 1))
                
                anomalies.append({
                    'unique_id': vehicle['unique_id'],
                    'speed': round(speed, 2),
                    'type_of_vehicle': vehicle['type_of_vehicle'],
                    'side': vehicle['side'],
                    'severity_level': severity_level,
                    'first_cross': round(vehicle['timestamps']['first_cross'], 2),
                    'last_cross': round(vehicle['timestamps']['last_cross'], 2),
                    'confidence': round(vehicle['confidence'], 3)
                })
        return sorted(anomalies, key=lambda x: x['severity_level'], reverse=True)
    
    def _get_severity_color(self, level):
        colors = {
            1: '#FFA500',  # Orange
            2: '#FF6B35',  # Red-Orange
            3: '#FF4444',  # Red
            4: '#CC0000'   # Dark Red
        }
        return colors.get(level, '#FFA500')
    
    def _get_severity_label(self, level):
        labels = {
            1: 'Baixa',
            2: 'Média',
            3: 'Alta',
            4: 'Crítica'
        }
        return labels.get(level, 'Baixa')
    
    def generate_html_report(self, output_file='traffic_report_with_video.html'):
        total_vehicles = self.data['vehicle_count']['total']
        up_vehicles = self.data['vehicle_count']['up']
        down_vehicles = self.data['vehicle_count']['down']
        total_anomalies = len(self.anomalies)
        
        # Calculate average speed
        speeds = [v['registered_speed'] for v in self.data['vehicles']]
        avg_speed = sum(speeds) / len(speeds) if speeds else 0
        
        html_content = f"""
<!DOCTYPE html>
<html lang="pt-BR">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Relatório de Tráfego - Dashboard com Vídeo</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }}
        
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 15px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.1);
            overflow: hidden;
        }}
        
        .header {{
            background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }}
        
        .header h1 {{
            font-size: 2.5em;
            margin-bottom: 10px;
            font-weight: 300;
        }}
        
        .header .subtitle {{
            font-size: 1.1em;
            opacity: 0.8;
        }}
        
        .main-content {{
            display: grid;
            grid-template-columns: 1fr 400px;
            gap: 30px;
            padding: 30px;
        }}
        
        .left-panel {{
            display: flex;
            flex-direction: column;
            gap: 30px;
        }}
        
        .right-panel {{
            background: #f8f9fa;
            padding: 20px;
            border-radius: 15px;
            max-height: calc(100vh - 200px);
            overflow-y: auto;
        }}
        
        .video-section {{
            background: #000;
            border-radius: 15px;
            overflow: hidden;
            box-shadow: 0 10px 30px rgba(0,0,0,0.3);
        }}
        
        .video-container {{
            position: relative;
            width: 100%;
            padding-bottom: 56.25%; /* 16:9 aspect ratio */
            height: 0;
        }}
        
        /* Updated iframe styles */
        .video-container iframe {{
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            border: none;
        }}
        
        .video-controls {{
            background: rgba(0,0,0,0.8);
            color: white;
            padding: 15px;
            display: flex;
            align-items: center;
            gap: 15px;
        }}
        
        .video-info {{
            flex: 1;
            font-size: 0.9em;
        }}
        
        .current-time {{
            font-weight: bold;
            color: #4CAF50;
        }}
        
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
        }}
        
        .stat-card {{
            background: white;
            padding: 25px;
            border-radius: 10px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.08);
            text-align: center;
            transition: transform 0.3s ease;
        }}
        
        .stat-card:hover {{
            transform: translateY(-5px);
        }}
        
        .stat-number {{
            font-size: 2.2em;
            font-weight: bold;
            color: #2c3e50;
            margin-bottom: 10px;
        }}
        
        .stat-label {{
            color: #7f8c8d;
            font-size: 1em;
        }}
        
        .anomalies-section {{
            background: white;
            border-radius: 15px;
            padding: 25px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.08);
        }}
        
        .section-title {{
            font-size: 1.8em;
            color: #2c3e50;
            margin-bottom: 20px;
            text-align: center;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 10px;
        }}
        
        .panel-title {{
            font-size: 1.5em;
            color: #2c3e50;
            margin-bottom: 20px;
            text-align: center;
        }}
        
        .anomaly-card {{
            background: white;
            border-left: 5px solid;
            margin-bottom: 15px;
            padding: 15px;
            border-radius: 0 10px 10px 0;
            box-shadow: 0 3px 10px rgba(0,0,0,0.1);
            cursor: pointer;
            transition: all 0.3s ease;
        }}
        
        .anomaly-card:hover {{
            transform: translateX(5px);
            box-shadow: 0 5px 20px rgba(0,0,0,0.15);
        }}
        
        .anomaly-card.playing {{
            background: #e8f5e8;
            border-left-width: 8px;
        }}
        
        .anomaly-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 8px;
        }}
        
        .anomaly-id {{
            font-weight: bold;
            font-size: 1.1em;
            color: #2c3e50;
        }}
        
        .severity-badge {{
            padding: 4px 12px;
            border-radius: 15px;
            color: white;
            font-weight: bold;
            font-size: 0.8em;
        }}
        
        .anomaly-speed {{
            font-size: 1.2em;
            font-weight: bold;
            color: #e74c3c;
            margin-bottom: 8px;
        }}
        
        .anomaly-timestamp {{
            font-size: 0.9em;
            color: #7f8c8d;
            display: flex;
            align-items: center;
            gap: 5px;
        }}
        
        .play-icon {{
            width: 16px;
            height: 16px;
            background: #4CAF50;
            border-radius: 50%;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-size: 10px;
        }}
        
        .anomaly-details {{
            display: none;
            margin-top: 15px;
            padding-top: 15px;
            border-top: 1px solid #ecf0f1;
        }}
        
        .anomaly-details.show {{
            display: block;
            animation: slideDown 0.3s ease;
        }}
        
        @keyframes slideDown {{
            from {{
                opacity: 0;
                max-height: 0;
            }}
            to {{
                opacity: 1;
                max-height: 200px;
            }}
        }}
        
        .detail-row {{
            display: flex;
            justify-content: space-between;
            margin-bottom: 8px;
            font-size: 0.9em;
        }}
        
        .detail-label {{
            font-weight: bold;
            color: #7f8c8d;
        }}
        
        .detail-value {{
            color: #2c3e50;
        }}
        
        .footer {{
            background: #34495e;
            color: white;
            text-align: center;
            padding: 20px;
            font-size: 0.9em;
        }}
        
        .no-anomalies {{
            text-align: center;
            padding: 40px;
            color: #27ae60;
            font-size: 1.2em;
        }}
        
        @media (max-width: 1200px) {{
            .main-content {{
                grid-template-columns: 1fr;
            }}
            
            .right-panel {{
                max-height: 500px;
            }}
        }}
        
        @media (max-width: 768px) {{
            .stats-grid {{
                grid-template-columns: repeat(2, 1fr);
            }}
            
            .stat-number {{
                font-size: 1.8em;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Dashboard de Tráfego com Vídeo</h1>
            <div class="subtitle">Relatório de Análise de Velocidade - {self.data['video_file']}</div>
            <div class="subtitle">Processado em: {datetime.datetime.fromisoformat(self.data['processing_date'].replace('Z', '+00:00')).strftime('%d/%m/%Y às %H:%M')}</div>
        </div>
        
        <div class="main-content">
            <div class="left-panel">
                <div class="video-section">
                    <div class="video-container">
                        <!-- Replaced video element with iframe -->
                        <iframe id="trafficVideo" 
                                src="{self.video_path}" 
                                frameborder="0" 
                                allowfullscreen
                                allow="autoplay">
                        </iframe>
                    </div>
                    <div class="video-controls">
                        <div class="video-info">
                            <div>Clique nas anomalias para navegar no vídeo</div>
                        </div>
                    </div>
                </div>
                
                <div class="stats-grid">
                    <div class="stat-card">
                        <div class="stat-number">{total_vehicles}</div>
                        <div class="stat-label">Total de Veículos</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-number">{up_vehicles}</div>
                        <div class="stat-label">Direção Norte</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-number">{down_vehicles}</div>
                        <div class="stat-label">Direção Sul</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-number">{round(avg_speed, 1)}</div>
                        <div class="stat-label">Velocidade Média (km/h)</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-number">{total_anomalies}</div>
                        <div class="stat-label">Anomalias Detectadas</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-number">{self.data['model_info']['distance_M']}m</div>
                        <div class="stat-label">Distância de Medição</div>
                    </div>
                </div>
            </div>
            
            <div class="right-panel">
                <h2 class="panel-title">🚨 Anomalias de Velocidade</h2>
"""
        
        if self.anomalies:
            for anomaly in self.anomalies:
                color = self._get_severity_color(anomaly['severity_level'])
                severity_label = self._get_severity_label(anomaly['severity_level'])
                
                html_content += f"""
                <div class="anomaly-card" style="border-left-color: {color};" onclick="seekToAnomaly({anomaly['first_cross']}, {anomaly['unique_id']})">
                    <div class="anomaly-header">
                        <div class="anomaly-id">Veículo #{anomaly['unique_id']}</div>
                        <div class="severity-badge" style="background-color: {color};">
                            {severity_label}
                        </div>
                    </div>
                    <div class="anomaly-speed">{anomaly['speed']} km/h</div>
                    <div class="anomaly-timestamp">
                        <span class="play-icon">▶</span>
                        {anomaly['first_cross']}s - {anomaly['last_cross']}s
                    </div>
                    <div class="anomaly-details" id="anomaly-{anomaly['unique_id']}">
                        <div class="detail-row">
                            <span class="detail-label">Tipo:</span>
                            <span class="detail-value">{anomaly['type_of_vehicle'].title()}</span>
                        </div>
                        <div class="detail-row">
                            <span class="detail-label">Direção:</span>
                            <span class="detail-value">{'Norte' if anomaly['side'] == 'up' else 'Sul'}</span>
                        </div>
                        <div class="detail-row">
                            <span class="detail-label">Confiança:</span>
                            <span class="detail-value">{anomaly['confidence'] * 100:.1f}%</span>
                        </div>
                        <div class="detail-row">
                            <span class="detail-label">Severidade:</span>
                            <span class="detail-value">{anomaly['severity_level']}/4</span>
                        </div>
                    </div>
                </div>
"""
        else:
            html_content += """
                <div class="no-anomalies">
                    ✅ Nenhuma anomalia detectada!
                </div>
"""
        
        html_content += f"""
            </div>
        </div>
        
        <div class="footer">
            Relatório gerado automaticamente usando modelo {self.data['model_info']['model_name']} 
            com limiar de confiança de {self.data['model_info']['confidence_threshold']}
        </div>
    </div>
    
    <script>
        const iframe = document.getElementById('trafficVideo');
        let currentPlayingCard = null;
        
        // Function to seek to anomaly timestamp
        function seekToAnomaly(timestamp, vehicleId) {{
            // Remove previous playing indicator
            if (currentPlayingCard) {{
                currentPlayingCard.classList.remove('playing');
            }}
            
            // Add playing indicator to current card
            const currentCard = event.currentTarget;
            currentCard.classList.add('playing');
            currentPlayingCard = currentCard;
            
            // Create new source URL with timestamp parameter
            const baseUrl = "{self.video_path}";
            const newUrl = `${{baseUrl}}#t=${{timestamp}}`;
            
            // Update iframe source
            iframe.src = newUrl;
            
            // Toggle details
            toggleDetails(`anomaly-${{vehicleId}}`);
            
            // Remove playing indicator after 3 seconds
            setTimeout(() => {{
                if (currentPlayingCard) {{
                    currentPlayingCard.classList.remove('playing');
                }}
            }}, 3000);
        }}
        
        function toggleDetails(elementId) {{
            const details = document.getElementById(elementId);
            if (details.classList.contains('show')) {{
                details.classList.remove('show');
            }} else {{
                // Hide all other details first
                document.querySelectorAll('.anomaly-details.show').forEach(el => {{
                    el.classList.remove('show');
                }});
                // Show the clicked one
                details.classList.add('show');
            }}
        }}
        
        // Add hover effects
        document.addEventListener('DOMContentLoaded', function() {{
            const cards = document.querySelectorAll('.anomaly-card');
            cards.forEach(card => {{
                card.addEventListener('mouseenter', function() {{
                    if (!this.classList.contains('playing')) {{
                        this.style.transform = 'translateX(5px)';
                    }}
                }});
                card.addEventListener('mouseleave', function() {{
                    if (!this.classList.contains('playing')) {{
                        this.style.transform = 'translateX(0)';
                    }}
                }});
            }});
        }});
    </script>
</body>
</html>
"""
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        return output_file

if __name__ == '__main__':
    # Create the reports directory if it doesn't exist
    Path('reports').mkdir(exist_ok=True)
    
    # Copy the JSON file to the reports directory if it doesn't exist
    if not Path('reports/traffic3_report.json').exists():
        import shutil
        shutil.copy('/home/ubuntu/upload/pasted_content.txt', 'reports/traffic3_report.json')
    
    # Generate the report with video
    generator = TrafficReportGeneratorWithVideo('reports/traffic3_report.json')
    output_file = generator.generate_html_report('reports/traffic_dashboard_with_video.html')
    
    print(f"Relatório HTML com vídeo gerado com sucesso: {output_file}")
    print(f"Total de anomalias encontradas: {len(generator.anomalies)}")
    print("\nFuncionalidades do vídeo:")
    print("- Clique nas anomalias para navegar para o timestamp")
    print("- Os vídeos agora são incorporados usando iframes para melhor compatibilidade")