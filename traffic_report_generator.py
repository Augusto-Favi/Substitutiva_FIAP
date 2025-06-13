import json
import datetime
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import io
import base64

class TrafficReportGeneratorWithVideo:
    def __init__(self, json_file_path, video_path):
        with open(json_file_path, 'r') as f:
            self.data = json.load(f)
        self.video_path = video_path
        self.anomalies = self._analyze_anomalies()
    
    def _analyze_anomalies(self):
        anomalies = []
        for vehicle in self.data['vehicles']:
            speed = vehicle['registered_speed']
            if speed > 100:
                severity_level = min(4, max(1, int((speed - 100) // 10) + 1))
                
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
            1: "#2451A7",  # Orange
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

    def _analyze_anomalies_by_direction(self):
        anomalies_by_direction = defaultdict(int)
        for anomaly in self.anomalies:
            direction = 'Norte' if anomaly['side'] == 'up' else 'Sul'
            anomalies_by_direction[direction] += 1
        return dict(anomalies_by_direction)

    def _analyze_mean_speed_by_direction(self):
        speed_by_direction = defaultdict(lambda: {'total_speed': 0, 'count': 0})
        for vehicle in self.data['vehicles']:
            direction = 'Norte' if vehicle['side'] == 'up' else 'Sul'
            speed_by_direction[direction]['total_speed'] += vehicle['registered_speed']
            speed_by_direction[direction]['count'] += 1
        
        mean_speed_by_direction = {}
        for direction, data in speed_by_direction.items():
            mean_speed_by_direction[direction] = round(data['total_speed'] / data['count'], 1) if data['count'] > 0 else 0
        return mean_speed_by_direction

    def _analyze_vehicle_counts_by_type(self):
        vehicle_counts_by_type = defaultdict(int)
        for vehicle in self.data['vehicles']:
            vehicle_counts_by_type[vehicle['type_of_vehicle']] += 1
        return dict(vehicle_counts_by_type)

    def _analyze_anomalies_by_vehicle_type(self):
        anomalies_by_type = defaultdict(int)
        for anomaly in self.anomalies:
            anomalies_by_type[anomaly['type_of_vehicle']] += 1
        return dict(anomalies_by_type)

    def _generate_anomaly_percentage_plot(self):
        anomaly_types = defaultdict(int)
        severity_levels_by_label = {}

        for anomaly in self.anomalies:
            label = self._get_severity_label(anomaly['severity_level'])
            anomaly_types[label] += 1
            severity_levels_by_label[label] = anomaly['severity_level']  # Garantir o mapeamento do label ao nível

        if not anomaly_types:
            return ""

        labels = list(anomaly_types.keys())
        sizes = list(anomaly_types.values())
        colors = [self._get_severity_color(severity_levels_by_label[label]) for label in labels]

        fig, ax = plt.subplots(figsize=(8, 6))
        wedges, texts, autotexts = ax.pie(sizes, labels=labels, autopct='%1.1f%%', 
                                         startangle=90, colors=colors)

        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')

        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        plt.close(fig)
        return base64.b64encode(buf.getvalue()).decode('utf-8')

    def _generate_vehicle_type_chart(self):
        vehicle_counts = self._analyze_vehicle_counts_by_type()
        
        if not vehicle_counts:
            return ""

        labels = list(vehicle_counts.keys())
        sizes = list(vehicle_counts.values())
        colors = ['#3498db', '#2ecc71', '#f39c12', '#e74c3c', '#9b59b6']

        fig, ax = plt.subplots(figsize=(8, 6))
        bars = ax.bar(labels, sizes, color=colors[:len(labels)])
        
        # Adicionar valores nas barras
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height)}', ha='center', va='bottom', fontweight='bold')
        
        ax.set_ylabel('Número de Veículos', fontweight='bold')
        ax.set_xlabel('Tipo de Veículo', fontweight='bold')
        
        plt.xticks(rotation=45)
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        plt.close(fig)
        return base64.b64encode(buf.getvalue()).decode('utf-8')
    
    def generate_html_report(self, output_file='traffic_report_with_video.html'):
        total_vehicles = self.data['vehicle_count']['total']
        up_vehicles = self.data['vehicle_count']['up']
        down_vehicles = self.data['vehicle_count']['down']
        total_anomalies = len(self.anomalies)
        
        # Calculate average speed
        speeds = [v['registered_speed'] for v in self.data['vehicles']]
        avg_speed = sum(speeds) / len(speeds) if speeds else 0

        anomalies_by_direction = self._analyze_anomalies_by_direction()
        mean_speed_by_direction = self._analyze_mean_speed_by_direction()
        vehicle_counts_by_type = self._analyze_vehicle_counts_by_type()
        anomalies_by_vehicle_type = self._analyze_anomalies_by_vehicle_type()
        anomaly_plot_base64 = self._generate_anomaly_percentage_plot()
        vehicle_chart_base64 = self._generate_vehicle_type_chart()
        
        html_content = f"""
<!DOCTYPE html>
<html lang="pt-BR">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Relatório de Tráfego</title>
    <link rel="stylesheet" href="style.css">
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
</head>
<body>
    <div class="container">
        <div class="header">
            <h1><i class="fas fa-road"></i> Relatório de Análise de Tráfego</h1>
            <p class="subtitle">Vídeo: <b>{self.data['video_file']}</b> - Processado em: <b>{datetime.datetime.now().strftime('%d/%m/%Y %H:%M:%S')}</b></p>
        </div>
        
        <div class="main-content">
            <div class="left-panel">
                <!-- Seção de Vídeo -->
                <div class="video-section">
                    <div class="video-container">
                        <video controls>
                            <source src="{self.video_path}" type="video/mp4">
                            Seu navegador não suporta o elemento de vídeo.
                        </video>
                    </div>
                </div>

                <!-- Estatísticas Gerais -->
                <div class="stats-section">
                    <div class="section-title">
                        <i class="fas fa-chart-bar"></i> Estatísticas Gerais
                    </div>
                    <div class="stats-grid">
                        <div class="stat-card">
                            <div class="stat-number">{total_vehicles}</div>
                            <div class="stat-label">Total de Veículos</div>
                        </div>
                        <div class="stat-card">
                            <div class="stat-number">{up_vehicles}</div>
                            <div class="stat-label">Sentido Norte</div>
                        </div>
                        <div class="stat-card">
                            <div class="stat-number">{down_vehicles}</div>
                            <div class="stat-label">Sentido Sul</div>
                        </div>
                        <div class="stat-card">
                            <div class="stat-number">{avg_speed:.1f}</div>
                            <div class="stat-label">Velocidade Média (km/h)</div>
                        </div>
                        <div class="stat-card alert">
                            <div class="stat-number">{total_anomalies}</div>
                            <div class="stat-label">Anomalias Detectadas</div>
                        </div>
                    </div>
                </div>

                <!-- Análises Segmentadas -->
                <div class="analysis-grid">
                    <div class="analysis-card">
                        <h3><i class="fas fa-compass"></i> Anomalias por Direção</h3>
                        <div class="direction-stats">
                            {''.join([f'<div class="direction-item"><span class="direction-label">{direction}:</span><span class="direction-value">{count}</span></div>' for direction, count in anomalies_by_direction.items()])}
                        </div>
                    </div>

                    <div class="analysis-card">
                        <h3><i class="fas fa-tachometer-alt"></i> Velocidade Média por Direção</h3>
                        <div class="speed-stats">
                            {''.join([f'<div class="speed-item"><span class="speed-label">{direction}:</span><span class="speed-value">{speed} km/h</span></div>' for direction, speed in mean_speed_by_direction.items()])}
                        </div>
                    </div>

                    <div class="analysis-card">
                        <h3><i class="fas fa-car"></i> Veículos por Tipo</h3>
                        <div class="vehicle-stats">
                            {''.join([f'<div class="vehicle-item"><span class="vehicle-label">{vehicle_type}:</span><span class="vehicle-value">{count}</span></div>' for vehicle_type, count in vehicle_counts_by_type.items()])}
                        </div>
                    </div>

                    <div class="analysis-card">
                        <h3><i class="fas fa-exclamation-triangle"></i> Anomalias por Tipo de Veículo</h3>
                        <div class="anomaly-vehicle-stats">
                            {''.join([f'<div class="anomaly-vehicle-item"><span class="anomaly-vehicle-label">{vehicle_type}:</span><span class="anomaly-vehicle-value">{count}</span></div>' for vehicle_type, count in anomalies_by_vehicle_type.items()])}
                        </div>
                    </div>
                </div>

                <!-- Gráficos -->
                <div class="charts-section">
                    <div class="chart-container">
                        <h3><i class="fas fa-chart-pie"></i> Distribuição de Anomalias por Severidade</h3>
                        <img src="data:image/png;base64,{anomaly_plot_base64}" alt="Gráfico de Porcentagem de Anomalias" class="chart-image">
                    </div>
                    
                    <div class="chart-container">
                        <h3><i class="fas fa-chart-bar"></i> Contagem de Veículos por Tipo</h3>
                        <img src="data:image/png;base64,{vehicle_chart_base64}" alt="Gráfico de Veículos por Tipo" class="chart-image">
                    </div>
                </div>
            </div>

            <div class="right-panel">
                <div class="panel-title">
                    <i class="fas fa-exclamation-circle"></i> Anomalias Detectadas
                </div>
                <div class="anomalies-list">
                    {''.join([f'''
                    <div class="anomaly-card" style="border-left-color: {self._get_severity_color(anomaly['severity_level'])};" onclick="playAnomalyVideo({anomaly['first_cross']})">
                        <div class="anomaly-header">
                            <span class="anomaly-id">#{anomaly['unique_id']}</span>
                            <span class="severity-badge" style="background-color: {self._get_severity_color(anomaly['severity_level'])};">
                                {self._get_severity_label(anomaly['severity_level'])}
                            </span>
                        </div>
                        <div class="anomaly-speed">
                            <i class="fas fa-tachometer-alt"></i> {anomaly['speed']} km/h
                        </div>
                        <div class="anomaly-details">
                            <div class="detail-row">
                                <span class="detail-label"><i class="fas fa-car"></i> Tipo:</span>
                                <span class="detail-value">{anomaly['type_of_vehicle']}</span>
                            </div>
                            <div class="detail-row">
                                <span class="detail-label"><i class="fas fa-compass"></i> Direção:</span>
                                <span class="detail-value">{'Norte' if anomaly['side'] == 'up' else 'Sul'}</span>
                            </div>
                            <div class="detail-row">
                                <span class="detail-label"><i class="fas fa-clock"></i> Tempo:</span>
                                <span class="detail-value">{anomaly['first_cross']}s</span>
                            </div>
                            <div class="detail-row">
                                <span class="detail-label"><i class="fas fa-percentage"></i> Confiança:</span>
                                <span class="detail-value">{anomaly['confidence']}</span>
                            </div>
                        </div>
                        <div class="play-button">
                            <i class="fas fa-play"></i> Reproduzir
                        </div>
                    </div>''' for anomaly in self.anomalies])}
                </div>
            </div>
        </div>

        <div class="footer">
            <p>&copy; {datetime.datetime.now().year} Sistema de Monitoramento de Tráfego. Todos os direitos reservados.</p>
        </div>
    </div>

    <script>
        function playAnomalyVideo(timestamp) {{
            timestamp = timestamp - 1
            const video = document.querySelector('video');
            if (video) {{
                video.currentTime = timestamp;
                video.play();
                
                // Highlight the playing anomaly
                document.querySelectorAll('.anomaly-card').forEach(card => {{
                    card.classList.remove('playing');
                }});
                event.currentTarget.classList.add('playing');
            }}
        }}

        // Add hover effects and interactions
        document.addEventListener('DOMContentLoaded', function() {{
            const anomalyCards = document.querySelectorAll('.anomaly-card');
            anomalyCards.forEach(card => {{
                card.addEventListener('mouseenter', function() {{
                    this.style.transform = 'translateX(10px)';
                }});
                card.addEventListener('mouseleave', function() {{
                    this.style.transform = 'translateX(0)';
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
