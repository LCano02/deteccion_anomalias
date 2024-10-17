from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import os
import matplotlib.pyplot as plt
import base64
from io import BytesIO

app = Flask(__name__)

# Ruta del archivo CSV
data_file_path = '/home/awitadecoco/proyecto2/transacciones.csv'

# Cargar los datos
if not os.path.exists(data_file_path):
    raise FileNotFoundError(f"El archivo {data_file_path} no existe. Asegúrate de que el CSV esté disponible.")

data = pd.read_csv(data_file_path)

# Variables globales
scaler = None
kmeans = None

def preprocesar_datos(data):
    if data['hora'].dtype == object:
        data['hora'] = pd.to_datetime(data['hora'], format='%H:%M', errors='coerce').dt.hour + \
                       pd.to_datetime(data['hora'], format='%H:%M', errors='coerce').dt.minute / 60.0
        data['hora'] = data['hora'].fillna(0)

    data['fecha'] = pd.to_datetime(data['fecha'])
    data['hora_sin'] = np.sin(2 * np.pi * data['hora'] / 24)
    data['hora_cos'] = np.cos(2 * np.pi * data['hora'] / 24)
    data['dia'] = data['fecha'].dt.day
    data['mes'] = data['fecha'].dt.month
    data['anio'] = data['fecha'].dt.year
    data['dia_semana'] = data['fecha'].dt.dayofweek

    # Retorna todas las columnas menos 'fecha' para el escalado
    return data[['monto', 'hora', 'dia', 'mes', 'anio', 'dia_semana', 'hora_sin', 'hora_cos', 'fecha']]

data = preprocesar_datos(data)

def escalar_datos(data):
    global scaler
    features = ['monto', 'hora', 'dia', 'mes', 'anio', 'dia_semana', 'hora_sin', 'hora_cos']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(data[features])
    return X_scaled

X_scaled = escalar_datos(data)

def determinar_k_optimo(X_scaled):
    silhouette_scores = []
    k_values = range(2, 51)

    for k in k_values:
        kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
        kmeans.fit(X_scaled)
        silhouette_scores.append(silhouette_score(X_scaled, kmeans.labels_))

    return k_values[silhouette_scores.index(max(silhouette_scores))]

optimal_k = determinar_k_optimo(X_scaled)
kmeans = KMeans(n_clusters=optimal_k, n_init=10, random_state=42)
data['cluster'] = kmeans.fit_predict(X_scaled)

def crear_grafico(nuevo_monto):
    plt.figure(figsize=(12, 6))
    plt.plot(data['fecha'], data['monto'], color='blue', label='Monto Total', linewidth=2, marker='o', markersize=5, alpha=0.7)
    plt.axhline(y=nuevo_monto, color='green', linestyle='--', linewidth=2, label='Monto Insertado')
    plt.title('Gráfico Lineal de Transacciones', fontsize=18, fontweight='bold')
    plt.xlabel('Fecha', fontsize=14)
    plt.ylabel('Monto Total', fontsize=14)
    plt.xticks(rotation=45)
    plt.legend(fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.ylim(0, data['monto'].max() * 1.1)
    plt.tight_layout()

    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()

    return base64.b64encode(buf.read()).decode('utf-8')

@app.route('/', methods=['GET', 'POST'])
def index():
    resultado = None
    graph = None
    nuevo_monto = None
    anomalias = []  # Lista para almacenar las transacciones anómalas

    if request.method == 'POST':
        try:
            fecha_str = request.form['fecha']
            hora_str = request.form['hora']
            monto_str = request.form['monto']

            fecha = pd.to_datetime(fecha_str)
            horas, minutos = map(int, hora_str.split(':'))
            hora = horas + minutos / 60.0
            nuevo_monto = float(monto_str)

            input_data = pd.DataFrame({
                'fecha': [fecha],
                'hora': [hora],
                'monto': [nuevo_monto]
            })

            input_data_processed = preprocesar_datos(input_data)
            X_input_scaled = scaler.transform(input_data_processed[['monto', 'hora', 'dia', 'mes', 'anio', 'dia_semana', 'hora_sin', 'hora_cos']])

            cluster_prediction = kmeans.predict(X_input_scaled)[0]
            cluster_data = data[data['cluster'] == cluster_prediction]
            rango_monto = (cluster_data['monto'].mean() - 2 * cluster_data['monto'].std(),
                           cluster_data['monto'].mean() + 2 * cluster_data['monto'].std())

            if nuevo_monto < rango_monto[0] or nuevo_monto > rango_monto[1]:
                resultado = "La transacción es anómala."
                anomalias.append({'fecha': fecha_str, 'hora': hora_str, 'monto': nuevo_monto})
            else:
                resultado = "La transacción es normal."

            graph = crear_grafico(nuevo_monto)

        except Exception as e:
            resultado = f"Error: {str(e)}"

    return render_template('index.html', resultado=resultado, graph=graph, anomalias=anomalias)

if __name__ == '__main__':
    app.run(debug=True)
