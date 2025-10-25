from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import joblib
import matplotlib
matplotlib.use('Agg')  # ✅ Hindari error GUI Matplotlib di Flask
import matplotlib.pyplot as plt
import os
import json
import plotly
import plotly.graph_objs as go
from sklearn.tree import plot_tree

app = Flask(__name__)
model = joblib.load('model_diabetes_tree.pkl')

# Baca dataset hanya sekali
DATA_PATH = 'diabetes.csv'
df = pd.read_csv(DATA_PATH)

# Buat folder static kalau belum ada
if not os.path.exists('static'):
    os.makedirs('static')

# =========================
# 🏠 HALAMAN UTAMA
# =========================
@app.route('/')
def home():
    # Kirim daftar indeks data agar user bisa pilih baris dataset
    dataset_options = df.index.tolist()
    return render_template('index.html', dataset_options=dataset_options)


# =========================
# 🔹 API untuk ambil 1 baris dataset berdasarkan indeks
# =========================
@app.route('/get_dataset/<int:row_id>', methods=['GET'])
def get_dataset(row_id):
    try:
        if row_id < 0 or row_id >= len(df):
            return jsonify({'error': 'Indeks di luar jangkauan dataset'})
        row = df.iloc[row_id].to_dict()
        return jsonify(row)
    except Exception as e:
        return jsonify({'error': str(e)})


# =========================
# 🔹 PROSES PREDIKSI
# =========================
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Ambil data dari form input
        input_data = [
            float(request.form['kehamilan']),
            float(request.form['glukosa']),
            float(request.form['tekanan_darah']),
            float(request.form['ketebalan_kulit']),
            float(request.form['insulin']),
            float(request.form['BMI']),
            float(request.form['fungsi_silsilah_diabetes']),
            float(request.form['usia'])
        ]

        # ✅ Gunakan DataFrame agar feature names valid
        columns = ['kehamilan', 'glukosa', 'tekanan_darah', 'ketebalan_kulit',
           'insulin', 'BMI', 'fungsi_silsilah_diabetes', 'usia']
        final_input = pd.DataFrame([input_data], columns=columns)

        # Prediksi
        prediction = model.predict(final_input)[0]
        probability = model.predict_proba(final_input)[0][1] * 100

        # Tentukan hasil
        if prediction == 1:
            result_text = "Pasien Berisiko Diabetes (Positif)"
            color = "danger"
        else:
            result_text = "Pasien Tidak Berisiko Diabetes (Negatif)"
            color = "success"

        # =========================
        # 📊 Bar Chart
        # =========================
        plt.figure(figsize=(4, 3))
        plt.bar(['Tidak Berisiko', 'Berisiko'], [100 - probability, probability], color=['green', 'red'])
        plt.title('Risiko Diabetes (%)')
        plt.ylabel('Probabilitas (%)')
        plt.tight_layout()
        bar_chart_file = 'static/bar_chart.png'
        plt.savefig(bar_chart_file)
        plt.close()

        # =========================
        # 🎯 Gauge Chart
        # =========================
        gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=probability,
            title={'text': "Risiko Diabetes (%)"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "red" if prediction == 1 else "green"},
                'steps': [
                    {'range': [0, 50], 'color': "lightgreen"},
                    {'range': [50, 75], 'color': "yellow"},
                    {'range': [75, 100], 'color': "red"}
                ]
            }
        ))
        gauge_json = json.dumps(gauge, cls=plotly.utils.PlotlyJSONEncoder)

        # =========================
        # 🧩 Pie Chart
        # =========================
        pie = go.Figure(data=[go.Pie(
            labels=['Tidak Berisiko', 'Berisiko'],
            values=[100 - probability, probability],
            marker=dict(colors=['green', 'red']),
            hole=0.3
        )])
        pie.update_layout(title='Proporsi Risiko Diabetes (%)')
        pie_json = json.dumps(pie, cls=plotly.utils.PlotlyJSONEncoder)

        # =========================
        # 🌳 Pohon Keputusan
        # =========================
        plt.figure(figsize=(20, 10))
        plot_tree(
            model,
            feature_names=columns,
            class_names=['Tidak Diabetes', 'Diabetes'],
            filled=True,
            rounded=True
        )
        plt.title("Visualisasi Pohon Keputusan untuk Dataset Diabetes")
        plt.tight_layout()
        tree_plot_file = 'static/tree_plot.png'
        plt.savefig(tree_plot_file, dpi=100)
        plt.close()

        return render_template(
            'index.html',
            prediction_text=result_text,
            probability_text=f"Probabilitas: {probability:.2f}%",
            color=color,
            bar_chart_file='bar_chart.png',
            gauge_json=gauge_json,
            pie_json=pie_json,
            tree_plot_file='tree_plot.png',
            dataset_options=df.index.tolist()
        )

    except Exception as e:
        return render_template(
            'index.html',
            prediction_text=f"Terjadi kesalahan: {str(e)}",
            color="warning",
            dataset_options=df.index.tolist()
        )


if __name__ == "__main__":
    app.run(debug=True)