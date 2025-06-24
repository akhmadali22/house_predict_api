from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)
CORS(app)

# Load model yang telah disimpan
model = joblib.load('model_rr.pkl')
model_lr = joblib.load('model_lr.pkl')

# Mapping lokasi
lokasi_mapping = {
    'Jakarta Barat': 1,
    'Jakarta Pusat': 2,
    'Jakarta Selatan': 3,
    'Jakarta Timur': 4,
    'Jakarta Utara': 5
}

@app.route('/')
def home():
    return "🏡 API Prediksi Harga Rumah Aktif!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    # Ambil nilai input
    lokasi_str = data.get('Lokasi')
    lokasi = lokasi_mapping.get(lokasi_str)
    luas_bangunan = data.get('Luas_Bangunan')
    luas_tanah = data.get('Luas_Tanah')
    kamar_tidur = data.get('Kamar_Tidur')
    kamar_mandi = data.get('Kamar_Mandi')
    garasi = data.get('Garasi')


    # Validasi input
    if None in [
        kamar_tidur, 
        kamar_mandi, 
        garasi,
        lokasi, 
        luas_bangunan, 
        luas_tanah, 
    ]:
        return jsonify({'error': 'Input tidak lengkap'}), 400

    rasio_bangunan = luas_bangunan / luas_tanah

    # Siapkan DataFrame
    input_df = pd.DataFrame([{
        'Luas Tanah': luas_tanah,
        'Luas Bangunan': luas_bangunan,
        'Kamar Tidur': kamar_tidur,
        'Kamar Mandi': kamar_mandi,
        'Garasi': garasi,
        'Lokasi': lokasi,
        'Rasio_Bangunan': rasio_bangunan,
    }])

    # Prediksi
    pred_log = model.predict(input_df)
    harga_prediksi = np.expm1(pred_log)[0]

    pred_log_lr = model_lr.predict(input_df)
    harga_prediksi_lr = np.expm1(pred_log_lr)[0]

    return jsonify({
        'predicted_price': round(harga_prediksi, 2),
        'predicted_price_up': round(harga_prediksi + harga_prediksi * 24 / 100, 2),
        'predicted_price_down': round(harga_prediksi - harga_prediksi * 24 / 100, 2),
        'predicted_price_lr': round(harga_prediksi_lr, 2),
        'predicted_price_lr_up': round(harga_prediksi_lr + harga_prediksi_lr * 24 / 100, 2),
        'predicted_price_lr_down': round(harga_prediksi_lr - harga_prediksi_lr * 24 / 100, 2)
    })

if __name__ == '__main__':
    app.run(debug=True)
