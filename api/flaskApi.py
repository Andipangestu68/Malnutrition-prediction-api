from flask import Flask, request, jsonify
import pandas as pd
import joblib
import traceback

# Memuat model, scaler, dan label encoder
model = joblib.load("malnutrition_random_forest_model.pkl")
scaler = joblib.load("scaler.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# Mapping status gizi
status_gizi_mapping = {
    0: 'Normal',
    1: 'Sangat pendek (severely stunted)',
    2: 'Tinggi',
    3: 'Pendek (stunted)'
}

# Inisialisasi aplikasi Flask
app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict_status_gizi():
    try:
        # Mengambil data dari request JSON
        data = request.get_json()

        # Validasi data input
        if not all(key in data for key in ["Umur (bulan)", "Jenis Kelamin", "Tinggi Badan (cm)"]):
            return jsonify({"error": "Input harus memiliki 'Umur (bulan)', 'Jenis Kelamin', dan 'Tinggi Badan (cm)'"}), 400

        # Membuat DataFrame dari data input
        new_data = pd.DataFrame({
            "Umur (bulan)": [data["Umur (bulan)"]],
            "Jenis Kelamin": [data["Jenis Kelamin"]],
            "Tinggi Badan (cm)": [data["Tinggi Badan (cm)"]]
        })

        # Transformasi label encoding untuk "Jenis Kelamin"
        new_data["Jenis Kelamin"] = label_encoder.transform(new_data["Jenis Kelamin"])

        # Normalisasi data
        new_data_scaled = scaler.transform(new_data[["Umur (bulan)", "Jenis Kelamin", "Tinggi Badan (cm)"]])

        # Prediksi status gizi
        new_prediction = model.predict(new_data_scaled)

        # Menghitung probabilitas prediksi
        prediction_proba = model.predict_proba(new_data_scaled)[0]

        # Decode hasil prediksi
        predicted_status_gizi = status_gizi_mapping[new_prediction[0]]

        # Mengembalikan hasil prediksi dan probabilitas sebagai JSON
        return jsonify({
            "predicted_status_gizi": predicted_status_gizi,
            "probabilities": {
                status_gizi_mapping[i]: float(prediction_proba[i]) for i in range(len(prediction_proba))
            }
        })

    except Exception as e:
        # Menangani error dan memberikan traceback untuk debug
        error_message = traceback.format_exc()
        return jsonify({"error": error_message}), 500

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
