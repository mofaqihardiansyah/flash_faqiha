from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load model dan scaler
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Ambil inputan 8 fitur dari form
        pregnancies = float(request.form['pregnancies'])
        glucose = float(request.form['glucose'])
        bloodpressure = float(request.form['bloodpressure'])
        skinthickness = float(request.form['skinthickness'])
        insulin = float(request.form['insulin'])
        bmi = float(request.form['bmi'])
        pedigree = float(request.form['pedigree'])
        age = float(request.form['age'])
        
        # Bentuk matriks array 2D dengan urutan yang BENAR
        fitur = np.array([[pregnancies, glucose, bloodpressure, skinthickness, insulin, bmi, pedigree, age]])
        
        # Normalisasi/scaling array dengan scaler.pkl
        fitur_scaled = scaler.transform(fitur)
        
        # Prediksi hasil
        prediksi = model.predict(fitur_scaled)
        
        # Interpretasi hasil
        if prediksi[0] == 1:
            hasil = "Positif Terkena Diabetes"
            warna = "#dc3545" # Merah
        else:
            hasil = "Negatif (Tidak Terkena Diabetes)"
            warna = "#28a745" # Hijau
            
        return render_template('index.html', hasil_prediksi=hasil, warna=warna)

if __name__ == '__main__':
    app.run(debug=True)