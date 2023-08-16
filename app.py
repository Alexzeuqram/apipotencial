from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
import joblib

app = Flask(__name__)

# Cargar y procesar los datos al iniciar la aplicación
ruta_archivo = 'historical_data.csv'
df = pd.read_csv(ruta_archivo)
X = df[['age']].values
y = df['potential'].values
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X)
model = MLPRegressor(hidden_layer_sizes=(50, 50, 50), max_iter=1000, random_state=42)
model.fit(X_train_scaled, y)

# Cargar el modelo entrenado y el escalador al iniciar la aplicación
loaded_model = model
loaded_scaler = scaler

@app.route('/predict', methods=['POST'])
def predict_potential():
    data = request.json
    edad_actual = data['edad_actual']
    potencial_actual = data['potencial_actual']
    edad_futura = data['edad_futura']
    
    if edad_futura <= edad_actual:
        return jsonify({"error": "La edad futura debe ser mayor que la edad actual."}), 400
    
    edad_futura_scaled = loaded_scaler.transform(np.array([[edad_futura]]))
    potencial_futuro_modelo = loaded_model.predict(edad_futura_scaled)[0]
    potencial_estimado = min(potencial_actual + (potencial_futuro_modelo - potencial_actual) * 0.5, 100)
    
    response = {
        "prediccion": potencial_estimado
    }
    
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
