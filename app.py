from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
import joblib

app = Flask(__name__)

# Cargar y procesar los datos (realizado una sola vez al iniciar la aplicación)
ruta_archivo = 'historical_data.csv'
df = pd.read_csv(ruta_archivo)
X = df[['age']].values
y = df['potential'].values

# Dividir los datos en conjuntos de entrenamiento y prueba (80% - 20%) (realizado una sola vez al iniciar la aplicación)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Escalar los datos (realizado una sola vez al iniciar la aplicación)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Entrenar un modelo MLPRegressor con los datos de entrenamiento (realizado una sola vez al iniciar la aplicación)
model = MLPRegressor(hidden_layer_sizes=(50, 50, 50), max_iter=1000, random_state=42)
model.fit(X_train_scaled, y_train)

# Guardar el modelo entrenado y el escalador (realizado una sola vez al iniciar la aplicación)
joblib.dump(model, 'model.pkl')
joblib.dump(scaler, 'scaler.pkl')

# Cargar el modelo y el escalador (realizado una sola vez al iniciar la aplicación)
loaded_model = joblib.load('model.pkl')
loaded_scaler = joblib.load('scaler.pkl')

@app.route('/predict', methods=['POST'])
def predict_potential():
    data = request.json
    edad_actual = data['edad_actual']
    potencial_actual = data['potencial_actual']
    edad_futura = data['edad_futura']
    
    if edad_futura <= edad_actual:
        return jsonify({"error": "La edad futura debe ser mayor que la edad actual."}), 400
    
    # Escalar la edad de entrada
    edad_actual_scaled = loaded_scaler.transform(np.array([[edad_actual]]))
    
    # Escalar la edad futura de entrada
    edad_futura_scaled = loaded_scaler.transform(np.array([[edad_futura]]))
    
    # Predecir el potencial futuro utilizando el modelo entrenado y la edad futura
    predicciones = []
    margenes_error = []
    
    # Realizar 5 predicciones y calcular sus márgenes de error
    for _ in range(5):
        potencial_futuro_modelo = loaded_model.predict(edad_futura_scaled)[0]
        
        # Calcular el potencial estimado considerando el potencial actual y el potencial futuro del modelo
        potencial_estimado = min(potencial_actual + (potencial_futuro_modelo - potencial_actual) * 0.5, 100)
        
        predicciones.append(potencial_estimado)
    
    # Calcular el margen de error de la predicción utilizando los datos de prueba
    X_test_scaled = loaded_scaler.transform(X_test)
    prediccion_prueba = loaded_model.predict(X_test_scaled)
    mse = mean_squared_error(y_test, prediccion_prueba)
    margen_error = np.sqrt(mse)
    
    margenes_error = [margen_error] * 5  # Mismo margen de error para todas las predicciones
    
    response = {
        "predicciones": predicciones,
        "margenes_error": margenes_error
    }
    
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
