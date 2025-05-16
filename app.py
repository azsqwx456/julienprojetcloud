import os
import joblib
import pandas as pd
import shap

from flask import Flask, jsonify, request, make_response
from sklearn.preprocessing import StandardScaler
from flask_cors import CORS

# Initialiser Flask
app = Flask(__name__)
CORS(app)  # Activer CORS pour permettre les requêtes depuis Streamlit

# Créer le dossier data s'il n'existe pas
os.makedirs('data', exist_ok=True)

# Charger les données et le modèle
@app.before_first_request
def load_model_and_data():
    global df, df_reel, pipeline, scaler, model, explainer
    try:
        # Charger les données
        df = pd.read_pickle("dataframeP7.pkl")
        df_reel = df[df["TARGET"].isna()]
        
        # Charger le modèle
        pipeline = joblib.load("modele_pipeline.pkl")
        scaler = pipeline.named_steps['scaler']
        model = pipeline.named_steps['classifier']
        
        # Initialiser l'explainer SHAP
        explainer = shap.TreeExplainer(model)
        
        print("Modèle et données chargés avec succès")
    except Exception as e:
        print(f"Erreur lors du chargement du modèle ou des données: {str(e)}")

# Charger le modèle
pipeline = joblib.load("modele_pipeline.pkl")
scaler = pipeline.named_steps['scaler']
model = pipeline.named_steps['classifier']

@app.route("/", methods=['GET'])
def home():
    return jsonify({
        'status': 'API en ligne',
        'message': 'Utilisez le endpoint /predict pour obtenir des prédictions'
    })

@app.route("/predict", methods=['POST', 'OPTIONS'])
def predict():
    # Gérer les requêtes OPTIONS pour CORS
    if request.method == 'OPTIONS':
        response = make_response()
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
        response.headers.add('Access-Control-Allow-Methods', 'POST')
        return response
        
    try:
        data = request.json
        sk_id_curr = data['SK_ID_CURR']

        sample = df_reel[df_reel['SK_ID_CURR'] == sk_id_curr]
        if sample.empty:
            return jsonify({'error': 'ID non trouvé dans les données'}), 404

        sample = sample.drop(columns=['TARGET'])
        sample_scaled = scaler.transform(sample)

        prediction = model.predict_proba(sample_scaled)
        proba = prediction[0][1] * 100

        # SHAP values
        shap_values = explainer.shap_values(sample_scaled)[0][0].tolist()

        return jsonify({
            'probability': proba,
            'shap_values': shap_values,
            'feature_names': sample.columns.tolist(),
            'feature_values': sample.values[0].tolist()
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Lancer l'API
if __name__ == "__main__":
    # Charger le modèle et les données au démarrage
    load_model_and_data()
    
    # Utiliser le port défini par l'environnement ou 5000 par défaut
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port)
