import os
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import joblib
import shap

app = Flask(__name__)
CORS(app)

MODEL_URL = "https://www.dropbox.com/scl/fi/xxxxxx/modele_pipeline.pkl?rlkey=xxxxxx&dl=1"
DATA_URL = "https://www.dropbox.com/scl/fi/yyyyyy/dataframeP7_light.pkl?rlkey=yyyyyy&dl=1"
MODEL_PATH = "modele_pipeline.pkl"
DATA_PATH = "dataframeP7_light.pkl"

# Télécharge les fichiers si absents
for url, path in [(MODEL_URL, MODEL_PATH), (DATA_URL, DATA_PATH)]:
    if not os.path.exists(path):
        r = requests.get(url)
        with open(path, 'wb') as f:
            f.write(r.content)

model = joblib.load(MODEL_PATH)
df = pd.read_pickle(DATA_PATH)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    sk_id_curr = data['SK_ID_CURR']
    client_data = df[df['SK_ID_CURR'] == sk_id_curr]
    if client_data.empty:
        return jsonify({'error': 'Client non trouvé'}), 404
    X = client_data.drop(columns=["SK_ID_CURR", "TARGET"], errors='ignore')
    proba = float(model.predict_proba(X)[0][1] * 100)
    explainer = shap.TreeExplainer(model.named_steps['classifier'])
    shap_values = explainer.shap_values(X)
    return jsonify({
        'probability': proba,
        'shap_values': shap_values[1][0].tolist(),
        'feature_names': X.columns.tolist(),
        'feature_values': X.iloc[0].tolist()
    })

@app.route('/')
def home():
    return 'API de scoring client OK!'

if __name__ == '__main__':
    app.run(debug=True)
