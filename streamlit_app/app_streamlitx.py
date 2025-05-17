import streamlit as st
import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Dashboard Prédictif", layout="wide")

API_URL = st.sidebar.text_input(
    "URL de l'API", 
    value="https://julien-api.onrender.com/predict", 
    help="Entrez l'URL de l'API déployée dans le cloud"
)

@st.cache_data
def load_data():
    try:
        return pd.read_pickle("dataframeP7_light.pkl")
    except Exception as e:
        st.error(f"Impossible de charger le fichier dataframeP7_light.pkl : {e}")
        return pd.DataFrame()

df = load_data()
if df.empty:
    st.stop()
df_clients = df[df["TARGET"].isna()]
id_list = df_clients["SK_ID_CURR"].sort_values().unique()

st.title("🔍 Dashboard Prédictif - Score Client")
selected_id = st.selectbox("Sélectionnez un client :", id_list)
client_data = df_clients[df_clients["SK_ID_CURR"] == selected_id]
st.subheader("Informations du client sélectionné :")
st.dataframe(client_data.drop(columns=["TARGET"]), use_container_width=True)

with st.spinner("Récupération du score et des interprétations..."):
    response = requests.post(API_URL, json={"SK_ID_CURR": int(selected_id)})
    if response.status_code == 200:
        result = response.json()
        proba = result["probability"]
        shap_values = result["shap_values"]
        feature_names = result["feature_names"]
        feature_values = result["feature_values"]

        st.subheader("Score de probabilité de défaut de paiement")
        st.metric("Probabilité (%)", f"{proba:.2f}%", delta=None)
        threshold = 50
        if proba >= threshold:
            st.warning(f"⚠️ Ce client dépasse le seuil de {threshold}%.")
        else:
            st.success(f"✅ Ce client est en dessous du seuil de {threshold}%.")
        # Affichage SHAP simplifié
        st.subheader("Interprétabilité (top 10 variables)")
        shap_df = pd.DataFrame({
            'feature': feature_names,
            'value': feature_values,
            'shap': shap_values
        }).sort_values(by='shap', key=abs, ascending=False).head(10)
        fig, ax = plt.subplots()
        sns.barplot(x='shap', y='feature', data=shap_df, ax=ax, palette='coolwarm')
        ax.set_xlabel('Valeur SHAP (impact)')
        ax.set_ylabel('Variable')
        st.pyplot(fig)
    else:
        st.error(f"Erreur API : {response.text}")
