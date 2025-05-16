import streamlit as st
import pandas as pd
import numpy as np
import requests
import shap
import matplotlib.pyplot as plt
import seaborn as sns

# Config Streamlit
st.set_page_config(page_title="Dashboard Prédictif", layout="wide")

# API Endpoint
# Par défaut, on utilise l'API locale pour le développement
API_URL = st.sidebar.text_input(
    "URL de l'API", 
    value="http://127.0.0.1:5000/predict", 
    help="Entrez l'URL de l'API déployée dans le cloud"
)

# Chargement des données
@st.cache_data
def load_data():
    return pd.read_pickle("data/dataframeP7.pkl")

df = load_data()
df_clients = df[df["TARGET"].isna()]
id_list = df_clients["SK_ID_CURR"].sort_values().unique()

# 🧠 Titre principal
st.title("🔍 Dashboard Prédictif - Score Client")

# 🎯 Sélection d’un client
selected_id = st.selectbox("Sélectionnez un client :", id_list)

# 🎯 Récupération des données client
client_data = df_clients[df_clients["SK_ID_CURR"] == selected_id]

# 📊 Affichage d'infos descriptives
st.subheader("Informations du client sélectionné :")
st.dataframe(client_data.drop(columns=["TARGET"]), use_container_width=True)

# 🔍 Requête à l’API pour récupérer score et SHAP values
with st.spinner("Récupération du score et des interprétations..."):
    response = requests.post(API_URL, json={"SK_ID_CURR": int(selected_id)})
    if response.status_code == 200:
        result = response.json()
        proba = result["probability"]
        shap_values = result["shap_values"]
        feature_names = result["feature_names"]
        feature_values = result["feature_values"]

        # 📈 Score du client
        st.subheader("Score de probabilité de défaut de paiement")
        st.metric("Probabilité (%)", f"{proba:.2f}%", delta=None)
        threshold = 50
        if proba >= threshold:
            st.warning(f"⚠️ Ce client dépasse le seuil de {threshold}%.")
        else:
            st.success(f"✅ Ce client est en dessous du seuil de {threshold}%.")

        # 🌈 Graphique SHAP simplifié (barres)
        st.subheader("Principales variables influentes (SHAP)")
        shap_df = pd.DataFrame({
            "Feature": feature_names,
            "Value": feature_values,
            "SHAP": shap_values
        }).sort_values(by="SHAP", key=abs, ascending=False).head(10)

        fig, ax = plt.subplots()
        bars = ax.barh(shap_df["Feature"], shap_df["SHAP"], color="cornflowerblue")
        ax.invert_yaxis()
        ax.set_xlabel("Impact sur la prédiction")
        ax.set_title("Top 10 variables influentes")
        st.pyplot(fig)

    else:
        st.error("Erreur lors de la récupération du score. L’API n’a pas répondu.")

# 🔁 Comparaison à d'autres clients
st.subheader("Comparaison avec d'autres clients")

feature_to_compare = st.selectbox(
    "Choisissez une variable à comparer :", 
    [col for col in client_data.columns if col not in ['SK_ID_CURR', 'TARGET']]
)

fig2, ax2 = plt.subplots()
sns.histplot(df_clients[feature_to_compare].dropna(), label="Population globale", ax=ax2, color="lightgray", bins=30)
ax2.axvline(client_data[feature_to_compare].values[0], color="red", linestyle="--", label="Client sélectionné")
ax2.set_title(f"Distribution de {feature_to_compare}")
ax2.legend()
st.pyplot(fig2)

# ♿️ Accessibilité (adapté WCAG)
st.markdown("""
<p style='font-size: 16px; line-height: 1.6;'>
Ce dashboard a été conçu avec des couleurs contrastées, des polices lisibles et des graphiques interprétables, 
pour répondre à des critères d’accessibilité (WCAG).
</p>
""", unsafe_allow_html=True)

# ✍️ (Optionnel) Recalcul avec saisie manuelle
with st.expander("🔄 Modifier les données du client pour recalculer le score"):
    edited_values = {}
    for col in shap_df["Feature"]:
        val = st.number_input(f"{col}", value=float(client_data[col].values[0]))
        edited_values[col] = val

    if st.button("Recalculer le score avec les nouvelles valeurs"):
        # Construire le client modifié (simulation)
        modified_input = client_data.copy()
        for k, v in edited_values.items():
            modified_input[k] = v

        # Envoi vers API (à adapter côté backend pour gérer cet input)
        st.warning("Fonctionnalité à activer côté API Flask pour tester avec des valeurs modifiées.")

