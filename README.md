# Dashboard Prédictif & API Scoring Client

Ce projet propose une API Flask pour le scoring de clients (prédiction de défaut de paiement) et une interface utilisateur Streamlit pour tester et visualiser les résultats.

## 1. Utilisation de l'API (Render)

- **URL de base** : `https://julien-api.onrender.com/`
- **Endpoint principal** : `/predict`
- **Méthode** : POST
- **Payload attendu** :
  ```json
  { "SK_ID_CURR": 123456 }
  ```
- **Réponse** :
  ```json
  {
    "probability": 12.34,
    "shap_values": [...],
    "feature_names": [...],
    "feature_values": [...]
  }
  ```

## 2. Interface de test (Streamlit)

- **Fichier** : `app_streamlitx.py`
- **Lancement local** :
  ```bash
  streamlit run app_streamlitx.py
  ```
- **Connexion à l'API** :
  - Par défaut, l'interface peut pointer vers l'API locale ou cloud.
  - Pour la version cloud, renseignez l'URL :
    ```
    https://julien-api.onrender.com/predict
    ```

## 3. Déploiement de l'interface sur Streamlit Cloud

- Rendez-vous sur [Streamlit Cloud](https://share.streamlit.io/)
- Connectez votre compte GitHub.
- Cliquez sur “New app” ou “Déployer une nouvelle application”.
- Sélectionnez ce repository.
- Indiquez `app_streamlitx.py` comme fichier principal.
- Lancez le déploiement.
- Dans la barre latérale de l’interface Streamlit Cloud, renseignez l’URL de l’API cloud :
  ```
  https://julien-api.onrender.com/predict
  ```

## 4. Dépendances

Toutes les dépendances sont listées dans `requirements.txt`.

## 5. Exemple de test avec curl
```bash
curl -X POST https://julien-api.onrender.com/predict -H "Content-Type: application/json" -d '{"SK_ID_CURR": 123456}'
```

---

**Auteur** : Julien
