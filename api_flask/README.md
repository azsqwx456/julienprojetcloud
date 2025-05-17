# API Flask de Scoring Client

Ce dossier contient l’API Flask déployable sur Render pour le scoring de clients.

## Déploiement sur Render

1. **Créer un nouveau service web** sur https://dashboard.render.com/
2. **Connecter le repo GitHub** et choisir le dossier `api_flask/` comme racine.
3. **Build Command** : (laisser vide ou `pip install -r requirements.txt`)
4. **Start Command** :
   ```
   gunicorn app:app
   ```
5. **Fichiers nécessaires** :
    - `app.py` : code de l’API
    - `requirements.txt` : dépendances complètes
    - `Procfile` : indique à Render comment lancer l’API
    - `dataframeP7_light.pkl` : à placer dans ce dossier (copie depuis la racine)

## Endpoints
- `/predict` : POST, payload `{ "SK_ID_CURR": 123456 }`
- `/` : GET, healthcheck

## Notes
- L’API charge le modèle et les données au démarrage (téléchargement auto si besoin).
- Les dépendances lourdes (shap, lightgbm…) sont nécessaires ici.
