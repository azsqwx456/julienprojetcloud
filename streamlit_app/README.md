# Interface Streamlit de Scoring Client

Ce dossier contient l’interface utilisateur Streamlit pour tester l’API de scoring client.

## Déploiement sur Streamlit Cloud

1. Aller sur https://share.streamlit.io/
2. Connecter le repo GitHub et choisir le dossier `streamlit_app/` comme racine.
3. Indiquer `app_streamlitx.py` comme fichier principal.
4. Vérifier que `requirements.txt` est allégé (pas de Flask, gunicorn, shap…)
5. Placer une copie de `dataframeP7_light.pkl` dans ce dossier.

## Utilisation
- Lancer l’interface :
  ```bash
  streamlit run app_streamlitx.py
  ```
- Entrer l’URL de l’API Flask déployée (ex : https://julien-api.onrender.com/predict)
- Sélectionner un client et visualiser le score + interprétabilité.

## Notes
- Le fichier `dataframeP7_light.pkl` doit être présent dans ce dossier.
- Si l’API n’est pas accessible, l’interface affichera une erreur.
