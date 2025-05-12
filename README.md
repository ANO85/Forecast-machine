# Prévisions de Ventes Avancées

## Description
Une application Streamlit pour générer des prévisions de ventes avancées avec multiple méthodes de prévision.

## Fonctionnalités
- Upload de fichiers Excel historiques
- Prévisions par moyenne mobile
- Prévisions par lissage exponentiel
- Prévisions SARIMA
- Visualisation interactive des prévisions
- Sélection personnalisée des méthodes de prévision

## Prérequis
- Python 3.8+
- Streamlit
- Pandas
- NumPy
- Statsmodels
- Scikit-learn
- Plotly

## Installation locale
```bash
git clone https://github.votre-depot/sales-forecast-app.git
cd sales-forecast-app
pip install -r requirements.txt
streamlit run app.py
```

## Déploiement Streamlit Cloud
1. Créez un compte sur Streamlit Cloud
2. Connectez votre compte GitHub
3. Sélectionnez ce dépôt
4. Choisissez la branche principale
5. Sélectionnez `app.py` comme fichier principal

## Structure du Projet
- `app.py`: Interface Streamlit principale
- `forecasting_methods.py`: Méthodes de prévision
- `file_processing.py`: Traitement des fichiers
- `utils.py`: Fonctions utilitaires
- `requirements.txt`: Dépendances du projet

## Utilisation
1. Téléchargez un fichier Excel de ventes historiques
2. Sélectionnez les clients et articles
3. Choisissez les méthodes de prévision
4. Générez et visualisez les prévisions
5. Téléchargez le fichier de résultats

## Licence
[À compléter - par exemple MIT, Apache, etc.]

## Contributions
Les contributions sont les bienvenues. Veuillez ouvrir une issue ou proposer une pull request.
