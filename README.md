# Outil de Prévisions de Ventes

## Description
Cette application Streamlit permet de générer des prévisions de ventes selon trois méthodes différentes :
- Moyenne mobile sur 6 mois
- Lissage exponentiel
- SARIMA (Seasonal ARIMA)

## Fonctionnalités
- Téléchargement de fichiers Excel historiques
- Calcul automatique des prévisions par couple Client/Article
- Génération d'un fichier Excel avec 3 onglets de prévisions

## Prérequis
- Python 3.8+
- Bibliothèques : Streamlit, Pandas, NumPy, Statsmodels

## Utilisation
1. Préparez votre fichier Excel avec les colonnes :
   - Date
   - Customer group
   - Item
   - Qty

2. Lancez l'application Streamlit
3. Téléchargez votre fichier
4. Générez et téléchargez les prévisions

## Déploiement
L'application peut être facilement déployée sur Streamlit Cloud.

## Licence
[Votre licence ici]
