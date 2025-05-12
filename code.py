import streamlit as st
import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Configuration de l'application Streamlit
st.set_page_config(page_title="Prévisions de Ventes", page_icon="📊")

# Titre de l'application
st.title("Prévisions de Ventes - Outil de Forecast")

# Fonction pour charger et prétraiter les données
def load_and_preprocess_data(uploaded_file):
    try:
        # Lecture du fichier Excel
        df = pd.read_excel(uploaded_file)
        
        # Vérification des colonnes requises
        required_columns = ['Date', 'Customer group', 'Item', 'Qty']
        for col in required_columns:
            if col not in df.columns:
                st.error(f"Colonne manquante : {col}")
                return None
        
        # Conversion de la colonne Date en datetime
        df['Date'] = pd.to_datetime(df['Date'])
        
        # Tri des données
        df = df.sort_values('Date')
        
        return df
    except Exception as e:
        st.error(f"Erreur de chargement du fichier : {e}")
        return None

# Méthode 1 : Moyenne mobile sur 6 mois
def moving_average_forecast(group, months_to_forecast=12):
    # Groupement par mois
    monthly_sales = group.groupby(pd.Grouper(key='Date', freq='M'))['Qty'].sum().reset_index()
    
    # Calcul de la moyenne mobile sur 6 mois
    monthly_sales['MA_6'] = monthly_sales['Qty'].rolling(window=6, min_periods=1).mean()
    
    # Dernière date connue
    last_date = monthly_sales['Date'].max()
    
    # Génération des prévisions
    forecast_dates = pd.date_range(start=last_date + timedelta(days=1), 
                                   periods=months_to_forecast, 
                                   freq='M')
    
    forecast_df = pd.DataFrame({
        'Date': forecast_dates,
        'Qty_Forecast_MA': [monthly_sales['MA_6'].iloc[-1]] * months_to_forecast
    })
    
    return forecast_df

# Méthode 2 : Lissage exponentiel
def exponential_smoothing_forecast(group, months_to_forecast=12):
    # Groupement par mois
    monthly_sales = group.groupby(pd.Grouper(key='Date', freq='M'))['Qty'].sum().reset_index()
    
    # Modèle de lissage exponentiel
    model = ExponentialSmoothing(monthly_sales['Qty'], 
                                 trend='add', 
                                 seasonal='add', 
                                 seasonal_periods=12).fit()
    
    # Prévisions
    forecast = model.forecast(steps=months_to_forecast)
    
    # Dernière date connue
    last_date = monthly_sales['Date'].max()
    forecast_dates = pd.date_range(start=last_date + timedelta(days=1), 
                                   periods=months_to_forecast, 
                                   freq='M')
    
    forecast_df = pd.DataFrame({
        'Date': forecast_dates,
        'Qty_Forecast_ES': forecast
    })
    
    return forecast_df

# Méthode 3 : SARIMA
def sarima_forecast(group, months_to_forecast=12):
    # Groupement par mois
    monthly_sales = group.groupby(pd.Grouper(key='Date', freq='M'))['Qty'].sum().reset_index()
    
    try:
        # Modèle SARIMA
        model = ARIMA(monthly_sales['Qty'], 
                      order=(1,1,1), 
                      seasonal_order=(1,1,1,12)).fit()
        
        # Prévisions
        forecast = model.forecast(steps=months_to_forecast)
        
        # Dernière date connue
        last_date = monthly_sales['Date'].max()
        forecast_dates = pd.date_range(start=last_date + timedelta(days=1), 
                                       periods=months_to_forecast, 
                                       freq='M')
        
        forecast_df = pd.DataFrame({
            'Date': forecast_dates,
            'Qty_Forecast_SARIMA': forecast
        })
        
        return forecast_df
    except Exception as e:
        st.warning(f"Impossible de calculer SARIMA : {e}")
        return None

# Interface Streamlit principale
def main():
    # Téléchargement du fichier
    uploaded_file = st.file_uploader("Charger le fichier historique des ventes", 
                                     type=['xlsx', 'xls'], 
                                     help="Fichier Excel avec colonnes : Date, Customer group, Item, Qty")
    
    if uploaded_file is not None:
        # Chargement et prétraitement des données
        df = load_and_preprocess_data(uploaded_file)
        
        if df is not None:
            # Affichage des données chargées
            st.write("Données chargées :")
            st.dataframe(df.head())
            
            # Récupération des groupes uniques
            customer_groups = df['Customer group'].unique()
            items = df['Item'].unique()
            
            # Dictionnaires pour stocker les résultats
            ma_results = {}
            es_results = {}
            sarima_results = {}
            
            # Calcul des prévisions pour chaque combinaison
            for group in customer_groups:
                for item in items:
                    # Filtrage des données
                    subset = df[(df['Customer group'] == group) & (df['Item'] == item)]
                    
                    # Calcul des prévisions
                    ma_forecast = moving_average_forecast(subset)
                    es_forecast = exponential_smoothing_forecast(subset)
                    sarima_forecast_result = sarima_forecast(subset)
                    
                    # Stockage des résultats
                    key = f"{group} - {item}"
                    ma_results[key] = ma_forecast
                    es_results[key] = es_forecast
                    if sarima_forecast_result is not None:
                        sarima_results[key] = sarima_forecast_result
            
            # Bouton de téléchargement des résultats
            if st.button("Générer et Télécharger les Prévisions"):
                # Création d'un fichier Excel avec plusieurs onglets
                with pd.ExcelWriter('Previsions_Ventes.xlsx') as writer:
                    # Onglet Moyenne Mobile
                    ma_combined = pd.concat([
                        pd.DataFrame({'Groupe Client': [k.split(' - ')[0]] * len(v), 
                                      'Article': [k.split(' - ')[1]] * len(v), 
                                      **v}) 
                        for k, v in ma_results.items()
                    ])
                    ma_combined.to_excel(writer, sheet_name='Moyenne_Mobile', index=False)
                    
                    # Onglet Lissage Exponentiel
                    es_combined = pd.concat([
                        pd.DataFrame({'Groupe Client': [k.split(' - ')[0]] * len(v), 
                                      'Article': [k.split(' - ')[1]] * len(v), 
                                      **v}) 
                        for k, v in es_results.items()
                    ])
                    es_combined.to_excel(writer, sheet_name='Lissage_Exponentiel', index=False)
                    
                    # Onglet SARIMA
                    sarima_combined = pd.concat([
                        pd.DataFrame({'Groupe Client': [k.split(' - ')[0]] * len(v), 
                                      'Article': [k.split(' - ')[1]] * len(v), 
                                      **v}) 
                        for k, v in sarima_results.items()
                    ])
                    sarima_combined.to_excel(writer, sheet_name='SARIMA', index=False)
                
                # Téléchargement du fichier
                with open('Previsions_Ventes.xlsx', 'rb') as f:
                    st.download_button(
                        label="Télécharger les prévisions",
                        data=f,
                        file_name='Previsions_Ventes.xlsx',
                        mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
                    )

# Configuration requise
if __name__ == "__main__":
    main()