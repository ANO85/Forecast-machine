import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
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
        
        # Nettoyage des noms de colonnes (suppression des espaces, conversion en minuscules)
        df.columns = df.columns.str.strip().str.lower()
        
        # Vérification des colonnes requises
        required_columns = ['date', 'customer group', 'item', 'qty']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            st.error(f"Colonnes manquantes : {', '.join(missing_columns)}")
            st.write("Colonnes actuellement présentes :", list(df.columns))
            return None
        
        # Renommage des colonnes pour standardisation
        df.columns = ['Date', 'Customer group', 'Item', 'Qty']
        
        # Conversion de la colonne Date en datetime
        df['Date'] = pd.to_datetime(df['Date'])
        
        # Conversion Qty en numérique, gestion des valeurs non-numériques
        df['Qty'] = pd.to_numeric(df['Qty'], errors='coerce').fillna(0)
        
        # Tri des données
        df = df.sort_values('Date')
        
        return df
    except Exception as e:
        st.error(f"Erreur de chargement du fichier : {e}")
        return None

# Méthode 1 : Moyenne mobile sur 6 mois
def moving_average_forecast(group, last_date, months_to_forecast=12):
    # Groupement par mois
    try:
        monthly_sales = group.groupby(pd.Grouper(key='Date', freq='M'))['Qty'].sum().reset_index()
        
        # Si moins de 6 mois de données, utiliser la moyenne totale
        if len(monthly_sales) < 6:
            avg_sales = monthly_sales['Qty'].mean()
            ma_forecast = [max(0, avg_sales)] * months_to_forecast
        else:
            # Calcul de la moyenne mobile sur 6 mois
            monthly_sales['MA_6'] = monthly_sales['Qty'].rolling(window=6, min_periods=1).mean()
            ma_forecast = [max(0, monthly_sales['MA_6'].iloc[-1])] * months_to_forecast
        
        # Génération des prévisions
        forecast_dates = pd.date_range(start=last_date.replace(day=1), 
                                       periods=months_to_forecast, 
                                       freq='M')
        
        forecast_df = pd.DataFrame({
            'Date': forecast_dates,
            'Qty_Forecast_MA': ma_forecast
        })
        
        # Formatage des dates
        forecast_df['Date'] = forecast_df['Date'].dt.strftime('%d/%m/%Y')
        
        return forecast_df
    except Exception as e:
        st.warning(f"Erreur de prévision par moyenne mobile : {e}")
        return None

# Méthode 2 : Lissage exponentiel simple
def exponential_smoothing_forecast(group, last_date, months_to_forecast=12):
    try:
        # Groupement par mois
        monthly_sales = group.groupby(pd.Grouper(key='Date', freq='M'))['Qty'].sum().reset_index()
        
        # Si pas assez de données, utiliser la moyenne
        if len(monthly_sales) < 2:
            avg_sales = monthly_sales['Qty'].mean()
            es_forecast = [max(0, avg_sales)] * months_to_forecast
        else:
            # Modèle de lissage exponentiel simple
            model = SimpleExpSmoothing(monthly_sales['Qty']).fit(smoothing_level=0.3, optimized=False)
            
            # Prévisions
            es_forecast = model.forecast(steps=months_to_forecast)
            es_forecast = [max(0, x) for x in es_forecast]  # Éviter les valeurs négatives
        
        # Génération des prévisions
        forecast_dates = pd.date_range(start=last_date.replace(day=1), 
                                       periods=months_to_forecast, 
                                       freq='M')
        
        forecast_df = pd.DataFrame({
            'Date': forecast_dates,
            'Qty_Forecast_ES': es_forecast
        })
        
        # Formatage des dates
        forecast_df['Date'] = forecast_df['Date'].dt.strftime('%d/%m/%Y')
        
        return forecast_df
    except Exception as e:
        st.warning(f"Erreur de prévision par lissage exponentiel : {e}")
        return None

# Méthode 3 : Régression linéaire simple
def linear_regression_forecast(group, last_date, months_to_forecast=12):
    try:
        # Groupement par mois
        monthly_sales = group.groupby(pd.Grouper(key='Date', freq='M'))['Qty'].sum().reset_index()
        
        # Si pas assez de données, utiliser la moyenne
        if len(monthly_sales) < 2:
            avg_sales = monthly_sales['Qty'].mean()
            lr_forecast = [max(0, avg_sales)] * months_to_forecast
        else:
            # Préparation des données pour régression
            X = np.arange(len(monthly_sales)).reshape(-1, 1)
            y = monthly_sales['Qty'].values
            
            # Régression linéaire
            slope, intercept, r_value, p_value, std_err = stats.linregress(X.ravel(), y)
            
            # Prévision des prochains mois
            last_index = len(monthly_sales)
            lr_forecast = [max(0, slope * (last_index + i) + intercept) for i in range(months_to_forecast)]
        
        # Génération des prévisions
        forecast_dates = pd.date_range(start=last_date.replace(day=1), 
                                       periods=months_to_forecast, 
                                       freq='M')
        
        forecast_df = pd.DataFrame({
            'Date': forecast_dates,
            'Qty_Forecast_LR': lr_forecast
        })
        
        # Formatage des dates
        forecast_df['Date'] = forecast_df['Date'].dt.strftime('%d/%m/%Y')
        
        return forecast_df
    except Exception as e:
        st.warning(f"Erreur de prévision par régression linéaire : {e}")
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
            # Trouver la dernière date dans le jeu de données
            last_date = df['Date'].max()
            
            # Affichage des données chargées
            st.write("Données chargées :")
            st.dataframe(df.head())
            st.write(f"Dernière date dans les données : {last_date.strftime('%d/%m/%Y')}")
            
            # Récupération des groupes uniques
            customer_groups = df['Customer group'].unique()
            items = df['Item'].unique()
            
            # Dictionnaires pour stocker les résultats
            ma_results = {}
            es_results = {}
            lr_results = {}
            
            # Calcul des prévisions pour chaque combinaison
            for group in customer_groups:
                for item in items:
                    # Filtrage des données
                    subset = df[(df['Customer group'] == group) & (df['Item'] == item)]
                    
                    # Calcul des prévisions
                    ma_forecast = moving_average_forecast(subset, last_date)
                    es_forecast = exponential_smoothing_forecast(subset, last_date)
                    lr_forecast = linear_regression_forecast(subset, last_date)
                    
                    # Stockage des résultats
                    key = f"{group} - {item}"
                    
                    if ma_forecast is not None:
                        ma_results[key] = ma_forecast
                    
                    if es_forecast is not None:
                        es_results[key] = es_forecast
                    
                    if lr_forecast is not None:
                        lr_results[key] = lr_forecast
            
            # Bouton de téléchargement des résultats
            if st.button("Générer et Télécharger les Prévisions"):
                # Création d'un fichier Excel avec plusieurs onglets
                with pd.ExcelWriter('Previsions_Ventes.xlsx') as writer:
                    # Onglet Moyenne Mobile
                    if ma_results:
                        ma_combined = pd.concat([
                            pd.DataFrame({'Groupe Client': [k.split(' - ')[0]] * len(v), 
                                          'Article': [k.split(' - ')[1]] * len(v), 
                                          **v}) 
                            for k, v in ma_results.items()
                        ])
                        ma_combined.to_excel(writer, sheet_name='Moyenne_Mobile', index=False)
                    
                    # Onglet Lissage Exponentiel
                    if es_results:
                        es_combined = pd.concat([
                            pd.DataFrame({'Groupe Client': [k.split(' - ')[0]] * len(v), 
                                          'Article': [k.split(' - ')[1]] * len(v), 
                                          **v}) 
                            for k, v in es_results.items()
                        ])
                        es_combined.to_excel(writer, sheet_name='Lissage_Exponentiel', index=False)
                    
                    # Onglet Régression Linéaire
                    if lr_results:
                        lr_combined = pd.concat([
                            pd.DataFrame({'Groupe Client': [k.split(' - ')[0]] * len(v), 
                                          'Article': [k.split(' - ')[1]] * len(v), 
                                          **v}) 
                            for k, v in lr_results.items()
                        ])
                        lr_combined.to_excel(writer, sheet_name='Regression_Lineaire', index=False)
                
                # Téléchargement du fichier
                with open('Previsions_Ventes.xlsx', 'rb') as f:
                    st.download_button(
                        label="Télécharger les prévisions",
                        data=f,
                        file_name='Previsions_Ventes.xlsx',
                        mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
                    )

# Point d'entrée principal
if __name__ == "__main__":
    main()