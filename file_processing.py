import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from forecasting_methods import (
    moving_average_forecast, 
    exponential_smoothing_forecast, 
    sarima_forecast,
    select_best_method
)

def load_sales_data(uploaded_file):
    """
    Charge le fichier Excel et prépare les données
    
    Args:
        uploaded_file (file): Fichier Excel uploadé
    
    Returns:
        pd.DataFrame: Données de ventes préparées
    """
    # Charger le fichier Excel
    df = pd.read_excel(uploaded_file)
    
    # Nettoyer et préparer les colonnes
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')
    
    return df

def generate_forecasts(df):
    """
    Génère des prévisions pour chaque couple Client-Article
    
    Args:
        df (pd.DataFrame): Données de ventes historiques
    
    Returns:
        dict: Dictionnaire contenant différents DataFrames de prévisions
    """
    # Résultats à stocker
    results = {
        'Moving Average': [],
        'Exponential Smoothing': [],
        'SARIMA': [],
        'Best Method per Item': [],
        'Best Method per Customer': []
    }
    
    # Grouper par Client et Article
    grouped = df.groupby(['Customer group', 'Item'])
    
    # Liste pour stocker les recommendations
    item_recommendations = []
    customer_recommendations = {}
    
    for (customer, item), group in grouped:
        # Préparer la série temporelle groupée par mois
        monthly_sales = group.groupby(pd.Grouper(key='Date', freq='M'))['Qty'].sum()
        
        # Générer les prévisions
        ma_forecast = moving_average_forecast(monthly_sales)
        exp_forecast = exponential_smoothing_forecast(monthly_sales)
        sarima_forecast_result = sarima_forecast(monthly_sales)
        
        # Sélectionner la meilleure méthode
        best_method = select_best_method(
            monthly_sales, 
            ma_forecast, 
            exp_forecast, 
            sarima_forecast_result
        )
        
        # Générer les dates de prévision
        last_date = monthly_sales.index[-1]
        forecast_dates = [last_date + timedelta(days=30*(i+1)) for i in range(12)]
        
        # Stocker les prévisions
        ma_df = pd.DataFrame({
            'Date': forecast_dates,
            'Customer group': [customer] * 12,
            'Item': [item] * 12,
            'Qty': ma_forecast
        })
        results['Moving Average'].append(ma_df)
        
        exp_df = pd.DataFrame({
            'Date': forecast_dates,
            'Customer group': [customer] * 12,
            'Item': [item] * 12,
            'Qty': exp_forecast
        })
        results['Exponential Smoothing'].append(exp_df)
        
        sarima_df = pd.DataFrame({
            'Date': forecast_dates,
            'Customer group': [customer] * 12,
            'Item': [item] * 12,
            'Qty': sarima_forecast_result
        })
        results['SARIMA'].append(sarima_df)
        
        # Stocker la recommandation pour l'article
        item_recommendations.append({
            'Item': item,
            'Best Method': best_method
        })
        
        # Agréger les recommandations par client
        if customer not in customer_recommendations:
            customer_recommendations[customer] = {}
        customer_recommendations[customer][item] = best_method
    
    # Convertir les listes de DataFrames
    results['Moving Average'] = pd.concat(results['Moving Average'])
    results['Exponential Smoothing'] = pd.concat(results['Exponential Smoothing'])
    results['SARIMA'] = pd.concat(results['SARIMA'])
    
    # Ajouter les recommandations
    results['Best Method per Item'] = pd.DataFrame(item_recommendations)
    
    # Convertir les recommandations par client en DataFrame
    customer_recommendations_list = []
    for customer, methods in customer_recommendations.items():
        customer_recommendations_list.append({
            'Customer group': customer,
            'Recommended Method': max(set(methods.values()), key=list(methods.values()).count)
        })
    results['Best Method per Customer'] = pd.DataFrame(customer_recommendations_list)
    
    return results

def save_forecasts_to_excel(forecasts, output_filename):
    """
    Sauvegarde les prévisions dans un fichier Excel
    
    Args:
        forecasts (dict): Dictionnaire de DataFrames de prévisions
        output_filename (str): Nom du fichier de sortie
    
    Returns:
        str: Chemin du fichier sauvegardé
    """
    # Créer un writer Excel
    with pd.ExcelWriter(output_filename) as writer:
        # Sauvegarder chaque DataFrame dans un onglet
        forecasts['Moving Average'].to_excel(writer, sheet_name='Forecast Moving Average', index=False)
        forecasts['Exponential Smoothing'].to_excel(writer, sheet_name='Forecast Exponential', index=False)
        forecasts['SARIMA'].to_excel(writer, sheet_name='Forecast SARIMA', index=False)
        forecasts['Best Method per Item'].to_excel(writer, sheet_name='Item Method Recommendation', index=False)
        forecasts['Best Method per Customer'].to_excel(writer, sheet_name='Customer Method Recommendation', index=False)
        
        # Ajouter un onglet pour les prévisions manuelles
        pd.DataFrame(columns=['Date', 'Item', 'Customer group', 'Qty', 'Method']).to_excel(
            writer, 
            sheet_name='Manual Forecast', 
            index=False
        )
        
        # Ajouter un onglet de résultats final
        pd.DataFrame(columns=['Date', 'Item', 'Customer group', 'Qty', 'Method']).to_excel(
            writer, 
            sheet_name='Final Forecast', 
            index=False
        )
    
    return output_filename