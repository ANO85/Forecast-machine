import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from forecasting_methods import (
    moving_average_forecast, 
    exponential_smoothing_forecast, 
    sarima_forecast,
    select_best_method
)

def standardize_column_names(df):
    """
    Normalise les noms de colonnes pour une compatibilité maximale
    
    Args:
        df (pd.DataFrame): DataFrame d'entrée
    
    Returns:
        pd.DataFrame: DataFrame avec noms de colonnes standardisés
    """
    # Dictionnaire de mappage des noms de colonnes
    column_mapping = {
        # Variations possibles pour chaque colonne clé
        'date': ['date', 'dates', 'datetime', 'date_vente'],
        'customer group': ['customer group', 'customer', 'customer_group', 'groupe client', 'client'],
        'item': ['item', 'product', 'produit', 'article'],
        'qty': ['qty', 'quantity', 'quantite', 'qte', 'ventes']
    }
    
    # Normaliser les noms de colonnes
    column_names = df.columns.str.lower().str.strip()
    
    # Créer un nouveau dictionnaire de mapping
    new_columns = {}
    for standard_name, possible_names in column_mapping.items():
        # Trouver la première correspondance
        match = next((col for col in column_names if col in possible_names), None)
        
        if match:
            # Remplacer le nom de colonne par le nom standard
            original_index = column_names.tolist().index(match)
            new_columns[df.columns[original_index]] = standard_name
    
    # Renommer les colonnes
    df = df.rename(columns=new_columns)
    
    # Vérifier que toutes les colonnes requises sont présentes
    required_columns = ['date', 'customer group', 'item', 'qty']
    missing = [col for col in required_columns if col not in df.columns]
    
    if missing:
        raise ValueError(f"Colonnes manquantes : {', '.join(missing)}")
    
    return df

def load_sales_data(uploaded_file):
    """
    Charge le fichier Excel et prépare les données
    
    Args:
        uploaded_file (file): Fichier Excel uploadé
    
    Returns:
        pd.DataFrame: Données de ventes préparées
    """
    # Charger le fichier Excel avec différents moteurs
    try:
        # Essayer différents moteurs de lecture
        df = pd.read_excel(uploaded_file, engine='openpyxl')
    except Exception as e:
        try:
            df = pd.read_excel(uploaded_file, engine='xlrd')
        except Exception as e2:
            raise ValueError(f"Impossible de lire le fichier Excel : {e} et {e2}")
    
    # Normaliser les noms de colonnes
    df = standardize_column_names(df)
    
    # Convertir la colonne de date
    try:
        df['date'] = pd.to_datetime(df['date'])
    except Exception as e:
        raise ValueError(f"Erreur de conversion de la date : {e}")
    
    # Trier par date
    df = df.sort_values('date')
    
    # Nettoyer les données
    # Supprimer les lignes avec des valeurs négatives ou nulles
    df = df[df['qty'] >= 0]
    
    # Gérer les valeurs manquantes
    # Remplacer les valeurs manquantes par 0 ou la médiane selon le contexte
    df['qty'] = df['qty'].fillna(0)
    
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
    grouped = df.groupby(['customer group', 'item'])
    
    # Liste pour stocker les recommendations
    item_recommendations = []
    customer_recommendations = {}
    
    for (customer, item), group in grouped:
        # Préparer la série temporelle groupée par mois
        monthly_sales = group.groupby(pd.Grouper(key='date', freq='M'))['qty'].sum()
        
        # S'assurer d'avoir suffisamment de données historiques
        if len(monthly_sales) < 12:
            print(f"Données insuffisantes pour {customer} - {item}")
            continue
        
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
        forecast_dates = [last_date + pd.offsets.MonthBegin(i+1) for i in range(12)]
        
        # Stocker les prévisions
        ma_df = pd.DataFrame({
            'date': forecast_dates,
            'customer group': [customer] * 12,
            'item': [item] * 12,
            'qty': ma_forecast
        })
        results['Moving Average'].append(ma_df)
        
        exp_df = pd.DataFrame({
            'date': forecast_dates,
            'customer group': [customer] * 12,
            'item': [item] * 12,
            'qty': exp_forecast
        })
        results['Exponential Smoothing'].append(exp_df)
        
        sarima_df = pd.DataFrame({
            'date': forecast_dates,
            'customer group': [customer] * 12,
            'item': [item] * 12,
            'qty': sarima_forecast_result
        })
        results['SARIMA'].append(sarima_df)
        
        # Stocker la recommandation pour l'article
        item_recommendations.append({
            'item': item,
            'best_method': best_method
        })
        
        # Agréger les recommandations par client
        if customer not in customer_recommendations:
            customer_recommendations[customer] = {}
        customer_recommendations[customer][item] = best_method
    
    # Convertir les listes de DataFrames
    results['Moving Average'] = pd.concat(results['Moving Average']) if results['Moving Average'] else pd.DataFrame()
    results['Exponential Smoothing'] = pd.concat(results['Exponential Smoothing']) if results['Exponential Smoothing'] else pd.DataFrame()
    results['SARIMA'] = pd.concat(results['SARIMA']) if results['SARIMA'] else pd.DataFrame()
    
    # Ajouter les recommandations
    results['Best Method per Item'] = pd.DataFrame(item_recommendations)
    
    # Convertir les recommandations par client en DataFrame
    customer_recommendations_list = []
    for customer, methods in customer_recommendations.items():
        customer_recommendations_list.append({
            'customer group': customer,
            'recommended_method': max(set(methods.values()), key=list(methods.values()).count)
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
        pd.DataFrame(columns=['date', 'item', 'customer group', 'qty', 'method']).to_excel(
            writer, 
            sheet_name='Manual Forecast', 
            index=False
        )
        
        # Ajouter un onglet de résultats final
        pd.DataFrame(columns=['date', 'item', 'customer group', 'qty', 'method']).to_excel(
            writer, 
            sheet_name='Final Forecast', 
            index=False
        )
    
    return output_filename