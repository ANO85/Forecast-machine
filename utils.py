import pandas as pd
import numpy as np
from typing import Dict, Any, List

def clean_sales_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Nettoie et prépare les données de ventes
    
    Args:
        df (pd.DataFrame): DataFrame des ventes brutes
    
    Returns:
        pd.DataFrame: DataFrame nettoyé
    """
    # Nettoyer les noms de colonnes
    df.columns = df.columns.str.strip().str.lower()
    
    # Convertir les dates
    date_columns = [col for col in df.columns if 'date' in col]
    for col in date_columns:
        df[col] = pd.to_datetime(df[col], errors='coerce')
    
    # Gérer les valeurs manquantes
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    for col in numeric_columns:
        df[col] = df[col].fillna(df[col].median())
    
    # Supprimer les lignes avec des valeurs nulles dans les colonnes clés
    key_columns = ['date', 'customer group', 'item', 'qty']
    df = df.dropna(subset=key_columns)
    
    return df

def calculate_forecast_accuracy(actual: pd.Series, forecast: pd.Series) -> Dict[str, float]:
    """
    Calcule différentes métriques de précision de prévision
    
    Args:
        actual (pd.Series): Valeurs réelles
        forecast (pd.Series): Valeurs prévues
    
    Returns:
        Dict[str, float]: Métriques de précision
    """
    # Mean Absolute Error (MAE)
    mae = np.mean(np.abs(actual - forecast))
    
    # Mean Absolute Percentage Error (MAPE)
    mape = np.mean(np.abs((actual - forecast) / actual)) * 100
    
    # Root Mean Squared Error (RMSE)
    rmse = np.sqrt(np.mean((actual - forecast)**2))
    
    return {
        'MAE': mae,
        'MAPE': mape,
        'RMSE': rmse
    }

def detect_seasonality(df: pd.DataFrame, date_column: str, value_column: str) -> Dict[str, Any]:
    """
    Détecte les caractéristiques saisonnières dans les données
    
    Args:
        df (pd.DataFrame): DataFrame des ventes
        date_column (str): Nom de la colonne de date
        value_column (str): Nom de la colonne de valeurs
    
    Returns:
        Dict[str, Any]: Informations sur la saisonnalité
    """
    # Assurer que la date est au format datetime
    df[date_column] = pd.to_datetime(df[date_column])
    
    # Grouper par mois et calculer la moyenne
    monthly_sales = df.groupby(pd.Grouper(key=date_column, freq='M'))[value_column].sum()
    
    # Calculer la variation mensuelle
    monthly_variation = monthly_sales.groupby(monthly_sales.index.month).mean()
    
    # Identifier le mois avec le pic et le creux de ventes
    peak_month = monthly_variation.idxmax()
    lowest_month = monthly_variation.idxmin()
    
    return {
        'monthly_variation': monthly_variation.to_dict(),
        'peak_month': peak_month,
        'lowest_month': lowest_month,
        'seasonal_strength': monthly_variation.max() / monthly_variation.min()
    }

def validate_sales_data(df: pd.DataFrame) -> List[str]:
    """
    Valide les données de ventes et retourne une liste de problèmes
    
    Args:
        df (pd.DataFrame): DataFrame des ventes
    
    Returns:
        List[str]: Liste des problèmes détectés
    """
    problems = []
    
    # Vérifier les colonnes requises
    required_columns = ['date', 'customer group', 'item', 'qty']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        problems.append(f"Colonnes manquantes : {', '.join(missing_columns)}")
    
    # Vérifier les valeurs négatives
    negative_qty = df[df['qty'] < 0]
    if not negative_qty.empty:
        problems.append(f"{len(negative_qty)} lignes avec des quantités négatives")
    
    # Vérifier les dates
    try:
        df['date'] = pd.to_datetime(df['date'])
    except:
        problems.append("Format de date invalide")
    
    # Vérifier les plages de données
    if len(df) == 0:
        problems.append("Aucune donnée dans le fichier")
    
    # Vérifier les doublons
    duplicates = df.duplicated(subset=['date', 'customer group', 'item']).sum()
    if duplicates > 0:
        problems.append(f"{duplicates} lignes dupliquées")
    
    return problems

def generate_sales_report(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Génère un rapport récapitulatif des données de ventes
    
    Args:
        df (pd.DataFrame): DataFrame des ventes
    
    Returns:
        Dict[str, Any]: Rapport de ventes
    """
    # Résumé par client
    customer_summary = df.groupby('customer group')['qty'].agg([
        ('total_sales', 'sum'),
        ('avg_sales', 'mean'),
        ('max_sales', 'max'),
        ('min_sales', 'min')
    ]).reset_index()
    
    # Résumé par article
    item_summary = df.groupby('item')['qty'].agg([
        ('total_sales', 'sum'),
        ('avg_sales', 'mean'),
        ('max_sales', 'max'),
        ('min_sales', 'min')
    ]).reset_index()
    
    return {
        'total_sales': df['qty'].sum(),
        'customer_summary': customer_summary.to_dict(orient='records'),
        'item_summary': item_summary.to_dict(orient='records'),
        'date_range': {
            'start_date': df['date'].min(),
            'end_date': df['date'].max()
        }
    }