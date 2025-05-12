import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_percentage_error

def moving_average_forecast(data, periods=6):
    """
    Calcule les prévisions par moyenne mobile sur 6 mois
    
    Args:
        data (pd.Series): Données historiques de ventes
        periods (int): Nombre de périodes pour la moyenne mobile
    
    Returns:
        pd.Series: Prévisions pour les 12 prochains mois
    """
    # Calcul de la moyenne mobile
    rolling_mean = data.rolling(window=periods).mean()
    
    # Utiliser la dernière moyenne mobile comme prévision
    last_mean = rolling_mean.iloc[-1]
    
    # Générer les prévisions pour les 12 prochains mois
    forecast = [last_mean] * 12
    
    return pd.Series(forecast)

def exponential_smoothing_forecast(data):
    """
    Calcule les prévisions par lissage exponentiel
    
    Args:
        data (pd.Series): Données historiques de ventes
    
    Returns:
        pd.Series: Prévisions pour les 12 prochains mois
    """
    # Ajustement du modèle de lissage exponentiel
    model = ExponentialSmoothing(
        data, 
        trend='add', 
        seasonal='add', 
        seasonal_periods=12
    ).fit()
    
    # Prévisions pour les 12 prochains mois
    forecast = model.forecast(12)
    
    return pd.Series(forecast)

def sarima_forecast(data):
    """
    Calcule les prévisions par modèle SARIMA
    
    Args:
        data (pd.Series): Données historiques de ventes
    
    Returns:
        pd.Series: Prévisions pour les 12 prochains mois
    """
    try:
        # Ajustement automatique du modèle SARIMA
        model = SARIMAX(
            data, 
            order=(1, 1, 1),  # p,d,q 
            seasonal_order=(1, 1, 1, 12)  # P,D,Q,s
        ).fit(disp=False)
        
        # Prévisions pour les 12 prochains mois
        forecast = model.forecast(12)
        
        return pd.Series(forecast)
    except:
        # Fallback sur la moyenne mobile si SARIMA échoue
        return moving_average_forecast(data)

def select_best_method(actual, ma_forecast, exp_forecast, sarima_forecast):
    """
    Sélectionne la meilleure méthode de prévision basée sur l'MAPE
    
    Args:
        actual (pd.Series): Données historiques réelles
        ma_forecast (pd.Series): Prévisions par moyenne mobile
        exp_forecast (pd.Series): Prévisions par lissage exponentiel
        sarima_forecast (pd.Series): Prévisions SARIMA
    
    Returns:
        str: Nom de la meilleure méthode
    """
    # Calcul du MAPE pour chaque méthode
    # Note : on utilise les dernières données disponibles comme "actual"
    mape_methods = {
        'Moving Average': mean_absolute_percentage_error(actual, ma_forecast),
        'Exponential Smoothing': mean_absolute_percentage_error(actual, exp_forecast),
        'SARIMA': mean_absolute_percentage_error(actual, sarima_forecast)
    }
    
    # Retourne la méthode avec le MAPE le plus bas
    return min(mape_methods, key=mape_methods.get)