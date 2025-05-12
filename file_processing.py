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
        'date': ['Date', 'date', 'dates', 'datetime', 'date_vente'],
        'customer group': ['Customer Group', 'Customer', 'customer group', 'customer_group', 'groupe client', 'client'],
        'item': ['Item', 'item', 'product', 'produit', 'article'],
        'qty': ['Qty', 'qty', 'quantity', 'quantite', 'qte', 'ventes']
    }
    
    # Créer un nouveau dictionnaire de mapping
    new_columns = {}
    for standard_name, possible_names in column_mapping.items():
        # Trouver la première correspondance
        match = next((col for col in df.columns if col in possible_names), None)
        
        if match:
            # Remplacer le nom de colonne par le nom standard
            new_columns[match] = standard_name
    
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
    
    # Imprimer les colonnes originales pour le débogage
    print("Colonnes originales :", list(df.columns))
    
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

# Le reste du code reste identique aux versions précédentes (generate_forecasts et save_forecasts_to_excel)