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
        
        # Renommage des colonnes pour standardisation - Correction de l'inversion
        col_mapping = {
            'date': 'Date',
            'customer group': 'Item',  # Inverser Customer group et Item
            'item': 'Customer group',  # Inverser Customer group et Item
            'qty': 'Qty'
        }
        df = df.rename(columns=col_mapping)
        
        # Conversion de la colonne Date en datetime
        df['Date'] = pd.to_datetime(df['Date'])
        
        # Conversion Qty en numérique, gestion des valeurs non-numériques
        df['Qty'] = pd.to_numeric(df['Qty'], errors='coerce').fillna(0)
        
        # Exclusion des valeurs aberrantes (plus de 3 écarts-types de la moyenne)
        mean_qty = df['Qty'].mean()
        std_qty = df['Qty'].std()
        if std_qty > 0:  # Éviter la division par zéro
            df = df[abs(df['Qty'] - mean_qty) <= 3 * std_qty]
        
        # Tri des données par date
        df = df.sort_values('Date')
        
        # Afficher l'aperçu des données avec les colonnes correctes
        st.write("Aperçu des données après prétraitement et correction des colonnes:")
        st.dataframe(df[['Date', 'Item', 'Customer group', 'Qty']].head())
        
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
            # Calcul de la moyenne mobile sur les 6 derniers mois
            recent_sales = monthly_sales['Qty'].tail(6)
            avg_sales = recent_sales.mean()
            
            # Vérification: si la moyenne est significativement différente de la dernière valeur
            last_value = monthly_sales['Qty'].iloc[-1]
            if last_value > 0 and (avg_sales > last_value * 2 or avg_sales < last_value * 0.5):
                # Si la moyenne est très différente, utiliser une moyenne pondérée
                # qui donne plus de poids aux mois récents
                weights = [0.1, 0.1, 0.15, 0.15, 0.2, 0.3]  # Plus de poids aux derniers mois
                weighted_avg = sum(recent_sales.iloc[i] * weights[i] for i in range(6))
                ma_forecast = [max(0, weighted_avg)] * months_to_forecast
            else:
                ma_forecast = [max(0, avg_sales)] * months_to_forecast
        
        # Génération des prévisions avec le premier jour de chaque mois
        # Premier jour du mois suivant le dernier mois de données
        forecast_start = last_date.replace(day=1) + pd.offsets.MonthBegin(1)
        forecast_dates = [forecast_start] * months_to_forecast
        
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
        if len(monthly_sales) < 3:  # On a besoin d'au moins 3 points pour un bon lissage
            avg_sales = monthly_sales['Qty'].mean()
            es_forecast = [max(0, avg_sales)] * months_to_forecast
        else:
            # Amélioration du lissage exponentiel simple
            # Pour plus de robustesse, nous utilisons les 12 derniers mois si disponibles
            train_data = monthly_sales['Qty'].tail(min(12, len(monthly_sales)))
            
            # On optimise le paramètre alpha pour un meilleur ajustement
            model = SimpleExpSmoothing(train_data).fit(optimized=True)
            
            # On vérifie que la dernière valeur lissée n'est pas trop éloignée
            last_smoothed = model.fittedvalues[-1]
            last_actual = train_data.iloc[-1]
            
            # Si l'écart est trop grand, on utilise la moyenne des 3 derniers mois
            if abs(last_smoothed - last_actual) > last_actual * 0.5:
                last_value = train_data.tail(3).mean()
                es_forecast = [max(0, last_value)] * months_to_forecast
            else:
                # Prévisions
                es_forecast = model.forecast(steps=months_to_forecast).tolist()
                es_forecast = [max(0, x) for x in es_forecast]  # Éviter les valeurs négatives
        
        # Génération des prévisions avec le premier jour de chaque mois
        forecast_start = last_date.replace(day=1) + pd.offsets.MonthBegin(1)
        forecast_dates = [forecast_start] * months_to_forecast
        
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
        if len(monthly_sales) < 3:  # Au moins 3 points pour une régression fiable
            avg_sales = monthly_sales['Qty'].mean()
            lr_forecast = [max(0, avg_sales)] * months_to_forecast
        else:
            # Préparation des données pour régression
            # Limiter aux 12 derniers mois pour éviter une influence excessive de données trop anciennes
            recent_data = monthly_sales.tail(min(12, len(monthly_sales)))
            X = np.arange(len(recent_data)).reshape(-1, 1)
            y = recent_data['Qty'].values
            
            # Régression linéaire
            slope, intercept, r_value, p_value, std_err = stats.linregress(X.ravel(), y)
            
            # Vérifier si la régression est significative
            if p_value > 0.3 or r_value**2 < 0.3:  # Si non significative ou faible R²
                # Utiliser la moyenne des derniers mois plutôt que la régression
                avg_recent = recent_data['Qty'].tail(3).mean()
                lr_forecast = [max(0, avg_recent)] * months_to_forecast
            else:
                # Prévision des prochains mois basée sur la régression
                # Le dernier index est la longueur des données récentes
                last_index = len(recent_data) - 1
                lr_forecast = [max(0, slope * (last_index + 1 + i) + intercept) for i in range(months_to_forecast)]
                
                # Vérifier que les prévisions ne sont pas aberrantes (pas plus de 2x la moyenne récente)
                avg_recent = recent_data['Qty'].mean()
                max_reasonable = avg_recent * 2 if avg_recent > 0 else 100
                lr_forecast = [min(max_reasonable, f) for f in lr_forecast]
        
        # Génération des prévisions avec le premier jour de chaque mois
        forecast_start = last_date.replace(day=1) + pd.offsets.MonthBegin(1)
        forecast_dates = [forecast_start] * months_to_forecast
        
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
            st.write(f"Dernière date dans les données : {last_date.strftime('%d/%m/%Y')}")
            st.write(f"Date de début des prévisions : {(last_date.replace(day=1) + pd.offsets.MonthBegin(1)).strftime('%d/%m/%Y')}")
            
            # Affichage des statistiques descriptives
            st.write("Statistiques descriptives des ventes:")
            st.dataframe(df.groupby('Item')['Qty'].describe())
            
            # Récupération des groupes uniques
            customer_groups = df['Customer group'].unique()
            items = df['Item'].unique()
            
            # Dictionnaires pour stocker les résultats
            ma_results = {}
            es_results = {}
            lr_results = {}
            
            # Barre de progression
            progress_bar = st.progress(0)
            total_combinations = len(items) * len(customer_groups)
            counter = 0
            
            # Calcul des prévisions pour chaque combinaison
            for item in items:
                for group in customer_groups:
                    # Mise à jour de la barre de progression
                    counter += 1
                    progress_bar.progress(counter / total_combinations)
                    
                    # Filtrage des données
                    subset = df[(df['Customer group'] == group) & (df['Item'] == item)]
                    
                    # Vérifier s'il y a des données pour cette combinaison
                    if not subset.empty:
                        # Calcul des prévisions
                        ma_forecast = moving_average_forecast(subset, last_date)
                        es_forecast = exponential_smoothing_forecast(subset, last_date)
                        lr_forecast = linear_regression_forecast(subset, last_date)
                        
                        # Stockage des résultats
                        key = f"{item} - {group}"
                        
                        if ma_forecast is not None:
                            ma_results[key] = ma_forecast
                        
                        if es_forecast is not None:
                            es_results[key] = es_forecast
                        
                        if lr_forecast is not None:
                            lr_results[key] = lr_forecast
            
            # Exemple de prévisions pour un article spécifique
            if items.size > 0 and customer_groups.size > 0:
                sample_item = items[0]
                sample_group = customer_groups[0]
                sample_key = f"{sample_item} - {sample_group}"
                
                if sample_key in ma_results:
                    st.write(f"Exemple de prévisions pour {sample_item} (Groupe: {sample_group}):")
                    st.write("Moyenne mobile:")
                    st.dataframe(ma_results[sample_key])
            
            # Bouton de téléchargement des résultats
            if st.button("Générer et Télécharger les Prévisions"):
                # Création d'un fichier Excel avec plusieurs onglets
                with pd.ExcelWriter('Previsions_Ventes.xlsx') as writer:
                    # Onglet Moyenne Mobile
                    if ma_results:
                        ma_combined = pd.concat([
                            pd.DataFrame({'Article': [k.split(' - ')[0]] * len(v), 
                                          'Groupe Client': [k.split(' - ')[1]] * len(v), 
                                          **v}) 
                            for k, v in ma_results.items()
                        ])
                        ma_combined.to_excel(writer, sheet_name='Moyenne_Mobile', index=False)
                    
                    # Onglet Lissage Exponentiel
                    if es_results:
                        es_combined = pd.concat([
                            pd.DataFrame({'Article': [k.split(' - ')[0]] * len(v), 
                                          'Groupe Client': [k.split(' - ')[1]] * len(v), 
                                          **v}) 
                            for k, v in es_results.items()
                        ])
                        es_combined.to_excel(writer, sheet_name='Lissage_Exponentiel', index=False)
                    
                    # Onglet Régression Linéaire
                    if lr_results:
                        lr_combined = pd.concat([
                            pd.DataFrame({'Article': [k.split(' - ')[0]] * len(v), 
                                          'Groupe Client': [k.split(' - ')[1]] * len(v), 
                                          **v}) 
                            for k, v in lr_results.items()
                        ])
                        lr_combined.to_excel(writer, sheet_name='Regression_Lineaire', index=False)
                    
                    # Onglet de comparaison des méthodes
                    if ma_results and es_results and lr_results:
                        # Prendre un échantillon pour comparer
                        sample_comparisons = []
                        for key in list(ma_results.keys())[:10]:  # Limiter à 10 combinaisons pour éviter la surcharge
                            item, group = key.split(' - ')
                            if key in ma_results and key in es_results and key in lr_results:
                                ma_val = ma_results[key]['Qty_Forecast_MA'][0]
                                es_val = es_results[key]['Qty_Forecast_ES'][0]
                                lr_val = lr_results[key]['Qty_Forecast_LR'][0]
                                
                                sample_comparisons.append({
                                    'Article': item,
                                    'Groupe Client': group,
                                    'Moyenne Mobile': ma_val,
                                    'Lissage Exponentiel': es_val,
                                    'Régression Linéaire': lr_val
                                })
                        
                        if sample_comparisons:
                            comparison_df = pd.DataFrame(sample_comparisons)
                            comparison_df.to_excel(writer, sheet_name='Comparaison_Methodes', index=False)
                
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