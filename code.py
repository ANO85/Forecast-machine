import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import io

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
        
        # Renommage des colonnes pour standardisation - CORRECTION: NE PAS INVERSER
        col_mapping = {
            'date': 'Date',
            'customer group': 'Customer group',  # Correction: ne pas inverser
            'item': 'Item',                      # Correction: ne pas inverser
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
        st.write("Aperçu des données après prétraitement:")
        st.dataframe(df[['Date', 'Customer group', 'Item', 'Qty']].head())
        
        return df
    except Exception as e:
        st.error(f"Erreur de chargement du fichier : {e}")
        return None

# Méthode 1 : Moyenne mobile sur 6 mois
def moving_average_forecast(group, last_date, months_to_forecast=12):
    # Groupement par mois
    try:
        # Grouper par mois pour obtenir la somme mensuelle des quantités
        monthly_sales = group.groupby(pd.Grouper(key='Date', freq='M'))['Qty'].sum().reset_index()
        
        # Afficher les données mensuelles pour debug
        print(f"Données mensuelles: {monthly_sales}")
        
        # Si moins de 6 mois de données, utiliser la moyenne totale
        if len(monthly_sales) < 6:
            avg_sales = monthly_sales['Qty'].mean()
            ma_forecast = [max(0, avg_sales)] * months_to_forecast
        else:
            # CORRECTION: Utiliser seulement les 6 derniers mois pour la moyenne mobile
            recent_sales = monthly_sales['Qty'].tail(6)
            avg_sales = recent_sales.mean()
            ma_forecast = [max(0, avg_sales)] * months_to_forecast
            
            # Debug
            print(f"Moyenne mobile sur les 6 derniers mois: {avg_sales}")
        
        # Générer des dates correctes pour les 12 prochains mois
        forecast_dates = []
        forecast_start = last_date.replace(day=1) + pd.offsets.MonthBegin(1)
        for i in range(months_to_forecast):
            forecast_dates.append(forecast_start + pd.offsets.MonthBegin(i))
        
        forecast_df = pd.DataFrame({
            'Date': forecast_dates,
            'Qty_Forecast_MA': ma_forecast
        })
        
        # Formatage des dates
        forecast_df['Date'] = forecast_df['Date'].dt.strftime('%d/%m/%Y')
        
        return forecast_df
    except Exception as e:
        print(f"Erreur de prévision par moyenne mobile : {e}")
        return None

# Méthode 2 : Lissage exponentiel simple
def exponential_smoothing_forecast(group, last_date, months_to_forecast=12):
    try:
        # Groupement par mois
        monthly_sales = group.groupby(pd.Grouper(key='Date', freq='M'))['Qty'].sum().reset_index()
        
        # Debug: afficher les données mensuelles
        print(f"Données mensuelles (lissage exponentiel): {monthly_sales}")
        
        # Si pas assez de données, utiliser la moyenne
        if len(monthly_sales) < 3:  # On a besoin d'au moins 3 points pour un bon lissage
            avg_sales = monthly_sales['Qty'].mean()
            es_forecast = [max(0, avg_sales)] * months_to_forecast
            print(f"Pas assez de données, utilisation de la moyenne simple: {avg_sales}")
        else:
            # CORRECTION: Utiliser uniquement les données récentes (jusqu'à 12 derniers mois)
            train_data = monthly_sales['Qty'].tail(min(12, len(monthly_sales)))
            
            # Paramètre alpha plus petit pour moins de réactivité aux dernières observations
            # (0.2 est une valeur standard pour donner plus de poids à l'historique)
            alpha = 0.2  # Utiliser une valeur fixe au lieu de l'optimisation qui peut être instable
            model = SimpleExpSmoothing(train_data).fit(smoothing_level=alpha, optimized=False)
            
            es_forecast = model.forecast(steps=months_to_forecast).tolist()
            
            # Éviter les valeurs négatives et les valeurs aberrantes
            # Limiter les prévisions à 150% de la moyenne historique pour éviter les surestimations
            avg_historical = train_data.mean()
            max_forecast = avg_historical * 1.5
            es_forecast = [max(0, min(x, max_forecast)) for x in es_forecast]
            
            print(f"Lissage exponentiel avec alpha={alpha}, moyenne={avg_historical}, max_forecast={max_forecast}")
        
        # Générer des dates correctes pour les prochains mois
        forecast_dates = []
        forecast_start = last_date.replace(day=1) + pd.offsets.MonthBegin(1)
        for i in range(months_to_forecast):
            forecast_dates.append(forecast_start + pd.offsets.MonthBegin(i))
        
        forecast_df = pd.DataFrame({
            'Date': forecast_dates,
            'Qty_Forecast_ES': es_forecast
        })
        
        # Formatage des dates
        forecast_df['Date'] = forecast_df['Date'].dt.strftime('%d/%m/%Y')
        
        return forecast_df
    except Exception as e:
        print(f"Erreur de prévision par lissage exponentiel : {e}")
        return None

# Méthode 3 : Régression linéaire simple
def linear_regression_forecast(group, last_date, months_to_forecast=12):
    try:
        # Groupement par mois
        monthly_sales = group.groupby(pd.Grouper(key='Date', freq='M'))['Qty'].sum().reset_index()
        
        # Debug: afficher les données mensuelles
        print(f"Données mensuelles (régression linéaire): {monthly_sales}")
        
        # Si pas assez de données, utiliser la moyenne
        if len(monthly_sales) < 3:  # Au moins 3 points pour une régression fiable
            avg_sales = monthly_sales['Qty'].mean()
            lr_forecast = [max(0, avg_sales)] * months_to_forecast
            print(f"Pas assez de données pour régression, utilisation de la moyenne: {avg_sales}")
        else:
            # CORRECTION: Utiliser uniquement les données récentes (jusqu'à 12 derniers mois)
            recent_data = monthly_sales.tail(min(12, len(monthly_sales)))
            X = np.arange(len(recent_data)).reshape(-1, 1)
            y = recent_data['Qty'].values
            
            # Régression linéaire
            slope, intercept, r_value, p_value, std_err = stats.linregress(X.ravel(), y)
            print(f"Régression: pente={slope}, intercept={intercept}, r²={r_value**2}")
            
            # Si la tendance est négative ou trop accentuée, limiter son impact
            if slope < 0:
                # Si tendance négative, atténuer pour éviter des prévisions trop pessimistes
                # Réduire la pente négative de moitié
                slope = slope / 2
            elif slope > 0 and slope > np.mean(y) / 12:
                # Si la pente est trop forte (croissance > moyenne/an), la limiter
                slope = np.mean(y) / 12
            
            # Calculer la moyenne des données récentes pour référence
            avg_recent = np.mean(y)
            
            # Prévision des prochains mois basée sur la régression modifiée
            last_index = len(recent_data) - 1
            lr_forecast = []
            
            for i in range(months_to_forecast):
                # Calculer la valeur prédite
                predicted = slope * (last_index + 1 + i) + intercept
                
                # Limiter la prévision à 150% de la moyenne récente pour éviter les surestimations
                max_forecast = avg_recent * 1.5
                
                # Appliquer les limites (jamais négatif, jamais > 150% de la moyenne)
                lr_forecast.append(max(0, min(predicted, max_forecast)))
            
            print(f"Moyenne récente: {avg_recent}, valeurs prédites: {lr_forecast}")
        
        # Générer des dates correctes pour les prochains mois
        forecast_dates = []
        forecast_start = last_date.replace(day=1) + pd.offsets.MonthBegin(1)
        for i in range(months_to_forecast):
            forecast_dates.append(forecast_start + pd.offsets.MonthBegin(i))
        
        forecast_df = pd.DataFrame({
            'Date': forecast_dates,
            'Qty_Forecast_LR': lr_forecast
        })
        
        # Formatage des dates
        forecast_df['Date'] = forecast_df['Date'].dt.strftime('%d/%m/%Y')
        
        return forecast_df
    except Exception as e:
        print(f"Erreur de prévision par régression linéaire : {e}")
        return None

# Interface Streamlit principale
def main():
    # Initialisation des variables de session pour stocker les résultats
    if 'forecast_calculated' not in st.session_state:
        st.session_state.forecast_calculated = False
    if 'excel_data' not in st.session_state:
        st.session_state.excel_data = None
    
    # Téléchargement du fichier
    uploaded_file = st.file_uploader("Charger le fichier historique des ventes", 
                                     type=['xlsx', 'xls'], 
                                     help="Fichier Excel avec colonnes : Date, Customer group, Item, Qty")
    
    # Ajout d'un sélecteur pour choisir l'horizon des prévisions
    months_to_forecast = st.selectbox(
        "Horizon de prévision (mois)",
        options=[3, 6, 12, 24],
        index=1  # 6 mois par défaut
    )
    
    # Paramètre pour contrôler le nombre de mois historiques à utiliser
    historic_months = st.slider(
        "Nombre de mois d'historique à utiliser",
        min_value=3, 
        max_value=24, 
        value=6,
        help="Nombre de mois d'historique à utiliser pour calculer les prévisions"
    )
    
    # Option pour filtrer les valeurs extrêmes
    filter_outliers = st.checkbox("Filtrer les valeurs extrêmes", value=True, 
                                 help="Supprimer les valeurs qui dépassent 3 écarts-types de la moyenne")
    
    if uploaded_file is not None:
        # Réinitialiser les calculs si un nouveau fichier est chargé
        if 'current_file' not in st.session_state or st.session_state.current_file != uploaded_file.name:
            st.session_state.forecast_calculated = False
            st.session_state.excel_data = None
            st.session_state.current_file = uploaded_file.name
            
        # Chargement et prétraitement des données
        df = load_and_preprocess_data(uploaded_file)
        
        if df is not None and not st.session_state.forecast_calculated:
            # Trouver la dernière date dans le jeu de données
            last_date = df['Date'].max()
            
            # Affichage des informations utiles
            with st.expander("Informations sur les données"):
                st.write(f"Dernière date dans les données : {last_date.strftime('%d/%m/%Y')}")
                st.write(f"Date de début des prévisions : {(last_date.replace(day=1) + pd.offsets.MonthBegin(1)).strftime('%d/%m/%Y')}")
                st.write("Statistiques descriptives des ventes:")
                st.dataframe(df.groupby(['Customer group', 'Item'])['Qty'].describe())
            
            # Récupération des groupes uniques
            customer_groups = df['Customer group'].unique()
            items = df['Item'].unique()
            
            # Limiter les données historiques en fonction du nombre de mois sélectionné
            if historic_months > 0:
                cutoff_date = last_date - pd.DateOffset(months=historic_months)
                df_filtered = df[df['Date'] >= cutoff_date]
                if len(df_filtered) > 0:  # Vérifier qu'il reste des données
                    df = df_filtered
                    st.info(f"Utilisation des {historic_months} derniers mois d'historique (à partir de {cutoff_date.strftime('%d/%m/%Y')})")
            
            # Dictionnaires pour stocker les résultats
            ma_results = {}
            es_results = {}
            lr_results = {}
            
            # Barre de progression
            st.write("Calcul des prévisions en cours...")
            progress_bar = st.progress(0)
            progress_status = st.empty()
            total_combinations = len(items) * len(customer_groups)
            counter = 0
            
            # Calcul des prévisions pour chaque combinaison
            for item in items:
                for group in customer_groups:
                    # Mise à jour de la barre de progression
                    counter += 1
                    progress_bar.progress(counter / total_combinations)
                    progress_status.text(f"Traitement: {counter}/{total_combinations} - Item: {item}, Groupe: {group}")
                    
                    # Filtrage des données
                    subset = df[(df['Customer group'] == group) & (df['Item'] == item)]
                    
                    # Vérifier s'il y a des données pour cette combinaison
                    if not subset.empty:
                        # Ajout d'un état d'avancement plus détaillé
                        with st.expander(f"Détails pour {item} (Groupe: {group})", expanded=False):
                            st.write(f"Nombre d'enregistrements: {len(subset)}")
                            # Afficher les ventes mensuelles pour inspection
                            monthly_data = subset.groupby(pd.Grouper(key='Date', freq='M'))['Qty'].sum().reset_index()
                            monthly_data['Date'] = monthly_data['Date'].dt.strftime('%m/%Y')
                            monthly_avg = monthly_data['Qty'].mean()
                            st.write(f"Moyenne mensuelle: {monthly_avg:.2f} unités")
                            st.dataframe(monthly_data)
                        
                        # Calcul des prévisions avec le nombre de mois spécifié
                        ma_forecast = moving_average_forecast(subset, last_date, months_to_forecast)
                        es_forecast = exponential_smoothing_forecast(subset, last_date, months_to_forecast)
                        lr_forecast = linear_regression_forecast(subset, last_date, months_to_forecast)
                        
            # Stockage des résultats
                        # Construction de la clé assurant un format uniforme "item - group"
                        key = f"{item} - {group}"
                        
                        if ma_forecast is not None:
                            ma_results[key] = ma_forecast
                        
                        if es_forecast is not None:
                            es_results[key] = es_forecast
                        
                        if lr_forecast is not None:
                            lr_results[key] = lr_forecast
            
            # Effacer le statut de progression une fois terminé
            progress_status.empty()
            st.success("Calcul des prévisions terminé!")
            
            # Exemple de prévisions pour un article spécifique
            if items.size > 0 and customer_groups.size > 0:
                sample_item = items[0]
                sample_group = customer_groups[0]
                sample_key = f"{sample_item} - {sample_group}"
                
                with st.expander("Exemple de prévisions"):
                    if sample_key in ma_results:
                        st.write(f"Prévisions pour {sample_item} (Groupe: {sample_group}):")
                        st.write("Moyenne mobile:")
                        st.dataframe(ma_results[sample_key])
            
            # Générer le fichier Excel en mémoire
            excel_buffer = io.BytesIO()
            
            # CORRECTION: Utiliser openpyxl au lieu de xlsxwriter
            try:
                # Vérifier d'abord si openpyxl est disponible
                import openpyxl
                excel_engine = 'openpyxl'
            except ImportError:
                # Si openpyxl n'est pas disponible, utiliser le moteur par défaut
                excel_engine = None
            
            with pd.ExcelWriter(excel_buffer, engine=excel_engine) as writer:
                # Onglet Moyenne Mobile
                if ma_results:
                    ma_combined = pd.concat([
                        pd.DataFrame({
                            'Item': [k.split(' - ')[0]] * len(v),  # Correction: item en premier 
                            'Customer group': [k.split(' - ')[1]] * len(v),  # Correction: groupe client en second
                            **v
                        }) 
                        for k, v in ma_results.items()
                    ])
                    ma_combined.to_excel(writer, sheet_name='Moyenne_Mobile', index=False)
                
                # Onglet Lissage Exponentiel
                if es_results:
                    es_combined = pd.concat([
                        pd.DataFrame({
                            'Item': [k.split(' - ')[0]] * len(v),  # Correction: item en premier
                            'Customer group': [k.split(' - ')[1]] * len(v),  # Correction: groupe client en second
                            **v
                        }) 
                        for k, v in es_results.items()
                    ])
                    es_combined.to_excel(writer, sheet_name='Lissage_Exponentiel', index=False)
                
                # Onglet Régression Linéaire
                if lr_results:
                    lr_combined = pd.concat([
                        pd.DataFrame({
                            'Item': [k.split(' - ')[0]] * len(v),  # Correction: item en premier
                            'Customer group': [k.split(' - ')[1]] * len(v),  # Correction: groupe client en second
                            **v
                        }) 
                        for k, v in lr_results.items()
                    ])
                    lr_combined.to_excel(writer, sheet_name='Regression_Lineaire', index=False)
                
                # Onglet de comparaison des méthodes
                if ma_results and es_results and lr_results:
                    # Prendre un échantillon pour comparer
                    sample_comparisons = []
                    for key in list(ma_results.keys())[:10]:  # Limiter à 10 combinaisons
                        # Gérer le cas où le format de clé est incorrect ou ne contient pas un délimiteur ' - '
                        key_parts = key.split(' - ', 1)  # Limite le split à 1 seule division
                        if len(key_parts) != 2:
                            # Si la clé n'a pas le bon format, passez à la suivante
                            continue
                            
                        item, group = key_parts
                        if key in ma_results and key in es_results and key in lr_results:
                            # Pour chaque élément, nous voulons récupérer la prévision pour chaque mois
                            # Vérifier qu'il y a bien 12 mois de données disponibles
                            try:
                                for month in range(min(12, len(ma_results[key]['Date']))):
                                    date_str = ma_results[key]['Date'][month]
                                    ma_val = ma_results[key]['Qty_Forecast_MA'][month]
                                    es_val = es_results[key]['Qty_Forecast_ES'][month]
                                    lr_val = lr_results[key]['Qty_Forecast_LR'][month]
                                    
                                    sample_comparisons.append({
                                        'Item': item,
                                        'Customer group': group,
                                        'Date': date_str,
                                        'Moyenne Mobile': ma_val,
                                        'Lissage Exponentiel': es_val,
                                        'Régression Linéaire': lr_val
                                    })
                            except (IndexError, KeyError) as e:
                                # En cas d'erreur d'accès aux données, enregistrer l'erreur et continuer
                                print(f"Erreur lors de l'accès aux données de prévision pour {key}: {e}")
                    
                    if sample_comparisons:
                        comparison_df = pd.DataFrame(sample_comparisons)
                        comparison_df.to_excel(writer, sheet_name='Comparaison_Methodes', index=False)
            
            # Enregistrement du résultat dans la session
            excel_buffer.seek(0)
            st.session_state.excel_data = excel_buffer.getvalue()
            st.session_state.forecast_calculated = True
        
        # Affichage du bouton de téléchargement si le calcul est terminé
        if st.session_state.forecast_calculated and st.session_state.excel_data is not None:
            st.download_button(
                label="📊 Télécharger les prévisions",
                data=st.session_state.excel_data,
                file_name='Previsions_Ventes.xlsx',
                mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                key='download_excel'
            )

# Point d'entrée principal
if __name__ == "__main__":
    main()