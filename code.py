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
        
        # Renommage des colonnes pour standardisation
        col_mapping = {
            'date': 'Date',
            'customer group': 'Customer group',
            'item': 'Item',
            'qty': 'Qty'
        }
        df = df.rename(columns=col_mapping)
        
        # Conversion de la colonne Date en datetime
        df['Date'] = pd.to_datetime(df['Date'])
        
        # Conversion Qty en numérique, gestion des valeurs non-numériques
        df['Qty'] = pd.to_numeric(df['Qty'], errors='coerce').fillna(0)
        
        # Exclusion des valeurs aberrantes (plus de 3 écarts-types de la moyenne)
        # SUPPRESSION de ce filtrage ici pour éviter un double filtrage
        # Les filtrages spécifiques seront appliqués dans les fonctions de prévision si nécessaire
        
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
        # CORRECTION: Vérifier d'abord si le DataFrame est vide
        if group.empty:
            return None
            
        # Afficher les données brutes pour debug
        st.write(f"DEBUG - Données brutes pour l'article sélectionné:")
        st.dataframe(group[['Date', 'Qty']].sort_values('Date'))
            
        # CORRECTION: Agréger les données quotidiennes en total mensuel
        # Grouper par mois pour obtenir la somme mensuelle des quantités
        monthly_sales = group.groupby(pd.Grouper(key='Date', freq='M'))['Qty'].sum()
        
        # CORRECTION: Convertir en DataFrame pour une meilleure manipulation
        monthly_sales = monthly_sales.reset_index()
        
        # Afficher les données mensuelles pour vérification
        st.write("DEBUG - Données mensuelles agrégées:")
        monthly_view = monthly_sales.copy()
        monthly_view['Date'] = monthly_view['Date'].dt.strftime('%m/%Y')
        st.dataframe(monthly_view)
        
        # Si moins de 6 mois de données, utiliser la moyenne totale
        if len(monthly_sales) < 6:
            avg_sales = monthly_sales['Qty'].mean()
            ma_forecast = [max(0, avg_sales)] * months_to_forecast
            st.write(f"DEBUG - Moins de 6 mois de données, moyenne utilisée: {avg_sales}")
        else:
            # CORRECTION: Utiliser les 6 derniers mois pour la moyenne mobile
            recent_sales = monthly_sales.tail(6)['Qty']
            st.write("DEBUG - 6 derniers mois utilisés pour la moyenne mobile:")
            st.dataframe(monthly_sales.tail(6))
            
            avg_sales = recent_sales.mean()
            ma_forecast = [max(0, avg_sales)] * months_to_forecast
            
            st.write(f"DEBUG - Moyenne mobile calculée sur les 6 derniers mois: {avg_sales}")
        
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
        st.error(f"Erreur de prévision par moyenne mobile : {e}")
        print(f"Erreur détaillée: {e}")
        import traceback
        st.code(traceback.format_exc())
        return None

# Méthode 2 : Lissage exponentiel simple
def exponential_smoothing_forecast(group, last_date, months_to_forecast=12):
    try:
        # CORRECTION: Vérifier d'abord si le DataFrame est vide
        if group.empty:
            return None
            
        # Groupement par mois
        monthly_sales = group.groupby(pd.Grouper(key='Date', freq='M'))['Qty'].sum().reset_index()
        
        # Si pas assez de données, utiliser la moyenne
        if len(monthly_sales) < 3:  # On a besoin d'au moins 3 points pour un bon lissage
            avg_sales = monthly_sales['Qty'].mean()
            es_forecast = [max(0, avg_sales)] * months_to_forecast
        else:
            # CORRECTION: Utiliser uniquement les données récentes (jusqu'à 12 derniers mois)
            train_data = monthly_sales['Qty'].tail(min(12, len(monthly_sales)))
            
            # Paramètre alpha plus petit pour moins de réactivité aux dernières observations
            alpha = 0.2  # Valeur fixe au lieu de l'optimisation qui peut être instable
            model = SimpleExpSmoothing(train_data).fit(smoothing_level=alpha, optimized=False)
            
            es_forecast = model.forecast(steps=months_to_forecast).tolist()
            
            # Éviter les valeurs négatives et les valeurs aberrantes
            # Limiter les prévisions à 150% de la moyenne historique pour éviter les surestimations
            avg_historical = train_data.mean()
            max_forecast = avg_historical * 1.5
            es_forecast = [max(0, min(x, max_forecast)) for x in es_forecast]
        
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
        st.error(f"Erreur de prévision par lissage exponentiel : {e}")
        return None

# Méthode 3 : Régression linéaire simple
def linear_regression_forecast(group, last_date, months_to_forecast=12):
    try:
        # CORRECTION: Vérifier d'abord si le DataFrame est vide
        if group.empty:
            return None
            
        # Groupement par mois
        monthly_sales = group.groupby(pd.Grouper(key='Date', freq='M'))['Qty'].sum().reset_index()
        
        # Si pas assez de données, utiliser la moyenne
        if len(monthly_sales) < 3:  # Au moins 3 points pour une régression fiable
            avg_sales = monthly_sales['Qty'].mean()
            lr_forecast = [max(0, avg_sales)] * months_to_forecast
        else:
            # CORRECTION: Utiliser uniquement les données récentes (jusqu'à 12 derniers mois)
            recent_data = monthly_sales.tail(min(12, len(monthly_sales)))
            X = np.arange(len(recent_data)).reshape(-1, 1)
            y = recent_data['Qty'].values
            
            # Régression linéaire
            slope, intercept, r_value, p_value, std_err = stats.linregress(X.ravel(), y)
            
            # Si la tendance est négative ou trop accentuée, limiter son impact
            if slope < 0:
                # Si tendance négative, atténuer pour éviter des prévisions trop pessimistes
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
        st.error(f"Erreur de prévision par régression linéaire : {e}")
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
    
    # AJOUT: Élément de recherche pour un article spécifique
    article_search = st.text_input("Rechercher un article spécifique (optionnel)")
    
    if uploaded_file is not None:
        # Réinitialiser les calculs si un nouveau fichier est chargé
        if 'current_file' not in st.session_state or st.session_state.current_file != uploaded_file.name:
            st.session_state.forecast_calculated = False
            st.session_state.excel_data = None
            st.session_state.current_file = uploaded_file.name
            
        # Chargement et prétraitement des données
        df = load_and_preprocess_data(uploaded_file)
        
        if df is not None:
            # CORRECTION: Application du filtre des valeurs aberrantes APRÈS le chargement et SEULEMENT si demandé
            if filter_outliers:
                # Créer une copie avant filtrage pour comparaison
                df_before = df.copy()
                
                # Filtrer les valeurs aberrantes par article et groupe client
                for (item, group), subset in df.groupby(['Item', 'Customer group']):
                    if len(subset) > 10:  # Assez de données pour calculer des statistiques fiables
                        mean_qty = subset['Qty'].mean()
                        std_qty = subset['Qty'].std()
                        if std_qty > 0:  # Éviter la division par zéro
                            # Récupérer les indices des lignes à conserver
                            valid_idx = subset[abs(subset['Qty'] - mean_qty) <= 3 * std_qty].index
                            # Filtrer le DataFrame principal
                            df = df[df.index.isin(valid_idx) | ~((df['Item'] == item) & (df['Customer group'] == group))]
                
                # Afficher un résumé du filtrage
                removed = len(df_before) - len(df)
                if removed > 0:
                    st.info(f"{removed} enregistrements identifiés comme aberrants ont été supprimés ({removed/len(df_before)*100:.1f}%)")
            
            # Trouver la dernière date dans le jeu de données
            last_date = df['Date'].max()
            
            # AJOUT: Si recherche d'article spécifique, filtrer les données
            if article_search:
                df_search = df[df['Item'].str.contains(article_search, case=False)]
                if not df_search.empty:
                    st.success(f"Article trouvé: {len(df_search)} enregistrements correspondant à '{article_search}'")
                    st.write("Aperçu des données pour cet article:")
                    st.dataframe(df_search)
                    
                    # Afficher les ventes mensuelles pour cet article
                    st.write("Ventes mensuelles pour cet article:")
                    monthly_data = df_search.groupby([pd.Grouper(key='Date', freq='M'), 'Customer group'])['Qty'].sum().reset_index()
                    monthly_data['Date'] = monthly_data['Date'].dt.strftime('%m/%Y')
                    st.dataframe(monthly_data)
                else:
                    st.warning(f"Aucun article correspondant à '{article_search}' trouvé dans les données.")
            
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
            
            # Si les calculs n'ont pas encore été effectués, les lancer
            if not st.session_state.forecast_calculated or st.button("Recalculer les prévisions"):
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
                        
                        # CORRECTION: Ajouter un debug spécifique pour l'article recherché
                        is_target_item = article_search and item.lower() == article_search.lower()
                        
                        # Vérifier s'il y a des données pour cette combinaison
                        if not subset.empty:
                            # Ajout d'un état d'avancement plus détaillé
                            with st.expander(f"Détails pour {item} (Groupe: {group})", expanded=is_target_item):
                                st.write(f"Nombre d'enregistrements: {len(subset)}")
                                # Afficher les ventes mensuelles pour inspection
                                monthly_data = subset.groupby(pd.Grouper(key='Date', freq='M'))['Qty'].sum().reset_index()
                                monthly_data['Date'] = monthly_data['Date'].dt.strftime('%m/%Y')
                                monthly_avg = monthly_data['Qty'].mean()
                                st.write(f"Moyenne mensuelle: {monthly_avg:.2f} unités")
                                st.dataframe(monthly_data)
                                
                                # SI c'est l'article recherché, afficher plus de détails
                                if is_target_item:
                                    st.write("Analysons en détail cet article spécifique")
                                    # Afficher tous les enregistrements bruts
                                    st.write("Données brutes:")
                                    st.dataframe(subset[['Date', 'Qty']].sort_values('Date'))
                            
                            # Calcul des prévisions avec le nombre de mois spécifié
                            ma_forecast = moving_average_forecast(subset, last_date, months_to_forecast)
                            es_forecast = exponential_smoothing_forecast(subset, last_date, months_to_forecast)
                            lr_forecast = linear_regression_forecast(subset, last_date, months_to_forecast)
                            
                            # Stockage des résultats
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
                    # Si un article est recherché, l'utiliser comme exemple
                    if article_search and article_search in items:
                        sample_item = article_search
                    else:
                        sample_item = items[0]
                    
                    sample_group = customer_groups[0]
                    sample_key = f"{sample_item} - {sample_group}"
                    
                    with st.expander("Exemple de prévisions", expanded=True):
                        if sample_key in ma_results:
                            st.write(f"Prévisions pour {sample_item} (Groupe: {sample_group}):")
                            st.write("Moyenne mobile:")
                            st.dataframe(ma_results[sample_key])
                
                # Générer le fichier Excel en mémoire
                excel_buffer = io.BytesIO()
                
                # Utiliser openpyxl si disponible
                try:
                    import openpyxl
                    excel_engine = 'openpyxl'
                except ImportError:
                    excel_engine = None
                
                with pd.ExcelWriter(excel_buffer, engine=excel_engine) as writer:
                    # Onglet Moyenne Mobile
                    if ma_results:
                        ma_combined = pd.concat([
                            pd.DataFrame({
                                'Item': [k.split(' - ')[0]] * len(v),
                                'Customer group': [k.split(' - ')[1]] * len(v),
                                **v
                            }) 
                            for k, v in ma_results.items()
                        ])
                        ma_combined.to_excel(writer, sheet_name='Moyenne_Mobile', index=False)
                    
                    # Onglet Lissage Exponentiel
                    if es_results:
                        es_combined = pd.concat([
                            pd.DataFrame({
                                'Item': [k.split(' - ')[0]] * len(v),
                                'Customer group': [k.split(' - ')[1]] * len(v),
                                **v
                            }) 
                            for k, v in es_results.items()
                        ])
                        es_combined.to_excel(writer, sheet_name='Lissage_Exponentiel', index=False)
                    
                    # Onglet Régression Linéaire
                    if lr_results:
                        lr_combined = pd.concat([
                            pd.DataFrame({
                                'Item': [k.split(' - ')[0]] * len(v),
                                'Customer group': [k.split(' - ')[1]] * len(v),
                                **v
                            }) 
                            for k, v in lr_results.items()
                        ])
                        lr_combined.to_excel(writer, sheet_name='Regression_Lineaire', index=False)
                    
                    # Onglet de comparaison des méthodes
                    if ma_results and es_results and lr_results:
                        # Prendre un échantillon pour comparer
                        sample_comparisons = []
                        
                        # Si un article est recherché, l'utiliser en priorité
                        if article_search:
                            target_keys = [k for k in ma_results.keys() if article_search.lower() in k.lower()]
                            keys_to_process = target_keys[:5] + [k for k in list(ma_results.keys())[:10] if k not in target_keys]
                        else:
                            keys_to_process = list(ma_results.keys())[:10]
                            
                        for key in keys_to_process:
                            key_parts = key.split(' - ', 1)
                            if len(key_parts) != 2:
                                continue
                                
                            item, group = key_parts
                            if key in ma_results and key in es_results and key in lr_results:
                                try:
                                    for month in range(min(12, len(ma_results[key]))):
                                        date_str = ma_results[key]['Date'].iloc[month]
                                        ma_val = ma_results[key]['Qty_Forecast_MA'].iloc[month]
                                        es_val = es_results[key]['Qty_Forecast_ES'].iloc[month]
                                        lr_val = lr_results[key]['Qty_Forecast_LR'].iloc[month]
                                        
                                        sample_comparisons.append({
                                            'Item': item,
                                            'Customer group': group,
                                            'Date': date_str,
                                            'Moyenne Mobile': ma_val,
                                            'Lissage Exponentiel': es_val,
                                            'Régression Linéaire': lr_val
                                        })
                                except (IndexError, KeyError) as e:
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