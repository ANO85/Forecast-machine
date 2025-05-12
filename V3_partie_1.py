import streamlit as st
import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import io

st.set_page_config(page_title="Prévision Ventes - Étape 1", layout="wide")
st.title("📊 Étape 1 – Génération des prévisions initiales")

# === Upload du fichier source ===
uploaded_file = st.file_uploader("Chargez le fichier de ventes (Excel)", type=["xlsx"])

if uploaded_file is not None:
    try:
        df = pd.read_excel(uploaded_file, sheet_name='DEMANDPLANNINGSOITEMDETAILYTDR')

        # Vérification de colonnes nécessaires
        required_cols = {'Date', 'Item', 'Customer Group', 'Qty'}
        if not required_cols.issubset(df.columns):
            st.error(f"Le fichier doit contenir les colonnes suivantes : {', '.join(required_cols)}")
        else:
            # Vérification des doublons
            duplicated_rows = df.duplicated(subset=['Date', 'Item', 'Customer Group'])
            if duplicated_rows.any():
                st.warning("Des doublons ont été détectés sur (Date, Item, Customer Group). Ils vont être agrégés.")
                df = df.groupby(['Date', 'Item', 'Customer Group'], as_index=False).agg({'Qty': 'sum'})

            # **Suppression explicite des doublons après agrégation**
            df = df.drop_duplicates(subset=['Date', 'Item', 'Customer Group'])

            st.success("Fichier chargé et vérifié avec succès.")
            # Tu peux ensuite appeler ici le reste de ton pipeline de prévision (partie 2, etc.)
    
    except Exception as e:
        st.error(f"Erreur pendant le traitement : {e}")

# === Fonctions de prévision ===
def moving_average_forecast(series, window):
    return series.rolling(window=window).mean()

def exponential_smoothing_forecast(series):
    try:
        model = ExponentialSmoothing(series, trend="add", seasonal=None)
        model_fit = model.fit()
        return model_fit.forecast(12)
    except:
        return pd.Series([np.nan] * 12)

def calculate_mape(actual, predicted):
    actual, predicted = np.array(actual), np.array(predicted)
    min_len = min(len(actual), len(predicted))
    if min_len == 0:
        return np.nan
    actual, predicted = actual[:min_len], predicted[:min_len]
    mask = actual != 0
    if not np.any(mask):
        return np.nan
    return np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100

# === Traitement après upload ===
if uploaded_file is not None:
    try:
        df = pd.read_excel(uploaded_file, sheet_name="DEMANDPLANNINGSOITEMDETAILYTDR")
        st.success("Fichier chargé avec succès.")
        
        # Vérification des colonnes nécessaires
        expected_cols = ['Date', 'Item', 'Customer Group', 'Qty']
        if not all(col in df.columns for col in expected_cols):
            st.error(f"Le fichier doit contenir les colonnes suivantes : {expected_cols}")
        else:
            df['Date'] = pd.to_datetime(df['Date'])
            forecast_6m, forecast_12m, forecast_exp, evaluation = [], [], [], []

            group_cols = ['Item', 'Customer Group']
            grouped = df.groupby(group_cols)

            for keys, group in grouped:
                group_sorted = group.sort_values('Date')
                series = group_sorted.set_index('Date')['Qty'].asfreq('MS').fillna(0)

                # **Suppression explicite des doublons après `set_index` pour éviter l'erreur**
                series = series[~series.index.duplicated(keep='first')]  # Supprime les doublons dans l'index

                # Prévisions des 12 mois à venir
                ma_6 = moving_average_forecast(series, 6).dropna().iloc[-1] if len(series) >= 6 else np.nan
                ma_12 = moving_average_forecast(series, 12).dropna().iloc[-1] if len(series) >= 12 else np.nan
                exp = exponential_smoothing_forecast(series)

                future_dates = pd.date_range(series.index[-1] + pd.DateOffset(months=1), periods=12, freq='MS')

                for i, date in enumerate(future_dates):
                    forecast_6m.append([*keys, date, ma_6])
                    forecast_12m.append([*keys, date, ma_12])
                    forecast_exp.append([*keys, date, exp[i] if isinstance(exp, pd.Series) else np.nan])

                # Évaluation sur les 3 derniers mois (si possible)
                if len(series) >= 15:
                    last_12 = series.iloc[-15:-3]
                    true_vals = series.iloc[-3:]
                    ma6_vals = moving_average_forecast(last_12, 6).dropna().iloc[-3:]
                    ma12_vals = moving_average_forecast(last_12, 12).dropna().iloc[-3:]
                    exp_vals = exponential_smoothing_forecast(last_12)

                    mape_6 = calculate_mape(true_vals, ma6_vals)
                    mape_12 = calculate_mape(true_vals, ma12_vals)
                    mape_exp = calculate_mape(true_vals, exp_vals[:3])

                    evaluation.append([*keys, mape_6, mape_12, mape_exp])

            # Création des DataFrames
            df_ma6 = pd.DataFrame(forecast_6m, columns=['Item', 'Customer Group', 'Date', 'Forecast_Qty'])
            df_ma12 = pd.DataFrame(forecast_12m, columns=['Item', 'Customer Group', 'Date', 'Forecast_Qty'])
            df_exp = pd.DataFrame(forecast_exp, columns=['Item', 'Customer Group', 'Date', 'Forecast_Qty'])
            df_eval = pd.DataFrame(evaluation, columns=['Item', 'Customer Group', 'MAPE_MA6', 'MAPE_MA12', 'MAPE_EXP'])

            # Sélection de la méthode recommandée
            df_eval['Recommended_Method'] = df_eval[['MAPE_MA6', 'MAPE_MA12', 'MAPE_EXP']].idxmin(axis=1).str.replace('MAPE_', '')

            # Préparation des feuilles Méthodes recommandées
            df_article_method = df_eval.groupby('Item')['Recommended_Method'].agg(lambda x: x.value_counts().idxmax()).reset_index()
            df_article_method.columns = ['Item', 'Recommended_Method']

            df_client_method = df_eval.groupby('Customer Group')['Recommended_Method'].agg(lambda x: x.value_counts().idxmax()).reset_index()
            df_client_method.columns = ['Customer Group', 'Recommended_Method']

            # === Export Excel final ===
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                df_ma6.to_excel(writer, index=False, sheet_name='Forecast_MA6')
                df_exp.to_excel(writer, index=False, sheet_name='Forecast_EXP')
                df_ma12.to_excel(writer, index=False, sheet_name='Forecast_MA12')
                df_article_method.to_excel(writer, index=False, sheet_name='Méthodes_Articles')
                df_client_method.to_excel(writer, index=False, sheet_name='Méthodes_Clients')

                # Feuille modèle MANUEL vide
                template = df_ma6[['Item', 'Customer Group', 'Date']].drop_duplicates().copy()
                template['Forecast_Qty'] = ''
                template.to_excel(writer, index=False, sheet_name='MANUEL')

            st.success("✅ Traitement terminé. Téléchargez le fichier généré ci-dessous.")
            st.download_button(
                label="📥 Télécharger le fichier de prévisions",
                data=output.getvalue(),
                file_name='previsions_initiales.xlsx',
                mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
            )

    except Exception as e:
        st.error(f"Erreur pendant le traitement : {e}")
else:
    st.info("Veuillez uploader un fichier Excel pour démarrer.")


