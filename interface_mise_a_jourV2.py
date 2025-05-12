import streamlit as st
import pandas as pd
from io import BytesIO
import datetime
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_absolute_percentage_error
import warnings

warnings.filterwarnings('ignore')

st.title("Prévision Catalogue Produits")

# 1. Upload du fichier Excel
uploaded_file = st.file_uploader("Uploader le fichier Excel source", type=["xlsx"])

if uploaded_file is not None:
    df = pd.read_excel(uploaded_file, sheet_name='DEMANDPLANNINGSOITEMDETAILYTDR')
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
    df['Qty'] = df['Qty'].astype(float)

    run_forecast = st.button("Lancer les prévisions")

    if run_forecast:
        placeholder = st.empty()  # Pour afficher les derniers logs
        today = datetime.datetime.today()
        forecast_start_date = today.replace(day=1)
        results = []

        for product in df['Item'].unique():
            for customer in df['Customer Group'].unique():
                temp_df = df[(df['Item'] == product) & (df['Customer Group'] == customer)]
                temp_df = temp_df.set_index('Date').resample('MS').sum().fillna(0)

                if len(temp_df) < 6:
                    continue

                try:
                    temp_df['Moving_Avg'] = temp_df['Qty'].rolling(window=6).mean()
                    forecast_qty_ma = np.repeat(temp_df['Moving_Avg'].iloc[-1], 12)
                    real_qty = temp_df['Qty'][-6:]
                    mape_ma = mean_absolute_percentage_error(real_qty, forecast_qty_ma[:6])
                except:
                    forecast_qty_ma = None
                    mape_ma = None

                try:
                    model = ExponentialSmoothing(temp_df['Qty'], trend='add', seasonal='add', seasonal_periods=12)
                    hw_fit = model.fit()
                    forecast_qty_hw = hw_fit.forecast(12).clip(lower=0)
                    mape_hw = mean_absolute_percentage_error(real_qty, forecast_qty_hw[:6])
                except:
                    forecast_qty_hw = None
                    mape_hw = None

                # Choix du meilleur modèle
                if mape_ma is not None and (mape_hw is None or mape_ma < mape_hw):
                    best_forecast_qty = forecast_qty_ma
                    best_model = "Moyenne Mobile"
                elif mape_hw is not None:
                    best_forecast_qty = forecast_qty_hw
                    best_model = "Lissage Exponentiel"
                else:
                    continue

                forecast_dates = pd.date_range(start=forecast_start_date, periods=12, freq='MS')
                customer_forecast = pd.DataFrame({
                    'Date': forecast_dates.strftime('%d/%m/%Y'),
                    'Item': product,
                    'Customer Group': customer,
                    'Forecasted Qty': best_forecast_qty
                })
                results.append(customer_forecast)

                # Affiche seulement la dernière mise à jour
                placeholder.text(f"{product} - {customer}: {best_model} sélectionné")

        if results:
            final_df = pd.concat(results)

            # Export Excel dans un buffer mémoire
            output = BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                final_df.to_excel(writer, sheet_name='Forecast_Fallback', index=False)
            output.seek(0)

            # ✅ Bouton de téléchargement
            st.download_button(
                label="📥 Télécharger le fichier avec prévisions",
                data=output,
                file_name="previsions_forecast.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
            placeholder.success("✅ Prévisions terminées avec succès.")

        else:
            placeholder.warning("⚠️ Aucun forecast n'a été généré.")
