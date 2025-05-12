import streamlit as st
import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from pmdarima import auto_arima
from io import BytesIO

def main():
    st.title("📈 Application de Prévisions de Vente")
    
    # ETAPE 1: Upload du fichier
    uploaded_file = st.file_uploader("Déposez votre fichier Excel (colonnes: Date, Item, Customer Group, Qty)", type="xlsx")
    
    if uploaded_file:
        df = pd.read_excel(uploaded_file, parse_dates=['Date'])
        df = df.sort_values(['Item', 'Customer Group', 'Date'])
        
        # ETAPE 2: Calcul des prévisions
        if st.button("Générer les prévisions"):
            with st.spinner('Calcul en cours...'):
                # Initialisation des DataFrames résultats
                ma_forecasts = []
                exp_forecasts = []
                sarima_forecasts = []
                items_reco = []
                clients_reco = []

                # Pour chaque couple produit-client
                groups = df.groupby(['Item', 'Customer Group'])
                for (item, client), group in groups:
                    # Préparation série temporelle
                    ts = group.set_index('Date')['Qty'].resample('MS').sum()
                    
                    # Moving Average (6 mois)
                    ma = ts.rolling(6).mean().iloc[-1]
                    ma_fcst = [ma] * 12
                    
                    # Lissage exponentiel
                    try:
                        model = ExponentialSmoothing(ts).fit()
                        exp_fcst = model.forecast(12).values
                    except:
                        exp_fcst = [np.nan] * 12
                    
                    # SARIMA
                    try:
                        model = auto_arima(ts, seasonal=True, m=12)
                        sarima_fcst = model.predict(n_periods=12)
                    except:
                        sarima_fcst = [np.nan] * 12
                    
                    # Stockage résultats
                    dates = pd.date_range(ts.index[-1] + pd.DateOffset(months=1), periods=12, freq='MS')
                    for i, date in enumerate(dates):
                        ma_forecasts.append([date, item, client, ma_fcst[i]])
                        exp_forecasts.append([date, item, client, exp_fcst[i]])
                        sarima_forecasts.append([date, item, client, sarima_fcst[i]])
                
                # Création DataFrames
                ma_df = pd.DataFrame(ma_forecasts, columns=['Date', 'Item', 'Customer Group', 'Qty'])
                exp_df = pd.DataFrame(exp_forecasts, columns=['Date', 'Item', 'Customer Group', 'Qty'])
                sarima_df = pd.DataFrame(sarima_forecasts, columns=['Date', 'Item', 'Customer Group', 'Qty'])
                
                # ETAPE 3: Génération fichier Excel
                output = BytesIO()
                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                    ma_df.to_excel(writer, sheet_name='Moving Average', index=False)
                    exp_df.to_excel(writer, sheet_name='Exponential Smoothing', index=False)
                    sarima_df.to_excel(writer, sheet_name='SARIMA', index=False)
                    pd.DataFrame(items_reco).to_excel(writer, sheet_name='Items Recommendation', index=False)
                    pd.DataFrame(clients_reco).to_excel(writer, sheet_name='Clients Recommendation', index=False)
                
                st.success('Calcul terminé!')
                st.download_button(
                    label="📥 Télécharger les prévisions",
                    data=output.getvalue(),
                    file_name='forecasts.xlsx',
                    mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
                )

if __name__ == "__main__":
    main()