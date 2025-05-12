import pandas as pd
import numpy as np
import datetime
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_absolute_percentage_error

# Fonction pour charger les données depuis un fichier Excel
def load_data(file_path):
    df = pd.read_excel(file_path, sheet_name='DEMANDPLANNINGSOITEMDETAILYTDR')
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
    df['Qty'] = df['Qty'].astype(float)
    return df

# Fonction pour calculer les prévisions avec une moyenne mobile sur 6 mois
def moving_average_forecast(df, months=6):
    df['Moving_Avg'] = df['Qty'].rolling(window=months).mean()
    forecast = np.repeat(df['Moving_Avg'].iloc[-1], 12)  # Prévision sur 12 mois
    return forecast

# Fonction pour calculer les prévisions avec lissage exponentiel de Holt-Winters
def holt_winters_forecast(df, months=12):
    model = ExponentialSmoothing(df['Qty'], trend='add', seasonal='add', seasonal_periods=12)
    hw_fit = model.fit()
    forecast = hw_fit.forecast(12).clip(lower=0)
    return forecast

# Fonction pour calculer les prévisions sur la base d'une autre méthode (par exemple, moyenne mobile sur 12 mois)
def custom_method_forecast(df, months=12):
    df['Custom_MA'] = df['Qty'].rolling(window=months).mean()
    forecast = np.repeat(df['Custom_MA'].iloc[-1], 12)
    return forecast

# Fonction pour évaluer la performance (MAPE) des prévisions
def evaluate_forecast(real_values, forecast_values):
    return mean_absolute_percentage_error(real_values, forecast_values)

# Fonction pour calculer et générer le fichier Excel de prévisions
def generate_forecast(df, file_path):
    results = []
    
    today = datetime.datetime.today()
    forecast_start_date = today.replace(day=1)  # Mois en cours
    
    for product in df['Item'].unique():
        for customer in df['Customer Group'].unique():
            temp_df = df[(df['Item'] == product) & (df['Customer Group'] == customer)]
            temp_df = temp_df.set_index('Date').resample('MS').sum()  # Resampling à fréquence mensuelle
            temp_df = temp_df.fillna(0)

            if len(temp_df) < 6:
                continue  # Si la série est trop courte, on passe

            # Prévisions avec la méthode de la moyenne mobile (6 mois)
            forecast_ma = moving_average_forecast(temp_df, 6)
            real_qty = temp_df['Qty'][-6:]  # Derniers 6 mois pour la comparaison

            mape_ma = evaluate_forecast(real_qty, forecast_ma[:6])  # MAPE pour les 6 derniers mois

            # Prévisions avec lissage exponentiel
            forecast_hw = holt_winters_forecast(temp_df, 12)
            mape_hw = evaluate_forecast(real_qty, forecast_hw[:6])

            # Prévisions avec la méthode personnalisée (moyenne mobile sur 12 mois)
            forecast_custom = custom_method_forecast(temp_df, 12)
            mape_custom = evaluate_forecast(real_qty, forecast_custom[:6])

            # Sélection de la meilleure méthode en fonction de la MAPE
            best_forecast_qty = None
            if mape_ma < mape_hw and mape_ma < mape_custom:
                best_forecast_qty = forecast_ma
            elif mape_hw < mape_ma and mape_hw < mape_custom:
                best_forecast_qty = forecast_hw
            else:
                best_forecast_qty = forecast_custom

            # Stockage des résultats pour chaque combinaison Product / Customer
            forecast_dates = pd.date_range(start=forecast_start_date, periods=12, freq='MS')
            customer_forecast = pd.DataFrame({
                'Date': forecast_dates.strftime('%d/%m/%Y'),
                'Item': product,
                'Customer Group': customer,
                'Moving_Avg_Forecast': forecast_ma,
                'Holt_Winters_Forecast': forecast_hw,
                'Custom_Forecast': forecast_custom,
                'Best_Forecast': best_forecast_qty
            })
            results.append(customer_forecast)

    # Exporter les résultats dans un fichier Excel avec plusieurs onglets
    if results:
        final_forecast_df = pd.concat(results)
        with pd.ExcelWriter(file_path, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
            final_forecast_df.to_excel(writer, sheet_name='Forecast_Results', index=False)
            print(f"✅ Export terminé dans l'onglet 'Forecast_Results'.")

# Exemple d'utilisation
file_path = "chemin_vers_fichier.xlsx"  # Remplace par le chemin réel du fichier
df = load_data(file_path)
generate_forecast(df, file_path)
