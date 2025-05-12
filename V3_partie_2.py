import pandas as pd
import numpy as np
import datetime
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_absolute_percentage_error

# Fonction pour charger les données du fichier Excel, incluant les prévisions et les modifications manuelles
def load_modified_data(file_path):
    df = pd.read_excel(file_path, sheet_name='Forecast_Results')
    manual_forecast = pd.read_excel(file_path, sheet_name='MANUEL')  # Si l'utilisateur a fourni des prévisions manuelles
    return df, manual_forecast

# Fonction pour calculer les prévisions en fonction de la méthode sélectionnée
def calculate_forecast(df, manual_forecast, months=12):
    forecast_results = []

    today = datetime.datetime.today()
    forecast_start_date = today.replace(day=1)  # Mois en cours

    for product in df['Item'].unique():
        for customer in df['Customer Group'].unique():
            temp_df = df[(df['Item'] == product) & (df['Customer Group'] == customer)]
            temp_df = temp_df.set_index('Date').resample('MS').sum()  # Resampling à fréquence mensuelle
            temp_df = temp_df.fillna(0)

            if len(temp_df) < 6:
                continue  # Si la série est trop courte, on passe

            # Lecture de la méthode choisie par l'utilisateur (si elle est modifiée)
            selected_method = temp_df['Best_Forecast'].iloc[-1]  # Colonne indiquant la méthode recommandée (par exemple)

            if selected_method == "MANUEL":
                # Si l'utilisateur a entré des prévisions manuelles, on les utilise
                manual_data = manual_forecast[(manual_forecast['Item'] == product) & 
                                              (manual_forecast['Customer Group'] == customer)]
                forecast_values = manual_data['Qty'].values[:12]
            else:
                # Si la méthode est automatique (moyenne mobile, Holt-Winters, etc.)
                if selected_method == "Moyenne Mobile":
                    forecast_values = moving_average_forecast(temp_df, 6)
                elif selected_method == "Holt-Winters":
                    forecast_values = holt_winters_forecast(temp_df, 12)
                elif selected_method == "Moyenne Mobile 12":
                    forecast_values = custom_method_forecast(temp_df, 12)
                else:
                    # Si aucune méthode n'est sélectionnée, on utilise la moyenne mobile par défaut
                    forecast_values = moving_average_forecast(temp_df, 6)

            forecast_dates = pd.date_range(start=forecast_start_date, periods=12, freq='MS')
            forecast_df = pd.DataFrame({
                'Date': forecast_dates.strftime('%d/%m/%Y'),
                'Item': product,
                'Customer Group': customer,
                'Forecast': forecast_values
            })
            forecast_results.append(forecast_df)

    # Retourner les prévisions recalculées avec la méthode sélectionnée
    return pd.concat(forecast_results)

# Fonction pour sauvegarder le fichier Excel avec les prévisions finales recalculées
def save_final_forecast(df, file_path):
    with pd.ExcelWriter(file_path, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
        df.to_excel(writer, sheet_name='Forecast_Results_Final', index=False)
        print(f"✅ Export final terminé dans l'onglet 'Forecast_Results_Final'.")

# Exemple d'utilisation
def generate_final_forecast(file_path):
    # Charger les données modifiées par l'utilisateur
    df, manual_forecast = load_modified_data(file_path)
    
    # Recalculer les prévisions en fonction de la méthode sélectionnée et des prévisions manuelles
    final_forecast = calculate_forecast(df, manual_forecast)
    
    # Sauvegarder le fichier avec les prévisions finales recalculées
    save_final_forecast(final_forecast, file_path)

# Exemple d'appel de la fonction (fichier Excel déjà téléchargé et modifié par l'utilisateur)
file_path = "chemin_vers_fichier_modifie.xlsx"  # Remplacer par le chemin réel du fichier modifié par l'utilisateur
generate_final_forecast(file_path)
