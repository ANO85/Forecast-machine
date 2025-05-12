import streamlit as st
import pandas as pd
import os
import plotly.express as px
import plotly.graph_objs as go
from file_processing import load_sales_data, generate_forecasts, save_forecasts_to_excel

def plot_sales_forecast(original_data, forecasts):
    """
    Crée un graphique interactif des ventes historiques et prévisions
    
    Args:
        original_data (pd.DataFrame): Données historiques
        forecasts (pd.DataFrame): Données de prévision
    
    Returns:
        plotly figure
    """
    # Préparer les données historiques
    historical_data = original_data.groupby(pd.Grouper(key='Date', freq='M'))['Qty'].sum().reset_index()
    
    # Créer des traces pour les données historiques et les prévisions
    fig = go.Figure()
    
    # Données historiques
    fig.add_trace(go.Scatter(
        x=historical_data['Date'], 
        y=historical_data['Qty'], 
        mode='lines+markers', 
        name='Ventes Historiques',
        line=dict(color='blue')
    ))
    
    # Prévisions
    fig.add_trace(go.Scatter(
        x=forecasts['Date'], 
        y=forecasts['Qty'], 
        mode='lines+markers', 
        name='Prévisions',
        line=dict(color='red', dash='dot')
    ))
    
    # Mise en page
    fig.update_layout(
        title='Ventes Historiques et Prévisions',
        xaxis_title='Date',
        yaxis_title='Quantité',
        hovermode='x unified'
    )
    
    return fig

def main():
    st.title('Prévisions de Ventes Avancées')
    
    # Section de téléchargement de fichier
    st.header('1. Télécharger les Données Historiques')
    uploaded_file = st.file_uploader(
        "Choisissez un fichier Excel de ventes historiques", 
        type=['xlsx', 'xls']
    )
    
    if uploaded_file is not None:
        try:
            # Charger les données
            df = load_sales_data(uploaded_file)
            st.success('Données chargées avec succès!')
            
            # Afficher un aperçu des données
            st.write('Aperçu des données:')
            st.dataframe(df.head())
            
            # Sélection des groupes de clients et articles
            st.header('2. Sélection des Prévisions')
            
            # Récupérer les groupes uniques
            unique_customers = df['Customer group'].unique()
            unique_items = df['Item'].unique()
            
            # Sélection multiple de clients et articles
            selected_customers = st.multiselect(
                'Sélectionnez les groupes de clients', 
                unique_customers, 
                default=unique_customers[:3]  # Sélection par défaut des 3 premiers
            )
            
            selected_items = st.multiselect(
                'Sélectionnez les articles', 
                unique_items, 
                default=unique_items[:3]  # Sélection par défaut des 3 premiers
            )
            
            # Filtrer les données
            filtered_df = df[
                (df['Customer group'].isin(selected_customers)) & 
                (df['Item'].isin(selected_items))
            ]
            
            # Méthodes de prévision
            st.header('3. Choix des Méthodes de Prévision')
            
            # Option de méthode globale
            global_method = st.selectbox(
                'Méthode de prévision par défaut',
                ['Automatique', 'Moyenne Mobile', 'Lissage Exponentiel', 'SARIMA']
            )
            
            # Options personnalisées par article
            st.subheader('Personnalisation par Article')
            custom_item_methods = {}
            
            for item in selected_items:
                custom_item_methods[item] = st.selectbox(
                    f'Méthode pour {item}', 
                    ['Défaut', 'Moyenne Mobile', 'Lissage Exponentiel', 'SARIMA'],
                    key=f'item_method_{item}'
                )
            
            # Bouton de génération des prévisions
            if st.button('Générer les Prévisions'):
                with st.spinner('Calcul des prévisions en cours...'):
                    # Générer les prévisions
                    forecasts = generate_forecasts(filtered_df)
                    
                    # Visualisation des prévisions
                    st.header('Visualisation des Prévisions')
                    
                    # Sélection du type de prévision à afficher
                    forecast_method = st.selectbox(
                        'Sélectionnez la méthode de prévision à visualiser',
                        ['Moyenne Mobile', 'Lissage Exponentiel', 'SARIMA']
                    )
                    
                    # Carte des méthodes
                    method_map = {
                        'Moyenne Mobile': 'Moving Average',
                        'Lissage Exponentiel': 'Exponential Smoothing',
                        'SARIMA': 'SARIMA'
                    }
                    
                    # Préparer le DataFrame de prévision sélectionné
                    selected_forecast = forecasts[method_map[forecast_method]]
                    
                    # Graphique des prévisions
                    fig = plot_sales_forecast(filtered_df, selected_forecast)
                    st.plotly_chart(fig)
                    
                    # Sauvegarder les prévisions
                    output_filename = 'previsions_ventes.xlsx'
                    save_forecasts_to_excel(forecasts, output_filename)
                    
                    # Téléchargement du fichier
                    with open(output_filename, 'rb') as f:
                        st.download_button(
                            label='Télécharger les Prévisions',
                            data=f,
                            file_name=output_filename,
                            mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
                        )
                    
                    # Nettoyer le fichier temporaire
                    os.remove(output_filename)
        
        except Exception as e:
            st.error(f'Erreur lors du traitement: {str(e)}')

if __name__ == '__main__':
    main()