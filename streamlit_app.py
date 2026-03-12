import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import statsmodels.api as sm
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression

import matplotlib.pyplot as plt

def main():
    st.set_page_config(page_title="mPredictive Dashboard", layout="wide")
    st.title("📊 mPredictive: Diagnóstico y Modelación")

    # --- 1. CARGA DE ARCHIVO ---
    st.sidebar.header("1. Configuración de Datos")
    uploaded_file = st.sidebar.file_uploader("Sube tu dataset (CSV)", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        
        # Limpieza y Formateo
        df.columns = df.columns.str.strip()
        for col in df.select_dtypes(include=['object']).columns:
            df[col] = df[col].str.strip()
        
        # Convertir fecha y extraer año
        df['MonthReport'] = pd.to_datetime(df['MonthReport'], dayfirst=True, errors='coerce')
        df['Year'] = df['MonthReport'].dt.year

        st.sidebar.success("✅ Datos cargados")

        # --- 2. FILTROS ---
        marcas_base = sorted(df['BrandStd'].unique())
        opciones_marca = ["Total Categoría"] + marcas_base
        selected_brand = st.sidebar.selectbox("Selecciona la Marca:", opciones_marca)

        if selected_brand == "Total Categoría":
            kpis_disponibles = sorted(df['KPI'].unique())
        else:
            kpis_disponibles = sorted(df[df['BrandStd'] == selected_brand]['KPI'].unique())
        
        selected_kpi = st.sidebar.selectbox("Selecciona el KPI:", kpis_disponibles)

        # Preparar DataFrames
        if selected_brand == "Total Categoría":
            df_model = df[df['KPI'] == selected_kpi].groupby(['MonthReport', 'Year']).agg({
                'Investment': 'sum', 'Value': 'mean'
            }).reset_index()
        else:
            df_model = df[(df['BrandStd'] == selected_brand) & (df['KPI'] == selected_kpi)].copy()

        df_bench = df[df['KPI'] == selected_kpi]

        # --- 3. INTERFAZ DE PESTAÑAS ---
        tab1, tab2, tab3 = st.tabs(["📈 Análisis Visual", "🎯 Modelo Predictivo", "📋 Validación de Datos"])

        with tab1:
            st.subheader(f"Contexto de la Categoría: {selected_kpi}")
            
            c1, c2 = st.columns(2)
            
            with c1:
                st.markdown("**Promedio KPI por Año y Marca**")
                avg_kpi_year = df_bench.groupby(['Year', 'BrandStd'])['Value'].mean().reset_index()
                fig_bar = px.bar(avg_kpi_year, x='Year', y='Value', color='BrandStd', 
                                barmode='group', text_auto='.3f', template="plotly_white")
                st.plotly_chart(fig_bar, use_container_width=True)

            with c2:
                st.markdown("**Evolución del Share Of Investment (SOI)**")
                # Cálculo de porcentajes para las etiquetas
                soi_data = df.groupby(['Year', 'BrandStd'])['Investment'].sum().reset_index()
                soi_data['Total_Year'] = soi_data.groupby('Year')['Investment'].transform('sum')
                soi_data['Percentage'] = (soi_data['Investment'] / soi_data['Total_Year']) * 100
                
                # Gráfica con etiquetas de texto
                fig_soi = px.bar(soi_data, x='Year', y='Investment', color='BrandStd',
                                 text=soi_data['Percentage'].apply(lambda x: f'{x:.1f}%'),
                                 labels={'Investment': 'Share %', 'Year': 'Año'},
                                 template="plotly_white")
                
                fig_soi.update_layout(barnorm='percent')
                fig_soi.update_traces(textposition='inside') # Poner etiquetas dentro de las barras
                st.plotly_chart(fig_soi, use_container_width=True)

            st.divider()
            st.subheader(f"Relación Inversión vs KPI ({selected_brand})")
            
            if not df_model.empty and len(df_model) > 2:
                # --- AJUSTE: MODELO LOGARÍTMICO ---
                # Filtrar inversiones > 0 para evitar errores matemáticos con logaritmos
                df_log = df_model[df_model['Investment'] > 0].copy()
                
                # Calcular la regresión logarítmica: y = a + b*ln(x)
                X_log = np.log(df_log[['Investment']]).values
                y = df_log['Value'].values
                log_reg = LinearRegression().fit(X_log, y)
                
                # Generar puntos para la curva de tendencia
                x_range = np.linspace(df_log['Investment'].min(), df_log['Investment'].max(), 100)
                y_pred = log_reg.predict(np.log(x_range.reshape(-1, 1)))
                
                # Crear la gráfica base de dispersión
                fig_scatter = px.scatter(df_log, x='Investment', y='Value', 
                                        hover_data=['MonthReport'], template="plotly_white",
                                        labels={'Investment': 'Inversión ($)', 'Value': f'Valor de {selected_kpi}'})
                
                # Añadir la línea del modelo logarítmico
                fig_scatter.add_trace(go.Scatter(x=x_range, y=y_pred, name='Modelo Logarítmico',
                                                line=dict(color='red', width=3)))
                
                st.plotly_chart(fig_scatter, use_container_width=True)
                st.caption(f"Nota: La línea roja representa un ajuste logarítmico (y = a + b * ln(Inversión)), ideal para observar la saturación de medios.")
            else:
                st.warning("No hay datos suficientes (inversión > 0) para generar el modelo logarítmico.")

        with tab2:

            st.header("🎯 Simulación de Impacto de Campaña")
            st.write("Configura los parámetros de tu próxima campaña para proyectar el Uplift esperado.")

            # --- INPUTS DE USUARIO ---
            col_input1, col_input2 = st.columns(2)

            with col_input1:
                campaign_investment = st.number_input(
                    "Inversión Total de Campaña ($)", 
                    min_value=0.0, 
                    value=1000000.0, 
                    step=10000.0,
                    format="%.2f"
                )

            with col_input2:
                campaign_duration = st.number_input(
                    "Duración (Meses)", 
                    min_value=1, 
                    max_value=12, 
                    value=1, 
                    step=1
                )
                monthly_investment = campaign_investment / campaign_duration
                st.info(f"💡 Simulación: **${monthly_investment:,.2f} mensuales**")

            # Parámetros fijos del modelo
            kpi_model = selected_kpi
            brand_model = selected_brand 
            simulated_investment_value = monthly_investment
            cal_Indx = 107
            beta_param = 5*10**-11 

            def s_curve(df, investment_column, min_alpha, max_alpha, beta, step=0.01):
                    """
                    Aplica la transformación de curva S a una columna de inversión para un rango de valores alpha.

                    Args:
                        df (pd.DataFrame): El DataFrame de entrada.
                        investment_column (str): El nombre de la columna de inversión.
                        min_alpha (float): El valor mínimo para el parámetro alpha.
                        max_alpha (float): El valor máximo para el parámetro alpha.
                        beta (float): El parámetro beta de la curva S.
                        step (float): El incremento para alpha en cada iteración.

                    Returns:
                        pd.DataFrame: Un nuevo DataFrame con columnas de curva S para cada alpha.
                    """
                    df_copy = df.copy()

                    # Calcular el promedio y la desviación estándar para la normalización una sola vez
                    mean_investment = df_copy[investment_column].mean()
                    std_investment = df_copy[investment_column].std()
                    max_investment = df_copy[investment_column].max()

                    # Normalizar la columna 'Investment' por Promedio y Desv. Std
                    # ================>
                    #normalized_investment = 100*(df_copy[investment_column] / (mean_investment + std_investment))

                    # Normalizar la columna 'Investment' por Máximo
                    # ================>
                    normalized_investment = 100*(df_copy[investment_column] / max_investment)

                    current_alpha = min_alpha
                    while current_alpha <= max_alpha:
                        # Crear un nombre de columna único para cada valor de alpha
                        s_curve_col_name = f'S_Curve_Investment_alpha_{current_alpha:.2f}'

                        # Aplicar la transformación de la curva S
                        df_copy[s_curve_col_name] = beta ** (current_alpha ** normalized_investment)

                        current_alpha += step

                    return df_copy                 
                    # Aplicación de S_Curve al Set de Datos

            if st.button("🚀 Calcular Impacto de Campaña"):
                
                # 1. Filtrado de datos
                if brand_model == 'Total Categoría':
                    ds_model = df[df['KPI'] == kpi_model].copy()
                else: 
                    ds_model = df[(df['BrandStd'] == brand_model) & (df['KPI'] == kpi_model)].copy()

                if ds_model.empty:
                    st.error("Error: No hay suficientes datos para los filtros seleccionados.")
                else:
                    # 2. Procesamiento y Ponderación
                    ds_model['YearReport'] = ds_model['MonthReport'].dt.year
                    ponderacion_df = pd.DataFrame({
                        'YearReport': [2022, 2023, 2024, 2025],
                        'ponderacion': [1.49, 1.49, 3.22, 2.76]
                    })
                    
                    ds_model_r = pd.merge(ds_model, ponderacion_df, on='YearReport', how='left')
                    ds_model_r['Inv_real'] = ds_model_r['Investment'] * ds_model_r['ponderacion']
                    
                    # 3. Estadísticas y Transformación S-Curve
                    std_kpi = ds_model['Value'].std()
                    year_stats_k_b = ds_model.groupby(['BrandStd'])['Value'].agg(['mean','std']).reset_index()
                    
                    # Evitar error si Coca-Cola no está en el set filtrado
                    try:
                        brand_std_val = year_stats_k_b[year_stats_k_b['BrandStd'] == 'Coca-Cola']['std'].iloc[0]
                        fact_trop = brand_std_val / std_kpi
                    except IndexError:
                        fact_trop = 1.0 # Valor por defecto si no se encuentra la marca benchmark

                    # Reutilizamos tu función lógica de s_curve
                    max_spend = ds_model_r['Inv_real'].max()
                    ds_model_transformed = s_curve(ds_model_r, 'Inv_real', 0.5, 0.9, beta_param)

                    # 4. Regresiones
                    s_curve_cols = [col for col in ds_model_transformed.columns if col.startswith('S_Curve_Investment_alpha_')]
                    regression_results = []

                    for col in s_curve_cols:
                        correlation = ds_model_transformed[col].corr(ds_model_transformed['Value'])
                        if correlation > 0:
                            y_reg = ds_model_transformed['Value']
                            X_reg = sm.add_constant(ds_model_transformed[col])
                            model = sm.OLS(y_reg, X_reg).fit()
                            
                            regression_results.append({
                                'Variable Independiente': col,
                                'Coeficiente de Regresión': model.params.iloc[1],
                                'Intervalo de Confianza Inferior': model.conf_int().iloc[1, 0],
                                'Attribution Puntual': model.params.iloc[1] * (beta_param**(float(col[-4:])**(100*simulated_investment_value/max_spend))) * fact_trop,
                                'Attribution Inferior': model.conf_int().iloc[1, 0] * (beta_param**(float(col[-4:])**(100*simulated_investment_value/max_spend))) * fact_trop,
                                'Brand Benchmark': brand_std_val if 'brand_std_val' in locals() else 0
                            })

                    regression_results_df = pd.DataFrame(regression_results)
                    
                    # 5. Selección de mejor Alpha e Indicadores
                    regression_results_df['Indx_i'] = 100 * (regression_results_df['Attribution Inferior'] / brand_std_val)
                    closest_idx = (regression_results_df['Indx_i'] - cal_Indx).abs().idxmin()
                    winner = regression_results_df.loc[closest_idx]
                    
                    # --- MOSTRAR MÉTRICAS ---
                    st.subheader("📊 Resultados de la Simulación")
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Attribution Esperada", f"{winner['Attribution Puntual']*100:.2f} PP")
                    m2.metric("Índice vs Benchmark", f"{winner['Indx_i']:.1f}", delta=f"{winner['Indx_i']-100:.1f}")
                    m3.metric("Alpha Seleccionado", winner['Variable Independiente'][-4:])

                    # --- GENERACIÓN DE GRÁFICA (Ajuste Matplotlib para Streamlit) ---
                    alpha_win = float(winner['Variable Independiente'][-4:])
                    
                    # Crear simulación de curva
                    sim_df = pd.DataFrame({'Idx': range(1, 71)})
                    sim_df['Media Spend'] = sim_df['Idx'] * (max_spend / 100)
                    inv_norm = 100 * sim_df['Media Spend'] / max_spend
                    curve_s = (beta_param**alpha_win**inv_norm) * fact_trop
                    
                    sim_df['Min'] = curve_s * winner['Intervalo de Confianza Inferior'] * 100
                    sim_df['Max'] = curve_s * winner['Coeficiente de Regresión'] * 100

                    # Configuración del gráfico
                    fig, ax = plt.subplots(figsize=(10, 5))
                    sns.lineplot(x=sim_df['Media Spend']/1e6, y=sim_df['Min'], ax=ax, label='Mínimo Esperado', color='orange')
                    sns.lineplot(x=sim_df['Media Spend']/1e6, y=sim_df['Max'], ax=ax, label='Máximo Esperado', color='green')
                    
                    # Benchmark line
                    ax.axhline(y=brand_std_val*100, color='black', linestyle='--', label='Benchmark Coca-Cola')
                    
                    # Puntos de la inversión actual
                    ax.scatter(simulated_investment_value/1e6, winner['Attribution Inferior']*100, color='orange', s=100, zorder=5)
                    ax.scatter(simulated_investment_value/1e6, winner['Attribution Puntual']*100, color='green', s=100, zorder=5)

                    ax.set_title(f'Curva de Respuesta mPredictive: {selected_kpi}')
                    ax.set_xlabel('Media Spend (Millones $)')
                    ax.set_ylabel('Impacto en KPI (PP)')
                    ax.legend()
                    
                    # MOSTRAR EN STREAMLIT
                    st.pyplot(fig)
                             
            else:
                st.error("Error: No hay suficientes datos históricos para este KPI/Marca para entrenar el modelo.")

        with tab3:
            st.subheader("Validación de Registros (Top 15)")
            st.dataframe(df_model.head(15).sort_values('MonthReport', ascending=False), use_container_width=True)

    else:
        st.info("👈 Por favor, carga el archivo CSV en la barra lateral.")

if __name__ == "__main__":
    main()  