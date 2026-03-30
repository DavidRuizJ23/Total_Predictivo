import streamlit as st
import requests
import pandas as pd
import numpy as np
import seaborn as sns
import statsmodels.api as sm
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression

import matplotlib.pyplot as plt

def get_github_content(url):
    try:
        response = requests.get(url)
        response.raise_for_status() # Lanza error si la URL está mal
        return response.content
    except Exception as e:
        st.error(f"Error al cargar archivo desde GitHub: {e}")
        return None

def main():
    st.set_page_config(page_title="mPredictive Dashboard", layout="wide")

    # --- URLs DE GITHUB (RECUERDA USAR EL LINK 'RAW') ---
    URL_TEMPLATE = "https://raw.githubusercontent.com/DavidRuizJ23/Total_Predictivo/main/Template_Datos_Predictive%20modelling.csv"
    URL_GUIA = "https://raw.githubusercontent.com/DavidRuizJ23/Total_Predictivo/main/Guia%20de%20Usuario%20Predictive%20Modelling%20App.docx"
    

    # --- CSS PARA AGRANDAR LAS PESTAÑAS (TABS) ---
    st.markdown("""
        <style>
            /* Cambia el tamaño del texto de las pestañas */
            button[data-baseweb="tab"] p {
                font-size: 20px !important;
                font-weight: bold !important;
            }
            
            /* Opcional: Cambia el tamaño cuando la pestaña está seleccionada */
            button[data-baseweb="tab"][aria-selected="true"] p {
                font-size: 22px !important;
                color: #ff4b4b !important; /* Color corporativo opcional */
            }
        </style>
    """, unsafe_allow_html=True)

    st.set_page_config(page_title="mPredictive Dashboard", layout="wide")

    # --- CONFIGURACIÓN DEL LOGO ---
    # 1. Pega aquí tu URL RAW de GitHub
    logo_url = "https://raw.githubusercontent.com/DavidRuizJ23/Total_Predictivo/main/Logo_WPP.png" 
    
    # 2. Define el tamaño (ancho) que desees para el logo (ej: 200px, 250px, etc.)
    ancho_logo = "200px" 

    # --- HACK DE DISEÑO: TÍTULO IZQ, LOGO GRANDE DER, ALINEADOS ---
    # Usamos Flexbox para control total del diseño
    st.markdown(
        f"""
        <div style="display: flex; align-items: center; justify-content: space-between; padding: 10px 0px;">
            <div style="flex: 1;">
                <h1 style="margin: 0; padding-bottom: 5px;">Predictive Modelling</h1>
                <h3 style="margin: 0; color: #555; font-weight: normal;">Análisis de Impacto Predictivo de Campaña</h3>
            </div>
            <div style="margin-left: 20px;">
                <img src="{logo_url}" width="{ancho_logo}">
            </div>
        </div>
        <hr style="margin-top: 0; margin-bottom: 25px;">
        """,
        unsafe_allow_html=True
    )


    # --- Actividades de lado izquierdo ---

    # --- Header  ---
    st.sidebar.header("Recursos y Carga")

    # --- 2: DESCARGAR TEMPLATE (AL PRINCIPIO) ---
    st.sidebar.subheader("Template de Datos")
    template_bytes = get_github_content(URL_TEMPLATE)
    
    if template_bytes:
        st.sidebar.download_button(
            label="📥 Descargar Template CSV",
            data=template_bytes,
            file_name="template_mpredictive.csv",
            mime="text/csv"
        )

    st.sidebar.divider()

    # --- 2. CARGA DE ARCHIVO ---

    st.sidebar.header("Configuración de Datos")
    uploaded_file = st.sidebar.file_uploader("Usa el formato .csv", type=["csv"])

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

        # --- 3. FILTROS ---
        st.sidebar.header("Calibración del Análisis")
        marcas_base = sorted(df['BrandStd'].unique())
        opciones_marca = ["Total Categoría"] + marcas_base
        selected_brand = st.sidebar.selectbox("Selecciona la Marca para modelar:", opciones_marca)

        # NUEVO Filtro: Marca propia o Benchmark (el que antes era fijo Coca-Cola)
        # Intentamos poner Coca-Cola por defecto si existe, si no, la primera de la lista
        default_idx = marcas_base.index('Coca-Cola') if 'Coca-Cola' in marcas_base else 0
        own_brand = st.sidebar.selectbox("Selecciona Marca Propia:", marcas_base, index=default_idx)

        if selected_brand == "Total Categoría":
            kpis_disponibles = sorted(df['KPI'].unique())
        else:
            kpis_disponibles = sorted(df[df['BrandStd'] == selected_brand]['KPI'].unique())
        
        selected_kpi = st.sidebar.selectbox("Selecciona el KPI a modelar:", kpis_disponibles)

        kpi_nature = st.sidebar.selectbox(
            "Naturaleza del KPI:", 
            ["Promedio", "Total"],
            help="Selecciona 'Promedio' para tasas (Brand Equity, CTR, ROI) o 'Total' para volúmenes (Ventas, Leads, Clics)."
        )

        # Preparar DataFrames
        if selected_brand == "Total Categoría":
            df_model = df[df['KPI'] == selected_kpi].groupby(['MonthReport', 'Year']).agg({
                'Investment': 'sum', 'Value': 'mean'
            }).reset_index()
        else:
            df_model = df[(df['BrandStd'] == selected_brand) & (df['KPI'] == selected_kpi)].copy()

        df_bench = df[df['KPI'] == selected_kpi]


        #  DESCARGAR GUÍA (AL FINAL DE LOS FILTROS) ---
        # Suponiendo que aquí terminan tus selectbox de Marcas y KPIs...
        st.sidebar.divider()
        st.sidebar.subheader("Ayuda")
        
        guia_bytes = get_github_content(URL_GUIA)
        if guia_bytes:
            st.sidebar.download_button(
                label="📖 Descargar Guía de Usuario",
                data=guia_bytes,
                file_name="Guia_Usuario_mPredictive.docx",
                mime="application/pdf" # Cambia a "text/plain" si es un .txt
            )

        # --- LEYENDA DE CONTACTO ---
        st.sidebar.markdown("---") # Una línea sutil adicional
        st.sidebar.caption("📧 **Contacto de soporte:**")
        st.sidebar.caption("david.ruizj@wppmedia.com")

        # --- 3. INTERFAZ DE PESTAÑAS ---
        tab1, tab2, tab3 = st.tabs(["📈 Análisis Visual", "🎯 Modelo Predictivo", "📋 Validación de Datos cargados"])

        with tab1:
            st.subheader("Exploración de Datos")
            
            # Usamos la paleta Safe que elegiste
            paleta_armoniosa = px.colors.qualitative.Safe 

            c1, c2 = st.columns(2)
            
            #Empieza gráfica de promedio anual de KPIs

            with c1:
                # 1. Definimos la lógica según la naturaleza seleccionada
                if kpi_nature == "Promedio":
                    # Agregamos por promedio
                    avg_kpi_year = df_bench.groupby(['Year', 'BrandStd'])['Value'].mean().reset_index()
                    # Formato de porcentaje (ej: 75.2%)
                    label_format = lambda x: f"{x*100:.1f}%"
                    display_title = f"Promedio {selected_kpi} por Año"
                else:
                    # Agregamos por suma total
                    avg_kpi_year = df_bench.groupby(['Year', 'BrandStd'])['Value'].sum().reset_index()
                    # Formato numérico con comas (ej: 1,500,000)
                    label_format = lambda x: f"{x:,.0f}"
                    display_title = f"Total {selected_kpi} por Año"

                st.markdown(f"**{display_title}**")
                
                avg_kpi_year = avg_kpi_year.sort_values("BrandStd")
                avg_kpi_year['text_label'] = avg_kpi_year['Value'].apply(label_format)

                fig_bar = px.bar(avg_kpi_year, 
                                x='Year', 
                                y='Value', 
                                color='BrandStd', 
                                barmode='group', 
                                text='text_label',
                                template="plotly_white",
                                color_discrete_sequence=paleta_armoniosa)
                
                fig_bar.update_traces(
                    textposition='outside',
                    textfont=dict(size=14, color="#2E4053", family="Arial")
                ) 
                
                fig_bar.update_layout(
                    showlegend=True,
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, title_text=""),
                    xaxis=dict(
                        title=dict(text="Año", font=dict(size=16)),
                        tickfont=dict(size=16),
                        showline=False
                    ),
                    yaxis=dict(
                        showgrid=False,
                        showticklabels=False,
                        title=None,
                        zeroline=False,
                        showline=False,
                        # Ajuste de rango dinámico para que no se corten las etiquetas
                        range=[0, avg_kpi_year['Value'].max() * 1.3] 
                    )
                )
                st.plotly_chart(fig_bar, use_container_width=True)
                
                
                # Termina gráfica de promedio anual de KPIs

            # Empieza gráfica de Share Of Investment (SOI)
            with c2:
                st.markdown("**Evolución del Share Of Investment (SOI)**")
                soi_data = df.groupby(['Year', 'BrandStd'])['Investment'].sum().reset_index()
                soi_data['Total_Year'] = soi_data.groupby('Year')['Investment'].transform('sum')
                soi_data['Percentage'] = (soi_data['Investment'] / soi_data['Total_Year']) * 100
                soi_data = soi_data.sort_values("BrandStd")
                
                fig_soi = px.bar(soi_data, 
                                 x='Year', 
                                 y='Investment', 
                                 color='BrandStd',
                                 text=soi_data['Percentage'].apply(lambda x: f'{x:.1f}%'),
                                 template="plotly_white",
                                 color_discrete_sequence=paleta_armoniosa)
                
                fig_soi.update_layout(
                    barnorm='percent', 
                    showlegend=True,
                    legend=dict(
                        orientation="h", 
                        yanchor="bottom", 
                        y=1.02, 
                        xanchor="right", 
                        x=1,
                        title_text=""
                    ),

                    # --- AJUSTE EJE X ---
                    xaxis=dict(
                        title=dict(text="Año", font=dict(size=16)), # Nombre del eje más grande
                        tickfont=dict(size=16),                    # Números de los años más grandes
                        showline=False
                    ),

                    yaxis=dict(
                        showgrid=False,
                        showticklabels=False,
                        title=None,
                        zeroline=False,
                        showline=False
                    )
                    #xaxis=dict(showline=False)
                )
                
                fig_soi.update_traces(
                    textposition='inside',
                    textfont=dict(
                        size=14,
                        color="white",
                        family="Arial"
                    ),
                    insidetextanchor='middle' # Para Centrar el texto dentro de las barras
                )
                
                st.plotly_chart(fig_soi, use_container_width=True)
                # Termina gráfica de Share Of Investment (SOI)

            # Inicia Tabal de Estadísticas de Inversión
            
            # --- TABLA DE ESTADÍSTICAS DE INVERSIÓN (CORREGIDA) ---
            st.divider()
            st.subheader("📊 Resumen Estadístico de Inversión por Marca")
            st.write("Cifras basadas en meses con inversión activa (desduplicado por KPI).")

            # 1. DESDUPLICACIÓN: Inversión única por Mes y Marca
            df_unique_inv = df[['MonthReport', 'BrandStd', 'Investment']].drop_duplicates()

            # 2. FILTRADO: Solo meses con inversión real
            # Aseguramos que Investment sea numérico por si acaso
            df_unique_inv['Investment'] = pd.to_numeric(df_unique_inv['Investment'], errors='coerce')
            df_active_inv = df_unique_inv[df_unique_inv['Investment'] > 0].dropna().copy()

            # 3. CÁLCULO DE ESTADÍSTICAS
            stats_df = df_active_inv.groupby('BrandStd')['Investment'].agg([
                ('Promedio', 'mean'),
                ('Desv. Est.', 'std'),
                ('Min', 'min'),
                ('Q25', lambda x: x.quantile(0.25)),
                ('Q50 (Mediana)', 'median'),
                ('Q75', lambda x: x.quantile(0.75)),
                ('Máx', 'max'),
                ('Meses Activos', 'count')
            ]).reset_index()

            # 4. RENDERIZADO CON FORMATO MONEDA ESTRICTO
            # Nota: El formato "$%,.0f" es la clave para que aparezca el signo de peso
            st.dataframe(
                stats_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "BrandStd": st.column_config.TextColumn("Marca"),
                    "Promedio": st.column_config.NumberColumn("Promedio", format="$%,.0f"),
                    "Desv. Est.": st.column_config.NumberColumn("Desv. Est.", format="$%,.0f"),
                    "Min": st.column_config.NumberColumn("Mínimo", format="$%,.0f"),
                    "Q25": st.column_config.NumberColumn("Q25", format="$%,.0f"),
                    "Q50 (Mediana)": st.column_config.NumberColumn("Mediana", format="$%,.0f"),
                    "Q75": st.column_config.NumberColumn("Q75", format="$%,.0f"),
                    "Máx": st.column_config.NumberColumn("Máximo", format="$%,.0f"),
                    "Meses Activos": st.column_config.NumberColumn("Meses Activos", format="%d")
                }
            )

            # Finaliza Tabla de Estadísticas de Inversión

            # Inicio de Tabla de Totales de Inversión

            # --- TABLA DE INVERSIÓN TOTAL POR AÑO ---
            st.divider()
            st.subheader("📅 Inversión Total Anual (Categoría)")
            st.write("Suma total de la inversión anual desduplicada, considerando todas las marcas y meses.")

            # 1. DESDUPLICACIÓN: Inversión única por Mes, Marca y Año
            df_unique_annual = df[['Year', 'MonthReport', 'BrandStd', 'Investment']].drop_duplicates()

            # 2. CÁLCULO: Suma total por año
            annual_inv_df = df_unique_annual.groupby('Year')['Investment'].sum().reset_index()
            
            # Ordenar por año para que la línea de tiempo sea clara
            annual_inv_df = annual_inv_df.sort_values('Year', ascending=False)

            # 3. RENDERIZADO CON FORMATO MONEDA
            st.dataframe(
                annual_inv_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Year": st.column_config.TextColumn("Año"),
                    "Investment": st.column_config.NumberColumn(
                        "Inversión Total ($)", 
                        format="$%,.0f",
                        help="Suma de la inversión de todas las marcas en el año seleccionado."
                    )
                }
            )


            # Fin de Tabla de Totales de Inversión


            # Gráfica de doble eje

            st.divider()
            st.subheader(f"Tendencia mensual: Inversión vs {selected_kpi} para {selected_brand}")

            # --- 1. PREPARACIÓN DE DATOS ---
            # Extraemos la paleta y seleccionamos colores individuales (un solo string cada uno)
            paleta_safe = px.colors.qualitative.Safe
            color_inv = paleta_safe[0]  # Tomamos el primer color de la lista (Azul claro)
            color_kpi = paleta_safe[1]  # Tomamos el segundo color de la lista (Rosa/Rojo)

            # Ordenamos cronológicamente
            df_line = df_model.sort_values('MonthReport')

            # --- 2. CREACIÓN DE LA GRÁFICA ---
            fig_dual = go.Figure()

            # Traza Inversión
            fig_dual.add_trace(go.Scatter(
                x=df_line['MonthReport'],
                y=df_line['Investment'],
                name="Inversión ($)",
                mode='lines+markers',
                line=dict(color=color_inv, width=3), # Ahora recibe un string, no una lista
                marker=dict(size=8)
            ))

            # Traza KPI
            fig_dual.add_trace(go.Scatter(
                x=df_line['MonthReport'],
                y=df_line['Value'] * 100, 
                name=f"{selected_kpi} (%)",
                mode='lines+markers',
                line=dict(color=color_kpi, width=3, dash='dash'),
                marker=dict(size=8),
                yaxis="y2" 
            ))

            # --- 3. CONFIGURACIÓN DEL DISEÑO ---
            fig_dual.update_layout(
                template="plotly_white",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                hovermode="x unified",
                
                xaxis=dict(
                    title=dict(text="Mes de Reporte", font=dict(size=16)),
                    tickfont=dict(size=16),
                    showgrid=False
                ),
                
                yaxis=dict(
                    title=dict(text="Inversión ($)", font=dict(size=18, color=color_inv)),
                    tickfont=dict(size=16, color=color_inv),
                    tickformat="$,.0f",
                    showgrid=False,
                ),
                
                yaxis2=dict(
                    title=dict(text=f"{selected_kpi} (%)", font=dict(size=18, color=color_kpi)),
                    tickfont=dict(size=16, color=color_kpi),
                    ticksuffix="%",
                    overlaying="y",
                    side="right",
                    showgrid=False,
                    rangemode="tozero"
                )
            )

            st.plotly_chart(fig_dual, use_container_width=True)

            # Fin de gráfica de doble eje
            # Grafica de Modelo Logaritmo

            st.divider()
            st.subheader(f"Relación Inversión vs KPI. Considerando {selected_brand}")
            
            if not df_model.empty and len(df_model) > 2:
                # --- PREPARACIÓN DE DATOS PARA LA REGRESIÓN ---
                df_log = df_model[df_model['Investment'] > 0].copy()
                X_log = np.log(df_log[['Investment']]).values
                y = df_log['Value'].values
                log_reg = LinearRegression().fit(X_log, y)
                
                x_range = np.linspace(df_log['Investment'].min(), df_log['Investment'].max(), 100)
                y_pred = log_reg.predict(np.log(x_range.reshape(-1, 1)))
                
                # --- CREACIÓN DEL GRÁFICO ---
                # Usamos el primer color de tu paleta Safe para los puntos
                fig_scatter = px.scatter(df_log, 
                                        x='Investment', 
                                        y='Value', 
                                        hover_data=['MonthReport'], 
                                        template="plotly_white",
                                        labels={'Investment': 'Inversión ($)', 'Value': f'{selected_kpi} (%)'},
                                        color_discrete_sequence=[paleta_armoniosa[0]]) # Color 1 de la paleta
                
                # Añadimos la curva logarítmica (en el segundo color de la paleta para contraste)
                fig_scatter.add_trace(go.Scatter(x=x_range, 
                                                y=y_pred, 
                                                name='Curva de Tendencia (Log)',
                                                line=dict(color=paleta_armoniosa[1], width=4)))
                
                # --- AJUSTES DE LOOK & FEEL (ESTILO MINIMALISTA) ---
                fig_scatter.update_layout(
                    hovermode="closest",
                    showlegend=True,
                    legend=dict(
                        orientation="h", 
                        yanchor="bottom", 
                        y=1.02, 
                        xanchor="right", 
                        x=1,
                        title_text=""
                    ),
                    
                    # Ajuste Eje X: Inversión
                    xaxis=dict(
                        title=dict(text="Inversión ($)", font=dict(size=18)),
                        tickfont=dict(size=16),
                        showgrid=False,       # Quitar cuadrícula
                        showline=False,       # Quita línea del eje
                        tickformat="$,.0f"    # Formato moneda sin decimales
                    ),
                    
                    # Ajuste Eje Y: KPI
                    yaxis=dict(
                        title=dict(text=f"{selected_kpi}", font=dict(size=18)),
                        tickfont=dict(size=16),
                        showgrid=False,       # Quitar cuadrícula
                        showline=False,       # Quita línea del eje
                        # Si tus datos son decimales (0.7), este formato pone 70.0%
                        tickformat=".1%"      
                    )
                )
                
                st.plotly_chart(fig_scatter, use_container_width=True)
                #st.caption(f"💡 Los puntos representan meses históricos. La curva indica la tendencia de retorno según el nivel de inversión.")
            
            # Fin de gráfica de Modelo Logaritmo
            else:
                st.warning("No hay datos suficientes para generar el modelo logarítmico.")
        with tab2:

            st.header("🎯 Simulación de Impacto de Campaña")
            st.write("Introduce la inversión Total de campaña y la duración para proyectar el Uplift esperado.")

            # --- INPUTS DE USUARIO ---
            # --- DEFINICIÓN DE COLUMNAS CON ALINEACIÓN INFERIOR ---
            col_input1, col_input2, col_input3 = st.columns(3, vertical_alignment="bottom")

            with col_input1:
                # Inyectamos el CSS una sola vez (afecta a todos los number_inputs)
                st.markdown("""
                    <style>
                    /* Estilo para el título (Label) */
                    div[data-testid="stNumberInput"] label p {
                        font-size: 22px !important;
                        font-weight: bold !important;
                        color: #2E4053 !important;
                        margin-bottom: 5px !important;
                    }

                    /* Estilo para el campo de texto (Input) */
                    div[data-testid="stNumberInput"] input {
                        font-size: 22px !important;
                        height: 60px !important;
                        color: #2E4053 !important;
                        background-color: #FBFCFC !important;
                    }
                    </style>
                    """, unsafe_allow_html=True)

                campaign_investment = st.number_input(
                    "Inversión", 
                    min_value=0.0, 
                    value=1000000.0, 
                    step=10000.0,
                    format="%.2f"
                )

            with col_input2:
                # Este input heredará el estilo de tamaño y fuente definido arriba
                campaign_duration = st.number_input(
                    "Duración (Meses)", 
                    min_value=1, 
                    max_value=12, 
                    value=1, 
                    step=1
                )

            with col_input3:
                monthly_investment = campaign_investment / campaign_duration
                # Quitamos el margin-bottom para que la alineación "bottom" sea exacta
                st.markdown(f"""
                    <div style='line-height: 1.1; margin-bottom: 5px;'>
                        <p style='color: #7f8c8d; font-size: 16px; margin-bottom: 14px; text-transform: capitalize; letter-spacing: 1px;'>
                            Inversión Mensual
                        </p>
                        <p style='color: #2E4053; font-size: 32px; font-weight: 800; margin-bottom: 22px;'>
                            ${monthly_investment:,.0f} <span style='font-size: 18px; font-weight: 400;'></span>
                        </p>
                    </div>
                    """, unsafe_allow_html=True)

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

            st.markdown("""
                <style>
                /* Estilo para el contenedor del botón */
                .stButton > button {
                    width: 250%; /* O puedes poner un ancho fijo como 300px */
                    height: 80px; /* Altura del botón */
                    border-radius: 10px; /* Bordes redondeados */
                    background-color: #2E4053; /* Color oscuro profesional */
                    color: white; /* Color de la fuente */
                    font-size: 20px; /* Tamaño de la letra */
                    font-weight: bold; /* Negrita */
                    border: none;
                    transition: 0.3s;
                }
                
                /* Efecto al pasar el mouse (hover) */
                .stButton > button:hover {
                    background-color: #E74C3C; /* Cambia a rojo al pasar el mouse */
                    color: white;
                    border: 2px solid #2E4053;
                }
                </style>
                """, unsafe_allow_html=True)        

            if st.button("Calcular Impacto de Campaña"):
                
                # 1. Filtrado de datos
                if brand_model == 'Total Categoría':
                    ds_model = df[df['KPI'] == kpi_model].copy()
                else: 
                    ds_model = df[(df['BrandStd'] == brand_model) & (df['KPI'] == kpi_model)].copy()

                if ds_model.empty:
                    st.error("ERROR")
                else:
                    # 2. Procesamiento y Ponderación
                    ds_model['YearReport'] = ds_model['MonthReport'].dt.year
                    ponderacion_df = pd.DataFrame({
                        'YearReport': [2022, 2023, 2024, 2025],
                        'ponderacion': [1.41, 2.05, 4.56, 4.56]
                    })
                    
                    ds_model_r = pd.merge(ds_model, ponderacion_df, on='YearReport', how='left')
                    ds_model_r['Inv_real'] = ds_model_r['Investment'] * ds_model_r['ponderacion']
                    
                    # 3. Estadísticas y Transformación S-Curve
                    std_kpi = ds_model['Value'].std()
                    year_stats_k_b = ds_model.groupby(['BrandStd'])['Value'].agg(['mean','std']).reset_index()

                    # LÍNEA AJUSTADA: Ahora es dinámica usando 'own_brand'
                    try:
                        brand_std_val = year_stats_k_b[year_stats_k_b['BrandStd'] == own_brand]['std'].iloc[0]          
                        fact_trop = brand_std_val / std_kpi
                    except IndexError:
                        st.warning(f"La marca {own_brand} no tiene suficientes datos en este recorte. Usando factor 1.0")
                        brand_std_val = std_kpi # Evita que se rompa la gráfica más abajo
                        fact_trop = 1.0
                    
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

                    # Extraemos el valor original de Alpha (ej: 0.70)
                    alpha_original = float(winner['Variable Independiente'][-4:])

                    # Realizamos tu nuevo cálculo: 100 * (1 - alpha)
                    valor_ajustado = 100 * (1 - alpha_original)
                    
                    # --- MOSTRAR MÉTRICAS ---
                    st.subheader("📊 Resultados del Impacto de Campaña")
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Uplift de Campaña", f"{winner['Attribution Puntual']*100:.2f} PP")
                    m2.metric("Benchmark de Crecimiento", f"{winner['Indx_i']:.1f}", delta=f"{winner['Indx_i']-100:.1f}")
                    m3.metric("Índice de Saturación", f"{valor_ajustado:.0f} idx")

                    # --- GENERACIÓN DE GRÁFICA (Ajuste Matplotlib para Streamlit) ---

                    # --- 1. PREPARACIÓN DE DATOS ---
                    alpha_win = float(winner['Variable Independiente'][-4:])
                    
                    # Creamos 100 puntos para que la curva se vea suave (no 71)
                    sim_df = pd.DataFrame({'Idx': range(1, 101)})
                    sim_df['Media Spend'] = sim_df['Idx'] * (max_spend / 100)
                    inv_norm = 100 * sim_df['Media Spend'] / max_spend
                    curve_s = (beta_param**alpha_win**inv_norm) * fact_trop
                    
                    sim_df['Min'] = curve_s * winner['Intervalo de Confianza Inferior'] * 100
                    sim_df['Max'] = curve_s * winner['Coeficiente de Regresión'] * 100
                    
                    # Inicio de la grafica de la curva S

                    # --- 1. CÁLCULO DE TECHO DINÁMICO REFORZADO ---
                    # Debemos asegurar que el eje Y cubra la curva, el benchmark y el punto proyectado
                    val_max_curva = sim_df['Max'].max()
                    val_benchmark = brand_std_val * 100
                    val_proyectado = winner['Attribution Puntual'] * 100                    

                    # El límite es el máximo de estos tres, más un 40% de margen para las etiquetas de texto
                    y_axis_upper_limit = max(val_max_curva, val_benchmark, val_proyectado) * 1.4

                    # --- 2. CREACIÓN DEL GRÁFICO ---
                    fig_model = go.Figure()

                    # Curva Máximo Esperado
                    fig_model.add_trace(go.Scatter(
                        x=sim_df['Media Spend']/1e6, 
                        y=sim_df['Max'],
                        name='Máximo Esperado',
                        line=dict(color='green', width=3),
                        mode='lines',
                        hovertemplate="Inv: $%{x:.2f}M<br>Uplift Max: %{y:.2f} PP<extra></extra>"
                    ))

                    # Curva Mínimo Esperado
                    fig_model.add_trace(go.Scatter(
                        x=sim_df['Media Spend']/1e6, 
                        y=sim_df['Min'],
                        name='Mínimo Esperado',
                        line=dict(color='orange', width=3),
                        mode='lines',
                        hovertemplate="Inv: $%{x:.2f}M<br>Uplift Min: %{y:.2f} PP<extra></extra>"
                    ))

                    # Línea de Benchmark
                    fig_model.add_hline(
                        y=brand_std_val*100, 
                        line_dash="dash", 
                        line_color="black",
                        annotation_text=f"Benchmark {own_brand}",
                        annotation_position="bottom right"
                    )

                    # --- 3. PUNTOS RESALTADOS ---
                    # Punto Máximo
                    fig_model.add_trace(go.Scatter(
                        x=[simulated_investment_value/1e6],
                        y=[winner['Attribution Puntual']*100],
                        mode='markers+text',
                        text=[f"${simulated_investment_value/1e6:.2f}M<br>{winner['Attribution Puntual']*100:.2f} PP"],
                        textposition='top center', # Cambiado a TOP para que se vea arriba
                        textfont=dict(size=12, color="green", family="Arial Black"),
                        cliponaxis=False, # Crucial: permite que el texto "salga" del eje si es necesario
                        marker=dict(color='green', size=14, symbol='diamond', line=dict(width=2, color='white')),
                        name='Resultado Proyectado (Max)',
                        hovertemplate="<b>SIMULACIÓN ACTUAL</b><br>Inversión: $%{x:.2f}M<br>Uplift: %{y:.2f} PP<extra></extra>"
                    ))

                    # Punto Mínimo
                    fig_model.add_trace(go.Scatter(
                        x=[simulated_investment_value/1e6],
                        y=[winner['Attribution Inferior']*100],
                        mode='markers+text',
                        text=[f"${simulated_investment_value/1e6:.2f}M<br>{winner['Attribution Inferior']*100:.2f} PP"],
                        textposition='bottom center',
                        textfont=dict(size=12, color="orange", family="Arial Black"),
                        cliponaxis=False,
                        marker=dict(color='orange', size=14, symbol='diamond', line=dict(width=2, color='white')),
                        name='Resultado Proyectado (Min)',
                        hovertemplate="<b>SIMULACIÓN ACTUAL</b><br>Inversión: $%{x:.2f}M<br>Uplift: %{y:.2f} PP<extra></extra>"
                    ))

                    # --- 4. AJUSTES DE DISEÑO FINAL ---
                    fig_model.update_layout(
                        title=f'Curva de Predicción: {selected_kpi}',
                        template="plotly_white",
                        height=600,
                        width=850,
                        margin=dict(t=100, b=50, l=50, r=50), # Aumentamos el margen superior (t=100)
                        hovermode="closest",
                        legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="right", x=1),
                        
                        xaxis=dict(
                            showgrid=False, 
                            tickfont=dict(size=14),
                            title=dict(text='Inv. Mensual (Millones)', font=dict(size=16))
                        ),
                        
                        yaxis=dict(
                            showgrid=False, 
                            tickfont=dict(size=14),
                            range=[0, y_axis_upper_limit], # Rango con margen de seguridad
                            autorange=False, # Forzamos el rango manual
                            title=dict(text='Impacto en KPI (PP)', font=dict(size=16))
                        )
                    )

                    st.plotly_chart(fig_model, use_container_width=False)

                    # Fin de la grafica de la curva S

                    # Inicio de Grafica de Proyección de Crecimiento

                    # --- 6. GRÁFICA DE IMPACTO PROYECTADO (HISTÓRICO VS SIMULACIÓN) ---
                    st.divider()
                    st.subheader(f"Impacto Proyectado: Escenario Base vs Con Campaña ({own_brand})")

                    # 1. Obtener históricos de la marca seleccionada como Benchmark
                    df_bench_own = df_bench[df_bench['BrandStd'] == own_brand].copy()
                    
                    if kpi_nature == "Promedio":
                        hist_data = df_bench_own.groupby('Year')['Value'].mean().reset_index()
                        label_fn = lambda x: f"{x*100:.1f}%"
                    else:
                        hist_data = df_bench_own.groupby('Year')['Value'].sum().reset_index()
                        label_fn = lambda x: f"{x:,.0f}"

                    # 2. Calcular el valor proyectado (Último año + Incremental)
                    last_year_val = hist_data.iloc[-1]['Value']
                    uplift_valor = winner['Attribution Puntual']
                    projected_val = last_year_val + uplift_valor
                    
                    # 3. Preparar DataFrame para la gráfica
                    # Convertimos años a string para que Plotly los trate como categorías
                    plot_df = hist_data.copy()
                    plot_df['Year'] = plot_df['Year'].astype(str)
                    plot_df['Tipo'] = "Histórico"

                    # Añadimos la fila de la simulación
                    sim_row = pd.DataFrame({
                        'Year': [f"Post Campaña"], 
                        'Value': [projected_val],
                        'Tipo': ["Proyectado"]
                    })
                    plot_df = pd.concat([plot_df, sim_row], ignore_index=True)
                    plot_df['Etiqueta'] = plot_df['Value'].apply(label_fn)

                    # 4. Crear la Gráfica
                    # Usamos colores contrastantes: Gris para histórico, el color de la marca para simulación
                    colors = {"Histórico": "#BDC3C7", "Proyectado": paleta_armoniosa[1]}

                    fig_projected = px.bar(
                        plot_df,
                        x='Year',
                        y='Value',
                        color='Tipo',
                        text='Etiqueta',
                        color_discrete_map=colors,
                        template="plotly_white"
                    )

                    fig_projected.update_traces(
                        textposition='outside',
                        textfont=dict(size=16, color="#2E4053", family="Arial Black")
                    )

                    fig_projected.update_layout(
                        showlegend=False,
                        height=500,
                        width=900,
                        margin=dict(t=50, b=50),
                        xaxis=dict(title="", tickfont=dict(size=14)),
                        yaxis=dict(
                            showgrid=False, 
                            showticklabels=False, 
                            title=None, 
                            range=[0, plot_df['Value'].max() * 1.3]
                        )
                    )

                    # Mostrar la gráfica centrada
                    col_g1, col_g2, col_g3 = st.columns([1, 4, 1])
                    with col_g2:
                        st.plotly_chart(fig_projected, use_container_width=True)
                        st.info(f"💡 La barra de **Post Campaña** toma como base el último año con datos y agrega el Uplift proyectado de **{uplift_valor*100:.2f} PP**.")
                        st.warning(f"💡 Consideraciones: La proyección asume que el resto de esfuerzos de Marketingo y Factores Externos se mantienen")

                    # Fin de grafica de Proyección de Crecimiento


                             
            else:
                #st.error("ERROR: No hay suficientes datos históricos para este KPI/Marca para entrenar el modelo.")
                st.warning("Introduce la inversión de Campaña y su Duración y luego haz clic en 'Calcular Impacto de Campaña' para ver los resultados.")
        with tab3:
            st.subheader("Validación de Registros (Top 15)")
            st.dataframe(df_model.head(15).sort_values('MonthReport', ascending=False), use_container_width=True)

    else:
        st.info("👈 Por favor, carga el archivo CSV en la barra lateral.")

if __name__ == "__main__":
    main()  