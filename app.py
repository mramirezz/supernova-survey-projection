import streamlit as st
import subprocess
import sys
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path
import json
from datetime import datetime
import glob

st.set_page_config(
    page_title="Supernova Projection Simulator",
    page_icon="",
    layout="wide"
)

# Título principal
st.title(" Simulador de Proyección de Supernovas")
st.markdown("Interfaz gráfica para ejecutar simulaciones multi-banda")

# ============================================================================
# SIDEBAR: CONFIGURACIÓN
# ============================================================================
with st.sidebar:
    st.header(" Configuración de Simulación")
    
    # Número de runs
    n_runs = st.number_input(
        "Número de simulaciones por filtro",
        min_value=1,
        max_value=10000,
        value=100,
        step=1,
        help="Cantidad de supernovas a simular en cada filtro"
    )
    
    st.markdown("---")
    
    # Filtros fotométricos
    st.subheader(" Filtros Fotométricos")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**ZTF/SDSS**")
        filter_g = st.checkbox("g (verde)", value=True)
        filter_r = st.checkbox("r (rojo)", value=True)
        filter_i = st.checkbox("i (IR cercano)", value=True)
        filter_u = st.checkbox("u (UV)")
        filter_z = st.checkbox("z (IR)")
    
    with col2:
        st.markdown("**Johnson-Cousins**")
        filter_U = st.checkbox("U (UV)")
        filter_B = st.checkbox("B (azul)")
        filter_V = st.checkbox("V (visual)")
        filter_R = st.checkbox("R (rojo)")
        filter_I = st.checkbox("I (IR)")
    
    # Construir lista de filtros seleccionados
    selected_filters = []
    if filter_g: selected_filters.append('g')
    if filter_r: selected_filters.append('r')
    if filter_i: selected_filters.append('i')
    if filter_u: selected_filters.append('u')
    if filter_z: selected_filters.append('z')
    if filter_U: selected_filters.append('U')
    if filter_B: selected_filters.append('B')
    if filter_V: selected_filters.append('V')
    if filter_R: selected_filters.append('R')
    if filter_I: selected_filters.append('I')
    
    st.markdown("---")
    
    # Tipos de SN
    st.subheader(" Tipos de Supernova")
    sn_Ia = st.checkbox("Tipo Ia", value=True, help="Termonucleares")
    sn_Ibc = st.checkbox("Tipo Ibc", help="Core-collapse sin H")
    sn_II = st.checkbox("Tipo II", help="Core-collapse con H")
    
    # Construir lista de tipos
    selected_sn_types = []
    if sn_Ia: selected_sn_types.append('Ia')
    if sn_Ibc: selected_sn_types.append('Ibc')
    if sn_II: selected_sn_types.append('II')
    
    st.markdown("---")
    
    # Parámetros cosmológicos
    st.subheader("🌌 Parámetros Cosmológicos")
    redshift_max = st.number_input(
        "Redshift máximo",
        min_value=0.01,
        max_value=2.0,
        value=0.3,
        step=0.01,
        format="%.2f",
        help="Redshift máximo para las simulaciones"
    )
    
    st.markdown("---")
    
    # Survey
    st.subheader(" Survey")
    survey = st.selectbox(
        "Seleccionar survey",
        options=["ZTF", "SUDARE"],
        index=0
    )
    
    st.markdown("---")
    
    # Semilla
    seed = st.number_input(
        "Semilla aleatoria",
        min_value=0,
        max_value=99999,
        value=42,
        help="Para reproducibilidad"
    )
    
    st.markdown("---")
    
    # Validación y resumen
    total_sims = n_runs * len(selected_filters)
    
    st.info(f"""
    **Resumen:**
    - Filtros: {len(selected_filters)}
    - Tipos SN: {len(selected_sn_types)}
    - **Total simulaciones: {total_sims}**
    - Tiempo estimado: ~{total_sims * 3 / 60:.1f} min
    """)
    
    # BOTÓN EJECUTAR
    run_simulation = st.button(
        " EJECUTAR SIMULACIÓN",
        type="primary",
        use_container_width=True,
        disabled=(len(selected_filters) == 0 or len(selected_sn_types) == 0)
    )

# ============================================================================
# PANEL PRINCIPAL
# ============================================================================

# Crear tabs
tab1, tab2, tab3, tab4 = st.tabs([
    " Ejecución",
    " Curvas de Luz",
    " Análisis Multi-Banda",
    " Resultados Guardados"
])

# ============================================================================
# TAB 1: EJECUCIÓN
# ============================================================================
with tab1:
    st.header("Monitor de Ejecución")
    
    if run_simulation:
        # Validar
        if len(selected_filters) == 0:
            st.error(" Debes seleccionar al menos un filtro fotométrico")
        elif len(selected_sn_types) == 0:
            st.error(" Debes seleccionar al menos un tipo de supernova")
        else:
            # Construir comando
            cmd = [
                sys.executable,
                "run_multiband.py",
                "--runs", str(n_runs),
                "--filters"
            ] + selected_filters + [
                "--sn-types"
            ] + selected_sn_types + [
                "--redshift-max", str(redshift_max),
                "--survey", survey,
                "--seed", str(seed)
            ]
            
            # Mostrar comando
            st.code(" ".join(cmd), language="bash")
            
            # Ejecutar
            st.info(" Iniciando simulación...")
            
            # Placeholder para output
            output_placeholder = st.empty()
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                # Ejecutar comando
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding='utf-8',
                    errors='replace'
                )
                
                # Capturar output en tiempo real
                output_lines = []
                for line in process.stdout:
                    output_lines.append(line)
                    output_placeholder.text_area(
                        "Logs de ejecución",
                        value="".join(output_lines[-50:]),  # Últimas 50 líneas
                        height=300
                    )
                    
                    # Actualizar progreso (estimación básica)
                    if "FILTRO" in line:
                        filter_num = len([l for l in output_lines if "FILTRO" in l])
                        progress = min(filter_num / len(selected_filters), 1.0)
                        progress_bar.progress(progress)
                        status_text.text(f"Procesando filtro {filter_num}/{len(selected_filters)}")
                
                process.wait()
                
                if process.returncode == 0:
                    progress_bar.progress(1.0)
                    st.success("✅ Simulación completada exitosamente!")
                    st.balloons()
                else:
                    st.error(f"❌ Error en la simulación (código: {process.returncode})")
                
            except Exception as e:
                st.error(f"❌ Error al ejecutar: {str(e)}")
    
    else:
        st.info(" Configura los parámetros en el panel lateral y presiona 'EJECUTAR SIMULACIÓN'")
        
        # Mostrar ejemplo de comando
        if selected_filters and selected_sn_types:
            example_cmd = f"""python run_multiband.py \\
    --runs {n_runs} \\
    --filters {' '.join(selected_filters)} \\
    --sn-types {' '.join(selected_sn_types)} \\
    --redshift-max {redshift_max} \\
    --survey {survey} \\
    --seed {seed}"""
            
            st.markdown("**Comando equivalente:**")
            st.code(example_cmd, language="bash")

# ============================================================================
# TAB 2: CURVAS DE LUZ
# ============================================================================
with tab2:
    st.header("Visualización de Curvas de Luz")
    
    # Buscar batches recientes
    batch_dir = Path("outputs/batch_runs")
    
    if batch_dir.exists():
        batch_dirs = sorted([d for d in batch_dir.iterdir() if d.is_dir()], reverse=True)
        
        if batch_dirs:
            # Selector de batch
            selected_batch = st.selectbox(
                "Seleccionar batch",
                options=[d.name for d in batch_dirs[:20]],
                format_func=lambda x: f"{x[:15]}... ({x.split('_')[0]})"
            )
            
            batch_path = batch_dir / selected_batch
            
            # Buscar archivos de proyección
            projection_files = list(batch_path.glob("**/projection_*.csv"))
            
            if projection_files:
                st.success(f"Encontrados {len(projection_files)} archivos de proyección")
                
                # Selector de archivo específico
                selected_file = st.selectbox(
                    "Seleccionar supernova",
                    options=projection_files,
                    format_func=lambda x: x.stem.replace("projection_", "")
                )
                
                # Cargar y visualizar
                try:
                    df = pd.read_csv(selected_file)
                    
                    # Crear gráfico con Plotly
                    fig = go.Figure()
                    
                    # Detectiones
                    df_detected = df[df['detected'] == True]
                    if len(df_detected) > 0:
                        fig.add_trace(go.Scatter(
                            x=df_detected['mjd'],
                            y=df_detected['magnitud_proyectada'],
                            mode='markers',
                            name='Detecciones',
                            marker=dict(size=8, color='blue'),
                        ))
                    
                    # Upper limits
                    df_upper = df[df['detected'] == False]
                    if len(df_upper) > 0:
                        fig.add_trace(go.Scatter(
                            x=df_upper['mjd'],
                            y=df_upper['maglimit'],
                            mode='markers',
                            name='Upper Limits',
                            marker=dict(size=8, symbol='triangle-down', color='red'),
                        ))
                    
                    fig.update_layout(
                        title=f"Curva de Luz: {selected_file.stem}",
                        xaxis_title="MJD (Modified Julian Date)",
                        yaxis_title="Magnitud",
                        yaxis_autorange="reversed",
                        hovermode='closest',
                        height=600
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Mostrar estadísticas
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Total Obs.", len(df))
                    with col2:
                        st.metric("Detecciones", len(df_detected))
                    with col3:
                        st.metric("Upper Limits", len(df_upper))
                    with col4:
                        detection_rate = len(df_detected) / len(df) if len(df) > 0 else 0
                        st.metric("Tasa Detección", f"{detection_rate:.1%}")
                    
                    # Mostrar tabla de datos
                    with st.expander("Ver datos"):
                        st.dataframe(df, use_container_width=True)
                
                except Exception as e:
                    st.error(f"Error al cargar archivo: {str(e)}")
            else:
                st.warning("No se encontraron archivos de proyección en este batch")
        else:
            st.info("No hay batches disponibles. Ejecuta una simulación primero.")
    else:
        st.info("No hay resultados guardados. Ejecuta una simulación primero.")

# ============================================================================
# TAB 3: ANÁLISIS MULTI-BANDA
# ============================================================================
with tab3:
    st.header("Comparación Multi-Banda")
    
    st.info(" En construcción: Comparación de curvas de luz en múltiples filtros")
    
    # TODO: Cargar múltiples filtros de la misma SN y compararlos

# ============================================================================
# TAB 4: RESULTADOS GUARDADOS
# ============================================================================
with tab4:
    st.header("Explorador de Resultados")
    
    batch_dir = Path("outputs/batch_runs")
    
    if batch_dir.exists():
        batch_dirs = sorted([d for d in batch_dir.iterdir() if d.is_dir()], reverse=True)
        
        if batch_dirs:
            st.success(f"Encontrados {len(batch_dirs)} batches")
            
            # Crear tabla de batches
            batch_data = []
            
            for batch_path in batch_dirs[:50]:  # Últimos 50
                # Buscar metadata
                metadata_file = batch_path / "batch_metadata.json"
                
                if metadata_file.exists():
                    try:
                        with open(metadata_file, 'r') as f:
                            metadata = json.load(f)
                        
                        stats = metadata.get('statistics', {})
                        run_stats = stats.get('run_statistics', {})
                        
                        batch_data.append({
                            'Batch ID': batch_path.name[:20] + '...',
                            'Fecha': batch_path.name.split('_')[0],
                            'Runs': run_stats.get('runs_completed', 0),
                            'Éxito': f"{run_stats.get('success_rate', 0):.1%}",
                            'Detecciones': stats.get('detection_statistics', {}).get('total_detections', 0),
                            'Ruta': str(batch_path)
                        })
                    except:
                        pass
            
            if batch_data:
                df_batches = pd.DataFrame(batch_data)
                
                # Mostrar tabla
                st.dataframe(
                    df_batches[['Batch ID', 'Fecha', 'Runs', 'Éxito', 'Detecciones']],
                    use_container_width=True,
                    hide_index=True
                )
                
                # Selector para explorar
                selected_row = st.selectbox(
                    "Seleccionar batch para explorar",
                    options=range(len(df_batches)),
                    format_func=lambda i: df_batches.iloc[i]['Batch ID']
                )
                
                if st.button("Abrir carpeta del batch"):
                    import os
                    batch_path = df_batches.iloc[selected_row]['Ruta']
                    os.startfile(batch_path)
                    st.success(f"Abriendo: {batch_path}")
        else:
            st.info("No hay batches disponibles")
    else:
        st.info("No hay resultados guardados")

# ============================================================================
# FOOTER
# ============================================================================
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <small>Supernova Survey Projection System | Powered by Streamlit</small>
</div>
""", unsafe_allow_html=True)
