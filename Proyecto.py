# app.py (Corregido - ValueError en plotly.update_yaxes)

# Importar librerías (igual que antes)
import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
import plotly.express as px
import time
from pythonping import ping
import psutil
import speedtest
import datetime

# --- Configuración (igual que antes) ---
CONTAMINACION_ESPERADA = 'auto'
ESTADO_ALEATORIO = 42
DURACION_MONITORIZACION_S = 15
USUARIOS_VALIDOS = { "Ivan123": "Ivan123", "Marcos123": "Marcos123" }
LATENCIA_RAPIDA_MS = 80
LATENCIA_ACEPTABLE_MS = 200
PERDIDA_PAQUETES_MAX_PERMITIDA = 0.5

# --- Funciones Auxiliares ---
def detectar_anomalias_serie(serie_datos):
    if serie_datos is None or len(serie_datos) < 5: return np.array([])
    try:
        X = np.array(serie_datos).reshape(-1, 1); X_clean = X[np.isfinite(X)].reshape(-1, 1)
        if len(X_clean) < 5: return np.array([])
        modelo = IsolationForest(contamination=CONTAMINACION_ESPERADA, random_state=ESTADO_ALEATORIO).fit(X_clean)
        predicciones = modelo.predict(X); return np.where(predicciones == -1)[0]
    except ValueError as ve:
        if "Input contains NaN, infinity or a value too large" in str(ve): st.warning("Advertencia ML: Datos con NaN/Inf."); return np.array([])
        else: st.error(f"Error ML (Valor): {ve}"); return np.array([])
    except Exception as e: st.error(f"Error ML: {e}"); return np.array([])

def sugerir_solucion_tasa(tasa_bps):
    if not isinstance(tasa_bps, (int, float)) or not np.isfinite(tasa_bps): return "Valor inválido."
    if tasa_bps > 10*1024*1024: return f"**Tasa MUY ALTA** ({tasa_bps/(1024*1024):.1f} MB/s). Causa: Descarga/subida intensiva, streaming 4K, backup, ¿proceso inesperado?"
    elif tasa_bps > 1*1024*1024: return f"**Tasa ELEVADA** ({tasa_bps/(1024*1024):.1f} MB/s). Causa: Actividad considerable (navegación, video HD, updates)."
    elif tasa_bps < 1*1024 and tasa_bps >= 0: return f"**Tasa MUY BAJA** ({tasa_bps/1024:.1f} KB/s). Normal si no hay actividad. Si se esperaba, ¿problema conexión?"
    else: return f"Tasa local: {tasa_bps/(1024*1024):.2f} MB/s (normal)."

# --- FUNCIÓN MODIFICADA ---
def crear_grafico_plotly_tasa(serie_tasas, anomalias_indices):
    if serie_tasas is None or len(serie_tasas) == 0: return None
    df = pd.DataFrame({'Segundo': np.arange(len(serie_tasas)), 'Tasa (Bytes/s)': serie_tasas}); df['Estado'] = 'Normal'
    valid_anomalias_indices = [idx for idx in anomalias_indices if idx < len(df)]
    if len(valid_anomalias_indices) > 0: df.loc[valid_anomalias_indices, 'Estado'] = 'Anomalía'
    fig = px.line(df, x='Segundo', y='Tasa (Bytes/s)', title=f'📈 Actividad de Red Local ({len(serie_tasas)} seg.)', markers=True, labels={'Segundo': 'Tiempo (s)', 'Tasa (Bytes/s)': 'Tasa (Bytes/s)'})
    df_anomalias = df[df['Estado'] == 'Anomalía']
    if not df_anomalias.empty: fig.add_scatter(x=df_anomalias['Segundo'], y=df_anomalias['Tasa (Bytes/s)'], mode='markers', marker=dict(color='red', size=10), name='Anomalía Detectada', hovertemplate="Seg: %{x}<br>Tasa: %{y:,.0f} B/s<br><b>Anomalía</b><extra></extra>")
    fig.update_traces(hovertemplate="Seg: %{x}<br>Tasa: %{y:,.0f} B/s<br>Normal<extra></extra>")
    fig.update_layout(hovermode='x unified', legend_title_text='Estado')

    # ***** INICIO DE LA CORRECCIÓN (Lógica de escala Y) *****
    if not df['Tasa (Bytes/s)'].empty:
        max_val = df['Tasa (Bytes/s)'].dropna().max()
        if pd.notna(max_val) and max_val > 0:
             # Asignar directamente los valores válidos para Plotly ('log' o 'linear')
             escala_plotly = 'log' if max_val > 50000 else 'linear' # 'linear' es el valor correcto
             # Crear texto para mostrar (con mayúscula inicial)
             escala_display = escala_plotly.capitalize() # Obtiene 'Log' o 'Linear'
             # Usar los valores correctos en update_yaxes
             fig.update_yaxes(type=escala_plotly, title_text=f"Tasa (Bytes/s) - Escala {escala_display}")
    # ***** FIN DE LA CORRECCIÓN *****
    return fig
# --- FIN FUNCIÓN MODIFICADA ---

def realizar_ping(host, count=4):
    try: return ping(host, count=count, verbose=False, timeout=2)
    except PermissionError: st.error(f"⛔ Error Permisos Ping a {host}."); return None
    except Exception as e: st.error(f"⛔ Error Ping a {host}: {e}"); return None

def realizar_speedtest():
    try:
        st_test = speedtest.Speedtest(secure=True); st_test.get_best_server(); st_test.download(); st_test.upload()
        return st_test.results.dict()
    except speedtest.SpeedtestException as e: st.error(f"⛔ Error Speedtest: {e}"); return None
    except Exception as e: st.error(f"⛔ Error inesperado Speedtest: {e}"); return None
# ---------------------------------------------------------------------

# --- Función para Cargar CSS desde archivo ---
def load_css_from_file(file_path):
    try:
        with open(file_path) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        st.error(f"Error: Archivo CSS no encontrado en {file_path}")

# --- Función para crear Tarjetas HTML ---
def create_metric_card(title, value, icon="", key_suffix=""):
    display_value = "N/A"
    if value is not None and np.isfinite(value):
        if isinstance(value, float):
            if "Perdidos" in title: display_value = f"{value:.1%}"
            elif "Mbps" in title: display_value = f"{value:.2f}<span style='font-size: 0.6em;'> Mbps</span>"
            elif "ms" in title: display_value = f"{value:.2f}<span style='font-size: 0.6em;'> ms</span>"
            else: display_value = f"{value:,.2f}"
        elif isinstance(value, int): display_value = f"{value:,}"
        else: display_value = str(value)
    card_html = f"""
    <div class="metric-card" key="card-{key_suffix}">
        <div class="icon">{icon}</div>
        <h3>{title}</h3>
        <div class="value">{display_value}</div>
    </div>"""
    return card_html

# --- Configuración de Página ---
st.set_page_config(
    layout="wide",
    page_title="Diagnóstico Red Pro",
    page_icon="🌐",
    initial_sidebar_state="expanded"
)

# --- Cargar CSS Personalizado desde archivo ---
load_css_from_file("style.css")

# --- Inicializar estados ---
if 'logged_in' not in st.session_state: st.session_state.logged_in = False
if 'username' not in st.session_state: st.session_state.username = ""
if 'ping_results' not in st.session_state: st.session_state.ping_results = None
if 'monitor_results' not in st.session_state: st.session_state.monitor_results = None
if 'speedtest_results' not in st.session_state: st.session_state.speedtest_results = None

# --- Pantalla de Login ---
if not st.session_state.logged_in:
    col1_login, col2_login = st.columns([1, 5])
    with col1_login: st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/c/c1/Google_%22G%22_logo.svg/120px-Google_%22G%22_logo.svg.png", width=80)
    with col2_login: st.title("Herramienta Diagnóstico de Red"); st.caption("Acceso Profesional")
    st.write("")
    with st.form("login_form"):
        st.subheader("🔑 Iniciar Sesión")
        username_introducido = st.text_input("👤 Usuario", key="login_user")
        password_introducida = st.text_input("🔒 Contraseña", type="password", key="login_pass")
        submitted = st.form_submit_button("➡️ Entrar")
        if submitted:
            if username_introducido in USUARIOS_VALIDOS and USUARIOS_VALIDOS[username_introducido] == password_introducida:
                st.session_state.logged_in = True; st.session_state.username = username_introducido
                for key in ['ping_results', 'monitor_results', 'speedtest_results']:
                    if key in st.session_state: del st.session_state[key]
                st.rerun()
            else: st.error("❌ Usuario o contraseña incorrectos.")

# --- Aplicación Principal ---
else:
    # --- Barra Lateral ---
    with st.sidebar:
        st.title(f"🌐 Diagnóstico Red"); st.success(f"✅ Conectado: **{st.session_state.username}**")
        # Assuming CEST is UTC+2. Adjust if needed for standard time or different timezone rules.
        now_spain = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=2)))
        st.info(f"🕒 Hora (ES): {now_spain.strftime('%Y-%m-%d %H:%M:%S')}")
        st.markdown('<hr class="custom-hr" style="margin: 1rem 0;">', unsafe_allow_html=True)
        if st.button("🚪 Cerrar Sesión", type="secondary"):
            st.session_state.logged_in = False; st.session_state.username = ""
            for key in ['ping_results', 'monitor_results', 'speedtest_results']:
                if key in st.session_state: del st.session_state[key]
            st.rerun()
        st.markdown('<hr class="custom-hr" style="margin: 1rem 0;">', unsafe_allow_html=True)
        st.caption("©️ 2025 - Herramienta Prototipo v1.3") # v1.3

    # --- Contenido Principal con Pestañas ---
    st.header(f"Panel de Control de Red")
    tab1, tab2, tab3 = st.tabs([
        "📊 Monitor Tráfico Local",
        "↔️ Comprobación Conexión (Ping)",
        "💨 Test Velocidad Internet"
    ])

    # Pestaña 1: Monitor Local
    with tab1:
        st.subheader("Monitorizar Actividad de Red Local")
        st.caption(f"Analiza la tasa de Bytes/s en tiempo real ({DURACION_MONITORIZACION_S} seg.) y detecta anomalías.")
        if st.button(f"⏱️ Iniciar Monitorización Local", key="start_monitor_tab1"):
            st.session_state.monitor_results = None
            tasas_bps_lista = []
            with st.status(f"Ejecutando monitorización local ({DURACION_MONITORIZACION_S}s)...", expanded=True) as status:
                try:
                    st.write("Obteniendo contadores iniciales...")
                    last_counters = psutil.net_io_counters(); last_time = time.time()
                    if not last_counters: raise Exception("No se pudieron obtener contadores iniciales.")
                    progress_bar_monitor = st.progress(0, text="Iniciando...")
                    for i in range(DURACION_MONITORIZACION_S):
                        progress_text = f"Monitorizando segundo {i+1}/{DURACION_MONITORIZACION_S}..."; st.write(progress_text); time.sleep(1)
                        current_counters = psutil.net_io_counters(); current_time = time.time()
                        if not current_counters: tasas_bps_lista.append(np.nan); continue
                        delta_time = current_time - last_time
                        delta_bytes = (current_counters.bytes_sent + current_counters.bytes_recv) - (last_counters.bytes_sent + last_counters.bytes_recv)
                        if delta_time > 0.1 and delta_bytes >= 0: tasas_bps_lista.append(delta_bytes / delta_time)
                        else: tasas_bps_lista.append(0)
                        last_counters = current_counters; last_time = current_time
                        progress_bar_monitor.progress((i + 1) / DURACION_MONITORIZACION_S, text=progress_text)
                    status.update(label="✔️ Monitorización Local Completada", state="complete", expanded=False)
                    if tasas_bps_lista:
                        serie_tasas_numeric = pd.to_numeric(np.array(tasas_bps_lista), errors='coerce')
                        anomalias_indices = detectar_anomalias_serie(serie_tasas_numeric)
                        st.session_state.monitor_results = {"serie_tasas": serie_tasas_numeric, "anomalias_indices": anomalias_indices}
                    else: st.warning("⚠️ No se recogieron datos."); st.session_state.monitor_results = None
                except Exception as e:
                    status.update(label="❌ Error en Monitorización", state="error", expanded=True); st.error(f"Detalle: {e}")
                    st.session_state.monitor_results = None

        if st.session_state.monitor_results:
            st.markdown('<hr class="custom-hr">', unsafe_allow_html=True)
            st.subheader("Resultados del Monitor Local")
            serie = st.session_state.monitor_results["serie_tasas"]; indices_anomalos = st.session_state.monitor_results["anomalias_indices"]
            # Llamada a la función ya corregida
            figura_plotly = crear_grafico_plotly_tasa(serie, indices_anomalos)
            if figura_plotly: st.plotly_chart(figura_plotly, use_container_width=True)
            else: st.warning("⚠️ No se pudo generar el gráfico.")

            with st.expander("🔍 Ver Detalles de Anomalías Detectadas"):
                indices_validos = [idx for idx in indices_anomalos if idx < len(serie)]
                if len(indices_validos) > 0:
                    st.write("**Segundos con actividad anómala y sugerencias:**")
                    for idx in indices_validos:
                        valor_tasa = serie[idx]
                        if pd.notna(valor_tasa):
                            solucion = sugerir_solucion_tasa(valor_tasa)
                            col1_exp, col2_exp = st.columns([1, 3])
                            with col1_exp:
                                st.warning(f"Seg. *{idx+1}* ➡️ `{valor_tasa:,.0f}` B/s")
                            with col2_exp:
                                st.info(f"💡 {solucion}")
                        else: # Manejo de NaN
                            col1_exp, col2_exp = st.columns([1, 3])
                            with col1_exp:
                                st.warning(f"Seg. *{idx+1}* ➡️ `NaN`")
                            with col2_exp:
                                st.info("Valor no numérico.")
                else:
                    st.success("✅ No se detectaron anomalías significativas.")
        else:
            st.info("ℹ️ Inicia la monitorización para ver resultados.")


    # Pestaña 2: Ping
    with tab2:
        st.subheader("Comprobación de Conexión (Ping)")
        st.caption("Mide la latencia y estabilidad hacia un servidor específico.")
        target_host_ping = st.text_input("🌐 Host o IP destino:", value="8.8.8.8", key="ping_target_tab2")
        if st.button("🚀 Realizar Prueba de Ping", key="start_ping_tab2"):
            st.session_state.ping_results = None
            if target_host_ping:
                with st.status(f"📡 Enviando pings a {target_host_ping}...", expanded=False) as status_ping:
                    ping_result_data = realizar_ping(target_host_ping); st.session_state.ping_results = ping_result_data
                    if ping_result_data: status_ping.update(label="✔️ Prueba Ping Completada", state="complete")
                    else: status_ping.update(label="❌ Error en Prueba Ping", state="error")
            else: st.warning("⚠️ Introduce un Host o IP.")
        if st.session_state.ping_results:
            st.markdown('<hr class="custom-hr">', unsafe_allow_html=True)
            st.subheader("Resultados del Ping")
            results_ping = st.session_state.ping_results
            avg_ms = getattr(results_ping, 'rtt_avg_ms', float('inf')); max_ms = getattr(results_ping, 'rtt_max_ms', float('inf')); loss = getattr(results_ping, 'packet_loss', 1.0)
            col_ping1, col_ping2, col_ping3 = st.columns(3)
            with col_ping1: st.markdown(create_metric_card("Latencia Media ms", avg_ms, "⏱️", "ping-avg"), unsafe_allow_html=True)
            with col_ping2: st.markdown(create_metric_card("Latencia Máxima ms", max_ms, "🐢", "ping-max"), unsafe_allow_html=True)
            with col_ping3: st.markdown(create_metric_card("Paquetes Perdidos", loss, "💔", "ping-loss"), unsafe_allow_html=True)
            with st.expander("💡 Ver Interpretación y Sugerencias del Ping"):
                velocidad = "Indeterminada"; sugerencia_ping = "No hay sugerencias."
                if loss > PERDIDA_PAQUETES_MAX_PERMITIDA:
                    velocidad = f"🔴 FALLO (> {PERDIDA_PAQUETES_MAX_PERMITIDA:.0%})"; sugerencia_ping = "Pérdida alta: Problemas serios. Sug: Reinicia, verifica, contacta ISP."; st.error(f"**Estado:** {velocidad}"); st.info(f"**Sugerencia:** {sugerencia_ping}")
                elif avg_ms == float('inf'):
                    velocidad = f"❓ INALCANZABLE"; sugerencia_ping = "Host no responde. Verifica IP/conexión/firewall."; st.error(f"**Estado:** {velocidad}"); st.info(f"**Sugerencia:** {sugerencia_ping}")
                elif avg_ms > LATENCIA_ACEPTABLE_MS:
                    velocidad = f"🐌 LENTA (> {LATENCIA_ACEPTABLE_MS} ms)"; sugerencia_ping = f"Latencia alta: Congestión/server lento. Sug: Reinicia, cierra apps, contacta ISP."; st.warning(f"**Estado:** {velocidad}"); st.info(f"**Sugerencia:** {sugerencia_ping}")
                elif avg_ms > LATENCIA_RAPIDA_MS:
                    velocidad = f"👍 ACEPTABLE ({LATENCIA_RAPIDA_MS}-{LATENCIA_ACEPTABLE_MS} ms)"; sugerencia_ping = "Latencia normal."; st.success(f"**Estado:** {velocidad}"); st.info(f"**Sugerencia:** {sugerencia_ping}")
                else:
                    velocidad = f"🚀 RÁPIDA (≤ {LATENCIA_RAPIDA_MS} ms)"; sugerencia_ping = "Latencia excelente."; st.success(f"**Estado:** {velocidad}"); st.info(f"**Sugerencia:** {sugerencia_ping}")
        else: st.info("ℹ️ Realiza una prueba de ping para ver resultados.")


    # Pestaña 3: Speedtest
    with tab3:
        st.subheader("Test de Velocidad de Conexión a Internet")
        st.caption("Mide tu velocidad real usando Speedtest.net (~30 seg).")
        st.warning("⚠️ Necesitas `pip install speedtest-cli`.", icon="⚙️")
        if st.button("💨 Iniciar Test de Velocidad", key="start_speedtest_tab3"):
            st.session_state.speedtest_results = None
            with st.status("⚙️ Ejecutando test de velocidad...", expanded=False) as status_speed:
                speed_results_data = realizar_speedtest(); st.session_state.speedtest_results = speed_results_data
                if speed_results_data: status_speed.update(label="✔️ Test de Velocidad Completado", state="complete")
                else: status_speed.update(label="❌ Error en Test de Velocidad", state="error")
        if st.session_state.speedtest_results:
            st.markdown('<hr class="custom-hr">', unsafe_allow_html=True)
            st.subheader("Resultados del Test de Velocidad")
            results_speed = st.session_state.speedtest_results
            download_mbps = results_speed.get('download', 0) / 1_000_000; upload_mbps = results_speed.get('upload', 0) / 1_000_000; ping_ms_speed = results_speed.get('ping', 0)
            col_spd1, col_spd2, col_spd3 = st.columns(3)
            with col_spd1: st.markdown(create_metric_card("Velocidad Descarga Mbps", download_mbps, "⬇️", "speed-dl"), unsafe_allow_html=True)
            with col_spd2: st.markdown(create_metric_card("Velocidad Subida Mbps", upload_mbps, "⬆️", "speed-ul"), unsafe_allow_html=True)
            with col_spd3: st.markdown(create_metric_card("Ping (Test Server) ms", ping_ms_speed, "↔️", "speed-ping"), unsafe_allow_html=True)
            with st.expander("📄 Ver Detalles del Test y Evaluación"):
                server_info = results_speed.get('server', {}); client_info = results_speed.get('client', {})
                server_loc = server_info.get('location', 'N/A') + ", " + server_info.get('country', 'N/A')
                st.write(f"**Servidor Test:** {server_info.get('name', 'N/A')} ({server_loc})")
                st.write(f"**Tu IP (Detectada):** {client_info.get('ip', 'N/A')} ({client_info.get('isp', 'N/A')})")
                st.write(f"**Fecha/Hora Test:** {results_speed.get('timestamp', 'N/A')}") # Consider formatting this timestamp
                st.markdown("**Evaluación General:**")
                if download_mbps == 0 and upload_mbps == 0: st.error("🔴 No se obtuvieron resultados.")
                else:
                    if download_mbps < 10: st.warning("Descarga baja (<10 Mbps).")
                    elif download_mbps < 50: st.info("Descarga moderada (10-50 Mbps).")
                    else: st.success("Descarga buena/excelente (≥50 Mbps).")
                    if upload_mbps < 2: st.warning("Subida baja (<2 Mbps).")
                    elif upload_mbps < 10: st.info("Subida moderada (2-10 Mbps).")
                    else: st.success("Subida buena/excelente (≥10 Mbps).")
        else: st.info("ℹ️ Inicia un test de velocidad para ver resultados.")

# --- Fin del Script ---