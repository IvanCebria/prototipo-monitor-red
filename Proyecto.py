# Importar librerías
import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
import plotly.express as px
import time
from pythonping import ping # Para la prueba de latencia
import psutil # Para estadísticas de red del sistema

# --- Configuración ---
CONTAMINACION_ESPERADA = 'auto' # O un valor como 0.05, 0.1
ESTADO_ALEATORIO = 42
DURACION_MONITORIZACION_S = 15 # Segundos para monitorizar con psutil
USUARIOS_VALIDOS = {
    "Ivan123": "Ivan123", # <-- ¡Cambia esto!
    "Marcos123": "Marcos123"
}
LATENCIA_RAPIDA_MS = 80
LATENCIA_ACEPTABLE_MS = 200
PERDIDA_PAQUETES_MAX_PERMITIDA = 0.5

# --- Funciones Auxiliares ---

def detectar_anomalias_serie(serie_datos):
    """Aplica Isolation Forest a una serie/array de datos numéricos."""
    if serie_datos is None or len(serie_datos) < 5:
        # st.warning("No hay suficientes datos para un análisis de anomalías fiable.") # Quitado para no repetir
        return np.array([])
    try:
        X = np.array(serie_datos).reshape(-1, 1)
        # Asegurarse de que no haya NaNs o Infs que rompan el modelo
        X_clean = X[np.isfinite(X)].reshape(-1, 1)
        if len(X_clean) < 5: return np.array([]) # Volver a comprobar tras limpiar

        modelo = IsolationForest(contamination=CONTAMINACION_ESPERADA, random_state=ESTADO_ALEATORIO).fit(X_clean)
        # Predecir sobre los datos originales (X), manejando NaNs si los hubiera al predecir
        # O mejor, devolver índices basados en la longitud de X_clean y mapearlos luego?
        # Por simplicidad ahora, asumimos que X no tiene NaNs problemáticos aquí
        # o que la función que llama maneja el mapeo de índices si limpiamos.
        # La forma más segura sería devolver las predicciones para X_clean y mapear.
        # Vamos a hacerlo simple por ahora: predecir sobre X original si es posible.
        predicciones = modelo.predict(X) # Puede fallar si X tiene NaN/inf
        anomalias_indices = np.where(predicciones == -1)[0]
        return anomalias_indices
    except ValueError as ve:
        # Capturar error común si hay NaN/Inf
        if "Input contains NaN, infinity or a value too large" in str(ve):
             st.warning("Advertencia ML: Los datos de tasa contienen valores no válidos (NaN/Inf). No se pudo realizar la detección de anomalías.")
             return np.array([])
        else:
             st.error(f"Error de valor durante el análisis de ML: {ve}")
             return np.array([])
    except Exception as e:
        st.error(f"Error durante el análisis de ML sobre la serie: {e}")
        return np.array([])

def sugerir_solucion_tasa(tasa_bps):
    """Devuelve sugerencias basadas en la tasa de Bytes/segundo."""
    if not isinstance(tasa_bps, (int, float)) or not np.isfinite(tasa_bps):
         return "Valor de tasa inválido."

    # Ajusta estos umbrales según tu red y lo que consideres normal/anómalo
    if tasa_bps > 10 * 1024 * 1024: # > 10 MB/s
        sugerencia = f"Tasa MUY ALTA ({tasa_bps / (1024*1024):.1f} MB/s). Causa: Descarga/subida intensiva, streaming HD/4K, backup, ¿proceso malicioso consumiendo red?"
    elif tasa_bps > 1 * 1024 * 1024: # > 1 MB/s
        sugerencia = f"Tasa ELEVADA ({tasa_bps / (1024*1024):.1f} MB/s). Causa: Actividad de red considerable (navegación, video, actualizaciones), ¿algo inesperado?"
    elif tasa_bps < 1 * 1024 and tasa_bps >=0 : # < 1 KB/s
        sugerencia = f"Tasa MUY BAJA ({tasa_bps / 1024:.1f} KB/s). Causa: Poca actividad (normal si no se usa la red), ¿problema de conexión si se esperaba tráfico?"
    else: # Tasas intermedias consideradas "normales" en este ejemplo
        sugerencia = f"Tasa de actividad local: {tasa_bps / (1024*1024):.2f} MB/s. (Considerada normal en este rango)."

    # Podríamos añadir lógica extra aquí si supiéramos más contexto
    return sugerencia

def crear_grafico_plotly_tasa(serie_tasas, anomalias_indices):
    """Crea gráfico Plotly para la serie de tasas Bps."""
    if serie_tasas is None or len(serie_tasas) == 0:
        return None
    df = pd.DataFrame({'Segundo': np.arange(len(serie_tasas)), 'Tasa (Bytes/s)': serie_tasas})
    df['Estado'] = 'Normal'
    if len(anomalias_indices) > 0: df.loc[anomalias_indices, 'Estado'] = 'Anomalía'
    fig = px.line(df, x='Segundo', y='Tasa (Bytes/s)', title=f'Actividad de Red Local Durante {len(serie_tasas)} Segundos', markers=True, labels={'Segundo': 'Segundo', 'Tasa (Bytes/s)': 'Tasa (Bytes/s)'})
    df_anomalias = df[df['Estado'] == 'Anomalía']
    if not df_anomalias.empty: fig.add_scatter(x=df_anomalias['Segundo'], y=df_anomalias['Tasa (Bytes/s)'], mode='markers', marker=dict(color='red', size=10), name='Anomalía Detectada', hovertemplate="Seg: %{x}<br>Tasa: %{y:,.0f} B/s<br>Anomalía<extra></extra>")
    fig.update_traces(hovertemplate="Seg: %{x}<br>Tasa: %{y:,.0f} B/s<br>Normal<extra></extra>")
    fig.update_layout(hovermode='x unified')
    max_val = df['Tasa (Bytes/s)'].max()
    if max_val > 0: fig.update_yaxes(type="log" if max_val > 10000 else "linear") # Escala log solo si hay valores grandes
    return fig

def realizar_ping(host, count=4):
    """Realiza un ping y devuelve resultados."""
    # ... (Igual que antes) ...
    try:
        resultado_ping = ping(host, count=count, verbose=False, timeout=2)
        return resultado_ping
    except PermissionError: st.error(f"Error Permisos Ping"); return None
    except Exception as e: st.error(f"Error Ping a {host}: {e}"); return None

# --- Lógica de la Aplicación Streamlit ---
st.set_page_config(layout="wide", page_title="Monitor Red ML v5")

# Inicializar estados
if 'logged_in' not in st.session_state: st.session_state.logged_in = False
if 'username' not in st.session_state: st.session_state.username = ""
if 'ping_results' not in st.session_state: st.session_state.ping_results = None
if 'monitor_results' not in st.session_state: st.session_state.monitor_results = None

# --- Pantalla de Login ---
if not st.session_state.logged_in:
    # ... (Mismo código de login que antes) ...
    st.title("🔒 Acceso Requerido - Prototipo Monitorización")
    st.write("Introduce tus credenciales para acceder.")
    with st.form("login_form"):
        username_introducido = st.text_input("👤 Usuario")
        password_introducida = st.text_input("🔑 Contraseña", type="password")
        submitted = st.form_submit_button("Entrar")
        if submitted:
            if username_introducido in USUARIOS_VALIDOS and USUARIOS_VALIDOS[username_introducido] == password_introducida:
                st.session_state.logged_in = True; st.session_state.username = username_introducido; st.rerun()
            else: st.error("Usuario o contraseña incorrectos.")

# --- Aplicación Principal ---
else:
    # Barra Lateral
    with st.sidebar:
        # ... (Info usuario, hora, logout) ...
        st.success(f"✅ Conectado como: {st.session_state.username}")
        st.write(f"Hora: {time.strftime('%Y-%m-%d %H:%M:%S')} CEST")
        if st.button("🚪 Cerrar Sesión"):
            for key in list(st.session_state.keys()):
                if key not in ['logged_in', 'username']: del st.session_state[key]
            st.session_state.logged_in = False; st.session_state.username = ""; st.rerun()
        st.markdown("---")
        st.header("Herramientas de Red")

    # Contenido Principal
    st.title(" Módulos de Análisis de Red")

    # --- Módulo 1: Monitorización de Actividad de Red Local (psutil) ---
    st.header("1. Monitorizar Actividad de Red Local")
    st.caption(f"Recopila la tasa de Bytes/s de tu máquina durante {DURACION_MONITORIZACION_S} segundos y busca anomalías.")

    if st.button(f"⏱️ Iniciar Monitorización ({DURACION_MONITORIZACION_S} segundos)"):
        st.session_state.monitor_results = None # Limpiar resultados anteriores
        tasas_bps_lista = []
        # Usar una barra de progreso y un placeholder para mensajes
        progress_bar = st.progress(0, text="Iniciando monitorización...")
        status_text = st.empty() # Placeholder para mensajes durante el bucle

        try:
            # El spinner es menos útil si actualizamos texto con la barra
            #with st.spinner(f"Monitorizando durante {DURACION_MONITORIZACION_S} segundos... ¡La app se bloqueará!"):
            last_counters = psutil.net_io_counters()
            last_time = time.time()
            if not last_counters: raise Exception("No se pudieron obtener contadores iniciales.")

            # Bucle de monitorización (BLOQUEANTE)
            for i in range(DURACION_MONITORIZACION_S):
                status_text.text(f"Monitorizando segundo {i+1}/{DURACION_MONITORIZACION_S}... Por favor, espera.")
                time.sleep(1) # Esperar 1 segundo
                current_counters = psutil.net_io_counters()
                current_time = time.time()
                if not current_counters:
                    tasas_bps_lista.append(np.nan) # Marcar como NaN si falla la lectura
                    continue

                delta_time = current_time - last_time
                # Manejar posible reinicio de contadores (delta_bytes < 0)
                delta_bytes = (current_counters.bytes_sent + current_counters.bytes_recv) - \
                              (last_counters.bytes_sent + last_counters.bytes_recv)

                if delta_time > 0.1 and delta_bytes >= 0: # Evitar división por cero y deltas negativos
                    tasa_actual_bps = delta_bytes / delta_time
                    tasas_bps_lista.append(tasa_actual_bps)
                else:
                     tasas_bps_lista.append(0) # Añadir 0 si no hay tiempo o delta raro

                last_counters = current_counters; last_time = current_time
                progress_bar.progress((i + 1) / DURACION_MONITORIZACION_S)

            progress_bar.empty(); status_text.empty() # Limpiar progreso y texto
            st.success(f"Monitorización completada. {len(tasas_bps_lista)} tasas Bps/s recogidas.")

            if tasas_bps_lista:
                serie_tasas = np.array(tasas_bps_lista)
                with st.spinner("🤖 Aplicando detección de anomalías..."):
                     anomalias_indices = detectar_anomalias_serie(serie_tasas)
                     st.session_state.monitor_results = {"serie_tasas": serie_tasas, "anomalias_indices": anomalias_indices}
            else: st.warning("No se recogieron datos de tasa."); st.session_state.monitor_results = None

        except Exception as e:
            progress_bar.empty(); status_text.empty()
            st.error(f"Error durante la monitorización con psutil: {e}")
            st.session_state.monitor_results = None

    # Mostrar resultados de la monitorización psutil si existen
    if st.session_state.monitor_results:
        st.subheader("Resultados de la Monitorización de Actividad")
        serie = st.session_state.monitor_results["serie_tasas"]
        indices_anomalos = st.session_state.monitor_results["anomalias_indices"]

        st.write("**Visualización de Tasa (Bytes/s):**")
        figura_plotly = crear_grafico_plotly_tasa(serie, indices_anomalos)
        if figura_plotly: st.plotly_chart(figura_plotly, use_container_width=True)
        st.markdown("<br>", unsafe_allow_html=True)

        st.write("**Detalle de Segundos Anómalos y Sugerencias:**")
        if len(indices_anomalos) > 0:
            col1, col2 = st.columns([1, 3])
            with col1: st.write("**Segundo / Tasa Anómala (B/s)**")
            with col2: st.write("**Sugerencia de Solución**") # Cabecera

            for idx in indices_anomalos:
                # Asegurarse de que el índice es válido
                if idx < len(serie):
                    valor_tasa = serie[idx]
                    # Llamar a la función para obtener la sugerencia
                    solucion = sugerir_solucion_tasa(valor_tasa)
                    with col1:
                        # Mostrar segundo (base 1) y valor formateado
                        st.warning(f"*{idx+1}* ➡️ `{valor_tasa:,.0f}`")
                    with col2:
                        # Mostrar la sugerencia obtenida
                        st.info(f"{solucion}") # <--- AQUÍ SE MUESTRAN LAS SUGERENCIAS
                else:
                     with col1: st.error(f"Índice anómalo {idx} fuera de rango.")


        else:
            st.success("✅ No se detectaron segundos con tasas anómalas significativas en este periodo.")
    # else: # Quitamos este else para no ser repetitivo si no hay resultados aún
    #    st.info("Resultados de monitorización aparecerán aquí.")

    st.markdown("---") # Separador

    # --- Módulo 2: Prueba de Latencia (Ping) ---
    st.header("2. Prueba de Velocidad de Red (Latencia por Ping)")
    # ... (Mismo código del ping que antes) ...
    target_host = st.text_input("Host o IP para hacer Ping:", value="8.8.8.8", key="ping_target")
    if st.button("🚀 Realizar Prueba de Ping"):
        if target_host:
            with st.spinner(f"Realizando ping a {target_host}..."): st.session_state.ping_results = realizar_ping(target_host)
            if st.session_state.ping_results: st.success("Prueba ping completada.")
        else: st.warning("Introduce Host/IP.")

    if st.session_state.ping_results:
        # ... (Mismo código de mostrar resultados y sugerencias del ping) ...
        results = st.session_state.ping_results; st.write("**Resultados del Ping:**")
        col_ping1, col_ping2, col_ping3 = st.columns(3)
        col_ping1.metric("Latencia Media", f"{results.rtt_avg_ms:.2f} ms"); col_ping2.metric("Latencia Máxima", f"{results.rtt_max_ms:.2f} ms"); col_ping3.metric("Paquetes Perdidos", f"{results.packet_loss:.1%}")
        velocidad = "Indeterminada"; sugerencia_ping = ""
        if results.packet_loss > PERDIDA_PAQUETES_MAX_PERMITIDA:
            velocidad = f"🔴 FALLO (> {PERDIDA_PAQUETES_MAX_PERMITIDA:.0%})"; sugerencia_ping = "Pérdida alta. Causas: Congestión severa, fallo HW, firewall. Sug: Reinicia equipos, verifica cables, prueba otros destinos, contacta ISP."
            st.error(f"⚠️ **Estado:** {velocidad}"); st.info(f"💡 **Sugerencia:** {sugerencia_ping}")
        elif results.rtt_avg_ms > LATENCIA_ACEPTABLE_MS:
            velocidad = f"🐌 LENTA (> {LATENCIA_ACEPTABLE_MS} ms)"; sugerencia_ping = f"Latencia alta. Causas: Congestión local/ISP, server lento/lejano. Sug: Reinicia router, cierra apps B/W, prueba otros pings, contacta ISP."
            st.warning(f"🚦 **Estado:** {velocidad}"); st.info(f"💡 **Sugerencia:** {sugerencia_ping}")
        elif results.rtt_avg_ms > LATENCIA_RAPIDA_MS:
            velocidad = f"👍 ACEPTABLE ({LATENCIA_RAPIDA_MS}-{LATENCIA_ACEPTABLE_MS} ms)"; st.info(f"✅ **Estado:** {velocidad}")
        else: velocidad = f"🚀 RÁPIDA (≤ {LATENCIA_RAPIDA_MS} ms)"; st.success(f"✅ **Estado:** {velocidad}")


# --- Fin ---
