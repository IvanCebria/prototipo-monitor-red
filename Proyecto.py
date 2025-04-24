# Importar librerías
import streamlit as st
import numpy as np
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
import time

# --- Configuración del Modelo ---
CONTAMINACION_ESPERADA = 'auto' # Proporción esperada de anomalías
ESTADO_ALEATORIO = 42         # Para reproducibilidad

# --- Funciones Auxiliares ---

def detectar_anomalias(datos):
    """Aplica Isolation Forest y devuelve datos, predicciones e índices anómalos."""
    # Comprobar si hay datos suficientes para evitar errores
    if datos is None or len(datos) < 2: # Isolation Forest necesita al menos 2 puntos
        return datos, np.array([]), []
    # Asegurarse de que los datos son numéricos (podría fallar con datos del text_area mal formateados)
    try:
        datos_numeric = np.array(datos, dtype=float)
    except ValueError:
        st.error("Error: Los datos contienen valores no numéricos.")
        return datos, np.array([]), []

    X = datos_numeric.reshape(-1, 1)
    try:
        modelo_iforest = IsolationForest(contamination=CONTAMINACION_ESPERADA,
                                         random_state=ESTADO_ALEATORIO)
        modelo_iforest.fit(X)
        predicciones = modelo_iforest.predict(X)
        # Obtener los índices donde la predicción es -1 (anomalía)
        anomalias_indices = np.where(predicciones == -1)[0]
        return datos_numeric, predicciones, anomalias_indices
    except Exception as e:
        st.error(f"Error durante el análisis de ML: {e}")
        return datos_numeric, np.array([]), []


def sugerir_solucion(valor_anomalo):
    """Devuelve un texto de solución basado en reglas simples para tráfico de red."""
    if not isinstance(valor_anomalo, (int, float)):
         return "Valor anómalo no numérico." # Añadir chequeo básico

    sugerencia = ""
    if valor_anomalo > 500000: # Umbral para pico extremo
        sugerencia = "Pico extremo de tráfico (>500KB/s). Posible DDoS o transferencia masiva. Sugerencias: Revisar IPs origen/destino (NetFlow/logs), aplicar ACLs/filtros, verificar actividad servidores."
    elif valor_anomalo < 1000: # Umbral para caída drástica
        sugerencia = "Caída drástica de tráfico (<1KB/s). Posible fallo de enlace/dispositivo. Sugerencias: Verificar estado físico interfaz, conectividad (ping), logs del dispositivo."
    elif valor_anomalo > 100000: # Umbral para tráfico elevado sostenido
         sugerencia = "Tráfico elevado sostenido (>100KB/s). Posible congestión o nueva carga. Sugerencias: Identificar aplicaciones/servicios responsables, revisar QoS, evaluar necesidad de más ancho de banda."
    else:
        # Regla por defecto para otras anomalías numéricas detectadas
        sugerencia = "Patrón de tráfico inusual detectado. Sugerencias: Analizar logs y métricas detalladas (NetFlow, SNMP) del periodo."
    return sugerencia

def crear_grafico(datos, anomalias_indices):
    """Crea y devuelve la figura de Matplotlib con el gráfico."""
    # Crear figura y ejes
    fig, ax = plt.subplots(figsize=(12, 6)) # Usar fig y ax es buena práctica

    # Comprobar si hay datos para graficar
    if datos is None or len(datos) == 0:
        ax.set_title("No hay datos para mostrar")
        return fig # Devuelve figura vacía o con mensaje

    indices_tiempo = np.arange(len(datos))
    anomalias_valores = datos[anomalias_indices]

    # Dibujar línea de datos
    ax.plot(indices_tiempo, datos, marker='.', linestyle='-', color='cornflowerblue', label='Tráfico de Red (Bytes/s)')

    # Marcar las anomalías si las hay
    if len(anomalias_indices) > 0:
        ax.scatter(anomalias_indices, anomalias_valores, color='red', s=100, label='Anomalía Detectada', zorder=5)

    # Configurar el gráfico
    ax.set_title('Detección de Anomalías de Red (Isolation Forest) - Prototipo')
    ax.set_xlabel('Índice de Tiempo (Simulado)')
    ax.set_ylabel('Tráfico de Red (Bytes/s)')
    ax.legend()
    ax.grid(True)
    # Usar escala logarítmica si ayuda a la visualización
    if np.any(datos > 0): # Evitar error si todos los datos son 0 o negativos
        ax.set_yscale('log')

    return fig # Devuelve el objeto figura

# --- Interfaz Gráfica con Streamlit ---

st.set_page_config(layout="wide") # Usar ancho completo de la página
st.title("📊 Prototipo: Monitorización de Red con ML (Simplificado)")
st.write(f"Fecha y hora actual: {time.strftime('%Y-%m-%d %H:%M:%S')} CEST") # Hora actual

# Usar st.session_state para guardar los datos y resultados entre interacciones
if 'datos' not in st.session_state:
    # Datos simulados iniciales
    st.session_state.datos = np.array([
        25000, 30000, 28000, 35000, 40000, 38000, 42000, 39000, 150000,
        160000, 45000, 40000, 38000, 950000, 41000, 37000, 39000, 43000,
        500, 35000, 38000, 40000, 41000
    ])
if 'resultados' not in st.session_state:
    st.session_state.resultados = None

st.subheader("1. Datos de Red Simulados")
# Permitir al usuario ver/editar los datos para probar
datos_entrada_str = ", ".join(map(str, st.session_state.datos)) # Convertir array a string
nuevo_datos_str = st.text_area("Valores simulados (Bytes/s, separados por coma):",
                               datos_entrada_str, height=100)
# Intentar actualizar los datos desde el text_area
try:
    # Quitar espacios extra y convertir a entero, ignorar si está vacío
    nuevos_datos_lista = [int(x.strip()) for x in nuevo_datos_str.split(',') if x.strip()]
    if nuevos_datos_lista: # Solo actualizar si no está vacío
         st.session_state.datos = np.array(nuevos_datos_lista)
except ValueError:
    st.error("¡Error! Asegúrate de que todos los valores sean números enteros separados por comas.")


st.subheader("2. Análisis de Anomalías")

# Botón para iniciar el análisis
if st.button("🔍 Analizar Datos Simulados"):
    # Ejecutar análisis solo si hay datos
    if 'datos' in st.session_state and len(st.session_state.datos) > 0:
        with st.spinner("Analizando..."): # Mostrar indicador de carga
            datos, preds, anom_idx = detectar_anomalias(st.session_state.datos)
            # Guardar los resultados en el estado de la sesión
            st.session_state.resultados = {
                "datos": datos,
                "predicciones": preds, # Guardamos predicciones por si las necesitamos
                "anomalias_indices": anom_idx
            }
        st.success("¡Análisis completado!")
    else:
        st.warning("No hay datos para analizar. Introduce datos en el área de texto.")

st.subheader("3. Resultados del Análisis")

# Mostrar los resultados solo si el análisis se ha realizado
if st.session_state.resultados:
    st.write("Visualización de los datos y anomalías detectadas:")
    # Crear y mostrar el gráfico usando st.pyplot
    figura_grafico = crear_grafico(st.session_state.resultados["datos"],
                                   st.session_state.resultados["anomalias_indices"])
    st.pyplot(figura_grafico)

    st.markdown("---") # Separador visual
    st.write("**Detalle de Anomalías y Sugerencias de Solución:**")
    indices_anomalos = st.session_state.resultados["anomalias_indices"]

    # Mostrar las sugerencias si se encontraron anomalías
    if len(indices_anomalos) > 0:
        for idx in indices_anomalos:
            valor = st.session_state.resultados["datos"][idx]
            solucion = sugerir_solucion(valor)

            # Mostrar cada anomalía y su sugerencia
            st.warning(f"**Anomalía Detectada en Índice {idx}:** Valor = {valor:,} Bytes/s") # Formato con comas
            st.info(f"**Sugerencia:** {solucion}")
            st.markdown("---") # Separador entre anomalías
    else:
        # Mensaje si no hubo anomalías
        st.success("✅ No se detectaron anomalías significativas en estos datos.")
else:
    # Mensaje inicial antes del primer análisis
    st.info("Haz clic en 'Analizar Datos Simulados' para ver los resultados.")


# --- Para ejecutar esta aplicación ---
# 1. Guarda este código en un archivo llamado, por ejemplo, `app_monitor_web.py`.
# 2. Abre tu terminal.
# 3. Asegúrate de tener activado tu entorno virtual (si usas uno).
# 4. Navega a la carpeta donde guardaste el archivo.
# 5. Ejecuta el comando: streamlit run app_monitor_web.py
# 6. Se abrirá automáticamente una pestaña en tu navegador web con la interfaz.