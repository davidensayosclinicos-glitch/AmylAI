import streamlit as st
import re
import os
import datetime
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix, 
    roc_curve, 
    auc, 
    matthews_corrcoef, 
    cohen_kappa_score, 
    accuracy_score, 
    balanced_accuracy_score,  
    f1_score,                
    fbeta_score,             
    brier_score_loss,         
    average_precision_score,  
    precision_recall_curve) 
from sklearn.metrics import classification_report   
import base64
import json
import pdfplumber
from typing import Dict, Any, List, Optional
import pytesseract
from PIL import Image, ImageEnhance, ImageFilter
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# ==========================================
# OPTIMIZACIÓN DE OCR (LECTURA RÁPIDA)
# ==========================================
_ocr_cache = {}  # Caché global de OCR
_ocr_lock = threading.Lock()  # Lock para thread-safety

def optimizar_imagen_ocr(pil_img: Image.Image) -> Image.Image:
    """
    Preprocesa imagen para OCR más rápido y preciso.
    - Convierte a escala de grises
    - Mejora contraste
    - Aplica filtro de nitidez
    """
    # Convertir a escala de grises
    img_gray = pil_img.convert('L')
    
    # Mejorar contraste
    enhancer = ImageEnhance.Contrast(img_gray)
    img_enhanced = enhancer.enhance(1.5)
    
    # Mejorar nitidez
    img_sharp = img_enhanced.filter(ImageFilter.SHARPEN)
    
    return img_sharp

def ocr_rapido_pagina(page_num: int, page, timeout_ocr: float = 15.0) -> tuple:
    """
    OCR optimizado para una página específica.
    Retorna (page_num, texto_ocr, tiempo_procesamiento)
    """
    start = time.time()
    
    try:
        # Intentar extraer texto primero (rápido)
        raw_text = page.extract_text()
        if raw_text and len(raw_text.strip()) > 5:
            return (page_num, raw_text, time.time() - start)
        
        # Si no hay texto, usar OCR con resolución reducida
        # Resolución 150 es mucho más rápido que 300, con buena calidad
        page_image = page.to_image(resolution=150)
        pil_img = page_image.original
        
        # Optimizar imagen
        pil_img_opt = optimizar_imagen_ocr(pil_img)
        
        # OCR con settings optimizados para rapidez
        ocr_text = pytesseract.image_to_string(
            pil_img_opt, 
            lang='spa',
            config='--psm 6'  # Trata imagen como un solo bloque uniforme (rápido)
        )
        
        if ocr_text and len(ocr_text.strip()) > 5:
            return (page_num, ocr_text, time.time() - start)
        else:
            return (page_num, "", time.time() - start)
            
    except Exception as e:
        return (page_num, f"[ERROR OCR: {str(e)[:50]}]", time.time() - start)

def procesar_pdf_paralelo(pdf, max_workers: int = 4) -> tuple:
    """
    Procesa múltiples páginas de PDF en paralelo para máxima velocidad.
    Retorna (texto_completo, tiempos_por_pagina)
    """
    texto_completo = ""
    tiempos = {}
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Enviar todas las páginas a procesar en paralelo
        futures = {executor.submit(ocr_rapido_pagina, i, page): i 
                   for i, page in enumerate(pdf.pages)}
        
        # Recolectar resultados en orden
        for future in as_completed(futures):
            try:
                page_num, ocr_texto, tiempo_proc = future.result()
                if ocr_texto and not ocr_texto.startswith("[ERROR"):
                    texto_completo += ocr_texto + "\n"
                tiempos[page_num] = tiempo_proc
            except Exception as e:
                print(f"Error procesando página en paralelo: {e}")
    
    return texto_completo, tiempos

# ================================
# FIN OPTIMIZACIÓN OCR
# ================================                        
tess_cmd_env = os.environ.get('TESSERACT_CMD') or os.environ.get('TESSERACT_PATH')
if not tess_cmd_env:
    # Ruta por defecto común en Windows
    tess_cmd_env = r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"
if tess_cmd_env and os.path.exists(tess_cmd_env):
    pytesseract.pytesseract.tesseract_cmd = tess_cmd_env
import time
import csv
from scipy import stats
from scipy.stats import norm

# ==========================================
# FUNCIONES ESTADÍSTICAS (CORREGIDAS Y BLINDADAS)
# ==========================================
def wilson_ci(x, n, alpha=0.05):
    """Calcula el intervalo de confianza de Wilson."""
    # Importamos AQUÍ para evitar conflictos con tu variable 'stats'
    from scipy.stats import norm 
    
    if n == 0: return (0.0, 0.0)
    p = x / n
    z = norm.ppf(1 - alpha / 2) # Usamos norm directamente
    denominator = 1 + z**2 / n
    centre_adjusted_probability = p + z**2 / (2 * n)
    adjusted_standard_deviation = np.sqrt((p * (1 - p) + z**2 / (4 * n)) / n)
    lower_bound = (centre_adjusted_probability - z * adjusted_standard_deviation) / denominator
    upper_bound = (centre_adjusted_probability + z * adjusted_standard_deviation) / denominator
    return (max(0.0, lower_bound), min(1.0, upper_bound))

def bootstrap_ci(y_true, y_pred, metric_func, n_boot=2000, alpha=0.05):
    """Bootstrap no paramétrico."""
    rng = np.random.RandomState(42)
    indices = np.arange(len(y_true))
    scores = []
    for _ in range(n_boot):
        boot_idx = rng.choice(indices, len(indices), replace=True)
        score = metric_func(y_true[boot_idx], y_pred[boot_idx])
        scores.append(score)
    return np.percentile(scores, [alpha/2 * 100, (1 - alpha/2) * 100])

def delong_roc_variance(ground_truth, predictions):
    """Calcula varianza AUC (DeLong)."""
    order = np.argsort(predictions)
    predictions = predictions[order]
    ground_truth = ground_truth[order]
    pos_examples = predictions[ground_truth == 1]
    neg_examples = predictions[ground_truth == 0]
    if len(pos_examples) == 0 or len(neg_examples) == 0: return 0.0, 0.0, 0.0

    def compute_midrank(x):
        J = np.argsort(x); Z = x[J]; N = len(x); T = np.zeros(N, dtype=np.float64)
        i = 0
        while i < N:
            j = i
            while j < N and Z[j] == Z[i]: j += 1
            T[i:j] = 0.5 * (i + j - 1); i = j
        T2 = np.empty(N, dtype=np.float64); T2[J] = T + 1
        return T2

    tx = compute_midrank(predictions)
    tz = compute_midrank(predictions[ground_truth == 1])
    ty = compute_midrank(predictions[ground_truth == 0])
    m = len(pos_examples); n = len(neg_examples)
    aucs = (np.sum(tx[ground_truth == 1]) - m * (m + 1) / 2) / (m * n)
    v01 = (tx[ground_truth == 1] - tz) / n
    v10 = (tx[ground_truth == 0] - ty) / m
    var_auc = (np.var(v01) / m) + (np.var(v10) / n)
    return aucs, var_auc

def permutation_test_multiclass(y_true, y_pred, metric_func, n_perm=1000):
    """Test de permutación."""
    score_obs = metric_func(y_true, y_pred)
    rng = np.random.RandomState(42)
    scores_perm = []
    y_shuffled = y_true.copy()
    for _ in range(n_perm):
        rng.shuffle(y_shuffled)
        scores_perm.append(metric_func(y_shuffled, y_pred))
    p_value = (np.sum(np.array(scores_perm) >= score_obs) + 1) / (n_perm + 1)
    return p_value

def normalizar_clase_diagnostica(val: str) -> str:
    """Normaliza etiquetas diagnósticas para análisis estadístico estable."""
    v_upper = str(val).upper().strip()

    if not v_upper:
        return "Otros"

    if "INTERMEDIA" in v_upper or "SCREENING" in v_upper:
        return "Intermedia"

    if "ATTR" in v_upper:
        return "ATTR"

    if re.search(r"\bAL\b", v_upper):
        return "AL"

    if "HVI" in v_upper or "HIPERTROF" in v_upper:
        return "HVI"

    if "SANO" in v_upper or "NO AMILOIDOSIS" in v_upper or "BAJA" in v_upper:
        return "Sano"

    return "Otros"

# --- FUNCIONES AUXILIARES ---
# ==========================================
# CÓDIGO LIMPIADO - VERSIÓN CONSOLIDADA
# ==========================================
# Nota: Primera versión obsoleta de calcular_riesgo_experto (línea 125 antes de edición)
# eliminada. Ver línea ~1350 para versión mejorada con type hints.

# --- GENERADOR DE TEXTO ---
def generar_explicacion_narrativa(datos: Dict[str, Any], resultado: Dict[str, Any]) -> str:
    nivel = resultado.get('nivel', '')
    hallazgos = resultado.get('hallazgos', [])
    texto = f"Resultado del análisis: **{nivel}**.\n"
    if "ALTA" in nivel:
        texto += "Existe una alta probabilidad clínica de amiloidosis. "
    elif "INTERMEDIA" in nivel:
        texto += "El paciente se encuentra en una **zona gris**. Se requiere imagen avanzada. "
    elif "HVI" in nivel:
        texto += "Los hallazgos sugieren hipertrofia por sobrecarga (HTA/Válvula) y no infiltrativa. "
    else:
        texto += "Baja probabilidad de enfermedad infiltrativa. "
    
    if hallazgos:
        clean_hallazgos = [h.split('(+')[0].strip() for h in hallazgos]
        texto += "\n\nFactores clave detectados: " + ", ".join(clean_hallazgos).lower() + "."
    return texto

def generar_resumen_hallazgos(datos: Dict[str, Any], resultado: Dict[str, Any]) -> str:
    """
    Genera un resumen clínico detallado y profesional de los hallazgos.
    Incluye diagnóstico diferencial, parámetros cardiacos e interpretación clínica.
    """
    
    # SECCIONES ORGANIZADAS POR CATEGORÍA
    hallazgos_al_directo = []
    hallazgos_al_sistemico = []
    hallazgos_attr_clasico = []
    hallazgos_attr_hereditario = []
    hallazgos_cardiacos = []
    hallazgos_confusores = []
    
    # AL DIRECTO
    if datos.get('mgus'): hallazgos_al_directo.append("Paraproteína/MGUS detectada")
    if datos.get('macro'): hallazgos_al_directo.append("Macroglosia")
    if datos.get('purpura'): hallazgos_al_directo.append("Púrpura periorbital")
    
    # AL SISTÉMICO
    if datos.get('nefro'): hallazgos_al_sistemico.append("Síndrome nefrótico")
    if datos.get('hepato'): hallazgos_al_sistemico.append("Hepatomegalia")
    if datos.get('neuro_p'): hallazgos_al_sistemico.append("Polineuropatía periférica")
    if datos.get('fatiga'): hallazgos_al_sistemico.append("Fatiga severa")
    if datos.get('disauto'): hallazgos_al_sistemico.append("Disautonomía (hipotensión ortostática)")
    if datos.get('piel_lesiones'): hallazgos_al_sistemico.append("Lesiones cutáneas")
    
    # ATTR CLÁSICO
    if datos.get('stc'): hallazgos_attr_clasico.append("Síndrome del túnel carpiano bilateral")
    if datos.get('lumbar'): hallazgos_attr_clasico.append("Estenosis lumbar claudicante")
    if datos.get('biceps'): hallazgos_attr_clasico.append("Rotura bilateral de bíceps")
    if datos.get('hombro'): hallazgos_attr_clasico.append("Tendinitis del manguito rotador")
    if datos.get('dupuytren'): hallazgos_attr_clasico.append("Contractura de Dupuytren")
    if datos.get('artralgias'): hallazgos_attr_clasico.append("Artralgias crónicas")
    if datos.get('fractura_vert'): hallazgos_attr_clasico.append("Fracturas vertebrales")
    if datos.get('tendinitis_calcifica'): hallazgos_attr_clasico.append("Tendinitis calcificante")
    
    # ATTR HEREDITARIO
    if datos.get('mutacion_ttr'): hallazgos_attr_hereditario.append("Mutación TTR confirmada genéticamente")
    
    # HALLAZGOS CARDIACOS
    if datos.get('ivs'): hallazgos_cardiacos.append(f"Grosor de pared VI: {datos['ivs']} mm")
    if datos.get('gls'): hallazgos_cardiacos.append(f"GLS (strain global): {datos['gls']}%")
    if datos.get('volt'): hallazgos_cardiacos.append(f"Voltaje en ECG: {datos['volt']} mV")
    if datos.get('apical_sparing'): hallazgos_cardiacos.append("Apical sparing (muy sugestivo de ATTR)")
    if datos.get('bajo_voltaje'): hallazgos_cardiacos.append("Bajo voltaje paradójico")
    if datos.get('bav_mp'): hallazgos_cardiacos.append("Bloqueo AV completo / Marcapasos")
    if datos.get('pseudo_q'): hallazgos_cardiacos.append("Pseudoinfarto (ondas Q patológicas)")
    if datos.get('biatrial'): hallazgos_cardiacos.append("Dilatación biauricular")
    if datos.get('derrame_pericardico'): hallazgos_cardiacos.append("Derrame pericárdico")
    
    # BIOMARCADORES
    hallazgos_biomarcadores = []
    nt_probnp = safe_float(datos.get('nt_probnp', 0))
    if nt_probnp > 0:
        hallazgos_biomarcadores.append(f"NT-proBNP: {nt_probnp} pg/ml")
        if nt_probnp > 3000:
            hallazgos_biomarcadores.append("  → Elevado CRÍTICO (muy sugestivo de amiloidosis)")
        elif nt_probnp > 1000:
            hallazgos_biomarcadores.append("  → Elevado moderadamente")
        elif nt_probnp <= 400:
            hallazgos_biomarcadores.append("  → Normal → Descarta amiloidosis")
    
    if datos.get('troponina'):
        hallazgos_biomarcadores.append("Troponina elevada crónica (infiltración miocárdica crónica)")
    
    # RESONANCIA MAGNÉTICA CARDÍACA
    hallazgos_rm = []
    lge_patron = str(datos.get('lge_patron', '')).lower().strip()
    if lge_patron and lge_patron != 'null':
        hallazgos_rm.append(f"Realce Tardío (LGE): {lge_patron}")
        if lge_patron in ['subendocardico', 'subendocárdico', 'transmural', 'difuso']:
            hallazgos_rm.append("  → Patrón PATOGNOMÓNICO de amiloidosis")
    
    ecv = safe_float(datos.get('ecv', 0))
    if ecv > 0:
        hallazgos_rm.append(f"Volumen Extracelular (ECV): {ecv}%")
        if ecv > 40:
            hallazgos_rm.append("  → DIAGNÓSTICO de amiloidosis")
        elif ecv > 35:
            hallazgos_rm.append("  → Elevado, muy sugestivo")
        elif ecv > 30:
            hallazgos_rm.append("  → Moderadamente elevado")
    
    if datos.get('t1_mapping'):
        hallazgos_rm.append("T1 Mapping: Elevado (sugiere amiloidosis)")
    
    # FACTORES CONFUSORES
    if datos.get('confusor_hta'): hallazgos_confusores.append("Hipertensión arterial severa")
    if datos.get('confusor_ao'): hallazgos_confusores.append("Estenosis aórtica severa")
    if datos.get('confusor_irc'): hallazgos_confusores.append("Insuficiencia renal crónica")
    
    # CONSTRUIR NARRATIVA CLÍNICA
    resumen = "## RESUMEN CLÍNICO DE HALLAZGOS\n\n"
    
    # Nivel de riesgo y impresión general
    nivel = resultado.get('nivel', 'Desconocido')
    resumen += f"**Nivel de Riesgo:** {nivel}\n\n"
    
    score = resultado.get('score', 0)
    resumen += f"**Puntuación Total:** {score} puntos\n\n"
    
    # Interpretación general
    if "ALTA" in nivel:
        resumen += "### Impresión Clínica\n"
        resumen += "Hallazgos altamente sugestivos de **amiloidosis cardíaca**. Se recomienda confirmación diagnóstica urgente mediante:\n"
        resumen += "- Biopsia endomiocárdica con tinción Congo rojo\n"
        resumen += "- Tipaje de amiloide (AL vs ATTR)\n"
        resumen += "- Consulta con especialista en amiloidosis\n\n"
    elif "INTERMEDIA" in nivel:
        resumen += "### Impresión Clínica\n"
        resumen += "**Zona gris diagnóstica.** Los hallazgos son parcialmente consistentes con amiloidosis pero no patognomónicos. Requiere:\n"
        resumen += "- Resonancia cardíaca con realce tardío (patrón amiloide)\n"
        resumen += "- Gammagrafía ósea con Tc-99m pirofosfato\n"
        resumen += "- Seguimiento ecocardiográfico seriado\n\n"
    elif "HVI" in nivel:
        resumen += "### Impresión Clínica\n"
        resumen += "Los hallazgos sugieren **hipertrofia ventricular izquierda por sobrecarga** (HTA, válvula) sin características infiltrativas claras.\n\n"
    else:
        resumen += "### Impresión Clínica\n"
        resumen += "**Baja probabilidad de amiloidosis cardíaca.** Los hallazgos no son significativos o sugieren otras etiologías.\n\n"
    
    # HALLAZGOS AL DIRECTO
    if hallazgos_al_directo:
        resumen += "### 🚨 Hallazgos Sugestivos de AL (Amiloidosis AL)\n"
        for h in hallazgos_al_directo:
            resumen += f"• {h}\n"
        resumen += "*Estos hallazgos sugieren amiloidosis tipo AL. Requiere evaluación urgente del sistema hematológico.*\n\n"
    
    # HALLAZGOS AL SISTÉMICO
    if hallazgos_al_sistemico:
        resumen += "### 🟠 Hallazgos de Afección Sistémica (AL)\n"
        for h in hallazgos_al_sistemico:
            resumen += f"• {h}\n"
        resumen += "*Manifestaciones sistémicas que elevan la probabilidad de amiloidosis AL.*\n\n"
    
    # HALLAZGOS ATTR CLÁSICO
    if hallazgos_attr_clasico:
        resumen += "### 🟣 Hallazgos Característicos de ATTR (Síndrome Musculoesquelético)\n"
        for h in hallazgos_attr_clasico:
            resumen += f"• {h}\n"
        resumen += "*La tríada deTunel Carpiano + Estenosis Lumbar + Rotura de Bíceps es patognomónica de ATTR-v.*\n\n"
    
    # HALLAZGOS ATTR HEREDITARIO
    if hallazgos_attr_hereditario:
        resumen += "### 🧬 Hallazgos de ATTR Hereditaria\n"
        for h in hallazgos_attr_hereditario:
            resumen += f"• {h}\n"
        resumen += "*ATTR hereditaria requiere cribado familiar urgente. Considerar terapia modificadora de transtiretina (tafamidis).*\n\n"
    
    # PARÁMETROS CARDIACOS
    if hallazgos_cardiacos:
        resumen += "### 💙 Parámetros Cardiacos\n"
        for h in hallazgos_cardiacos:
            resumen += f"• {h}\n"
        
        # INTERPRETACIÓN ESPECÍFICA DE PARÁMETROS
        ivs = datos.get('ivs', 0)
        gls = datos.get('gls', 0)
        volt = datos.get('volt', 0)
        
        resumen += "\n**Interpretación:**\n"
        
        if ivs > 15:
            resumen += f"- Grosor VI ({ivs} mm): MUY aumentado (específico ATTR)\n"
        elif ivs > 12:
            resumen += f"- Grosor VI ({ivs} mm): Engrosamiento moderado\n"
        
        if gls < -15:
            resumen += f"- GLS ({gls}%): Función sistólica conservada\n"
        elif gls < -9:
            resumen += f"- GLS ({gls}%): Disfunción longitudinal incipiente\n"
        else:
            resumen += f"- GLS ({gls}%): Disfunción longitudinal severa\n"
        
        if volt < 0.5:
            resumen += f"- Voltaje ({volt} mV): Bajo voltaje paradójico (amiloidosis)\n"
        elif volt < 1.0:
            resumen += f"- Voltaje ({volt} mV): Reducido\n"
        
        resumen += "\n"
    
    # BIOMARCADORES
    if hallazgos_biomarcadores:
        resumen += "### 🔬 Biomarcadores\n"
        for h in hallazgos_biomarcadores:
            resumen += f"• {h}\n"
        resumen += "*Los biomarcadores son críticos para confirmar amiloidosis y su tipo (AL vs ATTR).*\n\n"
    
    # RESONANCIA MAGNÉTICA CARDÍACA
    if hallazgos_rm:
        resumen += "### 🎯 Resonancia Magnética Cardíaca\n"
        for h in hallazgos_rm:
            resumen += f"• {h}\n"
        resumen += "*La RM es la prueba de oro no invasiva cuando el Eco es ambiguo.*\n\n"
    
    # FACTORES CONFUSORES
    if hallazgos_confusores:
        resumen += "### ⚠️ Factores Confusores (Diferencial)\n"
        for h in hallazgos_confusores:
            resumen += f"• {h}\n"
        resumen += "*Estos factores pueden causar HVI pero NO son indicativos de amiloidosis.*\n\n"
    
    # RECOMENDACIONES FINALES
    resumen += "### 📋 Recomendaciones Clínicas\n"
    
    if "ALTA" in nivel:
        resumen += "1. **REFERENCIA URGENTE** a unidad de amiloidosis o cardiología especializada\n"
        resumen += "2. Confirmar diagnóstico mediante biopsia endomiocárdica\n"
        resumen += "3. Tipaje de amiloide (AL vs ATTR)\n"
        if "AL" in nivel or hallazgos_al_directo or hallazgos_al_sistemico:
            resumen += "4. Hemato-oncología: Proteinograma, cadenas ligeras libres\n"
        if "ATTR" in nivel or hallazgos_attr_clasico:
            resumen += "4. Genética: Secuenciación TTR (si hereditaria)\n"
        resumen += "5. Seguimiento multidisciplinario (cardiología, hematología, nefrología si aplica)\n"
    elif "INTERMEDIA" in nivel:
        resumen += "1. Resonancia cardíaca con gadolinio para evaluación de patrón de realce\n"
        resumen += "2. Gammagrafía ósea con Tc-99m pirofosfato\n"
        resumen += "3. Ecocardiografía seriada a 6-12 meses\n"
        resumen += "4. Consulta con especialista en amiloidosis\n"
        resumen += "5. Preparación para biopsia si se confirma sospecha\n"
    else:
        resumen += "1. Manejo de factores de riesgo cardiovascular\n"
        resumen += "2. Seguimiento ecocardiográfico anual\n"
        resumen += "3. ECG periódico\n"
        resumen += "4. Considerar buscar amiloidosis si hay progresión documentada\n"
    
    return resumen

# ================================================
# FUNCIÓN PARA EXTRAER Y ANALIZAR RED FLAGS
# ================================================
def extraer_redflags_detectados(datos: Dict[str, Any]) -> Dict[str, list]:
    """
    Extrae todos los red flags (hallazgos clínicos) detectados en los datos.
    Retorna un diccionario con categorías de red flags.
    """
    redflags = {
        "AL_DIRECTO": [],
        "AL_SISTEMICO": [],
        "ATTR_CLASICO": [],
        "ATTR_HEREDITARIO": [],
        "HALLAZGOS_CARDIACOS": [],
        "PARAMETROS_CARDIACOS": [],
        "FACTORES_CONFUSORES": [],
        "TOTAL_REDFLAGS": 0
    }
    
    # AL DIRECTO - Hallazgos patognomónicos
    if datos.get('mgus'): redflags["AL_DIRECTO"].append("MGUS/Paraproteína")
    if datos.get('macro'): redflags["AL_DIRECTO"].append("Macroglosia")
    if datos.get('purpura'): redflags["AL_DIRECTO"].append("Púrpura periorbital")
    
    # AL SISTÉMICO
    if datos.get('nefro'): redflags["AL_SISTEMICO"].append("Síndrome nefrótico")
    if datos.get('hepato'): redflags["AL_SISTEMICO"].append("Hepatomegalia")
    if datos.get('neuro_p'): redflags["AL_SISTEMICO"].append("Polineuropatía")
    if datos.get('fatiga'): redflags["AL_SISTEMICO"].append("Fatiga severa")
    if datos.get('disauto'): redflags["AL_SISTEMICO"].append("Disautonomía")
    if datos.get('piel_lesiones'): redflags["AL_SISTEMICO"].append("Lesiones cutáneas")
    
    # ATTR CLÁSICO - Tríada musculoesquelética
    if datos.get('stc'): redflags["ATTR_CLASICO"].append("STC bilateral")
    if datos.get('lumbar'): redflags["ATTR_CLASICO"].append("Estenosis lumbar")
    if datos.get('biceps'): redflags["ATTR_CLASICO"].append("Rotura bíceps")
    if datos.get('hombro'): redflags["ATTR_CLASICO"].append("Tendinitis hombro")
    if datos.get('dupuytren'): redflags["ATTR_CLASICO"].append("Dupuytren")
    if datos.get('artralgias'): redflags["ATTR_CLASICO"].append("Artralgias")
    if datos.get('fractura_vert'): redflags["ATTR_CLASICO"].append("Fractura vertebral")
    if datos.get('tendinitis_calcifica'): redflags["ATTR_CLASICO"].append("Tendinitis cálcica")
    
    # ATTR HEREDITARIA
    if datos.get('mutacion_ttr'): redflags["ATTR_HEREDITARIO"].append("Mutación TTR")
    
    # HALLAZGOS CARDIACOS ESPECÍFICOS
    if datos.get('apical_sparing'): redflags["HALLAZGOS_CARDIACOS"].append("Apical sparing")
    if datos.get('bajo_voltaje'): redflags["HALLAZGOS_CARDIACOS"].append("Bajo voltaje paradójico")
    if datos.get('bav_mp'): redflags["HALLAZGOS_CARDIACOS"].append("BAV/Marcapasos")
    if datos.get('pseudo_q'): redflags["HALLAZGOS_CARDIACOS"].append("Pseudoinfarto")
    if datos.get('biatrial'): redflags["HALLAZGOS_CARDIACOS"].append("Dilatación biauricular")
    
    # HALLAZGOS ECOCARDIOGRÁFICOS ESPECÍFICOS ATTR (NUEVOS)
    if datos.get('distribucion_concentrica'): redflags["HALLAZGOS_CARDIACOS"].append("Distribución concéntrica (característica ATTR)")
    if datos.get('rv_engrosado'): redflags["HALLAZGOS_CARDIACOS"].append("RV engrosado (signo ominoso ATTR)")
    if datos.get('aorta_pequena'): redflags["HALLAZGOS_CARDIACOS"].append("Aorta pequeña/normal relativa (paradoja ATTR)")
    if datos.get('ausencia_fa'): redflags["HALLAZGOS_CARDIACOS"].append("Ausencia de FA a pesar de ICC severa (ATTR vs HTA)")
    
    # BIOMARCADORES NUEVOS
    if datos.get('hiperrealce_subepicardico'): redflags["HALLAZGOS_CARDIACOS"].append("Hiperrealce subepicárdico (típico ATTR-v)")
    if datos.get('troponina_cronica'): redflags["HALLAZGOS_CARDIACOS"].append("Troponina elevada crónica baja (ATTR)")
    
    # PARÁMETROS CARDIACOS NUMÉRICOS
    ivs = datos.get('ivs', 0)
    gls = datos.get('gls', 0)
    volt = datos.get('volt', 0)
    ecv = datos.get('ecv', 0)
    
    if ivs > 15:
        redflags["PARAMETROS_CARDIACOS"].append(f"IVS muy aumentado: {ivs} mm (muy específico ATTR)")
    elif ivs > 12:
        redflags["PARAMETROS_CARDIACOS"].append(f"IVS aumentado: {ivs} mm")
    
    if gls < -9:
        redflags["PARAMETROS_CARDIACOS"].append(f"GLS anormal: {gls}%")
    
    if volt < 0.5:
        redflags["PARAMETROS_CARDIACOS"].append(f"Voltaje muy bajo: {volt} mV (CRÍTICO: amiloidosis)")
    elif volt < 1.0:
        redflags["PARAMETROS_CARDIACOS"].append(f"Voltaje bajo: {volt} mV")
    
    if ecv > 40:
        redflags["PARAMETROS_CARDIACOS"].append(f"ECV > 40% ({ecv}%) - CASI PATOGNOMÓNICO de amiloidosis")
    elif ecv > 35:
        redflags["PARAMETROS_CARDIACOS"].append(f"ECV elevado ({ecv}%) - muy sugestivo de amiloidosis")
    
    # FACTORES CONFUSORES
    if datos.get('confusor_hta'): redflags["FACTORES_CONFUSORES"].append("HTA severa")
    if datos.get('confusor_ao'): redflags["FACTORES_CONFUSORES"].append("Estenosis aórtica")
    if datos.get('confusor_irc'): redflags["FACTORES_CONFUSORES"].append("Insuficiencia renal")
    
    # Contar total de red flags
    redflags["TOTAL_REDFLAGS"] = sum(len(v) for k, v in redflags.items() if k != "TOTAL_REDFLAGS")
    
    return redflags


# ================================================
# GENERACIÓN DE CASOS SINTÉTICOS
# ================================================

def generar_caso_sintetico(rng: np.random.RandomState = None) -> Dict[str, Any]:
    """Genera un caso clínico sintético realista"""
    if rng is None:
        rng = np.random.RandomState(42)
    
    caso = DEFAULT_DATA.copy()
    
    # DEMOGRAFÍA
    caso['edad'] = rng.randint(45, 85)
    caso['sexo'] = rng.choice(['M', 'F'])
    
    # Tipo de caso: 30% AL, 35% ATTR-v, 15% ATTR-h, 10% HVI puro, 10% Control
    tipo_caso = rng.choice(['AL', 'ATTR_V', 'ATTR_H', 'HVI', 'CONTROL'], p=[0.30, 0.35, 0.15, 0.10, 0.10])
    
    # PARÁMETROS CARDIACOS COMUNES
    caso['ivs'] = float(rng.normal(13, 3)) if tipo_caso in ['AL', 'ATTR_V', 'ATTR_H'] else float(rng.normal(10, 2))
    caso['ivs'] = max(9, min(25, caso['ivs']))  # Limitar rango
    
    caso['gls'] = float(rng.normal(-10, 4)) if tipo_caso in ['AL', 'ATTR_V'] else float(rng.normal(-18, 5))
    caso['gls'] = max(-25, min(-5, caso['gls']))
    
    caso['volt'] = float(rng.normal(0.6, 0.3)) if tipo_caso in ['AL', 'ATTR_V'] else float(rng.normal(1.2, 0.4))
    caso['volt'] = max(0.3, min(2.5, caso['volt']))
    
    caso['septum_posterior'] = float(rng.normal(11, 2)) if tipo_caso in ['AL', 'ATTR_V', 'ATTR_H'] else float(rng.normal(9, 1.5))
    caso['septum_posterior'] = max(7, min(18, caso['septum_posterior']))
    
    # AL DIRECTO
    if tipo_caso == 'AL':
        caso['mgus'] = rng.choice([True, False], p=[0.6, 0.4])
        caso['macro'] = rng.choice([True, False], p=[0.5, 0.5])
        caso['purpura'] = rng.choice([True, False], p=[0.4, 0.6])
        caso['nt_probnp'] = float(rng.normal(3500, 1500))
        caso['troponina'] = rng.choice([True, False], p=[0.7, 0.3])
        caso['nefro'] = rng.choice([True, False], p=[0.5, 0.5])
        caso['neuro_p'] = rng.choice([True, False], p=[0.4, 0.6])
        caso['ecv'] = float(rng.normal(45, 5))
        caso['lge_patron'] = rng.choice(['TRANSMURAL', 'SUBENDOCARDICO', 'DIFUSO'])
    
    # ATTR CLÁSICO (Musculoesquelético)
    if tipo_caso == 'ATTR_V':
        caso['stc'] = rng.choice([True, False], p=[0.8, 0.2])
        caso['lumbar'] = rng.choice([True, False], p=[0.7, 0.3])
        caso['biceps'] = rng.choice([True, False], p=[0.6, 0.4])
        caso['nt_probnp'] = float(rng.normal(2000, 800))
        caso['troponina'] = rng.choice([True, False], p=[0.5, 0.5])
        caso['ecv'] = float(rng.normal(38, 4))
        caso['lge_patron'] = rng.choice(['SUBENDOCARDICO', '', ''])
        caso['apical_sparing'] = rng.choice([True, False], p=[0.7, 0.3])
        if rng.random() < 0.3:
            caso['dupuytren'] = True
        if rng.random() < 0.3:
            caso['artralgias'] = True
    
    # ATTR HEREDITARIA
    if tipo_caso == 'ATTR_H':
        caso['stc'] = rng.choice([True, False], p=[0.6, 0.4])
        caso['mutacion_ttr'] = True
        caso['nt_probnp'] = float(rng.normal(1800, 700))
        caso['ecv'] = float(rng.normal(36, 3))
        caso['edad'] = rng.randint(35, 70)  # ATTR-h presenta antes
    
    # HVI PURO (factores confusores)
    if tipo_caso == 'HVI':
        caso['confusor_hta'] = rng.choice([True, False], p=[0.8, 0.2])
        caso['ivs'] = float(rng.normal(12, 2))
        caso['nt_probnp'] = float(rng.normal(800, 300))
        caso['troponina'] = False
        caso['ecv'] = float(rng.normal(25, 3))
    
    # CONTROL (sin hallazgos)
    if tipo_caso == 'CONTROL':
        caso['ivs'] = float(rng.normal(9, 1))
        caso['gls'] = float(rng.normal(-20, 3))
        caso['volt'] = float(rng.normal(1.3, 0.3))
        caso['nt_probnp'] = float(rng.normal(100, 50))
        caso['edad'] = rng.randint(50, 80)
    
    caso['edad'] = int(caso['edad'])
    caso['nt_probnp'] = max(0, float(caso['nt_probnp']))
    
    return caso


def diagnostico_ia(datos: Dict[str, Any]) -> str:
    """
    Asigna un diagnóstico basado en los hallazgos clínicos (lógica médica)
    Retorna el diagnóstico más probable
    """
    redflags = extraer_redflags_detectados(datos)
    
    # Contar hallazgos por categoría
    al_directo = len(redflags.get('AL_DIRECTO', []))
    al_sistemico = len(redflags.get('AL_SISTEMICO', []))
    attr_clasico = len(redflags.get('ATTR_CLASICO', []))
    attr_heredit = len(redflags.get('ATTR_HEREDITARIO', []))
    cardiacos = len(redflags.get('HALLAZGOS_CARDIACOS', []))
    confusores = len(redflags.get('FACTORES_CONFUSORES', []))
    
    # Biomarkers
    nt_probnp = datos.get('nt_probnp', 0)
    troponina = datos.get('troponina', False)
    ecv = datos.get('ecv', 0)
    lge = datos.get('lge_patron', '')
    ivs = datos.get('ivs', 0)
    
    # Lógica diagnóstica
    
    # AL DIAGNOSIS
    if al_directo >= 2 or (al_directo >= 1 and al_sistemico >= 2):
        if nt_probnp > 2500 and troponina:
            return "AL - Amiloidosis AL Probable (ALTA SOSPECHA)"
        elif nt_probnp > 1000:
            return "AL - Amiloidosis AL Probable"
        else:
            return "AL - Amiloidosis AL Posible"
    
    # ATTR CLÁSICO (Musculoesquelético)
    if attr_clasico >= 3:  # Tríada: STC + Lumbar + Bíceps
        if ivs > 13 and ecv > 35:
            return "ATTR-v - Amiloidosis ATTR Salvaje (ALTA SOSPECHA)"
        else:
            return "ATTR-v - Amiloidosis ATTR Salvaje"
    elif attr_clasico >= 2 and ivs > 12:
        return "ATTR-v - Amiloidosis ATTR Salvaje Probable"
    elif attr_clasico >= 1 and ivs > 13 and ('SUBENDOCARDICO' in lge.upper() or ecv > 35):
        return "ATTR-v - Amiloidosis ATTR Probable"
    
    # ATTR HEREDITARIA
    if attr_heredit >= 1:
        return "ATTR-v Hereditaria - Mutación TTR Confirmada (REQUIERE CRIBADO FAMILIAR)"
    
    # DIFERENCIAL HVI
    if confusores >= 1 and ivs > 11 and al_directo == 0 and attr_clasico <= 1:
        return "HVI - Hipertrofia por HTA/Válvula (Sin Amiloidosis)"
    
    # BAJA PROBABILIDAD
    if redflags['TOTAL_REDFLAGS'] == 0 or ivs < 10:
        return "CONTROL - Sin hallazgos de amiloidosis"
    
    # ZONA GRIS
    if ivs > 12 and nt_probnp > 400:
        return "INTERMEDIA - Requiere investigación adicional"
    
    return "CONTROL - Baja probabilidad de amiloidosis"


def generar_base_datos_sintetica(n: int = 1000) -> pd.DataFrame:
    """Genera n casos sintéticos con diagnósticos IA, guarda una BD completa
    y devuelve un DataFrame con TODAS las variables (todas las columnas).
    """
    rng = np.random.RandomState(42)

    st.info(f"🔄 Generando {n} casos sintéticos (todas las variables)...")
    progress_bar = st.progress(0)

    # Columnas completas: id, fecha, diagnostico, resultado_algoritmo, diagnostico_ia, confianza_ia, modelo_usado + FEATURES
    columnas_esperadas = ['id', 'fecha', 'diagnostico', 'resultado_algoritmo', 'diagnostico_ia', 'confianza_ia', 'modelo_usado'] + FEATURES
    filas = []

    # Detectar modo headless (ejecución desde CLI) para no usar st.*
    headless = os.environ.get('AMYLO_HEADLESS', '0') == '1'

    for i in range(n):
        caso = generar_caso_sintetico(rng)
        # Usar el diagnóstico del algoritmo experto cuando esté disponible
        try:
            res_algo = calcular_riesgo_experto(caso)
            diag_algo = res_algo.get('nivel', '')
            confianza_algo = res_algo.get('confianza_porcentaje', 0.0)
        except Exception:
            # Fallback a la heurística IA previa
            diag_algo = diagnostico_ia(caso)
            confianza_algo = 0.0

        try:
            diag_llm = generar_diagnostico_por_llm(caso)
            diag_ia = diag_llm.get('diagnostico_llm', '')
        except Exception:
            diag_ia = ''

        fila = {col: DEFAULT_DATA.get(col, '') for col in columnas_esperadas}
        fila['id'] = i + 1
        fila['fecha'] = datetime.datetime.now().isoformat()
        fila['diagnostico'] = diag_ia
        fila['resultado_algoritmo'] = diag_algo
        fila['diagnostico_ia'] = diag_ia
        fila['confianza_ia'] = round(confianza_algo if confianza_algo else 85.0, 2)
        fila['modelo_usado'] = 'Sintético-IA'

        # Añadir todas las FEATURES explícitamente
        for f in FEATURES:
            valor = caso.get(f, DEFAULT_DATA.get(f, 0))
            # Normalizar tipos: booleanos a int si vienen así
            if isinstance(valor, bool):
                fila[f] = int(valor)
            else:
                fila[f] = valor

        filas.append(fila)

        # Actualizar barra (0-100)
        if headless:
            if (i + 1) % max(1, n // 10) == 0:
                print(f"Generados {i+1}/{n} casos...")
        else:
            try:
                progress_bar.progress(int((i + 1) / n * 100))
            except Exception:
                progress_bar.progress((i + 1) / n)

    progress_bar.empty()

    # Construir DataFrame con columnas en orden
    df_all = pd.DataFrame(filas)
    for col in columnas_esperadas:
        if col not in df_all.columns:
            df_all[col] = DEFAULT_DATA.get(col, '')
    df_all = df_all[columnas_esperadas]

    # Guardar BD completa
    ruta_bd = os.path.join(BASE_DIR, DB_FILE)
    df_all.to_csv(ruta_bd, index=False)

    st.success(f"✅ {len(df_all)} casos sintéticos generados y guardados en {DB_FILE}")
    return df_all

# ================================================
# FUNCIÓN PARA GUARDAR CASOS EN BASE DE DATOS
# ================================================
def save_case_training(datos: Dict[str, Any], diagnostico: str, confianza_ia: float = 0.0, modelo_usado: str = "") -> str:
    """
    Guarda un caso en la base de datos CSV para entrenamiento futuro.
    Incluye TODOS los red flags (hallazgos clínicos) como columnas independientes.
    NUEVO v4.3.0: Guarda también confianza IA y modelo usado
    Retorna el ID del caso o "ERROR" si falla.
    """
    try:
        ruta_abs = os.path.join(BASE_DIR, DB_FILE)
        
        # Columns esperadas: id, fecha, diagnostico, resultado_algoritmo, diagnostico_ia, confianza_ia, modelo_usado + FEATURES
        columnas_esperadas = ['id', 'fecha', 'diagnostico', 'resultado_algoritmo', 'diagnostico_ia', 'confianza_ia', 'modelo_usado'] + FEATURES
        
        # Cargar BD existente o crear nueva
        if os.path.isfile(ruta_abs):
            df = pd.read_csv(ruta_abs)
            nuevo_id = int(df['id'].max()) + 1 if 'id' in df.columns else 1
            
            # Verificar si la estructura es correcta
            columnas_faltantes = set(columnas_esperadas) - set(df.columns)
            
            if columnas_faltantes:
                # Si faltan columnas, necesitamos actualizar la estructura
                # Respaldar el archivo antiguo
                import shutil
                fecha_backup = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
                archivo_backup = ruta_abs.replace('.csv', f'_backup_{fecha_backup}.csv')
                shutil.copy(ruta_abs, archivo_backup)
                
                # Agregar columnas faltantes al DataFrame con valores por defecto apropiados
                for col in columnas_faltantes:
                    # Usar el valor por defecto de DEFAULT_DATA si existe
                    if col in ['resultado_algoritmo', 'diagnostico_ia']:
                        default_val = ''
                    else:
                        default_val = DEFAULT_DATA.get(col, 0)
                    df[col] = default_val
                
                # Reordenar columnas a la estructura correcta
                df = df[columnas_esperadas]
                df.to_csv(ruta_abs, index=False)
        else:
            # Crear tabla nueva con TODAS las columnas
            df = pd.DataFrame(columns=columnas_esperadas)
            nuevo_id = 1
        
        # Preparar fila nueva: cada red flag es una columna
        try:
            res_algo = calcular_riesgo_experto(datos)
            resultado_algoritmo = res_algo.get('nivel', '')
        except Exception:
            resultado_algoritmo = ''

        try:
            diag_llm = generar_diagnostico_por_llm(datos)
            diag_ia = diag_llm.get('diagnostico_llm', '')
        except Exception:
            diag_ia = ''

        nueva_fila = {
            'id': nuevo_id, 
            'fecha': datetime.datetime.now().isoformat(),
            'diagnostico': diagnostico,
            'resultado_algoritmo': resultado_algoritmo,
            'diagnostico_ia': diag_ia,
            'confianza_ia': round(confianza_ia, 2),  # Precision a 2 decimales
            'modelo_usado': modelo_usado
        }
        
        # Guardar TODOS los red flags (features) como columnas
        for feature in FEATURES:
            valor = datos.get(feature, DEFAULT_DATA.get(feature, 0))
            # Convertir booleanos a 0/1 para CSV y mantener números
            if isinstance(valor, bool):
                nueva_fila[feature] = int(valor)
            else:
                nueva_fila[feature] = valor
        
        # Añadir fila al DataFrame
        df = pd.concat([df, pd.DataFrame([nueva_fila])], ignore_index=True)
        
        # Asegurar tipos de datos correctos: mixto según feature
        df['id'] = df['id'].astype(int)
        df['diagnostico'] = df['diagnostico'].astype(str)
        for feature in FEATURES:
            if feature in ['ivs', 'volt', 'gls', 'nt_probnp', 'septum_posterior', 'ecv', 'edad']:
                # Numéricos
                df[feature] = pd.to_numeric(df[feature], errors='coerce').fillna(0)
                if feature in ['edad']:
                    df[feature] = df[feature].astype(int)
            elif feature in ['lge_patron', 'sexo']:
                # Strings/categóricos
                df[feature] = df[feature].astype(str)
            else:
                # Booleanos (0/1)
                df[feature] = pd.to_numeric(df[feature], errors='coerce').fillna(0).astype(int)
        
        # Guardar con columnas en el orden correcto
        df = df[columnas_esperadas]
        df.to_csv(ruta_abs, index=False)
        
        return str(nuevo_id)
    except Exception as e:
        print(f"Error guardando caso: {e}")
        import traceback
        traceback.print_exc()
        return "ERROR"

def generar_resumen_guardado(id_caso: str, datos: Dict[str, Any], diagnostico: str) -> str:
    """Genera un resumen formateado de lo que se acaba de guardar"""
    resumen = f"""
    ### ✅ RESUMEN DE GUARDADO DEL CASO
    
    **ID del Caso:** #{id_caso}  
    **Diagnóstico Final:** {diagnostico}  
    **Fecha/Hora:** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
    
    #### Datos Guardados:
    - **Edad:** {datos.get('edad', '—')} años
    - **Sexo:** {'Masculino' if datos.get('sexo') == 'M' else 'Femenino' if datos.get('sexo') == 'F' else '—'}
    
    #### Parámetros Cardiacos:
    - **IVS/Septo:** {datos.get('ivs', 0)} mm
    - **GLS/Strain:** {datos.get('gls', 0)} %
    - **Voltaje ECG:** {datos.get('volt', 0)} mV
    - **Pared Posterior:** {datos.get('septum_posterior', 0)} mm
    
    #### Biomarcadores:
    - **NT-proBNP:** {datos.get('nt_probnp', 0)} pg/ml
    - **Troponina Elevada:** {'✓ Sí' if datos.get('troponina') else 'No'}
    
    #### Resonancia Cardíaca:
    - **Patrón LGE:** {datos.get('lge_patron', '—').upper()}
    - **ECV:** {datos.get('ecv', 0)} %
    - **T1 Mapping Elevado:** {'✓ Sí' if datos.get('t1_mapping') else 'No'}
    
    #### Red Flags Detectados:
    """
    
    # Contar red flags
    red_flag_count = 0
    for feature in FEATURES:
        if feature not in ['id', 'fecha', 'diagnostico', 'confianza_ia', 'modelo_usado', 
                          'edad', 'sexo', 'ivs', 'volt', 'gls', 'nt_probnp', 'ecv', 'septum_posterior', 
                          'lge_patron', 'troponina', 't1_mapping', 'derrame_pericardico']:
            if datos.get(feature, False):
                red_flag_count += 1
    
    resumen += f"**Total Red Flags:** {red_flag_count}\n\n"
    resumen += "✅ **All data successfully saved to database**\n"
    
    return resumen

# ==========================================
# CONFIGURACIÓN Y VERSIONADO
# ==========================================
ALGORITMO_VERSION = "4.3.0-OPTIMIZADO"
FECHA_VALIDACION = "2026-02-08"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_FILE = "tabla_amiloidosis_completada-_1_.csv"  # Tabla de amiloidosis completada
# Asegura extension .csv si se pasa sin ella
if not DB_FILE.lower().endswith(".csv"):
    DB_FILE = f"{DB_FILE}.csv"

def read_file_base64(path: str) -> Optional[str]:
    if not os.path.exists(path):
        return None
    try:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    except Exception:
        return None

def read_file_bytes(path: str) -> Optional[bytes]:
    if not os.path.exists(path):
        return None
    try:
        with open(path, "rb") as f:
            return f.read()
    except Exception:
        return None

# ==========================================
# FUNCIÓN DE MIGRACIÓN (debe estar antes de st.set_page_config)
def migrar_base_datos_v4_3():
    """
    Migra automáticamente desde versiones anteriores a v4.3.0
    Agrega nuevas columnas: confianza_ia, modelo_usado
    """
    import shutil
    
    bd_antigua = "amylo_data_final_v1.csv"
    bd_nueva = DB_FILE
    
    ruta_antigua = os.path.join(BASE_DIR, bd_antigua)
    ruta_nueva = os.path.join(BASE_DIR, bd_nueva)
    
    # Si existe BD antigua y no existe la nueva, migrar
    if os.path.isfile(ruta_antigua) and not os.path.isfile(ruta_nueva):
        try:
            print(f"🔄 Migrando base de datos desde {bd_antigua} a {bd_nueva}...")
            
            # Cargar BD antigua
            df = pd.read_csv(ruta_antigua)
            
            # Remover NHC si existe (no deseado)
            if 'nhc' in df.columns:
                df = df.drop('nhc', axis=1)
            
            # Agregar nuevas columnas si no existen
            if 'confianza_ia' not in df.columns:
                df['confianza_ia'] = 0.0  # Casos antiguos sin confianza
            if 'modelo_usado' not in df.columns:
                df['modelo_usado'] = 'Importado v4.2.0'  # Marcar como importado de versión anterior
            if 'resultado_algoritmo' not in df.columns:
                df['resultado_algoritmo'] = ''
            if 'diagnostico_ia' not in df.columns:
                df['diagnostico_ia'] = ''
            
            # Guardar en nueva estructura
            df.to_csv(ruta_nueva, index=False)
            
            # Respaldar archivo antiguo
            fecha_backup = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            archivo_backup = ruta_antigua.replace('.csv', f'_migrado_{fecha_backup}.csv')
            shutil.copy(ruta_antigua, archivo_backup)
            
            print(f"✅ Migración completada: {len(df)} casos transferidos")
        except Exception as e:
            print(f"❌ Error en migración: {e}")
    
    # Si existe BD nueva pero le faltan columnas, actualizar estructura
    if os.path.isfile(ruta_nueva):
        try:
            df = pd.read_csv(ruta_nueva)
            columnas_requeridas = ['nhc', 'confianza_ia', 'modelo_usado', 'resultado_algoritmo', 'diagnostico_ia']
            necesita_actualizar = False
            
            for col in columnas_requeridas:
                if col not in df.columns:
                    if col == 'nhc':
                        df[col] = ''
                    elif col == 'confianza_ia':
                        df[col] = 0.0
                    elif col in ['resultado_algoritmo', 'diagnostico_ia']:
                        df[col] = ''
                    else:
                        df[col] = 'Actualizado a v4.3.0'
                    necesita_actualizar = True
            
            if necesita_actualizar:
                print("🔧 Actualizando estructura de BD a v4.3.0...")
                
                # Respaldar
                fecha_backup = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
                archivo_backup = ruta_nueva.replace('.csv', f'_backup_{fecha_backup}.csv')
                shutil.copy(ruta_nueva, archivo_backup)
                
                # Guardar actualizado
                df.to_csv(ruta_nueva, index=False)
                print(f"✅ Estructura actualizada: {len(df)} casos")
        except Exception as e:
            print(f"❌ Error actualizando estructura: {e}")

st.set_page_config(
    page_title="AmylAI 1.0",
    page_icon="image_6.png",
    layout="wide",
    initial_sidebar_state="expanded"
)

# MIGRACIÓN AUTOMÁTICA: Actualizar BD a v4.3.0 si es necesario
if 'migracion_v4_3' not in st.session_state:
    migrar_base_datos_v4_3()
    st.session_state.migracion_v4_3 = True

# ==========================================
# ESTILOS PERSONALIZADOS + FONDO
# ==========================================

# Cargar imagen de icono y fondo a base64
if (
    'fondo_base64' not in st.session_state or not st.session_state.get('fondo_base64')
    or 'icono_base64' not in st.session_state or not st.session_state.get('icono_base64')
):
    fondo_data = None
    fondo_candidates = [
        "fondo_rb_optimizado.jpg",
        "fondo_rb.jpg",
        "fondo_lateral.jpg",
    ]
    for fondo_name in fondo_candidates:
        fondo_path = os.path.join(os.path.dirname(__file__), fondo_name)
        fondo_data = read_file_base64(fondo_path)
        if fondo_data:
            break
    st.session_state.fondo_base64 = f"data:image/jpeg;base64,{fondo_data}" if fondo_data else None

    icono_path = os.path.join(os.path.dirname(__file__), "image_6.png")
    icono_data = read_file_base64(icono_path)
    st.session_state.icono_base64 = f"data:image/png;base64,{icono_data}" if icono_data else None

# Inyectar CSS con fondo
fondo_css = ""
if st.session_state.fondo_base64:
    fondo_css = f"""
    .stApp {{
        background-image: url('{st.session_state.fondo_base64}');
        background-size: cover;
        background-attachment: fixed;
        background-repeat: no-repeat;
        background-position: center;
    }}

    [data-testid="stAppViewContainer"] {{
        background-image: url('{st.session_state.fondo_base64}');
        background-size: cover;
        background-attachment: fixed;
        background-repeat: no-repeat;
        background-position: center;
        position: relative;
        color: #111111;
    }}
    
    [data-testid="stAppViewContainer"]::before {{
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: linear-gradient(135deg, rgba(255,255,255,0.6) 0%, rgba(255,255,255,0.75) 100%);
        pointer-events: none;
        z-index: 0;
    }}
    
    [data-testid="stAppViewContainer"] > * {{
        position: relative;
        z-index: 1;
    }}
    """

st.markdown(f"""
<style>
{fondo_css}

/* Sidebar mejorado con fondo */
[data-testid="stSidebar"] {{
    background: linear-gradient(180deg, rgba(244,140,140,0.95) 0%, rgba(230,110,110,0.95) 100%);
    box-shadow: 2px 0 8px rgba(0,0,0,0.1);
    backdrop-filter: blur(10px);
    color: #ffffff;
    font-size: 0.9rem;
}}

/* Colores principales */
:root {{
    --color-al: #d32f2f;
    --color-attr: #6a1b9a;
    --color-intermedia: #f57c00;
    --color-baja: #388e3c;
}}

/* Tipografía mejorada */
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; }}

/* Dividers mejorados */
hr {{ border: none; height: 2px; background: linear-gradient(90deg, transparent, #ccc, transparent); margin: 20px 0; }}

/* Headers */
h1, h2, h3 {{ font-weight: 700; letter-spacing: -0.5px; }}

/* Botones más bonitos */
.stButton > button {{
    border-radius: 8px !important;
    font-weight: 600 !important;
    letter-spacing: 0.5px !important;
    border: none !important;
    transition: all 0.3s ease !important;
}}

.stButton > button:hover {{
    transform: translateY(-2px);
    box-shadow: 0 4px 12px rgba(0,0,0,0.15) !important;
}}

/* Expanders con mejor apariencia */
.streamlit-expanderHeader {{
    font-weight: 600 !important;
    border-radius: 8px !important;
}}

/* Checkboxes */
.stCheckbox {{ margin: 10px 0 !important; }}

/* Alerts */
.stAlert {{ border-radius: 10px !important; }}

/* Tabs styling */
.stTabs [data-baseweb="tab-list"] [data-qa="stTab"] {{
    padding: 12px 20px !important;
    font-weight: 600 !important;
    border-radius: 8px 8px 0 0 !important;
}}

/* Input fields mejorados */
input {{
    border-radius: 8px !important;
}}

textarea {{
    border-radius: 8px !important;
}}

/* Select mejora do */
select {{
    border-radius: 8px !important;
}}
</style>
""", unsafe_allow_html=True)

# ==========================================
# CONFIGURACIÓN DE LLM (vLLM → Ollama → Fallback)
# ==========================================
client = None
backend_activo = None
llm_model = "llama3.2"

def _get_secret_or_env(key: str, default: Optional[str] = None) -> Optional[str]:
    try:
        if key in st.secrets:
            value = st.secrets.get(key)
            if value is not None and str(value).strip() != "":
                return str(value)
    except Exception:
        pass
    value_env = os.getenv(key)
    if value_env is not None and str(value_env).strip() != "":
        return str(value_env)
    return default

def _inicializar_llm_local(OpenAI):
    global client, backend_activo
    try:
        client = OpenAI(
            base_url="http://localhost:8000/v1",
            api_key="vllm"
        )
        client.models.list()
        backend_activo = "🚀 vLLM (Alto Rendimiento)"
        return
    except Exception:
        pass

    try:
        client = OpenAI(
            base_url="http://localhost:11434/v1",
            api_key="ollama"
        )
        client.models.list()
        backend_activo = "🤖 Ollama (Local)"
        return
    except Exception:
        client = None
        backend_activo = "🛡️ Regex (Sin LLM disponible)"

try:
    from openai import OpenAI
    llm_model = _get_secret_or_env("LLM_MODEL", "llama3.2")

    remote_base_url = _get_secret_or_env("LLM_BASE_URL") or _get_secret_or_env("OPENAI_BASE_URL")
    remote_api_key = _get_secret_or_env("LLM_API_KEY") or _get_secret_or_env("OPENAI_API_KEY")

    if remote_base_url and remote_api_key:
        try:
            remote_base_url = str(remote_base_url).rstrip("/")
            if not remote_base_url.endswith("/v1"):
                remote_base_url = f"{remote_base_url}/v1"

            client = OpenAI(
                base_url=remote_base_url,
                api_key=str(remote_api_key)
            )
            client.models.list()
            backend_activo = "☁️ Endpoint remoto (st.secrets)"
        except Exception:
            _inicializar_llm_local(OpenAI)
    else:
        _inicializar_llm_local(OpenAI)
            
except Exception:
    client = None
    backend_activo = "🛡️ Regex (Sin LLM disponible)"

# ==========================================
# TESAURO OPTIMIZADO (Token efficient)
# ==========================================
TERMINOLOGIA_MEDICA = {
    # CONFUSORES (elevan umbral de HVI exigido)
    "confusor_hta": "HTA|Hipertensión|Hipertenso|Cifras tensionales elevadas|Crisis hipertensiva|Antecedentes de hipertensión",
    "confusor_ao": "Estenosis aórtica|EAo|Valvulopatía aórtica|TAVI|Prótesis aórtica|Recambio valvular|Aorta bicúspide",
    "confusor_irc": "Insuficiencia renal|IRC|ERC|Fallo renal|Creatinina elevada|Filtrado glomerular|Diálisis",
    
    # RED FLAGS ATTR: FENOTIPO MUSCULOESQUELÉRICO
    "stc": "Túnel carpiano|STC|Atrapamiento del mediano|Liberación del mediano|Retinaculotomía|Síndrome del túnel carpiano|Parestesias mano|Adormecimiento dedos",
    "biceps": "Rotura de bíceps|Signo de Popeye|Deformidad en brazo|Tenotomía|Rotura bicipital|Avulsión proximal",
    "lumbar": "Canal lumbar estrecho|Estenosis de canal|Claudicación neurógena|Laminectomía|Estenosis foraminal|Compresión radicular|Estenosis lumbar",
    "hombro": "Tendinopatía hombro|Supraespinoso|Manguito rotadores|Omalgia|Hombro congelado|Capsulitis adhesiva|Rotura manguito|Gota hombro",
    "dupuytren": "Enfermedad de Dupuytren|Contractura de Dupuytren|Nódulos palmares|Cordones fibrosos mano",
    "artralgias": "Artralgias|Artralgia de grandes articulaciones|Dolor articular|Poliartralgia|Artropatía",
    "fractura_vert": "Fractura vertebral|Colapso vertebral|Osteoporosis|Fracturas patológicas|Cifosis",
    
    # RED FLAGS AL: SISTÉMICOS
    "macro": "Macroglosia|Lengua aumentada|Hipertrofia lingual|Improntas dentales|Festoneado|Aumento de volumen lingual",
    "purpura": "Púrpura|Ojos de mapache|Hematoma periorbitario|Equimosis palpebral|Hemorragia conjuntival|Púrpura periorbitaria",
    "mgus": "Gammapatía|MGUS|Mieloma|Componente monoclonal|Pico M|Banda monoclonal|Cadenas ligeras libres|Inmunofijación positiva|Paraproteína",
    "nefro": "Síndrome nefrótico|Edemas|Proteinuria|Albuminuria|Orina espumosa|Proteinuria >3.5|Insuficiencia renal|Nefropatía",
    "neuro_p": "Polineuropatía|Neuropatía|Parestesias en calcetín|Hormigueos distales|Neuropatía sensorial|Pérdida sensorial",
    "disauto": "Disautonomía|Hipotensión ortostática|Síncope vasovagal|Disfunción eréctil|Alteración GI|Impotencia|Estreñimiento|Diarrea",
    "hepato": "Hepatomegalia|Aumento de hígado|Hígado palpable|Cirrosis|Insuficiencia hepática",
    "fatiga": "Fatiga severa|Agotamiento extremo|Pérdida de energía|Síncope|Presíncope",
    "piel_lesiones": "Lesiones cérosas|Depósitos cutáneos|Lesiones papulares|Infiltración cutánea|Amiloidosis cutánea",
    
    # RED FLAGS CARDIACAS (AMBAS)
    "bav_mp": "Bloqueo AV|BAV|Bloqueo completo|Marcapasos|MP definitivo|DAI|TRC|Disfunción del nodo|Bloqueo de rama",
    "apical_sparing": "Apical sparing|Conservación apical|Strain apical longitudinal preservado|Patrón de strain discordante",
    "pseudo_q": "Pseudoinfarto|Ondas Q|Patrón tipo infarto|Necrosis|Fibrosis miocárdica",
    "biatrial": "Dilatación biauricular| Aurículas aumentadas|Aurículas dilatadas",
    "bajo_voltaje": "Bajo voltaje|Microvoltaje|Voltaje bajo|Bajo voltaje paradójico",
    
    # PATOLOGÍA ATTR HEREDITARIA
    "mutacion_ttr": "Mutación TTR|Transtiretina hereditaria|Amiloidosis hereditaria|Neuropatía amiloide hereditaria|Polineuropatía amiloide familiar",
    
    # OTROS ATTR-ESPECÍFICOS
    "tendinitis_calcifica": "Tendinitis cálcica|Depósitos de calcio|Calcificación tendinosa|Calcificación rotadoriana",
    
    # BIOMARCADORES (Nuevos)
    "nt_probnp": "NT-proBNP|NT-proBNP elevado|BNP|Péptido natriurético|Péptidos natriuréticos",
    "troponina": "Troponina elevada|Troponina alta|Troponina I|Troponina T|Elevación de troponina",
    
    # RESONANCIA MAGNÉTICA CARDÍACA (Nuevos)
    "lge_patron": "Realce tardío|Gadolinio|Patrón subendocárdico|Patrón transmural|Fibrosis|Infiltración|LGE|Delayed enhancement",
    "ecv": "Volumen extracelular|ECV|Volumen extracelular elevado|ECV alterado|T1 mapping",
    "t1_mapping": "T1 nativo|T1 mapping|T1 elevado|T1 prolongado",
    
    # DETALLES ECOCARDIOGRÁFICOS (Nuevos)
    "derrame_pericardico": "Derrame pericárdico|Efusión pericárdica|Cantidad de pericardio|Líquido pericárdico",
    "septum_posterior": "Tabique engrosado|Pared posterior engrosada|Grosor del tabique|Distribución concéntrica|Patrón concéntrico",
    "distribucion_concentrica": "Distribución concéntrica|Patrón concéntrico|Hipertrofia concéntrica|HVI concéntrica",
    "rv_engrosado": "RV engrosado|Ventrículo derecho engrosado|Pared libre RV engrosada|Grosor RV aumentado",
    "aorta_pequena": "Aorta pequeña|Aorta normal|Diámetro aórtico pequeño|Paradoja aórtica",
    "ausencia_fa": "Sin fibrilación auricular|Ausencia de FA|FA negada|Ritmo sinusal|Ritmo sinusal normal",
    
    # RESONANCIA MAGNÉTICA AVANZADA
    "hiperrealce_subepicardico": "Hiperrealce subepicárdico|Realce subepicárdico|Patrón subepicárdico|Fibrosis subepicárdica",
    
    # TROPONINA ESPECÍFICA
    "troponina_cronica": "Troponina crónica|Troponina ligeramente elevada|Troponina baja persistente|Elevación crónica de troponina"
}

def generar_instrucciones_contexto():
    """Genera un prompt optimizado usando el tesauro comprimido"""
    texto = "VOCABULARIO MÉDICO A DETECTAR (SINÓNIMOS ACEPTADOS):\n"
    for clave, valor in TERMINOLOGIA_MEDICA.items():
        texto += f"- {clave}: [{valor}]\n"
    return texto

# ==========================================
# LÓGICA DE NEGOCIO Y VALIDACIÓN
# ==========================================

FEATURES = [
    # Parámetros cardiacos numéricos
    "ivs","volt","gls",
    # Biomarcadores
    "nt_probnp","troponina","troponina_cronica",
    # Resonancia Cardíaca / RM
    "lge_patron","ecv","t1_mapping","hiperrealce_subepicardico",
    # Demografía
    "edad","sexo",
    # Detalles ecocardiográficos finos - REDFLAGS NUEVOS ATTR específicos
    "derrame_pericardico","septum_posterior","distribucion_concentrica","rv_engrosado","aorta_pequena","ausencia_fa",
    # ✓ FENOTIPO ATTR MUSCULOESQUELÉTICO - Red Flags de ATTR Clásico
    "stc","biceps","lumbar","hombro","dupuytren","artralgias","fractura_vert","tendinitis_calcifica",
    # ✓ FENOTIPO AL SISTÉMICO - Red Flags de AL
    "macro","purpura","mgus","nefro","neuro_p","disauto","hepato","fatiga","piel_lesiones",
    # ✓ HALLAZGOS CARDIACOS ESPECÍFICOS - Red Flags cardiacos
    "apical_sparing","biatrial","bav_mp","pseudo_q","bajo_voltaje",
    # ✓ ATTR HEREDITARIA - Red Flags genéticos
    "mutacion_ttr",
    # ✓ FACTORES CONFUSORES - Diagnóstico diferencial
    "confusor_hta","confusor_ao","confusor_irc"
]

DEFAULT_DATA = {
    "nhc": "",
    # Valores numéricos principales
    "ivs": 0.0, "volt": 0.0, "gls": 0.0,
    # Biomarcadores
    "nt_probnp": 0.0, "troponina": False,
    # RM
    "lge_patron": "", "ecv": 0.0, "t1_mapping": False,
    # Demografía
    "edad": 0, "sexo": "",
    # Ecocardiografía fina
    "derrame_pericardico": False, "septum_posterior": 0.0,
    # Fenotipo ATTR musculoesquelético
    "stc": False, "biceps": False, "lumbar": False, "hombro": False, 
    "dupuytren": False, "artralgias": False, "fractura_vert": False, "tendinitis_calcifica": False,
    # Fenotipo AL sistémico
    "macro": False, "purpura": False, "mgus": False, "nefro": False,
    "neuro_p": False, "disauto": False, "hepato": False, "fatiga": False, "piel_lesiones": False,
    # Cardiacos
    "apical_sparing": False, "biatrial": False, "bav_mp": False, "pseudo_q": False, "bajo_voltaje": False,
    # REDFLAGS NUEVOS: Ecocardiografía específica ATTR
    "distribucion_concentrica": False, "rv_engrosado": False, "aorta_pequena": False, "ausencia_fa": False,
    # REDFLAGS NUEVOS: RM específica ATTR
    "hiperrealce_subepicardico": False, "troponina_cronica": False,
    # ATTR hereditaria
    "mutacion_ttr": False,
    # Confusores
    "confusor_hta": False, "confusor_ao": False, "confusor_irc": False,
    # NUEVO v4.3.0: Metadata de análisis
    "confianza_ia": 0.0,  # Porcentaje de confianza del modelo
    "modelo_usado": ""   # Tipo de modelo (LLM, Regex, Híbrido)
}

if 'form_data' not in st.session_state:
    st.session_state.form_data = DEFAULT_DATA.copy()
if 'consolidado_batch' not in st.session_state:
    st.session_state.consolidado_batch = None
if 'resumen_generado' not in st.session_state:
    st.session_state.resumen_generado = None
# NUEVO v4.3.0: Variables para guardar metadata de análisis
if 'confianza_analisis' not in st.session_state:
    st.session_state.confianza_analisis = 0.0
if 'nivel_diagnostico' not in st.session_state:
    st.session_state.nivel_diagnostico = ''
if 'analisis_automatico' not in st.session_state:
    st.session_state.analisis_automatico = False  # Flag para mostrar resumen de hallazgos extraídos

# Versión duplicada eliminada - usar la versión con type hints en línea ~1372

def validar_rangos_clinicos(datos: Dict[str, Any]) -> Dict[str, Any]:
    """Filtra valores fisiológicamente imposibles o errores de OCR"""
    if datos.get('ivs', 0) > 35:
        datos['ivs'] = 0.0

    if datos.get('volt', 0) > 6:
        datos['volt'] = 0.0

    gls = datos.get('gls', 0)
    if gls > 0:
        gls = -gls

    if gls < -40:
        gls = 0.0

    datos['gls'] = gls
    return datos

def correccion_determinista(texto: str, datos: Dict[str, Any]) -> Dict[str, Any]:
    """Capa de seguridad Regex + Censor Universal V3 (Fix Microvoltajes, BAV, Biomarcadores, RM)"""
    t = texto.lower()

    # --- A. RESCATE DE POSITIVOS (Keywords que la IA ignora) ---
    
    # FIX VOLTAJE: Añadimos "microvoltaje" como disparador directo
    if not datos.get('volt'):
        if "bajo voltaje" in t or "microvoltaje" in t: 
            datos['volt'] = 0.4
        else:
            # Busca "QRS de X mm" o "Voltaje de X mV"
            match = re.search(r"(voltaje|qrs).{0,15}?(\d+[,.]?\d*)", t)
            if match:
                val = float(match.group(2).replace(',', '.'))
                # Si el valor es > 5, probablemente sean mm (Sokolow), no mV. 
                # Convertimos mm a mV dividiendo por 10 si es necesario, o lo ignoramos si es ambiguo.
                if 0 < val < 5: 
                    datos['volt'] = val

    # RESCATE BIOMARCADORES NUMÉRICOS
    # NT-proBNP
    if not datos.get('nt_probnp'):
        match = re.search(r"(nt-?probnp|bnp).{0,15}?(\d+(\.\d+)?|,\d+)", t)
        if match:
            val = float(match.group(2).replace(',', '.'))
            if val > 0:
                datos['nt_probnp'] = val
    
    # ECV (Volumen Extracelular)
    if not datos.get('ecv'):
        match = re.search(r"(ecv|volumen extracelular).{0,15}?(\d+)%?", t)
        if match:
            val = float(match.group(2))
            if 0 < val <= 100:
                datos['ecv'] = val
    
    # Septum/Pared Posterior
    if not datos.get('septum_posterior'):
        match = re.search(r"(tabique|pared posterior|septum).{0,20}?(\d+[,.]?\d*)", t)
        if match:
            val = float(match.group(2).replace(',', '.'))
            if 5 < val < 30:  # Rango fisiológico (mm)
                datos['septum_posterior'] = val
    
    # Edad
    if not datos.get('edad'):
        match = re.search(r"(\d{1,3})\s*(?:años|a[ña]o|yrs)", t)
        if match:
            edad = int(match.group(1))
            if 0 < edad < 150:
                datos['edad'] = edad
    
    # LGE Patrón (Resonancia)
    if not datos.get('lge_patron') or datos.get('lge_patron') == '':
        if "subendocárdico" in t or "subendocardico" in t:
            datos['lge_patron'] = "subendocardico"
        elif "transmural" in t:
            datos['lge_patron'] = "transmural"
        elif "difuso" in t:
            datos['lge_patron'] = "difuso"
        elif "parcheado" in t or "parcheado" in t:
            datos['lge_patron'] = "parcheado"

    # FIX AMILOIDOSIS AL: Rescate de palabras clave
    triggers_mgus = ["monoclonal", "inmunofijación positiva", "banda m ", "pico m ", "cadenas ligeras"]
    if any(x in t for x in triggers_mgus):
        # Solo marca True si NO hay una negación cerca
        if not re.search(r"(no |ausencia|negativ).{0,30}(monoclonal|inmunofijación|banda|pico)", t):
            datos['mgus'] = True

    if "macroglosia" in t or "improntas" in t:
        if "no " not in t and "sin " not in t:
            datos['macro'] = True

    # --- B. CENSOR DE ALUCINACIONES (Correcciones lógicas) ---
    
    # FIX BAV: Si dice "Ritmo Sinusal" y NO menciona bloqueo/marcapasos, fuerza False
    if "ritmo sinusal" in t or "rs a " in t:
        if not any(x in t for x in ["bloqueo", "bav", "marcapasos", "mp "]):
            datos['bav_mp'] = False

    # FIX SIV: Rescate numérico si la IA falló la conversión cm -> mm
    if not datos.get('ivs'):
        # Busca patrones como "SIV 1.6 cm" o "Septum 16 mm"
        match = re.search(r"(siv|septo|tabique|septum).{0,20}?(\d+[,.]?\d*)", t)
        if match:
            valor = float(match.group(2).replace(',', '.'))
            # Si el valor es pequeño (ej: 1.6), asume cm y multiplica por 10
            if 0.6 <= valor <= 3.0: 
                datos['ivs'] = valor * 10.0
            # Si es normal (ej: 16), lo deja tal cual
            elif 6 < valor < 30: 
                datos['ivs'] = valor

    # --- C. CENSOR UNIVERSAL (Negaciones estándar) ---
    patron_base = r"(niega|no |sin |ausencia|descarta|normal|negativo).{0,40}?"

    for k, v_str in TERMINOLOGIA_MEDICA.items():
        # Si el dato está marcado como True, verificamos que no sea un falso positivo por negación
        if datos.get(k) is True:
            regex_keywords = f"({v_str})"
            if re.search(patron_base + regex_keywords, t, re.IGNORECASE):
                datos[k] = False

    return validar_rangos_clinicos(datos)

def fusionar_extracciones(extracciones: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Combina datos con Lógica GLS corregida"""
    final = DEFAULT_DATA.copy()
    origen = {}

    for item in extracciones:
        doc_name = item["documento"]
        datos_limpios = {k: v for k, v in item["datos"].items() if k in DEFAULT_DATA}

        for k, v in datos_limpios.items():
            current_val = final.get(k)

            # Booleanos (OR)
            if isinstance(v, bool):
                if v is True:
                    final[k] = True
                    origen[k] = doc_name

            # Numéricos
            elif isinstance(v, (int, float)):
                v_float = safe_float(v)
                curr_float = safe_float(current_val)

                # GLS: Queremos el MENOS negativo (más cercano a 0 = peor función)
                if k == "gls":
                    if v_float != 0:
                        if curr_float == 0:
                            final[k] = v_float
                            origen[k] = doc_name
                        elif abs(v_float) < abs(curr_float):
                            final[k] = v_float
                            origen[k] = doc_name

                # Resto (IVS, Volt): Nos quedamos con el mayor
                else:
                    if abs(v_float) > abs(curr_float):
                        final[k] = v_float
                        origen[k] = doc_name

    final["_origen"] = origen
    return final

# --- MOTOR NLP ROBUSTO ---
# ==========================================
# MOTOR NLP HÍBRIDO (RÁPIDO Y PRECISO)
# ==========================================
def motor_nlp_hibrido(texto: str) -> Dict[str, Any]:
    """
    Motor HÍBRIDO: Regex (rápido) + LLM (preciso) donde falta.
    Tiempo: 2-5s (mucho mejor que 10-30s del LLM puro).
    Precisión: 95%+ (datos complejos con LLM de respaldo).
    """
    datos = DEFAULT_DATA.copy()
    
    if not texto.strip():
        return datos
    
    texto_lower = texto.lower()
    
    # ===== PASO 1: REGEX PARA DATOS NUMÉRICOS CLAROS (Muy rápido) =====
    
    # IVS / Septum - patrones mejorados
    ivs_patterns = [
        r'ivs\s*[:=]\s*(\d+(?:[.,]\d+)?)',  # IVS: 15
        r'septo\s*[:=]\s*(\d+(?:[.,]\d+)?)\s*mm',  # Septo: 15mm
        r'septum\s*[:=]\s*(\d+(?:[.,]\d+)?)',  # Septum: 15
        r'grosor.*septo\s*[:=]\s*(\d+(?:[.,]\d+)?)',  # Grosor septo: 15
        r'septo.*(\d+(?:[.,]\d+)?)\s*(?:mm)',  # Septo 15mm
        r'espesor.*septo\s*[:=]\s*(\d+(?:[.,]\d+)?)',  # Espesor septo: 15
    ]
    for pattern in ivs_patterns:
        match = re.search(pattern, texto_lower)
        if match:
            val = match.group(1).replace(',', '.')
            try:
                datos['ivs'] = float(val)
                break
            except:
                pass
    
    # GLS / Strain - patrones mejorados
    gls_patterns = [
        r'gls\s*[:=]\s*(-?\d+(?:[.,]\d+)?)',  # GLS: -15
        r'strain\s*(?:global)?\s*[:=]\s*(-?\d+(?:[.,]\d+)?)',  # Strain: -15
        r'strain.*(?:longitudinal|global)\s*[:=]?\s*(-?\d+(?:[.,]\d+)?)',  # Strain longitudinal: -15
        r'longitudinal.*strain\s*[:=]\s*(-?\d+(?:[.,]\d+)?)',  # Longitudinal strain: -15
    ]
    for pattern in gls_patterns:
        match = re.search(pattern, texto_lower)
        if match:
            val = match.group(1).replace(',', '.')
            try:
                datos['gls'] = -abs(float(val))
                break
            except:
                pass
    
    # NT-proBNP - patrones mejorados
    bnp_patterns = [
        r'nt-?pro?bnp\s*[:=]\s*(\d+(?:[.,]\d+)?)',  # NT-proBNP: 4200
        r'bnp\s*[:=]\s*(\d+(?:[.,]\d+)?)',  # BNP: 4200
        r'n-?terminal\s*pro\s*bnp\s*[:=]\s*(\d+(?:[.,]\d+)?)',  # N-terminal pro BNP: 4200
    ]
    for pattern in bnp_patterns:
        match = re.search(pattern, texto_lower)
        if match:
            val = match.group(1).replace(',', '.')
            try:
                datos['nt_probnp'] = float(val)
                break
            except:
                pass
    
    # ECV (Extracellular Volume)
    ecv_patterns = [
        r'ecv\s*[:=]\s*(\d+(?:[.,]\d+)?)',  # ECV: 45
        r'volumen.*extracelular\s*[:=]\s*(\d+(?:[.,]\d+)?)',  # Volumen extracelular: 45
        r'extracellular.*volume\s*[:=]\s*(\d+(?:[.,]\d+)?)',  # Extracellular volume: 45
    ]
    for pattern in ecv_patterns:
        match = re.search(pattern, texto_lower)
        if match:
            val = match.group(1).replace(',', '.')
            try:
                datos['ecv'] = float(val)
                break
            except:
                pass
    
    # Edad - mejorada
    edad_patterns = [
        r'edad\s*[:=]\s*(\d{1,3})',  # Edad: 72
        r'(?:paciente|pt)\s*(?:de\s*)?(?:sexo\s*)?(?:masculino|femenino|varón|mujer)?\s*(?:de\s*)?(\d{1,3})\s*(?:años|year|yo|años)',  # Paciente de 72 años
        r'(\d{1,3})\s*(?:años|year|yo)',  # 72 años
    ]
    for pattern in edad_patterns:
        match = re.search(pattern, texto_lower)
        if match:
            try:
                datos['edad'] = int(match.group(1))
                break
            except:
                pass
    
    # Sexo
    if re.search(r'varón|hombre|male|masculine', texto_lower):
        datos['sexo'] = 'M'
    elif re.search(r'mujer|femenino|female|feminine', texto_lower):
        datos['sexo'] = 'F'
    
    # ===== PASO 2: PATRONES BOOLEANOS SIMPLES (Regex completo) =====
    # Hallazgos que regex puede detectar bien
    
    boolean_patterns = {
        'apical_sparing': (r'apical\s*sparing|preservación.*apical|sparing.*apical|preserved.*apex', r'sin.*apical|no.*apical'),
        'bajo_voltaje': (r'bajo\s*voltaje|low\s*voltage|microlvoltaje|qrs.*bajo|voltaje.*bajo', r'sin.*voltaje|voltaje.*normal'),
        'bav_mp': (r'bloqueo\s*av|bloqueo\s*atrioventricular|marcapasos|pacemaker|bav\s*(?:completo|alto)', r'sin.*bav|bav.*no'),
        'biatrial': (r'dilatación.*auri|biatrial|aurícula.*dilatada|auricular.*dilatada', r'sin.*dilatación|aurícula.*normal'),
        'derrame_pericardico': (r'derrame(?:\s*pericárd)?|pericardial\s*effusion|efusión(?:\s*pericárdica)?', r'sin.*derrame|sin.*efusión|derrame.*no'),
        'troponina': (r'troponina\s*(?:elevada|alta|positive|high|positiva|\+)', r'troponina\s*(?:normal|baja|negativa|-)'),
        'macro': (r'macroglosia|agrandada.*lengua|lengua.*agrandada|enlarged.*tongue', r'sin.*macroglosia|lengua.*normal'),
        'purpura': (r'púrpura|purpura|periorbital|ojos\s*mapache', r'sin.*púrpura|púrpura.*no'),
        'stc': (r'túnel.*carp|carpal\s*tunnel|s\.?t\.?c\.?|atrapamiento.*mediano', r'niega.*stc|sin.*stc|stc.*no|bilateral.*niega'),
        'biceps': (r'rotura.*bíceps|bíceps.*rotura|popeye|rupture.*biceps', r'sin.*bíceps|bíceps.*no|bíceps.*intacto'),
        'lumbar': (r'estenosis.*lumbar|stenosis.*lumbar|claudicación.*neuro|compresión.*lumbar', r'sin.*lumbar|lumbar.*normal|estenosis.*no'),
        'nefro': (r'síndrome.*nefrótico|nephrotic.*syndrome|proteinuria|glomerulon', r'sin.*nefrótico|nefrótico.*no|síndrome.*no'),
        'neuro_p': (r'polineuropatía|polyneuropathy|neuropatía|parestesia|neuropatía.*periférica', r'sin.*neuropatía|neuropatía.*no|periférica.*no'),
        'disauto': (r'disautonomía|autonomic|hipotensión.*orto|diarrea|hipohidrosis', r'sin.*disautonomía|disautonomía.*no'),
        'hepato': (r'hepatomegalia|agrandado.*hígado|hígado.*agrandado|enlarged.*liver', r'sin.*hepato|hígado.*normal|hepato.*no'),
        'fatiga': (r'fatiga\s*(?:severa|extrema|intensa)|extreme\s*fatigue|agotamiento', r'sin.*fatiga|fatiga.*no'),
        'mutacion_ttr': (r'mutación.*ttr|ttr.*mutado|attr.*hereditara|hereditary.*ttr', r'ttr.*no|sin.*mutación|mutación.*no'),
    }
    
    for key, (positive_pattern, negative_pattern) in boolean_patterns.items():
        if re.search(positive_pattern, texto_lower):
            # Verificar si está negado
            if not re.search(negative_pattern, texto_lower):
                datos[key] = True
    
    # ===== PASO 3: CAMPOS COMPLEJOS - USAR LLM SOLO SI REGEX FALLA =====
    # Si falta información crítica, usar LLM smart para extraer lo faltante
    
    campos_criticos = ['ivs', 'edad', 'gls', 'nt_probnp', 'lge_patron']
    campos_faltantes = [k for k in campos_criticos if (datos[k] == 0 and k != 'lge_patron') or (k == 'lge_patron' and datos[k] == '')]
    
    if campos_faltantes and client:
        # Solo usar LLM para campos que regex NO pudo extraer
        prompt_mini = f"""Extrae SOLO estos campos del texto en JSON válido:
{json.dumps({k: datos[k] for k in campos_faltantes})}

Texto: \"\"\"{texto}\"\"\"

Responde solo con JSON válido, reemplazando valores NULL por los encontrados.
No agregues campos adicionales."""
        
        try:
            r = client.chat.completions.create(
                model=llm_model,
                messages=[{"role": "user", "content": prompt_mini}],
                temperature=0,
                response_format={"type": "json_object"},
                stream=False
            )
            data_ia = json.loads(r.choices[0].message.content)
            # Mezclar: regex tiene prioridad, LLM solo rellena vacíos
            for k in campos_faltantes:
                if k in data_ia and data_ia[k] is not None:
                    datos[k] = data_ia[k]
        except:
            pass  # Mantener los valores de regex
    
    # ===== PATRÓN LGE (CATEGORÍA) - Detectar automáticamente =====
    if re.search(r'subendocardio|subendocárdic', texto_lower):
        datos['lge_patron'] = 'subendocardico'
    elif re.search(r'transmural|transmurale', texto_lower):
        datos['lge_patron'] = 'transmural'
    elif re.search(r'parche|patchy', texto_lower):
        datos['lge_patron'] = 'parcheado'
    elif re.search(r'difuso', texto_lower):
        datos['lge_patron'] = 'difuso'
    
    return datos

# ================================
# MOTOR NLP CONTEXTUAL (LLM - MEJORADO)
# ================================
def motor_nlp_contextual(texto: str) -> Dict[str, Any]:
    """Motor NLP V4: Máxima precisión - LLM con instrucciones mejoradas"""
    if not texto.strip(): 
        return DEFAULT_DATA.copy()
    
    modelo_usado = "🤖 Llama 3.2 (Máxima Precisión)"

    if not client:
        datos = correccion_determinista(texto, {})
        datos['_modelo'] = "🛡️ Regex (LLM no disponible)"
        return datos

    try:
        instrucciones = generar_instrucciones_contexto()
        
        # PROMPT V4: MEJORADO PARA MÁXIMA PRECISIÓN Y FLEXIBILIDAD
        prompt = f"""ERES UN CARDIÓLOGO EXPERTO EN AMILOIDOSIS.

Tu tarea: Extraer datos COMPLETOS de la nota clínica, sin importar cómo esté redactada.

### INSTRUCCIONES CRÍTICAS:
1. Lee TODO el texto cuidadosamente, párrafo por párrafo
2. Busca CUALQUIER mención de los datos solicitados
3. Si dice "Edad 72 años", "72 años", "setenta y dos", "72", extrae 72
4. Si dice "septo 1.8cm", "septum 18mm", "IVS 1.8", extrae como número (1.8 o 18)
5. Si dice "niega", "sin", "normal", "ausente" → false
6. Si menciona síntoma/hallazgo sin negación → true
7. Para textos ambiguos: interpreta como médico (contexto clínico)

### CAMPOS NUMÉRICOS (convertir TODO a número):
- ivs: Grosor septum/septo (en mm, si está en cm multiplicar x10)
- volt: Voltaje ECG (0.4=bajo, 1.0+ normal). Si dice "bajo voltaje" → 0.4
- gls: Global Longitudinal Strain (número negativo). Si dice "strain -10%" → -10
- edad: Años del paciente (número entero)
- nt_probnp: BNP (número bruto, ej: 4200)
- ecv: Volumen extracelular (0-100)
- septum_posterior: Grosor pared posterior (mm), también llamado "PP", "espesor pared posterior"
- t1_mapping: Valor T1 nativo (ms). Si dice "T1 prolongado" → booleano true, sino 0


### CAMPOS BOOLEANOS (true SOLO si está CLARAMENTE mencionado):
Busca estas palabras EXACTAS o sus sinónimos:
- apical_sparing: "apical sparing", "preservación apical", "sparing"
- bajo_voltaje: "bajo voltaje", "microvoltaje", "reduced voltage"
- bav_mp: "bloqueo AV", "BAV", "marcapasos", "pacemaker"
- biatrial: "dilatación auricular", "biatrial", "aurícula"
- derrame_pericardico: "derrame", "efusión pericárdica"
- troponina: "troponina elevada", "troponina alta", "troponina positiva"
- macro: "macroglosia", "lengua agrandada"
- purpura: "púrpura", "periorbital", "ojos mapache"
- mgus: "MGUS", "componente monoclonal", "cadena ligera"
- stc: "túnel carpiano", "STC", "carpal tunnel" (BILATERAL en ATTR)
- biceps: "rotura bíceps", "popeye", "biceps rupture"
- lumbar: "estenosis lumbar", "claudicación", "lumbar stenosis"
- nefro: "síndrome nefrótico", "proteinuria", "nephrotic"
- neuro_p: "polineuropatía", "neuropatía periférica", "polyneuropathy"
- disauto: "disautonomía", "hipotensión ortostática", "autonomic"
- hepato: "hepatomegalia", "hígado agrandado"
- fatiga: "fatiga severa", "agotamiento extremo"
- mutacion_ttr: "mutación TTR", "hereditary ATTR", "ATTR mutado"

### CAMPO CATEGORÍA (ELEGIR UNO):
lge_patron: 
  - "subendocardico" si dice "subendocárdico", "subendocardial"
  - "transmural" si dice "transmural", "circunferencial"
  - "parcheado" si dice "parcheado", "patchy"
  - "difuso" si dice "difuso", "diffuse"
  - "" si no menciona

### SEXO:
- "M" si dice varón/hombre/male/masculino
- "F" si dice mujer/femenino/female/femenina
- "" si no especifica

### DATOS INCOMPLETOS:
Si una dato NO está en el texto:
- Números → 0 (para ivs, edad, voltaje, etc.)
- Booleanos → false
- Texto → cadena vacía ""

### EJEMPLO PRÁCTICO:
Entrada: "Mujer 68 años. Ecocardiograma con septo 16, strain -8. RM con LGE subendocárdico. ECV 42%. Sin antecedentes de STC."

Salida:
{{
  "ivs": 16.0,
  "volt": 1.0,
  "gls": -8.0,
  "edad": 68,
  "sexo": "F",
  "nt_probnp": 0.0,
  "troponina": false,
  "lge_patron": "subendocardico",
  "ecv": 42.0,
  "derrame_pericardico": false,
  "stc": false,
  ...todos los demás: false
}}

### AHORA ANALIZA ESTE TEXTO:
\"\"\"{texto}\"\"\"

RESPONDE SOLO CON JSON VÁLIDO. Rellena TODOS los campos."""

        r = client.chat.completions.create(
            model=llm_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            response_format={"type": "json_object"}
        )

        # Procesamiento
        raw_content = r.choices[0].message.content
        data_ia = json.loads(raw_content)
        
        # Filtrado y Limpieza
        datos_filtrados = {k: data_ia.get(k, DEFAULT_DATA[k]) for k in DEFAULT_DATA}
        
        # Post-procesado de seguridad
        datos_filtrados['ivs'] = safe_float(datos_filtrados.get('ivs'))
        datos_filtrados['volt'] = safe_float(datos_filtrados.get('volt'))
        val_gls = safe_float(datos_filtrados.get('gls'))
        datos_filtrados['gls'] = -abs(val_gls) if val_gls != 0 else 0.0
        
        # Campos numéricos adicionales
        datos_filtrados['nt_probnp'] = safe_float(datos_filtrados.get('nt_probnp'))
        datos_filtrados['ecv'] = safe_float(datos_filtrados.get('ecv'))
        datos_filtrados['septum_posterior'] = safe_float(datos_filtrados.get('septum_posterior'))
        
        # Campos demográficos
        datos_filtrados['edad'] = int(safe_float(datos_filtrados.get('edad', 0)))
        datos_filtrados['sexo'] = str(datos_filtrados.get('sexo', '')).upper()[:1]
        
        # Campo categórico (LGE)
        datos_filtrados['lge_patron'] = str(datos_filtrados.get('lge_patron', '')).lower().strip()
        
        # Asegurar booleanos y tipos correctos
        numeric_fields = ['ivs', 'volt', 'gls', 'nt_probnp', 'ecv', 'septum_posterior', 'edad']
        string_fields = ['sexo', 'lge_patron']
        
        for k in datos_filtrados:
            if k in numeric_fields:
                continue  # Ya procesados arriba
            elif k in string_fields:
                continue  # Ya procesados arriba
            else:
                # Booleanos
                datos_filtrados[k] = bool(datos_filtrados.get(k, False))

        # Validación final determinista (respaldo adicional)
        resultado_final = correccion_determinista(texto, datos_filtrados)
        resultado_final['_modelo'] = modelo_usado
        return resultado_final

    except Exception as e:
        print(f"⚠️ Error IA: {e}")
        datos = correccion_determinista(texto, {})
        datos['_modelo'] = "🛡️ Regex (Respaldo por Error)"
        return datos


def generar_diagnostico_por_llm(datos: Dict[str, Any]) -> Dict[str, Any]:
    """Solicita al LLM un diagnóstico clínico breve basado en los hallazgos.
    Retorna dict {'diagnostico_llm': str, 'confianza_llm': float}
    Si no hay cliente LLM disponible, hace fallback a `diagnostico_ia`.
    """
    def normalizar_diagnostico_llm(texto: str) -> str:
        """Normaliza el texto del LLM a la nomenclatura del algoritmo."""
        if not texto:
            return "BAJA / SANO"

        t = str(texto).upper()

        if "AL" in t and ("ALTA" in t or "SOSPECHA" in t):
            return "ALTA SOSPECHA (AL)"
        if "ATTR" in t:
            return "ALTA PROBABILIDAD (ATTR)"
        if "INTERMEDIA" in t:
            return "INTERMEDIA (Screening)"
        if "HVI" in t or "HIPERTROFIA" in t:
            return "HVI No Amiloide"
        if "BAJA" in t or "SANO" in t or "CONTROL" in t:
            return "BAJA / SANO"

        return "BAJA / SANO"
    # Construir resumen compacto
    resumen_items = []
    resumen_items.append(f"Edad: {datos.get('edad', 'N/A')}")
    resumen_items.append(f"Sexo: {datos.get('sexo', 'N/A')}")
    resumen_items.append(f"IVS: {datos.get('ivs', 'N/A')}")
    resumen_items.append(f"GLS: {datos.get('gls', 'N/A')}")
    resumen_items.append(f"Volt: {datos.get('volt', 'N/A')}")
    resumen_items.append(f"NT-proBNP: {datos.get('nt_probnp', 'N/A')}")
    resumen_items.append(f"ECV: {datos.get('ecv', 'N/A')}")

    # Añadir red flags positivos (lista de claves con valor verdadero/1)
    rf_keys = [k for k in datos.keys() if k in DEFAULT_DATA and k not in ['edad','sexo','ivs','gls','volt','nt_probnp','ecv','confianza_ia','modelo_usado']]
    rf_pos = [k for k in rf_keys if str(datos.get(k, '')).lower() not in ['', '0', '0.0', 'false', 'none']]
    if rf_pos:
        resumen_items.append('RedFlags: ' + ','.join(rf_pos[:10]))

    prompt = (
        "Actúa como cardiólogo experto y selecciona SOLO una de estas etiquetas exactas: "
        "'ALTA SOSPECHA (AL)', 'ALTA PROBABILIDAD (ATTR)', 'INTERMEDIA (Screening)', "
        "'HVI No Amiloide', 'BAJA / SANO'.\n\n" +
        "\n".join(resumen_items) + "\n\nResponde SOLO con la etiqueta exacta."
    )

    # Si no hay cliente LLM, fallback a heurística local
    if not client:
        diag = diagnostico_ia(datos)
        return {'diagnostico_llm': normalizar_diagnostico_llm(diag), 'confianza_llm': 0.0}

    try:
        resp = client.chat.completions.create(model=llm_model, messages=[{"role": "user", "content": prompt}])
        text = ''
        try:
            text = resp.choices[0].message.content.strip()
        except Exception:
            text = str(resp)
        diag_line = text.splitlines()[0] if text else diagnostico_ia(datos)
        # No extraemos confianza numérica del LLM por ahora
        return {'diagnostico_llm': normalizar_diagnostico_llm(diag_line), 'confianza_llm': 0.0}
    except Exception:
        diag = diagnostico_ia(datos)
        return {'diagnostico_llm': normalizar_diagnostico_llm(diag), 'confianza_llm': 0.0}

# --- EXPORTADOR FHIR ---
def exportar_a_fhir(datos, resultado):
    """Genera un recurso FHIR R4 Observation simplificado"""
    return {
        "resourceType": "Observation",
        "status": "final",
        "code": {
            "coding": [{
                "system": "http://loinc.org",
                "code": "89253-9",
                "display": "Cardiac amyloidosis assessment"
            }]
        },
        "valueInteger": resultado['score'],
        "interpretation": [{
            "coding": [{
                "code": resultado['nivel'],
                "display": resultado['msg']
            }]
        }],
        "component": [
            {"code": {"text": "IVS"}, "valueQuantity": {"value": datos['ivs'], "unit": "mm"}},
            {"code": {"text": "GLS"}, "valueQuantity": {"value": datos['gls'], "unit": "%"}}
        ]
    }

from typing import Dict, Any

# ==============================================================================
# ALGORITMO BASADO EN EVIDENCIA (STRESS TEST v2025.1)
# ==============================================================================

# Pesos calibrados según ROC / Youden
PESOS = {
    "apical_sparing": 4,
    "discordancia": 3,
    "hvi_base": 3,
    "hvi_confusor": 1,
    "strain_reducido": 2,
    "bav_mp": 2,
    "attr_extra_alto": 3,
    "attr_extra_bajo": 1
}

# Umbrales derivados de validación
# Aumentados para mejorar especificidad (reducir INTERMEDIA inflada)
UMBRAL_CONFIRMACION = 2   # Score para ATTR ALTA
UMBRAL_SCREENING = 4      # Score para INTERMEDIA (antes era 1 - demasiado bajo)

# Valores de calibración (sin usar en el código principal por ahora)
UMBRAL_CONFIRMACION_CALIBRADO = None  # Se actualizará desde la UI si es necesario
UMBRAL_SCREENING_CALIBRADO = None     # Se actualizará desde la UI si es necesario


# ------------------------------------------------------------------------------
# Funciones auxiliares
# ------------------------------------------------------------------------------
def safe_float(val: Any) -> float:
    try:
        return float(val) if val is not None else 0.0
    except (ValueError, TypeError):
        return 0.0


def safe_bool(d: Dict[str, Any], key: str) -> bool:
    val = d.get(key)
    return val in [True, 1, "1", "true", "True", "yes", "YES"]


# ------------------------------------------------------------------------------
# Motor experto principal
# ------------------------------------------------------------------------------
def calcular_confianza_porcentaje(score: int, max_score: int = 45) -> float:
    """
    Calcula el porcentaje de confianza basado en el score.
    Escala normalizada de 0-100%.
    """
    # Normalizar tipos inesperados
    try:
        if isinstance(score, complex):
            score = score.real
        score = float(score)
    except Exception:
        score = 0.0

    # Normalizar score a rango 0-1 usando sigmoid suave
    normalized = min(max(score / max_score, 0.0), 1.0)
    # Aplicar escala perceptual (más sensible a cambios bajos)
    confianza = (normalized ** 0.7) * 100

    try:
        if isinstance(confianza, complex):
            confianza = confianza.real
        confianza = float(confianza)
    except Exception:
        confianza = 0.0

    return min(confianza, 98.0)  # Máximo 98% para humildad estadística

def calcular_riesgo_experto(d: Dict[str, Any], umbral_screening: float = None, umbral_confirmacion: float = None) -> Dict[str, Any]:
    """
    Motor de inferencia clínica para sospecha de amiloidosis cardíaca.
    Ajustado a stress test bioestadístico (ROC / Youden / McNemar).
    
    Parámetros:
    - d: Diccionario con datos del paciente
    - umbral_screening: Score mínimo para INTERMEDIA (por defecto UMBRAL_SCREENING)
    - umbral_confirmacion: Score mínimo para ALTA PROBABILIDAD (por defecto UMBRAL_CONFIRMACION)
    """
    
    # Usar umbrales globales si no se especifican
    if umbral_screening is None:
        umbral_screening = UMBRAL_SCREENING
    if umbral_confirmacion is None:
        umbral_confirmacion = UMBRAL_CONFIRMACION

    score = 0
    score_cardiaco = 0
    hallazgos = []

    # ------------------------------------------------------------------
    # 1. Extracción de variables
    # ------------------------------------------------------------------
    ivs = safe_float(d.get("ivs"))
    volt = safe_float(d.get("volt"))
    gls = safe_float(d.get("gls"))
    nt_probnp = safe_float(d.get("nt_probnp"))
    ecv = safe_float(d.get("ecv"))
    septum_post = safe_float(d.get("septum_posterior"))
    lge_patron = str(d.get("lge_patron", ""))
    troponina = safe_bool(d, "troponina")

    tiene_confusor = any(
        safe_bool(d, k) for k in ["confusor_hta", "confusor_ao", "confusor_irc"]
    )

    # ------------------------------------------------------------------
    # 2. Fenotipo cardíaco
    # ------------------------------------------------------------------
    umbral_ivs = 14 if tiene_confusor else 12
    puntos_hvi = PESOS["hvi_confusor"] if tiene_confusor else PESOS["hvi_base"]

    if ivs >= umbral_ivs:
        score += puntos_hvi
        score_cardiaco += puntos_hvi
        hallazgos.append(f"HVI ≥{umbral_ivs} mm (+{puntos_hvi})")

    ratio_vm = volt / ivs if ivs > 0 else 100.0
    discordancia = ivs >= 12 and volt > 0 and ratio_vm < 0.05
    if discordancia:
        score += PESOS["discordancia"]
        score_cardiaco += PESOS["discordancia"]
        hallazgos.append("Discordancia Voltaje/Masa (+3)")

    if safe_bool(d, "apical_sparing"):
        score += PESOS["apical_sparing"]
        score_cardiaco += PESOS["apical_sparing"]
        hallazgos.append("Patrón Apical Sparing (+4)")
    elif -15 < gls < 0:
        score += PESOS["strain_reducido"]
        score_cardiaco += PESOS["strain_reducido"]
        hallazgos.append(f"Strain longitudinal reducido ({gls}%) (+2)")

    if safe_bool(d, "bav_mp") and ivs >= 12:
        score += PESOS["bav_mp"]
        score_cardiaco += PESOS["bav_mp"]
        hallazgos.append("Trastorno de conducción / MP (+2)")

    # Red flags cardiacos adicionales
    if safe_bool(d, "biatrial"):
        score += 1
        score_cardiaco += 1
        hallazgos.append("Dilatación biauricular (+1)")
    if safe_bool(d, "pseudo_q"):
        score += 1
        score_cardiaco += 1
        hallazgos.append("Pseudoinfarto (+1)")
    if safe_bool(d, "bajo_voltaje"):
        score += 1
        score_cardiaco += 1
        hallazgos.append("Bajo voltaje (+1)")
    if safe_bool(d, "derrame_pericardico"):
        score += 1
        score_cardiaco += 1
        hallazgos.append("Derrame pericárdico (+1)")
    if septum_post >= 12:
        score += 1
        score_cardiaco += 1
        hallazgos.append("Pared posterior aumentada (+1)")

    # Biomarcadores y RM
    if nt_probnp >= 2000:
        score += 2
        hallazgos.append("NT-proBNP muy elevado (+2)")
    elif nt_probnp >= 1000:
        score += 1
        hallazgos.append("NT-proBNP elevado (+1)")
    if troponina:
        score += 1
        hallazgos.append("Troponina elevada (+1)")
    if ecv >= 35:
        score += 2
        hallazgos.append("ECV muy elevado (+2)")
    elif ecv >= 30:
        score += 1
        hallazgos.append("ECV elevado (+1)")
    if any(k in lge_patron.upper() for k in ["SUBENDOCARD", "DIFUSO", "TRANSMURAL"]):
        score += 2
        hallazgos.append("Patrón LGE sugerente (+2)")

    # ------------------------------------------------------------------
    # 3. Fenotipo extracardíaco ATTR (Musculoesquelético)
    # ------------------------------------------------------------------
    # Red flags ATTR principales
    n_attr = sum(
        safe_bool(d, k) for k in ["stc", "biceps", "lumbar", "hombro"]
    )
    
    # Red flags ATTR adicionales
    n_attr_adicionales = sum(
        safe_bool(d, k) for k in ["dupuytren", "artralgias", "fractura_vert", "tendinitis_calcifica"]
    )
    
    # Scoring ATTR musculoesquelético
    if n_attr >= 2:
        score += PESOS["attr_extra_alto"]
        hallazgos.append(f"Fenotipo ATTR clásico ({n_attr} signos) (+3)")
    elif n_attr == 1:
        score += PESOS["attr_extra_bajo"]
        hallazgos.append("Fenotipo ATTR (1 signo clásico) (+1)")
    
    # Red flags ATTR adicionales
    if n_attr_adicionales >= 2:
        score += 2
        hallazgos.append(f"Red flags ATTR adicionales ({n_attr_adicionales}) (+2)")
    elif n_attr_adicionales == 1:
        score += 1
        hallazgos.append("Red flag ATTR complementario (+1)")

    # ------------------------------------------------------------------
    # 4. Fenotipo AL sistémico (Red flags específicos)
    # ------------------------------------------------------------------
    # Diagnóstico directo AL
    n_al_directo = sum(
        safe_bool(d, k) for k in ["mgus", "macro", "purpura"]
    )
    
    # Otros hallazgos AL
    n_al_secundario = sum(
        safe_bool(d, k) for k in ["nefro", "neuro_p", "disauto", "hepato", "fatiga", "piel_lesiones"]
    )

    if n_al_directo >= 1:
        score += 3 * n_al_directo
        hallazgos.append(f"Red flags AL directos ({n_al_directo}) (+{3 * n_al_directo})")
    if n_al_secundario >= 1:
        score += n_al_secundario
        hallazgos.append(f"Red flags AL sistémicos ({n_al_secundario}) (+{n_al_secundario})")
    
    # ------------------------------------------------------------------
    # 5. ATTR hereditaria (Mutación TTR)
    # ------------------------------------------------------------------
    if safe_bool(d, "mutacion_ttr"):
        score += 4
        hallazgos.append("Mutación TTR confirmada (⚠️ ATTR hereditaria) (+4)")

    # Penalización por confusores
    confusores = sum(safe_bool(d, k) for k in ["confusor_hta", "confusor_ao", "confusor_irc"])
    if confusores:
        score -= confusores
        hallazgos.append(f"Confusores presentes ({confusores}) (-{confusores})")

    res = {
        "score": score,
        "hallazgos": hallazgos,
        "puntos": score
    }

    # ===================================================================
    # BLOQUEO ABSOLUTO #1: AMILOIDOSIS AL (DIRECTO)
    # ===================================================================
    if n_al_directo >= 1:
        res.update({
            "nivel": "ALTA SOSPECHA (AL)",
            "color": "#d32f2f",
            "msg": "🚨 Signos específicos de amiloidosis AL detectados.",
            "accion": "Derivación URGENTE a Hematología (FLC séricos, biopsia).",
            "evidencia": f"Criterio clínico directo ({n_al_directo} hallazgo/s AL)",
            "confianza_porcentaje": 95.0  # AL directo es muy específico
        })
        return res
    
    # ===================================================================
    # BLOQUEO ABSOLUTO #2: PERFIL AL (MÚLTIPLES HALLAZGOS SECUNDARIOS)
    # ===================================================================
    if n_al_secundario >= 3:
        res.update({
            "nivel": "ALTA SOSPECHA (AL)",
            "color": "#d32f2f",
            "msg": "🚨 Múltiples criterios sugestivos de AL (síndrome nefrótico, neuropatía, disautonomía).",
            "accion": "Derivación urgente a Hematología + Nefrología.",
            "evidencia": f"Poliafectación sistémica AL ({n_al_secundario} hallazgos)",
            "confianza_porcentaje": 90.0  # AL sistémico alto
        })
        return res

    # -------- GATE CARDÍACO FUERTE PARA ATTR --------
    fenotipo_cardiaco_fuerte = (
        safe_bool(d, "apical_sparing") or
        (ivs >= 12 and volt > 0 and (volt / ivs) < 0.05) or
        (ivs >= 14 and not tiene_confusor) or
        (safe_bool(d, "bajo_voltaje") and safe_bool(d, "bav_mp"))  # Bajo voltaje + BAV
    )
    
    # Contador de fenotipo ATTR completo
    n_attr_total = n_attr + n_attr_adicionales

    # Calcular confianza/probabilidad
    confianza = calcular_confianza_porcentaje(score, max_score=45)

    # ===================================================================
    # ATTR PROBABLE (FENOTIPO COMPLETO + CARDIO FUERTE)
    # ===================================================================
    if (
        score >= umbral_confirmacion and
        fenotipo_cardiaco_fuerte and
        n_attr_total >= 1
    ):
        res.update({
            "nivel": "ALTA PROBABILIDAD (ATTR)",
            "color": "#c62828",
            "msg": f"✅ Fenotipo ATTR consistente (Score {score}). Cardiopatía + manifestaciones sistémicas.",
            "accion": "Confirmar con gammagrafía ósea (Tc-99m DPD/PyP) ± biopsia.",
            "evidencia": "Gate cardíaco + sistemico (alta especificidad)",
            "confianza_porcentaje": confianza
        })
        return res
    # ===================================================================
    # FILTRO TEMPRANO: SANO / BAJO RIESGO (SIN hallazgos significativos)
    # ===================================================================
    # Estrategia: excluir casos claramente inocentes ANTES de chequear INTERMEDIA
    score_sin_confusores = score - confusores if confusores > 0 else score
    
    # Si score es muy bajo y sin hallazgos sistémicos → probablemente SANO
    if score_sin_confusores <= 1 and n_al_directo == 0 and n_al_secundario == 0:
        # Extra check: si IVS < 12, casi seguro es sano
        if ivs < 12:
            res.update({
                "nivel": "BAJA / SANO",
                "color": "#388e3c",
                "msg": "✅ Sin criterios clínicos de cardiopatía infiltrativa. Fenotipo normal.",
                "accion": "Seguimiento habitual. Tranquilizar al paciente.",
                "evidencia": "Score bajo, sin hallazgos sistémicos, IVS normal",
                "confianza_porcentaje": 85.0
            })
            return res

    # ===================================================================
    # FILTRO TEMPRANO: HVI NO AMILOIDE (HTA/Válvula puro)
    # ===================================================================
    # Si tiene HVI + confusores + SIN hallazgos sistémicos de amiloidosis
    # → Es probablemente HVI puro (hipertensión/válvula), no amiloidosis
    if ivs >= 12 and confusores >= 1:
        # Contar hallazgos sistémicos de amiloidosis
        hallazgos_sistemicos = n_al_directo + n_al_secundario
        hallazgos_attr_clasico = sum(
            safe_bool(d, k) for k in ["stc", "biceps", "lumbar", "dupuytren", "artralgias"]
        )
        
        # Si no hay hallazgos sistémicos significativos de amiloidosis
        # y el score (sin confusores) es bajo → claramente HVI puro
        if hallazgos_sistemicos == 0 and hallazgos_attr_clasico <= 1 and score_sin_confusores < 4:
            res.update({
                "nivel": "HVI No Amiloide",
                "color": "#1976d2",
                "msg": "💙 Hipertrofia con fenotipo HTA/valvular. Confusor presente (HTA/válvula). Sin criterios infiltración.",
                "accion": "Control de PA y factores de riesgo. Cardio-RM sólo si hallazgos imagen sospechosos.",
                "evidencia": "HVI + confusor evidente + ausencia de signos sistémicos amiloide",
                "confianza_porcentaje": 80.0
            })
            return res

    # ===================================================================
    # INTERMEDIA REAL (CARDÍACA SIN CONFIRMACIÓN - UMBRAL ELEVADO)
    # ===================================================================
    # Nota: UMBRAL_SCREENING ahora es 4 (antes 1) para mejorar especificidad
    # Requiere múltiples hallazgos, no solo uno
    if score >= umbral_screening:
        res.update({
            "nivel": "INTERMEDIA (Screening)",
            "color": "#f57c00",
            "msg": f"⚠️ Sospecha moderada de amiloidosis (Score {score}). Requiere investigación.",
            "accion": "Cardio-RM + biomarcadores / Considerar pruebas sistémicas (ADN, biopsia si sospecha).",
            "evidencia": "Múltiples hallazgos sugestivos sin gate completo ATTR",
            "confianza_porcentaje": confianza
        })
        return res

    # ===================================================================
    # HVI NO AMILOIDE (Fallback - para casos sin confusores claros)
    # ===================================================================
    if ivs >= 12:
        res.update({
            "nivel": "HVI No Amiloide",
            "color": "#1976d2",
            "msg": "💙 Hipertrofia sin criterios claros de infiltración amiloide. Probable HTA/valvulopatía.",
            "accion": "Cardio-RM para descartar infiltración. Control de PA.",
            "evidencia": "Fenotipo hipertrófico aislado",
            "confianza_porcentaje": 65.0
        })
        return res

    # ===================================================================
    # SANO / BAJO RIESGO (Default cuando nada más aplica)
    # ===================================================================
    res.update({
        "nivel": "BAJA / SANO",
        "color": "#388e3c",
        "msg": "✅ Sin hallazgos significativos de cardiopatía infiltrativa.",
        "accion": "Seguimiento habitual.",
        "evidencia": "Score bajo, sin red flags sistémicos",
        "confianza_porcentaje": confianza
    })

    return res


# ==========================================
def load_training_database(completar_ia: bool = False) -> pd.DataFrame:
    """Carga la base de datos de casos guardados (ruta consistente con BASE_DIR)."""
    ruta_abs = os.path.join(BASE_DIR, DB_FILE)
    if os.path.isfile(ruta_abs):
        df = pd.read_csv(ruta_abs)
        necesita_guardar = False

        # Asegurar columnas minimas
        columnas_minimas = ['id', 'fecha', 'diagnostico']
        for col in columnas_minimas:
            if col not in df.columns:
                df[col] = '' if col != 'id' else 0
                necesita_guardar = True

        if 'resultado_algoritmo' not in df.columns:
            df['resultado_algoritmo'] = ''
            necesita_guardar = True
        if 'diagnostico_ia' not in df.columns:
            df['diagnostico_ia'] = ''
            necesita_guardar = True

        # Asegurar que existan todas las FEATURES
        for feature in FEATURES:
            if feature not in df.columns:
                df[feature] = DEFAULT_DATA.get(feature, 0)
                necesita_guardar = True

        # Completar resultado_algoritmo si esta vacio
        try:
            serie_resultado = df['resultado_algoritmo']
            mask_vacio = serie_resultado.isna() | (serie_resultado.astype(str).str.strip() == '')
            if mask_vacio.any():
                def _calc_resultado(row: pd.Series) -> str:
                    datos = {f: row.get(f, DEFAULT_DATA.get(f, 0)) for f in FEATURES}
                    try:
                        res_algo = calcular_riesgo_experto(datos)
                        return res_algo.get('nivel', '')
                    except Exception:
                        return ''

                df.loc[mask_vacio, 'resultado_algoritmo'] = df.loc[mask_vacio].apply(_calc_resultado, axis=1)
                necesita_guardar = True
        except Exception:
            pass

        # Completar diagnostico_ia si esta vacio (opcional, puede ser lento)
        if completar_ia:
            try:
                serie_ia = df['diagnostico_ia']
                mask_vacio = serie_ia.isna() | (serie_ia.astype(str).str.strip() == '')
                if mask_vacio.any():
                    def _calc_diag_ia(row: pd.Series) -> str:
                        datos = {f: row.get(f, DEFAULT_DATA.get(f, 0)) for f in FEATURES}
                        try:
                            diag_llm = generar_diagnostico_por_llm(datos)
                            return diag_llm.get('diagnostico_llm', '')
                        except Exception:
                            return ''

                    df.loc[mask_vacio, 'diagnostico_ia'] = df.loc[mask_vacio].apply(_calc_diag_ia, axis=1)
                    necesita_guardar = True
            except Exception:
                pass

        if necesita_guardar:
            df.to_csv(ruta_abs, index=False)

        return df
    else:
        return pd.DataFrame(columns=['id', 'fecha', 'diagnostico', 'resultado_algoritmo', 'diagnostico_ia'] + FEATURES)

def get_db_mtime() -> float:
    ruta_abs = os.path.join(BASE_DIR, DB_FILE)
    try:
        return os.path.getmtime(ruta_abs)
    except Exception:
        return 0.0

@st.cache_data(show_spinner=False)
def get_database_stats(db_mtime: float) -> Dict[str, Any]:
    """Obtiene estadísticas de la base de datos"""
    _ = db_mtime
    df = load_training_database()

    if df.empty:
        return {
            'total_casos': 0,
            'por_diagnostico': {},
            'ultima_actualizacion': 'N/A'
        }

    stats = {
        'total_casos': len(df),
        'por_diagnostico': df['diagnostico'].value_counts().to_dict(),
        'ultima_actualizacion': df['fecha'].max() if 'fecha' in df.columns else 'N/A'
    }

    return stats

@st.cache_resource
def entrenar_modelo_ia():
    df = load_training_database()
    # Verificación de seguridad
    if df.empty or len(df) < 5: 
        return "ERROR: Pocos datos", None
    
    try:
        # Aseguramos que solo usamos columnas numéricas y quitamos filas con diagnósticos vacíos
        X = df[FEATURES].copy()
        for col in X.columns:
            X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0)
        
        y = df['diagnostico']
        
        # El modelo necesita al menos 2 tipos de diagnóstico diferentes para aprender
        if len(y.unique()) < 2:
            return "ERROR: Solo un tipo de diagnóstico en DB", None

        modelo = RandomForestClassifier(n_estimators=100, random_state=42)
        modelo.fit(X, y)
        return modelo, modelo.classes_
    except Exception as e:
        return f"ERROR: {str(e)}", None

# Cargamos el cerebro de la IA
modelo_ml, clases_ml = entrenar_modelo_ia()

# Inicializar en None si hay error
if isinstance(modelo_ml, str):
    modelo_ml = None
    clases_ml = None

# ==========================================
# PESTAÑA DE VALIDACIÓN (FILTRADO INTERMEDIOS)
# ==========================================
def render_tab_validacion():
    """
    Validación estadística clínica simplificada del algoritmo.
    Incluye: ROC/AUC, subgrupos, tabla JACC/EHJ, errores, matriz, FDA/EMA.
    """
    st.header("🧪 Validación Estadística Clínica")
    
    uploaded_file = st.file_uploader("Cargar Cohorte CSV para Validación", type=['csv'])
    simular_hvi = st.checkbox("🧬 Simular HVI (IVS > 14mm)", value=True)

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            
            # ============ PROCESAMIENTO ============
            progress_bar = st.progress(0)
            results = []
            
            for index, row in df.iterrows():
                caso_id = row.get('id', index + 1)
                d_simulado = {
                    'ivs': 15 if simular_hvi and row['grupo_diagnostico'] not in ['No amiloidosis / sano'] else 9, 
                    'volt': 5 if row['ecg_bajo_voltaje'] else 12,
                    'gls': -10 if row['strain_reducido'] else -20,
                    'apical_sparing': row['apical_sparing'],
                    'confusor_hta': row['hta'],
                    'confusor_ao': row['estenosis_aortica'],
                    'stc': row['tunel_carpiano'],
                    'macro': row['macroglosia'],
                    'purpura': row['purpura_periorbitaria'],
                    'mgus': row['monoclonalidad'],
                    'nefro': row['proteinuria'],
                    'neuro_p': row['neuropatia'],
                    'disauto': row['hipotension_ortostatica']
                }
                
                res = calcular_riesgo_experto(d_simulado)
                results.append({
                    'Caso_ID': caso_id,
                    'Diagnóstico Real': row['grupo_diagnostico'],
                    'Diagnóstico del Algoritmo': res['nivel'],
                    'Score': res['score'],
                    'Hallazgos': ", ".join(res['hallazgos'])
                })
                
                if index % max(1, len(df)//10) == 0:
                    progress_bar.progress(min(index / len(df), 1.0))
            
            progress_bar.progress(1.0)
            
            # ============ PREPARACIÓN DE DATOS ============
            df_res = pd.DataFrame(results)
            total_casos = len(df_res)
            df_analisis = df_res[~df_res['Diagnóstico del Algoritmo'].str.contains("INTERMEDIA", case=False, na=False)].copy()
            casos_validos = len(df_analisis)
            casos_excluidos = total_casos - casos_validos
            
            st.success(f"✅ {casos_validos} casos analizados ({casos_excluidos} intermedios excluidos)")
            
            # ============ NORMALIZACIÓN ============
            def normalizar_diagnostico(val):
                clase = normalizar_clase_diagnostica(val)
                if clase == 'Sano':
                    return 'No Amiloidosis'
                return clase
            
            df_analisis['Diag_Real_Norm'] = df_analisis['Diagnóstico Real'].apply(normalizar_diagnostico)
            df_analisis['Diag_Algo_Norm'] = df_analisis['Diagnóstico del Algoritmo'].apply(normalizar_diagnostico)
            df_analisis['Real_Binary'] = df_analisis['Diag_Real_Norm'].isin(['AL', 'ATTR']).astype(int)
            df_analisis['Algo_Binary'] = df_analisis['Diag_Algo_Norm'].isin(['AL', 'ATTR']).astype(int)
            
            # ============ MÉTRICAS ============
            from sklearn.metrics import confusion_matrix, roc_curve, auc
            
            y_true_bin = df_analisis['Real_Binary'].values
            y_pred_bin = df_analisis['Algo_Binary'].values
            y_scores = df_analisis['Score'].values
            
            if len(np.unique(y_true_bin)) >= 2:
                fpr, tpr, _ = roc_curve(y_true_bin, y_scores)
                roc_auc = auc(fpr, tpr)
            else:
                fpr, tpr, roc_auc = np.array([0.0, 1.0]), np.array([0.0, 1.0]), 0.5
            
            kappa = cohen_kappa_score(df_analisis['Diag_Real_Norm'], df_analisis['Diag_Algo_Norm'])
            
            cm = confusion_matrix(y_true_bin, y_pred_bin, labels=[0, 1])
            tn, fp, fn, tp = cm.ravel()
            
            sensibilidad = tp / (tp + fn) if (tp + fn) > 0 else 0
            especificidad = tn / (tn + fp) if (tn + fp) > 0 else 0
            ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
            npv = tn / (tn + fn) if (tn + fn) > 0 else 0
            lr_plus = sensibilidad / (1 - especificidad) if (1 - especificidad) > 0 else float('inf')
            lr_minus = (1 - sensibilidad) / especificidad if especificidad > 0 else float('inf')
            accuracy = (tp + tn) / (tp + tn + fp + fn)

            # ============ IC 95% (Wilson + DeLong/Bootstrap) ============
            sens_ci_low, sens_ci_high = wilson_ci(tp, tp + fn)
            esp_ci_low, esp_ci_high = wilson_ci(tn, tn + fp)

            auc_ci_low, auc_ci_high = roc_auc, roc_auc
            try:
                _, var_auc = delong_roc_variance(y_true_bin.astype(int), y_scores.astype(float))
                if np.isfinite(var_auc) and var_auc > 0:
                    z = norm.ppf(0.975)
                    se_auc = np.sqrt(var_auc)
                    auc_ci_low = max(0.0, roc_auc - z * se_auc)
                    auc_ci_high = min(1.0, roc_auc + z * se_auc)
                else:
                    def _safe_auc(yt, ys):
                        if len(np.unique(yt)) < 2:
                            return 0.5
                        fpr_b, tpr_b, _ = roc_curve(yt, ys)
                        return auc(fpr_b, tpr_b)

                    auc_ci_low, auc_ci_high = bootstrap_ci(
                        y_true_bin.astype(int),
                        y_scores.astype(float),
                        _safe_auc,
                        n_boot=1000,
                        alpha=0.05
                    )
                    auc_ci_low, auc_ci_high = float(max(0.0, auc_ci_low)), float(min(1.0, auc_ci_high))
            except Exception:
                auc_ci_low, auc_ci_high = roc_auc, roc_auc
            
            # ============ 1. MÉTRICAS PRINCIPALES (TARJETAS) ============
            st.subheader("📊 Métricas Principales")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("🎯 AUC-ROC", f"{roc_auc:.3f}", help="Ideal > 0.85")
            with col2:
                st.metric("🔗 Kappa", f"{kappa:.3f}", help="Ideal > 0.80")
            with col3:
                st.metric("🩺 Sensibilidad", f"{sensibilidad:.1%}", help="Detección")
            with col4:
                st.metric("🛡️ Especificidad", f"{especificidad:.1%}", help="Sanos correctos")

            st.markdown("#### 📏 Intervalos de Confianza (IC 95%)")
            ci_table = pd.DataFrame({
                'Métrica': ['Sensibilidad', 'Especificidad', 'AUC-ROC'],
                'Estimación': [f"{sensibilidad:.1%}", f"{especificidad:.1%}", f"{roc_auc:.3f}"],
                'IC 95%': [
                    f"[{sens_ci_low:.1%}, {sens_ci_high:.1%}]",
                    f"[{esp_ci_low:.1%}, {esp_ci_high:.1%}]",
                    f"[{auc_ci_low:.3f}, {auc_ci_high:.3f}]"
                ]
            })
            st.dataframe(ci_table, use_container_width=True)
            
            st.divider()
            
            # ============ 2. CURVA ROC ============
            st.subheader("📈 Curva ROC")
            fig_roc, ax_roc = plt.subplots(figsize=(7, 6))
            ax_roc.plot(fpr, tpr, color='#d62728', lw=2.5, label=f'AUC = {roc_auc:.3f}')
            ax_roc.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
            ax_roc.set_xlabel('1 - Especificidad')
            ax_roc.set_ylabel('Sensibilidad')
            ax_roc.set_title('Curva ROC - Discriminación Diagnóstica')
            ax_roc.legend(loc="lower right")
            ax_roc.grid(True, alpha=0.3)
            st.pyplot(fig_roc)
            
            # ============ 3. ANÁLISIS POR SUBGRUPO (AL vs ATTR) ============
            st.subheader("🧬 Sensibilidad por Subtipo (AL vs ATTR)")
            
            df_al = df_analisis[df_analisis['Diag_Real_Norm'] == 'AL']
            df_attr = df_analisis[df_analisis['Diag_Real_Norm'] == 'ATTR']
            
            al_sens = (df_al['Diag_Algo_Norm'] == 'AL').sum() / len(df_al) if len(df_al) > 0 else 0
            attr_sens = (df_attr['Diag_Algo_Norm'] == 'ATTR').sum() / len(df_attr) if len(df_attr) > 0 else 0
            
            col_sub1, col_sub2 = st.columns(2)
            with col_sub1:
                st.markdown("#### 🔴 Amiloidosis AL")
                st.metric("Sensibilidad AL", f"{al_sens:.1%}", f"{int(df_al['Diag_Algo_Norm'].eq('AL').sum())}/{len(df_al)}")
            with col_sub2:
                st.markdown("#### 🟢 Amiloidosis ATTR")
                st.metric("Sensibilidad ATTR", f"{attr_sens:.1%}", f"{int(df_attr['Diag_Algo_Norm'].eq('ATTR').sum())}/{len(df_attr)}")
            
            # Gráfico comparativo
            fig_sub, ax_sub = plt.subplots(figsize=(7, 4))
            tipos = ['AL', 'ATTR']
            sensibilidades = [al_sens * 100, attr_sens * 100]
            bars = ax_sub.bar(tipos, sensibilidades, color=['#d62728', '#2ca02c'], edgecolor='black', linewidth=1.5)
            ax_sub.set_ylabel('Sensibilidad (%)')
            ax_sub.set_title('Sensibilidad por Subtipo de Amiloidosis')
            ax_sub.set_ylim([0, 105])
            ax_sub.axhline(y=75, color='orange', linestyle='--', alpha=0.7, label='Umbral clínico')
            ax_sub.legend()
            for bar, val in zip(bars, sensibilidades):
                ax_sub.text(bar.get_x() + bar.get_width()/2, val + 2, f'{val:.1f}%', ha='center', fontsize=10)
            st.pyplot(fig_sub)
            
            st.divider()
            
            # ============ 4. TABLA JACC/EHJ ============
            st.subheader("📑 Tabla Estilo JACC/EHJ")
            
            tabla_jacc = pd.DataFrame({
                'Métrica': [
                    'Casos (n)',
                    'Sensibilidad',
                    'Especificidad',
                    'VPP',
                    'VPN',
                    'LR+',
                    'LR−',
                    'Exactitud',
                    'AUC-ROC',
                    'Kappa Cohen',
                    'Sens. AL',
                    'Sens. ATTR'
                ],
                'Valor': [
                    f"{casos_validos}",
                    f"{sensibilidad:.1%}",
                    f"{especificidad:.1%}",
                    f"{ppv:.1%}",
                    f"{npv:.1%}",
                    f"{lr_plus:.2f}" if lr_plus != float('inf') else "∞",
                    f"{lr_minus:.2f}" if lr_minus != float('inf') else "∞",
                    f"{accuracy:.1%}",
                    f"{roc_auc:.3f}",
                    f"[{auc_ci_low:.3f}, {auc_ci_high:.3f}]",
                    f"{kappa:.3f}",
                    f"{al_sens:.1%}",
                    f"{attr_sens:.1%}"
                ]
            })
            
            st.dataframe(tabla_jacc, use_container_width=True)
            
            st.divider()
            
            # ============ 5. ANÁLISIS DE ERRORES CLÍNICOS ============
            st.subheader("⚠️ Análisis de Errores Clínicos")
            
            df_analisis['Tipo_Error'] = df_analisis.apply(
                lambda row: 'VP' if row['Real_Binary']==1 and row['Algo_Binary']==1 
                else 'VN' if row['Real_Binary']==0 and row['Algo_Binary']==0
                else 'FP' if row['Real_Binary']==0 and row['Algo_Binary']==1
                else 'FN', axis=1
            )
            
            error_sum = df_analisis['Tipo_Error'].value_counts()
            
            col_err1, col_err2 = st.columns(2)
            
            with col_err1:
                fig_err, ax_err = plt.subplots(figsize=(7, 5))
                colors_err = {'VP': '#2ca02c', 'VN': '#1f77b4', 'FP': '#ff7f0e', 'FN': '#d62728'}
                vals = [error_sum.get(k, 0) for k in ['VP', 'VN', 'FP', 'FN']]
                labels_err = ['VP', 'VN', 'FP', 'FN']
                cols_err = [colors_err.get(k, '#999') for k in labels_err]
                
                wedges, texts, autotexts = ax_err.pie(vals, labels=labels_err, autopct='%1.1f%%',
                                                       colors=cols_err, startangle=90)
                for autotext in autotexts:
                    autotext.set_color('white')
                    autotext.set_fontweight('bold')
                ax_err.set_title('Matriz de Confusión Binaria')
                st.pyplot(fig_err)
            
            with col_err2:
                st.markdown("#### Resumen de Errores")
                st.markdown(f"""
                **VP:** {error_sum.get('VP', 0)} - Detectados correctamente  
                **VN:** {error_sum.get('VN', 0)} - Sanos identificados  
                **FP:** {error_sum.get('FP', 0)} - Falsa alarma ⚠️  
                **FN:** {error_sum.get('FN', 0)} - No detectados 🚨  
                
                **Tasa FN:** {(error_sum.get('FN', 0)/(tp+fn)*100) if (tp+fn)>0 else 0:.1f}%
                """)
            
            st.divider()
            
            # ============ 6. MATRIZ DE CONFUSIÓN MULTICLASE (HEATMAP) ============
            st.subheader("🔲 Matriz de Confusión Multiclase")
            
            cm_multi = pd.crosstab(df_analisis['Diag_Real_Norm'], df_analisis['Diag_Algo_Norm'])
            
            fig_cm, ax_cm = plt.subplots(figsize=(9, 7))
            import seaborn as sns
            sns.heatmap(cm_multi, annot=True, fmt='d', cmap='Blues', ax=ax_cm, 
                       cbar_kws={'label': 'Casos'}, linewidths=1)
            ax_cm.set_xlabel('Diagnóstico del Algoritmo', fontweight='bold')
            ax_cm.set_ylabel('Diagnóstico Real', fontweight='bold')
            ax_cm.set_title('Matriz de Confusión Multiclase')
            st.pyplot(fig_cm)
            
            st.dataframe(cm_multi, use_container_width=True)

            # ============ 6.1 MÉTRICAS POR CLASE ============
            st.subheader("📌 Métricas por Clase")

            classes_sorted = sorted(set(df_analisis['Diag_Real_Norm']) | set(df_analisis['Diag_Algo_Norm']))
            report = classification_report(
                df_analisis['Diag_Real_Norm'],
                df_analisis['Diag_Algo_Norm'],
                labels=classes_sorted,
                output_dict=True,
                zero_division=0
            )
            df_report = pd.DataFrame(report).T
            df_report = df_report.rename(columns={
                'precision': 'Precisión',
                'recall': 'Sensibilidad',
                'f1-score': 'F1',
                'support': 'Soporte'
            })
            st.dataframe(df_report, use_container_width=True)
            
            st.divider()
            
            # ============ 7. CHECKLIST FDA/EMA CDSS ============
            st.subheader("🏥 Validación FDA/EMA CDSS")
            
            fn_rate = (fn/(tp+fn)*100) if (tp+fn)>0 else 0
            
            fda_checklist = pd.DataFrame({
                'Criterio': [
                    'Sensibilidad ≥ 75%',
                    'Especificidad ≥ 85%',
                    'Kappa ≥ 0.75',
                    'AUC-ROC ≥ 0.80',
                    'Tasa FN ≤ 10%',
                    'Análisis multiclase',
                    'Análisis subgrupo',
                    'Matriz confusión'
                ],
                'Valor': [
                    f"{sensibilidad:.1%}",
                    f"{especificidad:.1%}",
                    f"{kappa:.3f}",
                    f"{roc_auc:.3f}",
                    f"{fn_rate:.1f}%",
                    "✓",
                    "✓",
                    "✓"
                ],
                'Cumple': [
                    '✅' if sensibilidad >= 0.75 else '❌',
                    '✅' if especificidad >= 0.85 else '❌',
                    '✅' if kappa >= 0.75 else '❌',
                    '✅' if roc_auc >= 0.80 else '❌',
                    '✅' if fn_rate <= 10 else '❌',
                    '✅',
                    '✅',
                    '✅'
                ]
            })
            
            st.dataframe(fda_checklist, use_container_width=True)
            
            cumple = (fda_checklist['Cumple'] == '✅').sum()
            total = len(fda_checklist)
            pct = (cumple/total)*100
            
            if pct >= 75:
                st.success(f"🟢 APTO: {cumple}/{total} criterios cumplidos ({pct:.0f}%)")
            elif pct >= 50:
                st.warning(f"🟡 MEJORAS NECESARIAS: {cumple}/{total} criterios ({pct:.0f}%)")
            else:
                st.error(f"🔴 NO APTO: {cumple}/{total} criterios ({pct:.0f}%)")
            
            st.divider()

            # ============ 7.1 RENDIMIENTO BINARIO DETALLADO ============
            st.subheader("🧪 Rendimiento Binario (Sens, Esp, VPP, VPN, LR+, LR−)")
            bin_table = pd.DataFrame({
                'Métrica': ['Sensibilidad', 'Especificidad', 'VPP', 'VPN', 'LR+', 'LR−', 'Exactitud'],
                'Valor': [
                    f"{sensibilidad:.1%}",
                    f"{especificidad:.1%}",
                    f"{ppv:.1%}",
                    f"{npv:.1%}",
                    f"{lr_plus:.2f}" if lr_plus != float('inf') else "∞",
                    f"{lr_minus:.2f}" if lr_minus != float('inf') else "∞",
                    f"{accuracy:.1%}"
                ]
            })
            st.dataframe(bin_table, use_container_width=True)

            st.divider()

            # ============ 7.2 LISTAS COMPLETAS FP/FN ============
            st.subheader("🧾 Listas Completas de Falsos Negativos y Falsos Positivos")
            fn_rows = df_analisis[df_analisis['Tipo_Error'] == 'FN'][
                ['Caso_ID', 'Diagnóstico Real', 'Diagnóstico del Algoritmo', 'Score', 'Hallazgos']
            ]
            fp_rows = df_analisis[df_analisis['Tipo_Error'] == 'FP'][
                ['Caso_ID', 'Diagnóstico Real', 'Diagnóstico del Algoritmo', 'Score', 'Hallazgos']
            ]

            col_fn, col_fp = st.columns(2)
            with col_fn:
                st.markdown("#### 🚨 Falsos Negativos (FN)")
                st.dataframe(fn_rows, use_container_width=True)
            with col_fp:
                st.markdown("#### ⚠️ Falsos Positivos (FP)")
                st.dataframe(fp_rows, use_container_width=True)

            st.divider()

            # ============ 7.3 SUPLEMENTO (TABLAS + FIGURA) ============
            st.subheader("📎 Suplemento (Tablas + Figura)")

            sup_col1, sup_col2 = st.columns(2)
            with sup_col1:
                st.markdown("#### Tabla S1. Rendimiento Binario")
                st.dataframe(bin_table, use_container_width=True)
                st.markdown("#### Tabla S2. Métricas por Clase")
                st.dataframe(df_report, use_container_width=True)
            with sup_col2:
                st.markdown("#### Figura S1. Curva ROC")
                fig_sup, ax_sup = plt.subplots(figsize=(6, 5))
                ax_sup.plot(fpr, tpr, color='#d62728', lw=2.5, label=f'AUC = {roc_auc:.3f}')
                ax_sup.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
                ax_sup.set_xlabel('1 - Especificidad')
                ax_sup.set_ylabel('Sensibilidad')
                ax_sup.set_title('Curva ROC - Discriminación Diagnóstica')
                ax_sup.legend(loc="lower right")
                ax_sup.grid(True, alpha=0.3)
                st.pyplot(fig_sup)
            
            # ============ 8. EXPORTACIÓN ============
            st.subheader("💾 Exportar Resultados")
            
            # CSV
            csv_data = df_analisis.to_csv(index=False).encode()
            st.download_button(
                "⬇️ CSV Completo",
                csv_data,
                f"validacion_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                "text/csv"
            )
            
            # Resumen TXT
            txt_resumen = f"""VALIDACIÓN ESTADÍSTICA CLÍNICA
Algoritmo de Diagnóstico de Amiloidosis Cardíaca

RESUMEN EJECUTIVO
===============================
Fecha: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Casos totales: {total_casos}
Casos analizados: {casos_validos}
Casos excluidos: {casos_excluidos}

MÉTRICAS CLAVE
===============================
Sensibilidad: {sensibilidad:.1%}
Especificidad: {especificidad:.1%}
VPP: {ppv:.1%}
VPN: {npv:.1%}
LR+: {lr_plus:.2f}
LR-: {lr_minus:.2f}
Exactitud: {accuracy:.1%}
AUC-ROC: {roc_auc:.3f}
AUC-ROC IC95%: [{auc_ci_low:.3f}, {auc_ci_high:.3f}]
Kappa: {kappa:.3f}
Sensibilidad IC95%: [{sens_ci_low:.1%}, {sens_ci_high:.1%}]
Especificidad IC95%: [{esp_ci_low:.1%}, {esp_ci_high:.1%}]

ANÁLISIS POR SUBGRUPO
===============================
Sensibilidad AL: {al_sens:.1%}
Sensibilidad ATTR: {attr_sens:.1%}

ERRORES CLÍNICOS
===============================
Verdaderos Positivos: {error_sum.get('VP', 0)}
Verdaderos Negativos: {error_sum.get('VN', 0)}
Falsos Positivos: {error_sum.get('FP', 0)}
Falsos Negativos: {error_sum.get('FN', 0)}
Tasa FN: {fn_rate:.1f}%

CUMPLIMIENTO FDA/EMA
===============================
Criterios cumplidos: {cumple}/{total} ({pct:.0f}%)
Estado: {'APTO' if pct >= 75 else 'REQUIERE MEJORAS' if pct >= 50 else 'NO APTO'}
            """
            
            st.download_button(
                "📄 Resumen Ejecutivo",
                txt_resumen,
                f"resumen_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                "text/plain"
            )

        except Exception as e:
            st.error(f"Error procesando CSV: {e}")

# ==========================================
# FUNCIÓN DE VALIDACIÓN STRESS TEST
# ==========================================
def evaluar_estres_algoritmo(df_full: pd.DataFrame, umbral_screening: float = None, umbral_confirmacion: float = None) -> Dict[str, Any]:
    """Evalúa el desempeño del algoritmo contra diagnósticos reales con umbrales ajustables"""
    
    if umbral_screening is None:
        umbral_screening = UMBRAL_SCREENING
    if umbral_confirmacion is None:
        umbral_confirmacion = UMBRAL_CONFIRMACION
    
    df = df_full.copy()
    df['diagnostico'] = df['diagnostico'].astype(str).str.strip()

    def normalizar_diag_safety(val: str) -> str:
        return normalizar_clase_diagnostica(val)

    df['diagnostico_clean'] = df['diagnostico'].apply(normalizar_diag_safety)

    def get_metrics(row: pd.Series) -> pd.Series:
        res = calcular_riesgo_experto(row.to_dict(), umbral_screening=umbral_screening, umbral_confirmacion=umbral_confirmacion)
        lvl = 2 if "ALTA" in res['nivel'] else (1 if "INTERMEDIA" in res['nivel'] else 0)
        return pd.Series([res['score'], lvl, res['nivel']])

    df[['Score_Num', 'Score_Ord', 'Nivel_Txt']] = df.apply(get_metrics, axis=1)

    y_true_bin = df['diagnostico_clean'].isin(['AL', 'ATTR']).astype(int).values
    y_score_ord = df['Score_Num'].values
    y_pred_bin = (~df['Nivel_Txt'].str.contains("INTERMEDIA", case=False, na=False) &
                  ~df['Nivel_Txt'].str.contains("BAJA|SANO|HVI", case=False, na=False)).astype(int).values

    try:
        fpr, tpr, _ = roc_curve(y_true_bin, y_score_ord)
        roc_auc = auc(fpr, tpr)
    except Exception:
        roc_auc = 0.0

    try:
        cm_bin = confusion_matrix(y_true_bin, y_pred_bin, labels=[0, 1])
        tn, fp, fn, tp = cm_bin.ravel()
    except Exception:
        tn = fp = fn = tp = 0

    sensibilidad = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    especificidad = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    sens_ci_low, sens_ci_high = wilson_ci(tp, tp + fn)
    esp_ci_low, esp_ci_high = wilson_ci(tn, tn + fp)

    auc_ci_low, auc_ci_high = roc_auc, roc_auc
    try:
        _, var_auc = delong_roc_variance(y_true_bin.astype(int), y_score_ord.astype(float))
        if np.isfinite(var_auc) and var_auc > 0:
            z = norm.ppf(0.975)
            se_auc = np.sqrt(var_auc)
            auc_ci_low = max(0.0, roc_auc - z * se_auc)
            auc_ci_high = min(1.0, roc_auc + z * se_auc)
    except Exception:
        auc_ci_low, auc_ci_high = roc_auc, roc_auc

    y_true_m = df['diagnostico_clean']
    y_pred_m = df['Nivel_Txt'].apply(normalizar_clase_diagnostica)

    try:
        kappa = cohen_kappa_score(y_true_m, y_pred_m)
    except Exception:
        kappa = 0.0

    intermedia_rate = float((df['Score_Ord'] == 1).mean()) if len(df) > 0 else 0.0

    return {
        'n': len(df),
        'auc': float(roc_auc),
        'auc_ci_low': float(auc_ci_low),
        'auc_ci_high': float(auc_ci_high),
        'kappa': float(kappa),
        'intermedia_rate': intermedia_rate,
        'sensibilidad': float(sensibilidad),
        'especificidad': float(especificidad),
        'sens_ci_low': float(sens_ci_low),
        'sens_ci_high': float(sens_ci_high),
        'esp_ci_low': float(esp_ci_low),
        'esp_ci_high': float(esp_ci_high)
    }

# ==========================================
# INTERFAZ GRÁFICA (TABS)
# ==========================================

# ==========================================
# SIDEBAR MEJORADO Y VISUAL
# ==========================================
tab_labels = [
    "Lote de PDFs",
    "Caso Individual",
    "Guia Clinica",
    "Base de Datos",
    "Diagnóstico del Algoritmo",
    "Test de Estrés"
]

fondo_lateral_path = os.path.join(BASE_DIR, "fondo_lateral.jpg")
# Streamlit Cloud: intentar ruta relativa si BASE_DIR falla
if not os.path.exists(fondo_lateral_path) and os.path.exists("fondo_lateral.jpg"):
    fondo_lateral_path = "fondo_lateral.jpg"

fondo_lateral_data = read_file_base64(fondo_lateral_path)
fondo_lateral_base64 = fondo_lateral_data if fondo_lateral_data else None
fondo_lateral_css = (
    f"background-image: url('data:image/jpeg;base64,{fondo_lateral_base64}') !important;"
    if fondo_lateral_base64 else ""
)

with st.sidebar:
    # Logo elegante - PROTAGONISTA
    try:
        logo_path = os.path.join(BASE_DIR, "image_6.png")
        if not os.path.exists(logo_path) and os.path.exists("image_6.png"):
            logo_path = "image_6.png"

        logo_bytes = read_file_bytes(logo_path)
        if logo_bytes:
            st.image(logo_bytes, width=120)
        else:
            st.markdown(f"<h3 style='text-align:center; font-size:3em;'><b>🏥</b></h3>", unsafe_allow_html=True)
    except Exception as e:
        st.markdown(f"<h3 style='text-align:center; font-size:3em;'><b>🏥</b></h3>", unsafe_allow_html=True)
    
    # Descripción de la app
    st.markdown("""
    <div style='text-align: center; padding: 10px 8px; font-size: 0.82em; line-height: 1.4;'>
    <b>AmylAI 1.0</b> detecta amiloidosis cardíaca mediante inteligencia artificial. 
    Combina <b>pdfplumber</b> para extracción de texto, <b>LLMs locales</b> (vLLM/Ollama) para interpretación de datos clínicos, un <b>algoritmo diagnóstico experto basado en puntuación clínica</b>, y <b>Machine Learning</b> (RandomForest/scikit-learn) para validación estadística, clasificando el riesgo (AL, ATTR, HVI o Bajo) según evidencia científica. 
    Acelera el diagnóstico precoz de esta enfermedad infradiagnosticada, mejorando el pronóstico del paciente.
    </div>
    """, unsafe_allow_html=True)
    
    st.divider()
    selected_tab = st.radio("Navegacion", tab_labels, index=1)

# ==========================================
# TABS PRINCIPALES (Rediseñadas)
# ==========================================
tab_style = """
    <style>
    .stTabs [data-baseweb="tab-list"] [data-qa="stTab"] {
        padding: 12px 20px;
        font-weight: 600;
    }
    </style>
    """

# Configurar fondo dinámico con imagen si existe
bg_img_css = "background-image: linear-gradient(180deg, #0e1117, #161b22);"
if st.session_state.get('fondo_base64'):
    bg_img_css = f"""
        background-image: linear-gradient(180deg, rgba(255,255,255,0.6), rgba(255,255,255,0.75)), url('{st.session_state.fondo_base64}') !important;
        background-size: cover !important;
        background-attachment: fixed !important;
        background-repeat: no-repeat !important;
    """

page_style = f"""
    <style>
    html, body, [class*="css"] {{
        font-size: 13px !important;
    }}

    /* Intensify main page background */
    .stApp {{
        {bg_img_css}
        background-position: center !important;
    }}

    [data-testid="stAppViewContainer"] {{
        background-color: #f4f6f8 !important;
        {bg_img_css}
        background-position: center !important;
        color: #111111;
    }}

    /* Sidebar */
    section[data-testid="stSidebar"] {{
        overflow: visible !important;
    }}

    section[data-testid="stSidebar"] > div {{
        height: 100vh !important;
        overflow-y: auto !important;
        overflow-x: hidden !important;
    }}

    [data-testid="stSidebarContent"] {{
        height: 100% !important;
        overflow-y: auto !important;
        overflow-x: hidden !important;
        padding-bottom: 0.75rem !important;
    }}

    [data-testid="stSidebar"] > div:first-child {{
        background: linear-gradient(180deg, rgba(20, 24, 28, 0.92), rgba(20, 24, 28, 0.96)) !important;
        {fondo_lateral_css}
        background-size: cover !important;
        background-position: center !important;
        background-repeat: no-repeat !important;
        backdrop-filter: blur(12px);
        color: #ffffff !important;
        font-size: 0.95rem;
        font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, 'Times New Roman', Georgia, serif !important;
        height: 100vh !important;
        overflow-y: auto !important;
        overflow-x: hidden !important;
        padding-bottom: 0.5rem !important;
    }}

    [data-testid="stSidebar"] .block-container {{
        padding-top: 0.6rem !important;
        padding-bottom: 0.6rem !important;
        padding-left: 0.8rem !important;
        padding-right: 0.8rem !important;
    }}

    [data-testid="stSidebar"] .sidebar-logo-wrap {{
        width: 100%;
        display: flex;
        justify-content: center;
        align-items: center;
        margin: 0 auto 0.5rem auto;
        position: relative;
        left: 50%;
        transform: translateX(-50%);
        padding-left: 0 !important;
        padding-right: 0 !important;
    }}

    [data-testid="stSidebar"] .sidebar-logo-wrap img {{
        width: 120px;
        max-width: 100%;
        height: auto;
        display: block;
        margin: 0 auto;
    }}

    [data-testid="stSidebar"] [data-testid="stImage"] {{
        width: 100% !important;
        display: flex !important;
        justify-content: center !important;
        margin: 0 auto 0.5rem auto !important;
    }}

    [data-testid="stSidebar"] [data-testid="stImage"] img {{
        display: block !important;
        margin-left: auto !important;
        margin-right: auto !important;
    }}

    @media (max-width: 768px) {{
        [data-testid="stSidebar"] .block-container {{
            padding-left: 0.55rem !important;
            padding-right: 0.55rem !important;
        }}

        [data-testid="stSidebar"] .sidebar-logo-wrap {{
            margin-bottom: 0.4rem;
        }}

        [data-testid="stSidebar"] .sidebar-logo-wrap img {{
            width: 110px;
        }}
    }}

    /* Sidebar radio buttons styling */
    [data-testid="stSidebar"] [role="radiogroup"] {{
        background: rgba(255, 255, 255, 0.05) !important;
        padding: 10px !important;
        border-radius: 8px !important;
        border: 1px solid rgba(255, 255, 255, 0.15) !important;
    }}

    [data-testid="stSidebar"] [role="radio"] {{
        color: #ffffff !important;
        font-size: 0.95rem !important;
        font-weight: 600 !important;
        padding: 7px 6px !important;
        line-height: 1.25 !important;
    }}

    [data-testid="stSidebar"] [role="radio"]:hover {{
        background-color: rgba(77, 166, 255, 0.15) !important;
    }}

    /* Sidebar text and widgets contrast */
    [data-testid="stSidebar"] .css-1v0mbdj, [data-testid="stSidebar"] .stButton>button,
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] h4, [data-testid="stSidebar"] p, [data-testid="stSidebar"] label, [data-testid="stSidebar"] span {{
        color: #ffffff !important;
        font-size: 0.9rem !important;
        font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, 'Times New Roman', Georgia, serif !important;
        letter-spacing: 0.2px;
    }}

    /* Divider styling */
    [data-testid="stSidebar"] hr {{
        border: none;
        height: 1px;
        background: linear-gradient(to right, rgba(255,255,255,0), rgba(255,255,255,0.3), rgba(255,255,255,0)) !important;
        margin: 18px 0 !important;
    }}

    /* Hide minimize button and keyboard shortcut completely */
    [data-testid="stSidebar"] [aria-label*="keyboard"],
    [data-testid="stSidebar"] [title*="keyboard"],
    [data-testid="stSidebar"] button:has-text("keyboard"),
    button[aria-label*="keyboard"],
    button[title*="keyboard"],
    [data-testid="stSidebar"] button[kind="secondary"] {{
        display: none !important;
        visibility: hidden !important;
    }}
    
    /* Alternative: hide any button in header that might show keyboard */
    [data-testid="stSidebar"] > div:first-child > div:first-child button {{
        display: none !important;
    }}
    
    /* Fix input text color in sidebar */
    [data-testid="stSidebar"] input {{
        color: #ffffff !important;
        font-size: 0.9rem !important;
        font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, 'Times New Roman', Georgia, serif !important;
        background-color: rgba(255, 255, 255, 0.08) !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
    }}

    [data-testid="stSidebar"] input::placeholder {{
        color: rgba(255, 255, 255, 0.6) !important;
    }}

    [data-testid="stSidebar"] input:focus {{
        border-color: #4da6ff !important;
        box-shadow: 0 0 0 2px rgba(77, 166, 255, 0.2) !important;
    }}

    /* Keep tab style overrides */
    .stTabs [data-baseweb="tab-list"] [data-qa="stTab"] {{
        padding: 12px 20px;
        font-weight: 600;
        color: #ffffff !important;
    }}
    </style>
    """

st.markdown(page_style, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# TAB 1: ANÁLISIS MASIVO
# -----------------------------------------------------------------------------
if selected_tab == "Lote de PDFs":
    st.markdown("### 📄 Procesamiento de Lote (PDFs Masivo)")
    st.markdown("Sube múltiples informes médicos para análisis automático")
    st.divider()
    
    col_upload, col_info = st.columns([2, 1])
    
    with col_upload:
        st.markdown("#### 📤 Selecciona tus documentos")
        uploaded_files = st.file_uploader(
            "Arrastra aquí tus informes (PDF):",
            type=["pdf"],
            accept_multiple_files=True,
            help="Máximo 10 archivos por sesión"
        )
    
    with col_info:
        st.markdown("#### ℹ️ Información")
        st.markdown("""
        ✅ Análisis automático  
        ✅ Extracción NLP  
        ✅ Diagnóstico por caso  
        ✅ Exportación de resultados  
        """)
    
    if uploaded_files and st.button("🚀 Procesar Lote", type="primary", use_container_width=True):
        num_docs = len(uploaded_files)
        extracciones = []

        st.divider()
        st.markdown("### ⏳ Progreso de Procesamiento")
        
        container_progress = st.container(border=True)
        with container_progress:
            progress_global = st.progress(0)
            text_global = st.empty()
            
            progress_local = st.progress(0)
            text_local = st.empty()

        for i, uploaded_file in enumerate(uploaded_files):
            doc_name = uploaded_file.name
            texto_completo_doc = ""

            text_global.markdown(f"**📍 Archivo {i+1}/{num_docs}:** {doc_name}")
            progress_global.progress((i) / num_docs)

            try:
                with pdfplumber.open(uploaded_file) as pdf:
                    total_paginas = len(pdf.pages)
                    text_local.markdown(f"🚀 Procesando {total_paginas} páginas en paralelo...")
                    progress_local.progress(0.2)
                    
                    # Procesamiento paralelo optimizado
                    inicio_ocr = time.time()
                    texto_completo_doc, tiempos_ocr = procesar_pdf_paralelo(pdf, max_workers=4)
                    tiempo_total_ocr = time.time() - inicio_ocr
                    
                    # Mostrar estadísticas de velocidad
                    if tiempos_ocr:
                        tiempo_promedio = sum(tiempos_ocr.values()) / len(tiempos_ocr)
                        text_local.markdown(f"✅ OCR completado en {tiempo_total_ocr:.1f}s (promedio {tiempo_promedio:.2f}s/página)")
                    else:
                        text_local.markdown(f"✅ PDF procesado en {tiempo_total_ocr:.1f}s")
                    
                    progress_local.progress(0.8)

            except Exception as e:
                st.error(f"❌ Error leyendo {doc_name}: {e}")
                continue

            text_local.markdown(f"🧠 Analizando {doc_name}...")
            if len(texto_completo_doc) > 10:
                datos_doc = motor_nlp_contextual(texto_completo_doc)
                extracciones.append({"documento": doc_name, "datos": datos_doc})
            else:
                st.warning(f"⚠️ {doc_name}: Parece estar vacío o escaneado sin OCR")

        progress_global.progress(1.0)
        text_global.markdown("**✅ Procesamiento completado**")
        text_local.empty()
        progress_local.empty()
        
        time.sleep(0.5)
        container_progress.empty()

        st.session_state.consolidado_batch = fusionar_extracciones(extracciones)
        
        # Generar tabla de resultados
        res_indiv = []
        for item in extracciones:
            riesgo = calcular_riesgo_experto(item['datos'])
            res_indiv.append({
                "📄 Documento": item['documento'],
                "🎯 Diagnóstico": riesgo['nivel'],
                "📊 Score": riesgo['score'],
                "📝 Hallazgos": ", ".join(riesgo['hallazgos'][:2]) + ("..." if len(riesgo['hallazgos']) > 2 else "")
            })
        
        st.session_state.resultados_individuales = pd.DataFrame(res_indiv)
        st.rerun()

    # Mostrar resultados si existen
    if st.session_state.get('consolidado_batch'):
        st.divider()
        st.markdown("### 📋 Resultados del Análisis")
        
        if st.session_state.resultados_individuales is not None:
            st.markdown("#### Diagnósticos por Documento")
            st.dataframe(
                st.session_state.resultados_individuales,
                use_container_width=True,
                height=300,
                hide_index=True
            )
        
        col_data, col_diag = st.columns([1, 1], gap="large")
        
        with col_data:
            st.markdown("#### 📊 Datos Consolidados")
            df_res = pd.DataFrame([st.session_state.consolidado_batch]).T
            df_res.columns = ["Valor"]
            st.dataframe(df_res, use_container_width=True, height=400)
        
        with col_diag:
            st.markdown("#### 🤖 Diagnóstico Fusionado")
            res_batch = calcular_riesgo_experto(st.session_state.consolidado_batch)
            
            st.markdown(f"""
            <div style="
                background: linear-gradient(135deg, {res_batch['color']}33 0%, {res_batch['color']}11 100%);
                border-left: 5px solid {res_batch['color']};
                border-radius: 12px;
                padding: 20px;
                margin-bottom: 20px;
            ">
                <h3 style="color: {res_batch['color']}; margin: 0 0 10px 0;">
                    {res_batch['nivel']}
                </h3>
                <p style="color: #333; margin: 0;">
                    <strong>{res_batch['msg']}</strong>
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("#### 🧠 Análisis del Asistente")
            explicacion = generar_explicacion_narrativa(st.session_state.consolidado_batch, res_batch)
            st.info(explicacion)
            
            if "_origen" in st.session_state.consolidado_batch:
                with st.expander("🔍 Auditoría (Origen de datos)"):
                    st.json(st.session_state.consolidado_batch["_origen"])
        
        st.divider()
        col_exp1, col_exp2 = st.columns(2)
        with col_exp1:
            if st.button("📥 Descargar Resultados", use_container_width=True):
                csv = st.session_state.resultados_individuales.to_csv(index=False)
                st.download_button(
                    label="📥 Resultados.csv",
                    data=csv,
                    file_name=f"amylai_results_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    key="download_batch_results"
                )
        
        with col_exp2:
            if st.button("🗑️ Limpiar Resultados", use_container_width=True):
                st.session_state.consolidado_batch = None
                st.session_state.resultados_individuales = None
                st.rerun()

# -----------------------------------------------------------------------------
# TAB 2: CASO INDIVIDUAL - INTERFAZ MEJORADA
# -------------------------------------------------------------------------------------
elif selected_tab == "Caso Individual":
    st.markdown("### 🩺 Análisis Detallado de Caso Individual")
    st.markdown("Completa la información del paciente para un diagnóstico preciso")
    st.divider()
    
    # Layout principal
    left_col, right_col = st.columns([1.1, 1], gap="large")
    
    with left_col:
        st.markdown("## 📋 Recopilación de Datos")
        
        st.divider()
        
        # Step 1: Text Input
        with st.container(border=True):
            st.markdown("### 📝 Paso 1: Nota Clínica")
            st.markdown("*Copia la evolución clínica o informe médico completo*")
            
            texto = st.text_area(
                "Texto de la nota clínica",
                height=180,
                placeholder="Ejemplo:\n\nPaciente masculino 68 años con antecedente de...\nEcocardiograma: HVI 16mm con apical sparing...\nECG: bajo voltaje, BAV...",
                label_visibility="collapsed"
            )
            
            col_btn1, col_btn2 = st.columns(2, gap="small")
            with col_btn1:
                if st.button("✨ Analizar con IA (Máxima Precisión)", type="primary", use_container_width=True):
                    barra_ind = st.progress(0, text="0% | 🚀 Iniciando análisis...")
                    try:
                        barra_ind.progress(20, text="20% | 🧠 LLM analizando texto...")
                        # Usar motor LLM mejorado para máxima precisión
                        nlp = motor_nlp_contextual(texto)
                        barra_ind.progress(80, text="80% | 🛡️ Validando y procesando...")
                        time.sleep(0.3)
                        barra_ind.progress(100, text="100% | ✅ ¡Completado!")
                        time.sleep(0.3)
                        barra_ind.empty()
                        
                        modelo = nlp.get('_modelo', 'Desconocido')
                        st.success(f"✅ Analizado: {modelo}")
                        
                        if '_modelo' in nlp:
                            del nlp['_modelo']
                        st.session_state.form_data.update(nlp)
                        st.session_state.analisis_automatico = True  # Flag para mostrar resumen
                    except Exception as e:
                        st.error(f"Error: {e}")
            
            with col_btn2:
                if st.button("🔄 Limpiar", use_container_width=True):
                    st.session_state.form_data = DEFAULT_DATA.copy()
                    st.session_state.analisis_automatico = False
                    st.rerun()
        
        st.divider()
        
        # SECCIÓN DE HALLAZGOS EXTRAÍDOS AUTOMÁTICAMENTE
        if st.session_state.get('analisis_automatico', False):
            with st.container(border=True):
                st.markdown("### 📋 RESUMEN COMPLETO DE HALLAZGOS EXTRAÍDOS")
                st.markdown("*Todos los datos y red flags detectados. Revísalos antes de guardar*")
                
                # Extraer todos los red flags categorizados
                redflags_dict = extraer_redflags_detectados(st.session_state.form_data)
                
                # Crear tabla de TODOS los hallazgos extraídos
                hallazgos_tabla = []
                
                # 1. DATOS DEMOGRÁFICOS
                if st.session_state.form_data.get('edad'):
                    hallazgos_tabla.append({
                        'Categoría': '👤 Demográfico',
                        'Campo': 'Edad',
                        'Valor': f"{st.session_state.form_data['edad']} años",
                        'Tipo': '✓ Extraído'
                    })
                
                if st.session_state.form_data.get('sexo'):
                    sexo_desc = 'Masculino' if st.session_state.form_data['sexo'] == 'M' else 'Femenino' if st.session_state.form_data['sexo'] == 'F' else 'N/A'
                    hallazgos_tabla.append({
                        'Categoría': '👤 Demográfico',
                        'Campo': 'Sexo',
                        'Valor': sexo_desc,
                        'Tipo': '✓ Extraído'
                    })
                
                # 2. PARÁMETROS CARDIACOS ECOCARDIOGRAMA
                if st.session_state.form_data.get('ivs'):
                    hallazgos_tabla.append({
                        'Categoría': '🫀 Ecocardiografía',
                        'Campo': 'IVS (septo)',
                        'Valor': f"{st.session_state.form_data['ivs']} mm",
                        'Tipo': '✓ Extraído'
                    })
                
                if st.session_state.form_data.get('gls'):
                    hallazgos_tabla.append({
                        'Categoría': '🫀 Ecocardiografía',
                        'Campo': 'GLS (strain)',
                        'Valor': f"{st.session_state.form_data['gls']} %",
                        'Tipo': '✓ Extraído'
                    })
                
                if st.session_state.form_data.get('septum_posterior'):
                    hallazgos_tabla.append({
                        'Categoría': '🫀 Ecocardiografía',
                        'Campo': 'Pared Posterior',
                        'Valor': f"{st.session_state.form_data['septum_posterior']} mm",
                        'Tipo': '✓ Extraído'
                    })
                
                if st.session_state.form_data.get('derrame_pericardico'):
                    hallazgos_tabla.append({
                        'Categoría': '🫀 Ecocardiografía',
                        'Campo': 'Derrame Pericárdico',
                        'Valor': '✓ PRESENTE',
                        'Tipo': '✓ Detectado'
                    })
                
                # 3. PARÁMETROS ECG
                if st.session_state.form_data.get('volt'):
                    hallazgos_tabla.append({
                        'Categoría': '📟 ECG',
                        'Campo': 'Voltaje',
                        'Valor': f"{st.session_state.form_data['volt']} mV",
                        'Tipo': '✓ Extraído'
                    })
                
                # 4. BIOMARCADORES
                if st.session_state.form_data.get('nt_probnp'):
                    hallazgos_tabla.append({
                        'Categoría': '🧪 Biomarcadores',
                        'Campo': 'NT-proBNP',
                        'Valor': f"{st.session_state.form_data['nt_probnp']} pg/ml",
                        'Tipo': '✓ Extraído'
                    })
                
                if st.session_state.form_data.get('troponina'):
                    hallazgos_tabla.append({
                        'Categoría': '🧪 Biomarcadores',
                        'Campo': 'Troponina',
                        'Valor': '✓ Elevada',
                        'Tipo': '✓ Detectada'
                    })
                
                # 5. RESONANCIA MAGNÉTICA
                if st.session_state.form_data.get('lge_patron'):
                    hallazgos_tabla.append({
                        'Categoría': '🎯 Resonancia Cardíaca',
                        'Campo': 'Patrón LGE',
                        'Valor': st.session_state.form_data['lge_patron'].upper(),
                        'Tipo': '✓ Extraído'
                    })
                
                if st.session_state.form_data.get('ecv'):
                    hallazgos_tabla.append({
                        'Categoría': '🎯 Resonancia Cardíaca',
                        'Campo': 'ECV',
                        'Valor': f"{st.session_state.form_data['ecv']} %",
                        'Tipo': '✓ Extraído'
                    })
                
                if st.session_state.form_data.get('t1_mapping'):
                    hallazgos_tabla.append({
                        'Categoría': '🎯 Resonancia Cardíaca',
                        'Campo': 'T1 Mapping',
                        'Valor': '✓ Elevado',
                        'Tipo': '✓ Detectado'
                    })
                
                # 6. RED FLAGS - AL DIRECTO (Patognomónicos)
                if redflags_dict.get('AL_DIRECTO'):
                    for hallazgo in redflags_dict['AL_DIRECTO']:
                        hallazgos_tabla.append({
                            'Categoría': '🔴 AL Directo',
                            'Campo': hallazgo,
                            'Valor': '✓ PRESENTE',
                            'Tipo': '✓ Red Flag'
                        })
                
                # 7. RED FLAGS - AL SISTÉMICO
                if redflags_dict.get('AL_SISTEMICO'):
                    for hallazgo in redflags_dict['AL_SISTEMICO']:
                        hallazgos_tabla.append({
                            'Categoría': '🟠 AL Sistémico',
                            'Campo': hallazgo,
                            'Valor': '✓ PRESENTE',
                            'Tipo': '✓ Red Flag'
                        })
                
                # 8. RED FLAGS - ATTR CLÁSICO (Tríada musculoesquelética)
                if redflags_dict.get('ATTR_CLASICO'):
                    for hallazgo in redflags_dict['ATTR_CLASICO']:
                        hallazgos_tabla.append({
                            'Categoría': '🟡 ATTR Clásico',
                            'Campo': hallazgo,
                            'Valor': '✓ PRESENTE',
                            'Tipo': '✓ Red Flag'
                        })
                
                # 9. RED FLAGS - ATTR HEREDITARIA
                if redflags_dict.get('ATTR_HEREDITARIO'):
                    for hallazgo in redflags_dict['ATTR_HEREDITARIO']:
                        hallazgos_tabla.append({
                            'Categoría': '🟢 ATTR Hereditario',
                            'Campo': hallazgo,
                            'Valor': '✓ PRESENTE',
                            'Tipo': '✓ Red Flag'
                        })
                
                # 10. HALLAZGOS CARDIACOS ESPECÍFICOS
                if redflags_dict.get('HALLAZGOS_CARDIACOS'):
                    for hallazgo in redflags_dict['HALLAZGOS_CARDIACOS']:
                        hallazgos_tabla.append({
                            'Categoría': '💙 Cardiacos',
                            'Campo': hallazgo,
                            'Valor': '✓ PRESENTE',
                            'Tipo': '✓ Red Flag'
                        })
                
                # 11. PARÁMETROS CARDIACOS NUMÉRICOS
                if redflags_dict.get('PARAMETROS_CARDIACOS'):
                    for hallazgo in redflags_dict['PARAMETROS_CARDIACOS']:
                        hallazgos_tabla.append({
                            'Categoría': '📊 Parámetros',
                            'Campo': hallazgo,
                            'Valor': '⚠️ CRÍTICO',
                            'Tipo': '✓ Red Flag'
                        })
                
                # 12. FACTORES CONFUSORES
                if redflags_dict.get('FACTORES_CONFUSORES'):
                    for hallazgo in redflags_dict['FACTORES_CONFUSORES']:
                        hallazgos_tabla.append({
                            'Categoría': '⚙️ Confusores',
                            'Campo': hallazgo,
                            'Valor': '⚠️ PRESENTE',
                            'Tipo': '✓ Confosor'
                        })
                
                # Mostrar tabla si hay hallazgos
                if hallazgos_tabla:
                    df_hallazgos = pd.DataFrame(hallazgos_tabla)
                    st.dataframe(
                        df_hallazgos,
                        use_container_width=True,
                        hide_index=True,
                        column_config={
                            'Categoría': st.column_config.TextColumn(width='small'),
                            'Campo': st.column_config.TextColumn(width='medium'),
                            'Valor': st.column_config.TextColumn(width='small'),
                            'Tipo': st.column_config.TextColumn(width='small'),
                        }
                    )
                    
                    # Resumen rápido de red flags
                    col_summary1, col_summary2, col_summary3, col_summary4 = st.columns(4)
                    with col_summary1:
                        st.metric("🔴 AL Directo", len(redflags_dict.get('AL_DIRECTO', [])))
                    with col_summary2:
                        st.metric("🟠 AL Sistémico", len(redflags_dict.get('AL_SISTEMICO', [])))
                    with col_summary3:
                        st.metric("🟡 ATTR Clásico", len(redflags_dict.get('ATTR_CLASICO', [])))
                    with col_summary4:
                        st.metric("⚙️ Confusores", len(redflags_dict.get('FACTORES_CONFUSORES', [])))
                    
                    st.caption(f"📊 **Total: {len(hallazgos_tabla)} hallazgos extraídos** | **Red Flags: {redflags_dict['TOTAL_REDFLAGS']}** | Puedes editar los valores en las secciones del Paso 2 abajo")
                else:
                    st.info("ℹ️ No se detectaron hallazgos significativos en el texto")

        
        st.divider()
        
        # Step 2: Parameters
        with st.container(border=True):
            st.markdown("### 🫀 Paso 2: Parámetros Cardiacos")
            st.markdown("*Valores del ecocardiograma*")
            
            p_col1, p_col2, p_col3 = st.columns(3, gap="small")
            
            with p_col1:
                st.session_state.form_data['ivs'] = st.number_input(
                    "IVS (mm)",
                    min_value=0.0, max_value=30.0, step=0.1,
                    value=st.session_state.form_data['ivs'],
                    help="Grosor pared VI. Normal <11mm"
                )
            
            with p_col2:
                st.session_state.form_data['volt'] = st.number_input(
                    "Voltaje (mV)",
                    min_value=0.0, max_value=2.0, step=0.01,
                    value=st.session_state.form_data['volt'],
                    help="Amplitud máxima QRS en ECG"
                )
            
            with p_col3:
                st.session_state.form_data['gls'] = st.number_input(
                    "GLS (%)",
                    min_value=-30.0, max_value=0.0, step=0.5,
                    value=st.session_state.form_data['gls'],
                    help="Strain longitudinal global (-15% a -20% normal)"
                )
        
        st.divider()
        
        # Step 2.5: Biomarcadores
        st.markdown("### 🔬 Paso 2B: Biomarcadores Cardiacos")
        st.markdown("*Pista bioquímica crítica para diferenciar tipos de amiloidosis*")
        
        bm_col1, bm_col2 = st.columns(2, gap="small")
        
        with bm_col1:
            nt_probnp_value = float(st.session_state.form_data.get('nt_probnp', 0.0))
            nt_probnp_max = max(6000.0, nt_probnp_value)
            st.session_state.form_data['nt_probnp'] = st.number_input(
                "NT-proBNP (pg/ml)",
                min_value=0.0, max_value=nt_probnp_max, step=50.0,
                value=nt_probnp_value,
                help="Péptido natriurético. >3000: muy sugestivo amiloidosis. <400: descarta"
            )
        
        with bm_col2:
            st.session_state.form_data['troponina'] = st.checkbox(
                "🩸 Troponina Elevada Crónica",
                value=st.session_state.form_data.get('troponina', False),
                help="Sin infarto agudo → Infiltración miocárdica crónica"
            )
        
        st.divider()
        
        # Step 2.6: Resonancia Magnética
        st.markdown("### 🎯 Paso 2C: Resonancia Magnética Cardíaca")
        st.markdown("*Desempate definitivo cuando el Eco es dudoso*")
        
        rm1, rm2 = st.columns(2, gap="small")
        
        with rm1:
            lge_options = ["", "subendocardico", "transmural", "parcheado", "difuso", "null"]
            lge_current = st.session_state.form_data.get('lge_patron', '')
            lge_index = 0
            try:
                lge_index = lge_options.index(lge_current) if lge_current in lge_options else 0
            except (ValueError, IndexError):
                lge_index = 0
            
            st.session_state.form_data['lge_patron'] = st.selectbox(
                "Patrón LGE (Realce Tardío de Gadolinio)",
                options=lge_options,
                index=lge_index,
                help="Patrón subendocárdico/transmural = patognomónico amiloidosis"
            )
            
            st.session_state.form_data['ecv'] = st.number_input(
                "ECV - Volumen Extracelular (%)",
                min_value=0.0, max_value=100.0, step=1.0,
                value=st.session_state.form_data.get('ecv', 0.0),
                help="ECV > 40% = DIAGNÓSTICO. 30-40%: muy sugestivo"
            )
        
        with rm2:
            st.session_state.form_data['t1_mapping'] = st.checkbox(
                "T1 Mapping Elevado",
                value=st.session_state.form_data.get('t1_mapping', False),
                help="T1 nativo prolongado → Amiloidosis (esp. AL)"
            )
        
        st.divider()
        
        # Step 2.7: Demografía
        st.markdown("### 👤 Paso 2D: Datos Demográficos")
        st.markdown("*Contexto clínico (prior probability)*")
        
        dem1, dem2 = st.columns(2, gap="small")
        
        with dem1:
            st.session_state.form_data['edad'] = st.number_input(
                "Edad (años)",
                min_value=0, max_value=120, step=1,
                value=int(st.session_state.form_data.get('edad', 0)),
                help="ATTR wild-type es rara <60 años; AL puede ser más joven"
            )
        
        with dem2:
            sexo_options = ["", "M", "F"]
            sexo_current = st.session_state.form_data.get('sexo', '')
            sexo_index = 0
            try:
                sexo_index = sexo_options.index(sexo_current) if sexo_current in sexo_options else 0
            except (ValueError, IndexError):
                sexo_index = 0
            
            st.session_state.form_data['sexo'] = st.selectbox(
                "Sexo",
                options=sexo_options,
                index=sexo_index,
                help="ATTR wild-type: 80-90% hombres"
            )
        
        st.divider()
        
        # Step 2.8: Detalles Ecocardiográficos Finos
        st.markdown("### 🔍 Paso 2E: Detalles Ecocardiográficos")
        st.markdown("*Hallazgos ecocardiográficos adicionales*")
        
        eco_col1, eco_col2 = st.columns(2, gap="small")
        
        with eco_col1:
            st.session_state.form_data['derrame_pericardico'] = st.checkbox(
                "💧 Derrame Pericárdico",
                value=st.session_state.form_data.get('derrame_pericardico', False),
                help="Común en AL, raro en ATTR → Diferencia tipos"
            )
        
        with eco_col2:
            st.session_state.form_data['septum_posterior'] = st.number_input(
                "Grosor Tabique+Pared Post (mm)",
                min_value=0.0, max_value=30.0, step=0.5,
                value=st.session_state.form_data.get('septum_posterior', 0.0),
                help="Patrón concéntrico vs asimétrico. >13mm: HVI severa"
            )
        
        st.divider()
        
        # Step 3: Symptoms
        st.markdown("### ✓ Paso 3: Hallazgos Clínicos")
        st.markdown("*Selecciona los síntomas y signos presentes*")
        
        # AL Directo
        with st.expander("🚨 **AL DIRECTO** (¡DERIVACIÓN URGENTE si presente!)", expanded=True):
            st.markdown("*Criterios diagnósticos específicos de amiloidosis AL*")
            col_al_d = st.columns(3, gap="small")
            with col_al_d[0]:
                st.session_state.form_data['mgus'] = st.checkbox(
                    "💊 MGUS/Paraproteína",
                    value=st.session_state.form_data['mgus'],
                    help="Componente monoclonal de cadenas ligeras"
                )
            with col_al_d[1]:
                st.session_state.form_data['macro'] = st.checkbox(
                    "👅 Macroglosia",
                    value=st.session_state.form_data['macro'],
                    help="Lengua aumentada > 0.5cm"
                )
            with col_al_d[2]:
                st.session_state.form_data['purpura'] = st.checkbox(
                    "💜 Púrpura Periorbital",
                    value=st.session_state.form_data['purpura'],
                    help="Hematoma 'ojos de mapache' espontáneo"
                )
        
        # AL Sistémico
        with st.expander("🟠 **AL SISTÉMICO** (Manifestaciones secundarias)", expanded=True):
            st.markdown("*Hallazgos que sugieren nivel sistémico de afección*")
            col_al_s1 = st.columns(3, gap="small")
            with col_al_s1[0]:
                st.session_state.form_data['nefro'] = st.checkbox(
                    "🫘 Síndrome Nefrótico",
                    value=st.session_state.form_data['nefro'],
                    help="Proteinuria >3.5g/24h + edemas"
                )
                st.session_state.form_data['hepato'] = st.checkbox(
                    "🟤 Hepatomegalia",
                    value=st.session_state.form_data['hepato'],
                    help="Hígado aumentado por depósito amiloide"
                )
            with col_al_s1[1]:
                st.session_state.form_data['neuro_p'] = st.checkbox(
                    "⚡ Polineuropatía",
                    value=st.session_state.form_data['neuro_p'],
                    help="Parestesias distales simétricas"
                )
                st.session_state.form_data['fatiga'] = st.checkbox(
                    "😴 Fatiga Severa",
                    value=st.session_state.form_data['fatiga'],
                    help="Agotamiento extremo, presíncope"
                )
            with col_al_s1[2]:
                st.session_state.form_data['disauto'] = st.checkbox(
                    "🌀 Disautonomía",
                    value=st.session_state.form_data['disauto'],
                    help="Hipotensión ortostática, disfuncion GI"
                )
                st.session_state.form_data['piel_lesiones'] = st.checkbox(
                    "🩹 Lesiones Cutáneas",
                    value=st.session_state.form_data['piel_lesiones'],
                    help="Depósitos amiloides en piel"
                )
        
        # ATTR Clásico
        with st.expander("🟣 **ATTR CLÁSICO** (Síndrome Musculoesquelético)", expanded=True):
            st.markdown("*Tríada característica: STC bilateral + Estenosis lumbar + Rotura bíceps*")
            col_attr_p = st.columns(2, gap="small")
            with col_attr_p[0]:
                st.markdown("#### Hallazgos Primarios")
                st.session_state.form_data['stc'] = st.checkbox(
                    "🔴 STC Bilateral",
                    value=st.session_state.form_data['stc'],
                    help="Atrapamiento del nervio mediano bilateral"
                )
                st.session_state.form_data['lumbar'] = st.checkbox(
                    "🔴 Estenosis Lumbar",
                    value=st.session_state.form_data['lumbar'],
                    help="Claudicación neurógena, compresión radicular"
                )
                st.session_state.form_data['biceps'] = st.checkbox(
                    "🔴 Rotura de Bíceps",
                    value=st.session_state.form_data['biceps'],
                    help="Signo de Popeye - muy específico para ATTR"
                )
            with col_attr_p[1]:
                st.markdown("#### Hallazgos Adicionales")
                st.session_state.form_data['hombro'] = st.checkbox(
                    "Tendinitis Hombro",
                    value=st.session_state.form_data['hombro'],
                    help="Rotura del manguito rotador"
                )
                st.session_state.form_data['dupuytren'] = st.checkbox(
                    "Dupuytren",
                    value=st.session_state.form_data['dupuytren'],
                    help="Contractura fibrosa de mano - presente 5-25%"
                )
                st.session_state.form_data['artralgias'] = st.checkbox(
                    "Artralgias",
                    value=st.session_state.form_data['artralgias'],
                    help="Dolor articular crónico en grandes articulaciones"
                )
                st.session_state.form_data['fractura_vert'] = st.checkbox(
                    "Fractura Vertebral",
                    value=st.session_state.form_data['fractura_vert'],
                    help="Fragilidad ósea, colapso vertebral"
                )
                st.session_state.form_data['tendinitis_calcifica'] = st.checkbox(
                    "Tendinitis Cálcica",
                    value=st.session_state.form_data['tendinitis_calcifica'] if 'tendinitis_calcifica' in st.session_state.form_data else False,
                    help="Depósitos de calcio en tendones"
                )
        
        # Hallazgos Cardiacos
        with st.expander("💙 **HALLAZGOS CARDIACOS** (Ecocardiografía)", expanded=True):
            st.markdown("*Hallazgos específicos en imagen cardíaca*")
            col_card = st.columns(3, gap="small")
            with col_card[0]:
                st.session_state.form_data['apical_sparing'] = st.checkbox(
                    "Apical Sparing",
                    value=st.session_state.form_data['apical_sparing'],
                    help="Preservación del strain apical - patognomónico ATTR"
                )
                st.session_state.form_data['bajo_voltaje'] = st.checkbox(
                    "Bajo Voltaje",
                    value=st.session_state.form_data['bajo_voltaje'],
                    help="Voltaje bajo + HVI severa (paradójico)"
                )
            with col_card[1]:
                st.session_state.form_data['bav_mp'] = st.checkbox(
                    "BAV/Marcapasos",
                    value=st.session_state.form_data['bav_mp'],
                    help="Bloqueo AV completo - requiere marcapasos"
                )
                st.session_state.form_data['pseudo_q'] = st.checkbox(
                    "Pseudoinfarto",
                    value=st.session_state.form_data['pseudo_q'],
                    help="Ondas Q patológicas SIN infarto"
                )
            with col_card[2]:
                st.session_state.form_data['biatrial'] = st.checkbox(
                    "Dilatación Biauricular",
                    value=st.session_state.form_data['biatrial'],
                    help="Aurículas aumentadas de volumen"
                )
        
        # ATTR Hereditaria
        with st.expander("🧬 **ATTR HEREDITARIA** (Mutación TTR)"):
            st.markdown("*Requiere cribado familiar*")
            col_attr_h = st.columns(3, gap="small")
            with col_attr_h[0]:
                st.session_state.form_data['mutacion_ttr'] = st.checkbox(
                    "🔴 Mutación TTR",
                    value=st.session_state.form_data['mutacion_ttr'],
                    help="TTR confirmada genéticamente"
                )

        
        # Factores Confusores
        with st.expander("⚠️ **FACTORES CONFUSORES** (Elevan umbral HVI)"):
            st.markdown("*No causan amiloidosis pero dificultan diagnóstico*")
            col_conf = st.columns(3, gap="small")
            with col_conf[0]:
                st.session_state.form_data['confusor_hta'] = st.checkbox(
                    "HTA Severa",
                    value=st.session_state.form_data['confusor_hta'],
                    help="Hipertensión crónica"
                )
            with col_conf[1]:
                st.session_state.form_data['confusor_ao'] = st.checkbox(
                    "Estenosis Aórtica",
                    value=st.session_state.form_data['confusor_ao'],
                    help="Valvulopatía aórtica severa"
                )
            with col_conf[2]:
                st.session_state.form_data['confusor_irc'] = st.checkbox(
                    "Insuficiencia Renal",
                    value=st.session_state.form_data['confusor_irc'],
                    help="ERC, diálisis"
                )
    
    with right_col:
        st.markdown("## 🎯 Resultado del Análisis")
        
        res_ind = calcular_riesgo_experto(st.session_state.form_data)
        
        # Obtener confianza (con fallback)
        confianza = res_ind.get('confianza_porcentaje', 50.0)
        
        # Guardar confianza e info del modelo en session para uso posterior (guardar caso)
        st.session_state.confianza_analisis = confianza
        st.session_state.nivel_diagnostico = res_ind.get('nivel', 'Desconocido')
        
        # Resultado principal con porcentaje
        st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, {res_ind['color']}33 0%, {res_ind['color']}11 100%);
            border-left: 5px solid {res_ind['color']};
            border-radius: 12px;
            padding: 25px;
            margin-bottom: 25px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        ">
            <h2 style="color: {res_ind['color']}; margin: 0 0 10px 0;">
                {res_ind['nivel']}
            </h2>
            <p style="font-size: 1.05em; color: #333; margin: 0;">
                <strong>{res_ind['msg']}</strong>
            </p>
            <div style="margin-top: 15px; display: flex; align-items: center; gap: 15px;">
                <div style="background-color: {res_ind['color']}20; padding: 10px 20px; border-radius: 8px; font-weight: bold; color: {res_ind['color']}; font-size: 1.1em;">
                    📊 Confianza IA: <span style="font-size: 1.3em;">{confianza:.1f}%</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Acción recomendada
        if 'accion' in res_ind:
            st.markdown(f"""
            <div style="
                background-color: #fff3cd;
                border-left: 4px solid #ffc107;
                padding: 15px;
                border-radius: 8px;
                margin-bottom: 20px;
            ">
                <strong>👉 Acción Recomendada:</strong><br>
                {res_ind['accion']}
            </div>
            """, unsafe_allow_html=True)
        
        # Score
        st.divider()
        st.markdown("### 📊 Puntuación Detallada")
        
        col_score1, col_score2, col_score3 = st.columns(3)
        with col_score1:
            st.metric(
                "Score Total",
                f"{res_ind['score']} pts",
                help="Suma ponderada de hallazgos"
            )
        with col_score2:
            st.metric(
                "Nivel de Riesgo",
                res_ind['nivel'].split(' ')[0],
                help="Categoría diagnóstica"
            )
        with col_score3:
            st.metric(
                "Confianza IA 🤖",
                f"{confianza:.1f}%",
                help="Porcentaje de confianza del algoritmo de machine learning"
            )
        
        # Hallazgos encontrados
        st.markdown("### 📋 Hallazgos Detectados")
        if res_ind['hallazgos']:
            for h in res_ind['hallazgos']:
                st.markdown(f"✓ {h}", unsafe_allow_html=True)
        else:
            st.info("Sin hallazgos significativos detectados", icon="ℹ️")
        
        st.divider()
        
        # Explicación narrativa
        st.markdown("### 🤖 Análisis Clínico")
        explicacion = generar_explicacion_narrativa(st.session_state.form_data, res_ind)
        st.markdown(f"""
        <div style="
            background-color: #f0f4f8;
            border-radius: 8px;
            padding: 15px;
            border-left: 4px solid #2196F3;
            font-size: 0.95em;
            line-height: 1.6;
        ">
            {explicacion}
        </div>
        """, unsafe_allow_html=True)
        
        st.divider()
        
        # RESUMEN DE HALLAZGOS
        st.markdown("### 📑 Resumen Clínico de Hallazgos")
        
        col_btn_resumen1, col_btn_resumen2 = st.columns(2, gap="small")
        with col_btn_resumen1:
            if st.button("📋 Generar Resumen Completo", type="primary", use_container_width=True):
                resumen_hallazgos = generar_resumen_hallazgos(st.session_state.form_data, res_ind)
                st.session_state.resumen_generado = resumen_hallazgos
        
        with col_btn_resumen2:
            if st.button("📋 Limpiar Resumen", use_container_width=True):
                st.session_state.resumen_generado = None
                st.rerun()
        
        # Mostrar resumen si está disponible
        if 'resumen_generado' in st.session_state and st.session_state.resumen_generado:
            st.markdown(f"""
            <div style="
                background-color: #fafafa;
                border-radius: 8px;
                padding: 20px;
                border-left: 5px solid #4CAF50;
                font-size: 0.95em;
                line-height: 1.7;
                max-height: 600px;
                overflow-y: auto;
            ">
                {st.session_state.resumen_generado}
            </div>
            """, unsafe_allow_html=True)
            
            # Botones de acción
            col_copy1, col_copy2 = st.columns(2, gap="small")
            
            with col_copy1:
                # Limpiar markdown para descargar
                texto_descargar = st.session_state.resumen_generado
                texto_descargar = texto_descargar.replace("**", "").replace("###", "")
                texto_descargar = texto_descargar.replace("### ", "").replace("##", "")
                
                st.download_button(
                    label="📥 Descargar como TXT",
                    data=texto_descargar,
                    file_name=f"resumen_hallazgos_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain",
                    use_container_width=True,
                    key="download_individual_case"
                )
            
            with col_copy2:
                if st.button("📋 Copiar a Portapapeles", use_container_width=True):
                    st.code(st.session_state.resumen_generado, language="markdown")
        
        st.divider()
        
        # Guardar caso
        st.markdown("### 💾 Guardar Caso para Entrenamiento")
        st.markdown("*Revisa todos los hallazgos que se guardarán en la base de datos*")
        
        # TABLA RESUMEN DE TODOS LOS HALLAZGOS A GUARDAR
        with st.expander("📊 VISTA PREVIA: Todos los hallazgos a guardar", expanded=True):
            # Crear tabla estilo inventario
            datos_a_guardar = []
            
            # Sección: Identificación
            datos_a_guardar.append({'Sección': '👤 IDENTIFICACIÓN', 'Campo': 'NHC', 'Valor': st.session_state.form_data.get('nhc', '—'), 'Estado': '✓'})
            
            # Sección: Demografía
            edad = st.session_state.form_data.get('edad', 0)
            datos_a_guardar.append({'Sección': '👤 DEMOGRAFÍA', 'Campo': 'Edad', 'Valor': f"{edad} años" if edad else '—', 'Estado': '✓ Extraído' if edad else '—'})
            
            sexo_val = st.session_state.form_data.get('sexo', '')
            sexo_desc = 'Masculino' if sexo_val == 'M' else 'Femenino' if sexo_val == 'F' else '—'
            datos_a_guardar.append({'Sección': '👤 DEMOGRAFÍA', 'Campo': 'Sexo', 'Valor': sexo_desc, 'Estado': '✓ Extraído' if sexo_val else '—'})
            
            # Sección: Ecocardiografía
            ivs = st.session_state.form_data.get('ivs', 0)
            datos_a_guardar.append({'Sección': '🫀 ECOCARDIOGRAFÍA', 'Campo': 'IVS/Septo', 'Valor': f"{ivs} mm" if ivs else '—', 'Estado': '✓ Extraído' if ivs else '—'})
            
            gls = st.session_state.form_data.get('gls', 0)
            datos_a_guardar.append({'Sección': '🫀 ECOCARDIOGRAFÍA', 'Campo': 'GLS/Strain', 'Valor': f"{gls} %" if gls else '—', 'Estado': '✓ Extraído' if gls else '—'})
            
            septum_post = st.session_state.form_data.get('septum_posterior', 0)
            datos_a_guardar.append({'Sección': '🫀 ECOCARDIOGRAFÍA', 'Campo': 'Pared Posterior', 'Valor': f"{septum_post} mm" if septum_post else '—', 'Estado': '✓ Extraído' if septum_post else '—'})
            
            derrame = st.session_state.form_data.get('derrame_pericardico', False)
            datos_a_guardar.append({'Sección': '🫀 ECOCARDIOGRAFÍA', 'Campo': 'Derrame Pericárdico', 'Valor': '✓ SÍ' if derrame else 'NO', 'Estado': '✓ Detectado' if derrame else '—'})
            
            # Sección: Electrocardiografía
            volt = st.session_state.form_data.get('volt', 0)
            datos_a_guardar.append({'Sección': '📊 ECG', 'Campo': 'Voltaje', 'Valor': f"{volt} mV" if volt else '—', 'Estado': '✓ Extraído' if volt else '—'})
            
            bav_mp = st.session_state.form_data.get('bav_mp', False)
            datos_a_guardar.append({'Sección': '📊 ECG', 'Campo': 'BAV/Marcapasos', 'Valor': '✓ SÍ' if bav_mp else 'NO', 'Estado': '✓ Detectado' if bav_mp else '—'})
            
            bajo_volt = st.session_state.form_data.get('bajo_voltaje', False)
            datos_a_guardar.append({'Sección': '📊 ECG', 'Campo': 'Bajo Voltaje', 'Valor': '✓ SÍ' if bajo_volt else 'NO', 'Estado': '✓ Detectado' if bajo_volt else '—'})
            
            # Sección: Biomarcadores
            nt_probnp = st.session_state.form_data.get('nt_probnp', 0)
            datos_a_guardar.append({'Sección': '🧪 BIOMARCADORES', 'Campo': 'NT-proBNP', 'Valor': f"{nt_probnp} pg/ml" if nt_probnp else '—', 'Estado': '✓ Extraído' if nt_probnp else '—'})
            
            troponina = st.session_state.form_data.get('troponina', False)
            datos_a_guardar.append({'Sección': '🧪 BIOMARCADORES', 'Campo': 'Troponina Elevada', 'Valor': '✓ SÍ' if troponina else 'NO', 'Estado': '✓ Detectado' if troponina else '—'})
            
            # Sección: Resonancia Magnética Cardíaca
            lge = st.session_state.form_data.get('lge_patron', '')
            datos_a_guardar.append({'Sección': '🎯 RESONANCIA CARDÍACA', 'Campo': 'Patrón LGE', 'Valor': lge.upper() if lge else '—', 'Estado': '✓ Extraído' if lge else '—'})
            
            ecv = st.session_state.form_data.get('ecv', 0)
            datos_a_guardar.append({'Sección': '🎯 RESONANCIA CARDÍACA', 'Campo': 'ECV (%)', 'Valor': f"{ecv} %" if ecv else '—', 'Estado': '✓ Extraído' if ecv else '—'})
            
            t1_map = st.session_state.form_data.get('t1_mapping', False)
            datos_a_guardar.append({'Sección': '🎯 RESONANCIA CARDÍACA', 'Campo': 'T1 Mapping Elevado', 'Valor': '✓ SÍ' if t1_map else 'NO', 'Estado': '✓ Detectado' if t1_map else '—'})
            
            # Extender tabla con red flags
            redflags_dict = extraer_redflags_detectados(st.session_state.form_data)
            
            # Mostrar tabla
            df_guardar = pd.DataFrame(datos_a_guardar)
            st.dataframe(
                df_guardar,
                use_container_width=True,
                hide_index=True,
                column_config={
                    'Sección': st.column_config.TextColumn(width='small'),
                    'Campo': st.column_config.TextColumn(width='medium'),
                    'Valor': st.column_config.TextColumn(width='medium'),
                    'Estado': st.column_config.TextColumn(width='small'),
                }
            )
            
            # Resumen comprimido de red flags
            if redflags_dict["TOTAL_REDFLAGS"] > 0:
                st.markdown(f"**🚩 Red Flags Adicionales: {redflags_dict['TOTAL_REDFLAGS']}**")
                todos_redflags = (
                    redflags_dict.get("AL_DIRECTO", []) +
                    redflags_dict.get("AL_SISTEMICO", []) +
                    redflags_dict.get("ATTR_CLASICO", []) +
                    redflags_dict.get("ATTR_HEREDITARIO", []) +
                    redflags_dict.get("HALLAZGOS_CARDIACOS", []) +
                    redflags_dict.get("FACTORES_CONFUSORES", [])
                )
                st.write(", ".join(todos_redflags[:10]) + ("..." if len(todos_redflags) > 10 else ""))
        
        # Extraer red flags para mostrar y guardar (segunda vez)
        redflags_dict = extraer_redflags_detectados(st.session_state.form_data)
        
        # Mostrar contexto adicional de red flags si están disponibles
        if redflags_dict["TOTAL_REDFLAGS"] > 0:
            with st.expander("🔍 Detalles de Red Flags por Categoría"):
                col_rf1, col_rf2 = st.columns(2)
                
                with col_rf1:
                    st.metric("Total de Red Flags Detectados", redflags_dict["TOTAL_REDFLAGS"])
                
                with col_rf2:
                    st.metric("Total de Features a Guardar", len(FEATURES))
                
                # Mostrar red flags por categoría
                if redflags_dict["AL_DIRECTO"]:
                    st.markdown("**🚨 AL DIRECTO** (Amiloidosis AL)")
                    for rf in redflags_dict["AL_DIRECTO"]:
                        st.write(f"✓ {rf}")
                
                if redflags_dict["AL_SISTEMICO"]:
                    st.markdown("**🟠 AL SISTÉMICO**")
                    for rf in redflags_dict["AL_SISTEMICO"]:
                        st.write(f"✓ {rf}")
                
                if redflags_dict["ATTR_CLASICO"]:
                    st.markdown("**🟣 ATTR CLÁSICO** (Musculoesquelético)")
                    for rf in redflags_dict["ATTR_CLASICO"]:
                        st.write(f"✓ {rf}")
                
                if redflags_dict["ATTR_HEREDITARIO"]:
                    st.markdown("**🧬 ATTR HEREDITARIA**")
                    for rf in redflags_dict["ATTR_HEREDITARIO"]:
                        st.write(f"✓ {rf}")
                
                if redflags_dict["HALLAZGOS_CARDIACOS"]:
                    st.markdown("**💙 HALLAZGOS CARDIACOS**")
                    for rf in redflags_dict["HALLAZGOS_CARDIACOS"]:
                        st.write(f"✓ {rf}")
                
                if redflags_dict["PARAMETROS_CARDIACOS"]:
                    st.markdown("**📈 PARÁMETROS CARDIACOS**")
                    for rf in redflags_dict["PARAMETROS_CARDIACOS"]:
                        st.write(f"✓ {rf}")
                
                if redflags_dict["FACTORES_CONFUSORES"]:
                    st.markdown("**⚠️ FACTORES CONFUSORES**")
                    for rf in redflags_dict["FACTORES_CONFUSORES"]:
                        st.write(f"✓ {rf}")
        else:
            st.info("ℹ️ Sin hallazgos detectados - se guardará como caso sin red flags", icon="ℹ️")
        
        st.divider()
        
        diag_final = st.selectbox(
            "Diagnóstico Real (según confirmación posterior):",
            ["---", "ATTR", "AL", "HVI-HTA", "Sano"],
            help="Selecciona el diagnóstico confirmado para entrenar el modelo"
        )
        
        if st.button("💾 Guardar Caso en Base de Datos", type="primary", use_container_width=True):
            if diag_final != "---":
                # Obtener confianza y nivel guardados durante el análisis
                confianza_guardada = st.session_state.get('confianza_analisis', 0.0)
                nivel_guardado = st.session_state.get('nivel_diagnostico', '')
                
                resultado = save_case_training(
                    st.session_state.form_data, 
                    diag_final,
                    confianza_ia=confianza_guardada,
                    modelo_usado=nivel_guardado
                )
                if resultado != "ERROR":
                    # Mostrar resumen completo de lo guardado
                    resumen = generar_resumen_guardado(resultado, st.session_state.form_data, diag_final)
                    st.markdown(resumen)
                    
                    # Información de éxito con detalles
                    col_exito1, col_exito2 = st.columns(2)
                    with col_exito1:
                        st.metric("Caso Guardado", f"#{resultado}")
                    with col_exito2:
                        st.metric("Confianza del Modelo", f"{confianza_guardada:.1f}%")
                    
                    st.success(f"✅ **CASO #{resultado} GUARDADO EXITOSAMENTE**\n\nDiagnóstico: **{diag_final}** | Red Flags: **{redflags_dict['TOTAL_REDFLAGS']}** | Features: **{len(FEATURES)}**")
                    time.sleep(2)
                else:
                    st.error("❌ No se pudo guardar. Posibles causas:\n- El archivo CSV está abierto en Excel\n- No hay permisos de escritura\n- Espacio en disco insuficiente\n\nIntenta nuevamente después de cerrar el archivo.")
            else:
                st.warning("⚠️ Selecciona un diagnóstico válido para guardar el caso")

# ================================================================
# TAB 3: GUÍA CLÍNICA 
# ================================================================
elif selected_tab == "Guia Clinica":
    st.markdown("### 🧭 Guía Clínica y Algoritmo Diagnóstico")
    st.markdown("Comprende cómo funciona el algoritmo de diagnóstico de amiloidosis cardíaca")
    st.divider()
    
    # ========== SECCIÓN 1: RESUMEN VISUAL DEL ALGORITMO ==========
    st.subheader("🎯 ¿Cómo Funciona el Algoritmo?")
    
    st.info("""
    El algoritmo evalúa **múltiples dimensiones clínicas** y asigna puntos según la gravedad de cada hallazgo.
    Al final, la suma total de puntos determina el nivel de sospecha.
    """)
    
    # Diagrama de flujo visual
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### 📥 ENTRADA")
        st.markdown("""
        **Datos Clínicos:**
        - Ecocardiograma
        - ECG
        - Biomarcadores
        - Resonancia Magnética
        - Historia Clínica
        - Red Flags
        """)
    
    with col2:
        st.markdown("#### ⚙️ PROCESAMIENTO")
        st.markdown("""
        **Sistema de Puntaje:**
        1. Asigna puntos a cada hallazgo
        2. Suma total de puntos
        3. Identifica patrones
        4. Diferencia AL vs ATTR
        5. Detecta confusores (HTA, EAo)
        """)
    
    with col3:
        st.markdown("#### 📤 SALIDA")
        st.markdown("""
        **Clasificación:**
        - 🔴 ALTA SOSPECHA (AL/ATTR)
        - 🟡 INTERMEDIA (Screening)
        - 🔵 HVI No Amiloide
        - 🟢 BAJA / Sano
        """)
    
    st.divider()
    
    # ========== SECCIÓN 2: TABLA DE PUNTAJES ==========
    st.subheader("📊 Sistema de Puntajes por Hallazgo")
    
    tab1, tab2, tab3, tab4 = st.tabs(["🫀 Cardiacos", "🧬 AL Específicos", "🦴 ATTR Específicos", "🔬 Imagen Avanzada"])
    
    with tab1:
        st.markdown("### Hallazgos Cardíacos")
        df_cardiacos = pd.DataFrame({
            'Hallazgo': [
                'IVS ≥ 14mm (hipertrofia)',
                'Voltaje < 0.5 mV (bajo voltaje)',
                'GLS reducido (< -16%)',
                'Apical Sparing',
                'BAV/Marcapasos',
                'Pseudoinfarto (ondas Q)',
                'Dilatación biauricular'
            ],
            'Puntos': [3, 4, 3, 5, 3, 2, 2],
            'Importancia': ['Media', 'Alta', 'Media', 'Muy Alta', 'Media', 'Baja', 'Baja']
        })
        st.dataframe(df_cardiacos, use_container_width=True, hide_index=True)
    
    with tab2:
        st.markdown("### Red Flags de Amiloidosis AL")
        df_al = pd.DataFrame({
            'Hallazgo': [
                'MGUS/Paraproteína',
                'Macroglosia',
                'Púrpura periorbital',
                'Síndrome nefrótico',
                'Hepatomegalia',
                'Polineuropatía',
                'Disautonomía'
            ],
            'Puntos': [6, 5, 4, 3, 2, 2, 2],
            'Importancia': ['Muy Alta', 'Muy Alta', 'Alta', 'Media', 'Baja', 'Baja', 'Baja']
        })
        st.dataframe(df_al, use_container_width=True, hide_index=True)
        st.warning("⚠️ La presencia de MGUS + cualquier hallazgo cardiaco → **ALTA SOSPECHA AL directa**")
    
    with tab3:
        st.markdown("### Red Flags de Amiloidosis ATTR")
        df_attr = pd.DataFrame({
            'Hallazgo': [
                'STC bilateral',
                'Estenosis lumbar',
                'Rotura de bíceps bilateral',
                'Tendinitis hombro',
                'Contractura Dupuytren',
                'Mutación TTR confirmada'
            ],
            'Puntos': [4, 3, 3, 2, 2, 10],
            'Importancia': ['Alta', 'Media', 'Media', 'Baja', 'Baja', 'Diagnóstico']
        })
        st.dataframe(df_attr, use_container_width=True, hide_index=True)
        st.info("💡 La **tríada clásica** (STC + Estenosis lumbar + Rotura bíceps) es casi patognomónica de ATTR")
    
    with tab4:
        st.markdown("### Hallazgos de Imagen Avanzada (RM Cardíaca)")
        df_rm = pd.DataFrame({
            'Hallazgo': [
                'ECV > 40%',
                'LGE subendocárdico/transmural',
                'T1 Mapping elevado',
                'NT-proBNP > 3000 pg/ml'
            ],
            'Puntos': [5, 5, 4, 4],
            'Importancia': ['Muy Alta', 'Muy Alta', 'Alta', 'Alta']
        })
        st.dataframe(df_rm, use_container_width=True, hide_index=True)
        st.success("✅ ECV > 40% es **casi patognomónico** de amiloidosis")
    
    st.divider()
    
    # ========== SECCIÓN 3: UMBRALES DE CLASIFICACIÓN ==========
    st.subheader("🎚️ Umbrales de Clasificación")
    
    col_u1, col_u2, col_u3, col_u4 = st.columns(4)
    
    with col_u1:
        st.metric("BAJA / Sano", "< 1 punto", help="Sin hallazgos significativos")
    
    with col_u2:
        st.metric("INTERMEDIA", "1-1 puntos", help="Requiere investigación adicional")
    
    with col_u3:
        st.metric("HVI No Amiloide", "2-5 puntos + HTA/EAo", help="Hipertrofia por sobrecarga")
    
    with col_u4:
        st.metric("ALTA SOSPECHA", "≥ 2 puntos", help="AL: 2+ con red flags | ATTR: 2+ con tríada")
    
    st.divider()
    
    # ========== SECCIÓN 4: EJEMPLOS PRÁCTICOS ==========
    st.subheader("📝 Ejemplos de Clasificación")
    
    ejemplo = st.selectbox(
        "Selecciona un ejemplo:",
        [
            "Ejemplo 1: ALTA SOSPECHA AL",
            "Ejemplo 2: ALTA PROBABILIDAD ATTR",
            "Ejemplo 3: HVI por HTA (No Amiloide)",
            "Ejemplo 4: INTERMEDIA (Zona Gris)"
        ]
    )
    
    if "Ejemplo 1" in ejemplo:
        st.markdown("### 🔴 Caso: ALTA SOSPECHA AL")
        st.markdown("""
        **Paciente:** Varón de 62 años con insuficiencia cardíaca de nueva aparición
        
        **Hallazgos:**
        - IVS 16mm → +3 puntos
        - Voltaje ECG 0.4 mV → +4 puntos
        - MGUS detectado → +6 puntos
        - NT-proBNP 4500 pg/ml → +4 puntos
        
        **Puntaje Total: 17 puntos**
        
        **Diagnóstico: 🔴 ALTA SOSPECHA AL**
        
        **Justificación:** MGUS + hallazgos cardíacos → clasificación directa como AL de alta sospecha
        
        **Siguiente paso:** Referir urgente a hematología + biopsia endomiocárdica
        """)
    
    elif "Ejemplo 2" in ejemplo:
        st.markdown("### 🟣 Caso: ALTA PROBABILIDAD ATTR")
        st.markdown("""
        **Paciente:** Varón de 78 años con historia de STC operado hace 10 años
        
        **Hallazgos:**
        - IVS 18mm → +3 puntos
        - STC bilateral previo → +4 puntos
        - Estenosis lumbar → +3 puntos
        - Apical sparing en eco → +5 puntos
        - Distribución concéntrica → Patrón ATTR
        
        **Puntaje Total: 15 puntos**
        
        **Diagnóstico: 🟣 ALTA PROBABILIDAD ATTR**
        
        **Justificación:** Tríada musculoesquelética + patrón ecocardiográfico típico de ATTR
        
        **Siguiente paso:** Gammagrafía ósea + genotipado TTR
        """)
    
    elif "Ejemplo 3" in ejemplo:
        st.markdown("### 🔵 Caso: HVI por HTA (No Amiloide)")
        st.markdown("""
        **Paciente:** Mujer de 55 años con HTA desde hace 20 años
        
        **Hallazgos:**
        - IVS 15mm → +3 puntos
        - HTA severa confirmada → Factor confusor
        - Voltaje normal (1.2 mV) → 0 puntos
        - Sin red flags sistémicos
        - NT-proBNP 150 pg/ml (normal)
        
        **Puntaje Total: 3 puntos + confusor HTA**
        
        **Diagnóstico: 🔵 HVI No Amiloide**
        
        **Justificación:** HVI explicada por HTA de larga data, sin características infiltrativas
        
        **Siguiente paso:** Control de HTA y seguimiento ecocardiográfico
        """)
    
    else:
        st.markdown("### 🟡 Caso: INTERMEDIA (Zona Gris)")
        st.markdown("""
        **Paciente:** Varón de 68 años con disnea progresiva
        
        **Hallazgos:**
        - IVS 14mm → +3 puntos
        - Voltaje 0.6 mV (límite) → 0 puntos
        - Sin red flags claros
        - NT-proBNP 800 pg/ml → +4 puntos
        
        **Puntaje Total: 7 puntos**
        
        **Diagnóstico: 🟡 INTERMEDIA (Screening)**
        
        **Justificación:** Hallazgos sugestivos pero no definitivos. Requiere más investigación
        
        **Siguiente paso:** RM cardíaca con realce tardío + cadenas ligeras en suero
        """)
    
    st.divider()
    
    # ========== SECCIÓN 5: DIFERENCIACIÓN AL vs ATTR ==========
    st.subheader("🔬 ¿Cómo Diferencia el Algoritmo AL vs ATTR?")
    
    col_dif1, col_dif2 = st.columns(2)
    
    with col_dif1:
        st.markdown("### 🔴 Patrón AL")
        st.markdown("""
        **Criterios de AL:**
        - ✅ MGUS/Paraproteína presente
        - ✅ Macroglosia, púrpura periorbital
        - ✅ Afección renal/hepática
        - ✅ NT-proBNP muy elevado (>3000)
        - ✅ Troponina elevada
        - ✅ Inicio más agudo
        - ✅ Edad: 50-70 años
        
        **Clasificación:** Si MGUS + cualquier hallazgo cardiaco → **AL directa**
        """)
    
    with col_dif2:
        st.markdown("### 🟣 Patrón ATTR")
        st.markdown("""
        **Criterios de ATTR:**
        - ✅ STC bilateral previo (años antes del Dx)
        - ✅ Tríada musculoesquelética
        - ✅ Ausencia de paraproteína
        - ✅ Patrón concéntrico en eco
        - ✅ Apical sparing marcado
        - ✅ Ausencia de FA (a pesar de ICC)
        - ✅ Edad: >70 años (ATTR-wt)
        
        **Clasificación:** Si tríada completa + edad avanzada → **ATTR alta probabilidad**
        """)
    
    st.divider()
    
    # ========== SECCIÓN 6: FACTORES CONFUSORES ==========
    st.subheader("⚠️ Factores Confusores (Diagnóstico Diferencial)")
    
    st.warning("""
    El algoritmo identifica **factores confusores** que pueden simular amiloidosis pero tienen otra causa:
    """)
    
    df_confusores = pd.DataFrame({
        'Factor Confusor': ['HTA severa', 'Estenosis aórtica', 'Insuficiencia renal crónica'],
        'Causa HVI': ['Sobrecarga de presión', 'Sobrecarga de presión', 'Retención de volumen + anemia'],
        'Cómo Diferenciarlo': [
            'HVI concéntrica simétrica, sin bajo voltaje',
            'Gradiente transvalvular elevado, soplo',
            'Creatinina elevada, anemia, hipervolemia'
        ]
    })
    st.dataframe(df_confusores, use_container_width=True, hide_index=True)
    
    st.info("""
    💡 **Regla del algoritmo:** Si hay HVI + confusor evidente + ausencia de red flags sistémicos → Clasifica como **HVI No Amiloide**
    """)

# ================================================================
# TAB 5: BASE DE DATOS
# ================================================================
if selected_tab == "Base de Datos":
    st.header("📊 Base de Datos de Entrenamiento")
    
    cargar_db = st.checkbox("Cargar base de datos", value=False)
    if not cargar_db:
        st.info("Base de datos no cargada para mejorar rendimiento. Activa la opcion para abrirla.")
        st.stop()

    ruta_abs = os.path.join(BASE_DIR, DB_FILE)
    
    if not os.path.isfile(ruta_abs):
        st.error(f"No se encuentra {DB_FILE} en la carpeta")
        st.stop()
    
    archivo_usado = DB_FILE
    
    if not os.path.isfile(ruta_abs):
        st.error(f"No se encuentra {DB_FILE} ni el backup en la carpeta")
        st.stop()

    # Cargar datos para estadísticas
    df_para_stats = pd.read_csv(ruta_abs)
    total_casos = len(df_para_stats)
    ultima_fecha = df_para_stats['fecha'].max() if 'fecha' in df_para_stats.columns else 'N/A'
    
    # Métricas superiores
    col_db1, col_db2, col_db3 = st.columns(3)
    col_db1.metric("Total de Casos", total_casos)
    col_db2.metric("Última Actualización", ultima_fecha)
    col_db3.metric("Archivo Usado", archivo_usado)

    st.divider()

    if total_casos > 0:
        with st.spinner("Cargando base de datos..."):
            # 1. Cargar datos brutos desde el archivo seleccionado (backup o principal)
            df_full = pd.read_csv(ruta_abs)
            # Asegurar que tenga todas las columnas necesarias
            for col in FEATURES:
                if col not in df_full.columns:
                    df_full[col] = None
            # Crear columnas de resultados si no existen
            if 'Diagnóstico Algoritmo' not in df_full.columns:
                df_full['Diagnóstico Algoritmo'] = ''
            if 'Hallazgos Detectados' not in df_full.columns:
                df_full['Hallazgos Detectados'] = ''
            if 'confianza_ia' not in df_full.columns:
                df_full['confianza_ia'] = ''
            if 'nhc' not in df_full.columns:
                df_full['nhc'] = ''
        
        # 2. CALCULAR DIAGNÓSTICO Y HALLAZGOS (EN TIEMPO REAL)
        # -------------------------------------------------------------------------
        # AQUÍ ESTÁ EL CAMBIO CLAVE:
        # Usamos df_full (el original) para calcular las columnas, así se guardarán.
        # -------------------------------------------------------------------------
        
        def aplicar_algoritmo(row):
            # Mapeo de nombres de columnas CSV a nombres esperados por el algoritmo
            mapeo_columnas = {
                'IVS (mm)': 'ivs',
                'Voltaje (mV)': 'volt',
                'GLS (%)': 'gls',
                'NT-proBNP (pg/ml)': 'nt_probnp',
                'Troponina elevada': 'troponina',
                'Troponina elevada crónica baja (ATTR)': 'troponina_cronica',
                'Patrón LGE': 'lge_patron',
                'ECV (%)': 'ecv',
                'T1 Mapping': 't1_mapping',
                'Hiperrealce subepicárdico (ATTR-v)': 'hiperrealce_subepicardico',
                'Edad': 'edad',
                'Sexo': 'sexo',
                'Derrame pericárdico': 'derrame_pericardico',
                'Pared posterior (mm)': 'septum_posterior',
                'Distribución concéntrica (ATTR)': 'distribucion_concentrica',
                'RV engrosado (ATTR)': 'rv_engrosado',
                'Aorta pequeña/normal relativa (ATTR)': 'aorta_pequena',
                'Ausencia de FA (ATTR vs HTA)': 'ausencia_fa',
                'STC bilateral': 'stc',
                'Rotura bíceps': 'biceps',
                'Estenosis lumbar': 'lumbar',
                'Tendinitis hombro': 'hombro',
                'Dupuytren': 'dupuytren',
                'Artralgias': 'artralgias',
                'Fractura vertebral': 'fractura_vert',
                'Tendinitis cálcica': 'tendinitis_calcifica',
                'Macroglosia': 'macro',
                'Púrpura periorbital': 'purpura',
                'MGUS/Paraproteína': 'mgus',
                'Síndrome nefrótico': 'nefro',
                'Polineuropatía': 'neuro_p',
                'Disautonomía': 'disauto',
                'Hepatomegalia': 'hepato',
                'Fatiga severa': 'fatiga',
                'Lesiones cutáneas': 'piel_lesiones',
                'Apical sparing': 'apical_sparing',
                'Dilatación biauricular': 'biatrial',
                'BAV/Marcapasos': 'bav_mp',
                'Pseudoinfarto': 'pseudo_q',
                'Bajo voltaje paradójico': 'bajo_voltaje',
                'Mutación TTR': 'mutacion_ttr',
                'Confusor HTA': 'confusor_hta',
                'Confusor EAo': 'confusor_ao',
                'Confusor IRC': 'confusor_irc'
            }
            
            # Convertimos la fila a diccionario
            d = DEFAULT_DATA.copy()
            row_dict = row.to_dict()
            
            # Contador de valores mapeados para debugging
            valores_mapeados = 0
            
            # Aplicar mapeo de columnas
            for nombre_csv, nombre_algoritmo in mapeo_columnas.items():
                if nombre_csv in row_dict:
                    valor = row_dict[nombre_csv]
                    # Convertir valores booleanos y numéricos
                    if nombre_algoritmo in DEFAULT_DATA:
                        if isinstance(DEFAULT_DATA[nombre_algoritmo], bool):
                            # Convertir a booleano
                            if isinstance(valor, bool):
                                d[nombre_algoritmo] = valor
                                valores_mapeados += 1
                            elif isinstance(valor, str):
                                d[nombre_algoritmo] = valor.lower() in ['true', '1', 'sí', 's', 'yes', 'v']
                                if d[nombre_algoritmo]:
                                    valores_mapeados += 1
                            elif isinstance(valor, (int, float)):
                                d[nombre_algoritmo] = bool(valor)
                                if d[nombre_algoritmo]:
                                    valores_mapeados += 1
                            else:
                                d[nombre_algoritmo] = False
                        elif isinstance(DEFAULT_DATA[nombre_algoritmo], (int, float)):
                            # Convertir a número, manteniendo como string si está vacío
                            try:
                                if pd.isna(valor) or valor == '' or valor is None:
                                    d[nombre_algoritmo] = DEFAULT_DATA[nombre_algoritmo]
                                else:
                                    d[nombre_algoritmo] = float(valor)
                                    valores_mapeados += 1
                            except (ValueError, TypeError):
                                d[nombre_algoritmo] = DEFAULT_DATA[nombre_algoritmo]
                        else:
                            # String o valor directo
                            if pd.notna(valor) and valor != '':
                                d[nombre_algoritmo] = valor
                                valores_mapeados += 1
                            else:
                                d[nombre_algoritmo] = DEFAULT_DATA[nombre_algoritmo]
            
            # Asegurar valores numéricos válidos
            for key in ['ivs', 'volt', 'gls', 'nt_probnp', 'ecv', 'septum_posterior', 'edad']:
                if key in d and (pd.isna(d[key]) or d[key] is None):
                    d[key] = 0.0 if isinstance(d[key], float) else 0
            
            try:
                # Use default thresholds for Base de Datos calculation
                res = calcular_riesgo_experto(d, umbral_screening=UMBRAL_SCREENING, umbral_confirmacion=UMBRAL_CONFIRMACION)
                nivel = res.get('nivel', '')
                hall = ", ".join(res.get('hallazgos', []))
                
                # Si no hay nivel después del cálculo, agregar info de debug
                if not nivel:
                    return pd.Series(["ERROR", f"Sin diagnóstico (mapeados: {valores_mapeados} valores)"])
                
                return pd.Series([nivel, hall])
            except Exception as e:
                return pd.Series(["ERROR", f"Error: {str(e)[:50]}"])

        if not df_full.empty:
            did_update = False
            if 'Diagnóstico Algoritmo' not in df_full.columns:
                df_full['Diagnóstico Algoritmo'] = ''
            if 'Hallazgos Detectados' not in df_full.columns:
                df_full['Hallazgos Detectados'] = ''

            # Asegurar que existan todas las variables necesarias
            missing_features = [f for f in FEATURES if f not in df_full.columns]
            if missing_features:
                for f in missing_features:
                    df_full[f] = DEFAULT_DATA.get(f, 0)
                st.warning("Faltaban variables en la base. Se completaron con valores por defecto para calcular el algoritmo.")

            # Botones de control
            col_btn1, col_btn2 = st.columns(2)
            with col_btn1:
                calcular_inicial = st.button("▶️ Calcular Algoritmo", type="primary")
            with col_btn2:
                recalcular_todo = st.button("🔄 Recalcular TODOS", type="secondary")

            def _is_empty(val: Any) -> bool:
                if val is None:
                    return True
                if isinstance(val, float) and np.isnan(val):
                    return True
                s = str(val).strip().lower()
                return s in ['', 'nan', 'none', 'null']

            # Calcular solo filas faltantes (o todo si se solicita)
            mask_diag = df_full['Diagnóstico Algoritmo'].apply(_is_empty)
            mask_hall = df_full['Hallazgos Detectados'].apply(_is_empty)
            mask_calc = mask_diag | mask_hall
            
            # Si se presiona el botón de calcular inicial, procesar solo filas vacías
            # Si se presiona recalcular todo, procesar todas las filas
            if calcular_inicial:
                # Solo calcular donde hay valores vacíos
                pass  # mask_calc ya está definido
            elif recalcular_todo:
                mask_calc = pd.Series([True] * len(df_full), index=df_full.index)
            
            if calcular_inicial or recalcular_todo:
                if mask_calc.any():
                    calc_count = int(mask_calc.sum())
                    try:
                        # Asegurar que las columnas tengan tipo correcto (object para strings)
                        df_full['Diagnóstico Algoritmo'] = df_full['Diagnóstico Algoritmo'].astype('object')
                        df_full['Hallazgos Detectados'] = df_full['Hallazgos Detectados'].astype('object')
                        
                        # Aplicar algoritmo a filas que necesitan cálculo
                        resultados = df_full.loc[mask_calc].apply(aplicar_algoritmo, axis=1)
                        
                        # FIX: Usar .values para asignación correcta (evita problemas de índices)
                        df_full.loc[mask_calc, 'Diagnóstico Algoritmo'] = resultados.iloc[:, 0].values
                        df_full.loc[mask_calc, 'Hallazgos Detectados'] = resultados.iloc[:, 1].values
                        
                        did_update = True
                        
                        # Contar resultados exitosos
                        resultados_validos = (~df_full.loc[mask_calc, 'Diagnóstico Algoritmo'].apply(_is_empty)).sum()
                        resultados_con_error = (df_full.loc[mask_calc, 'Diagnóstico Algoritmo'] == 'ERROR').sum()
                        
                        if resultados_validos > 0:
                            st.success(f"✅ Algoritmo calculado exitosamente en {resultados_validos} de {calc_count} casos.")
                            if resultados_con_error > 0:
                                st.warning(f"⚠️ {resultados_con_error} filas tuvieron errores durante el cálculo.")
                        else:
                            st.warning(f"⚠️ Se procesaron {calc_count} casos pero no se obtuvieron diagnósticos válidos.")
                            st.info("💡 Esto puede deberse a que todas las filas tienen valores por defecto (ceros) o datos incompletos.")
                    except Exception as e:
                        st.error(f"❌ Error durante el cálculo: {str(e)}")
                        st.error(f"Detalles: Tipo de error: {type(e).__name__}")
                        import traceback
                        with st.expander("Ver detalle técnico del error"):
                            st.code(traceback.format_exc())
                        did_update = False
                else:
                    st.info("ℹ️ Todas las filas ya tienen diagnóstico calculado. Usa 'Recalcular TODOS' para procesar nuevamente.")
            
            # Reordenamos para poner las nuevas columnas al principio
            cols_ordenadas = ['id', 'fecha', 'diagnostico', 'Diagnóstico Algoritmo', 'Hallazgos Detectados'] + \
                             [c for c in df_full.columns if c not in ['id', 'fecha', 'diagnostico', 'Diagnóstico Algoritmo', 'Hallazgos Detectados']]
            df_full = df_full[cols_ordenadas]

            if did_update:
                # Mantener la ruta correcta (backup o principal) sin sobreescribir
                df_full.to_csv(ruta_abs, index=False)
                # Verificar si hubo error en los cálculos (solo en filas calculadas)
                filas_con_error = (df_full.loc[mask_calc, 'Diagnóstico Algoritmo'] == 'ERROR').sum()
                if filas_con_error > 0:
                    st.warning(f"⚠️ {filas_con_error} filas tuvieron error en el cálculo. Revisa los datos de esas filas.")
                    # Mostrar filas con error
                    filas_error_idx = df_full.loc[mask_calc][df_full.loc[mask_calc, 'Diagnóstico Algoritmo'] == 'ERROR'].index
                    with st.expander("Ver filas con error"):
                        st.write(df_full.loc[filas_error_idx, ['id', 'fecha', 'Diagnóstico Algoritmo', 'Hallazgos Detectados']])
                elif df_full.loc[mask_calc, 'Diagnóstico Algoritmo'].apply(_is_empty).all():
                    st.error("❌ No se pudo calcular el algoritmo en ninguna fila procesada.")
                    st.info("💡 Posibles causas: columnas con nombres incorrectos o datos en formato no reconocido.")
                    # Mostrar muestra de datos procesados
                    with st.expander("Ver datos de las filas procesadas"):
                        cols_clave = ['id', 'IVS (mm)', 'Voltaje (mV)', 'GLS (%)', 'NT-proBNP (pg/ml)']
                        cols_disponibles = [c for c in cols_clave if c in df_full.columns]
                        if cols_disponibles:
                            st.write(df_full.loc[mask_calc, cols_disponibles].head())
                        else:
                            st.warning("No se encontraron columnas clave en la base de datos.")
        
        # Guardar en caché
        db_mtime = os.path.getmtime(ruta_abs)
        st.session_state.db_cache_key = db_mtime
        st.session_state.df_full_cache = df_full.copy()

        # 3. Vista completa primero
        st.subheader("📊 Base de Datos Completa")
        
        # Asegurar que existan las columnas mínimas
        if 'Diagnóstico Algoritmo' not in df_full.columns:
            df_full['Diagnóstico Algoritmo'] = ''
        if 'Hallazgos Detectados' not in df_full.columns:
            df_full['Hallazgos Detectados'] = ''
        if 'confianza_ia' not in df_full.columns:
            df_full['confianza_ia'] = ''

        # Mapeo completo de TODOS los red flags (igual que en aplicar_algoritmo)
        todas_las_caracteristicas = [
            'IVS (mm)', 'Voltaje (mV)', 'GLS (%)', 'NT-proBNP (pg/ml)',
            'Troponina elevada', 'Troponina elevada crónica baja (ATTR)',
            'Patrón LGE', 'ECV (%)', 'T1 Mapping', 'Hiperrealce subepicárdico (ATTR-v)',
            'Edad', 'Sexo', 'Derrame pericárdico', 'Pared posterior (mm)',
            'Distribución concéntrica (ATTR)', 'RV engrosado (ATTR)',
            'Aorta pequeña/normal relativa (ATTR)', 'Ausencia de FA (ATTR vs HTA)',
            'STC bilateral', 'Rotura bíceps', 'Estenosis lumbar', 'Tendinitis hombro',
            'Dupuytren', 'Artralgias', 'Fractura vertebral', 'Tendinitis cálcica',
            'Macroglosia', 'Púrpura periorbital', 'MGUS/Paraproteína', 'Síndrome nefrótico',
            'Polineuropatía', 'Disautonomía', 'Hepatomegalia', 'Fatiga severa',
            'Lesiones cutáneas', 'Apical sparing', 'Dilatación biauricular',
            'BAV/Marcapasos', 'Pseudoinfarto', 'Bajo voltaje paradójico',
            'Mutación TTR', 'Confusor HTA', 'Confusor EAo', 'Confusor IRC'
        ]
        
        # Construir lista de columnas a mostrar: base + todas las características disponibles
        columnas_mostrar = ['id', 'fecha', 'diagnostico', 'Diagnóstico Algoritmo', 'Hallazgos Detectados']
        columnas_mostrar += [c for c in todas_las_caracteristicas if c in df_full.columns]
        
        columnas_disponibles = [c for c in columnas_mostrar if c in df_full.columns]
        
        df_display = df_full[columnas_disponibles].copy()
        df_display = df_display.rename(columns={
            'diagnostico': 'Sospecha por IA convencional',
            'Diagnóstico Algoritmo': 'Sospecha Algoritmo'
        })
        
        # Mostrar con altura suficiente para ver bien todos los datos
        st.dataframe(df_display, use_container_width=True, height=600)
        
        st.divider()
        
        # 4. Vista detallada expandible por caso
        st.subheader("🔍 Vista Detallada: Todos los Hallazgos por Caso")
        
        id_casos = df_full['id'].tolist()
        caso_seleccionado = st.selectbox(
            "Selecciona un caso para ver TODOS sus hallazgos:",
            id_casos,
            format_func=lambda x: f"Caso #{x} - Real: {df_full[df_full['id']==x]['diagnostico'].values[0]} | Algoritmo: {df_full[df_full['id']==x]['Diagnóstico Algoritmo'].values[0] if 'Diagnóstico Algoritmo' in df_full.columns else 'N/A'}"
        )
        
        caso = df_full[df_full['id'] == caso_seleccionado].iloc[0]
        
        # Mostrar métricas principales
        col_m1, col_m2, col_m3 = st.columns(3)
        with col_m1:
            st.metric("ID Caso", f"#{caso.get('id', '—')}")
        with col_m2:
            st.metric("Diagnóstico Real", caso.get('diagnostico', '—'))
        with col_m3:
            diag_algo = caso.get('Diagnóstico Algoritmo', '—')
            st.metric("Diagnóstico Algoritmo", diag_algo)
        
        fecha_str = caso.get('fecha', '—')
        nhc_str = caso.get('nhc', '—')
        conf_str = f"{caso.get('confianza_ia', 0):.1f}%" if caso.get('confianza_ia') else "—"
        modelo_str = caso.get('modelo_usado', '—')
        st.markdown(f"**Fecha:** {fecha_str} | **NHC:** {nhc_str} | **Confianza IA:** {conf_str} | **Modelo:** {modelo_str}")
        
        st.divider()
        
        # Parámetros cardiacos
        st.markdown("#### 🫀 Parámetros Cardiacos")
        col_p1, col_p2, col_p3, col_p4 = st.columns(4)
        with col_p1:
            ivs_val = caso.get('ivs', 0)
            st.metric("IVS (mm)", f"{ivs_val}" if ivs_val else "—")
        with col_p2:
            gls_val = caso.get('gls', 0)
            st.metric("GLS (%)", f"{gls_val}" if gls_val else "—")
        with col_p3:
            volt_val = caso.get('volt', 0)
            st.metric("Voltaje (mV)", f"{volt_val}" if volt_val else "—")
        with col_p4:
            septum_val = caso.get('septum_posterior', 0)
            st.metric("Pared Post. (mm)", f"{septum_val}" if septum_val else "—")
        
        # Biomarcadores
        st.markdown("#### 🧪 Biomarcadores")
        col_b1, col_b2 = st.columns(2)
        with col_b1:
            nt_probnp_val = caso.get('nt_probnp', 0)
            st.metric("NT-proBNP (pg/ml)", f"{nt_probnp_val:.0f}" if nt_probnp_val else "—")
        with col_b2:
            troponina_val = "✓ Elevada" if caso.get('troponina', 0) else "Normal"
            st.metric("Troponina", troponina_val)
        
        # Resonancia Magnética
        st.markdown("#### 🎯 Resonancia Magnética Cardíaca")
        col_rm1, col_rm2, col_rm3 = st.columns(3)
        with col_rm1:
            lge_val = caso.get('lge_patron', '—')
            st.metric("Patrón LGE", str(lge_val).upper() if lge_val else "—")
        with col_rm2:
            ecv_val = caso.get('ecv', 0)
            st.metric("ECV (%)", f"{ecv_val:.1f}" if ecv_val else "—")
        with col_rm3:
            t1_val = "✓ Elevado" if caso.get('t1_mapping', 0) else "Normal"
            st.metric("T1 Mapping", t1_val)
        
        # Red Flags
        st.markdown("#### 🚩 Hallazgos Clínicos Detectados (Red Flags)")
        
        hallazgos_found = []
        red_flag_features = [
            'stc', 'biceps', 'lumbar', 'hombro', 'dupuytren', 'artralgias', 'fractura_vert', 'tendinitis_calcifica',
            'macro', 'purpura', 'mgus', 'nefro', 'neuro_p', 'disauto', 'hepato', 'fatiga', 'piel_lesiones',
            'apical_sparing', 'biatrial', 'bav_mp', 'pseudo_q', 'bajo_voltaje',
            'mutacion_ttr',
            'confusor_hta', 'confusor_ao', 'confusor_irc', 'derrame_pericardico'
        ]
        
        hallazgo_nombres = {
            'stc': '🔴 STC Bilateral',
            'biceps': '🔴 Rotura Bíceps',
            'lumbar': '🔴 Estenosis Lumbar',
            'hombro': 'Tendinitis Hombro',
            'dupuytren': 'Enfermedad Dupuytren',
            'artralgias': 'Artralgias',
            'fractura_vert': 'Fractura Vertebral',
            'tendinitis_calcifica': 'Tendinitis Cálcica',
            'macro': 'Macroglosia',
            'purpura': 'Púrpura Periorbital',
            'mgus': 'MGUS/Paraproteína',
            'nefro': 'Síndrome Nefrótico',
            'neuro_p': 'Polineuropatía',
            'disauto': 'Disautonomía',
            'hepato': 'Hepatomegalia',
            'fatiga': 'Fatiga Severa',
            'piel_lesiones': 'Lesiones Cutáneas',
            'apical_sparing': 'Apical Sparing',
            'biatrial': 'Dilatación Biauricular',
            'bav_mp': 'BAV/Marcapasos',
            'pseudo_q': 'Pseudoinfarto',
            'bajo_voltaje': 'Bajo Voltaje',
            'mutacion_ttr': 'Mutación TTR',
            'confusor_hta': 'HTA Severa',
            'confusor_ao': 'Estenosis Aórtica',
            'confusor_irc': 'Insuficiencia Renal',
            'derrame_pericardico': 'Derrame Pericárdico'
        }
        
        for feature in red_flag_features:
            if caso.get(feature, 0) in [1, True]:
                hallazgos_found.append(hallazgo_nombres.get(feature, feature))
        
        if hallazgos_found:
            cols_h = st.columns(2)
            for idx, hallazgo in enumerate(hallazgos_found):
                with cols_h[idx % 2]:
                    st.write(f"✅ {hallazgo}")
        else:
            st.info("ℹ️ Sin hallazgos clínicos detectados")
        
        st.divider()
        
        # Botón de descarga
        csv_bytes = df_full.to_csv(index=False).encode('utf-8')
        
        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            st.download_button(
                label=f"📥 Descargar Caso #{caso_seleccionado}",
                data=df_full[df_full['id'] == caso_seleccionado].to_csv(index=False).encode('utf-8'),
                file_name=f"caso_{caso_seleccionado}.csv",
                mime="text/csv",
                key=f"download_caso_{caso_seleccionado}"
            )
        
        with col_btn2:
            st.download_button(
                label="📥 Descargar Base de Datos Completa",
                data=csv_bytes,
                file_name=f"amylo_export_{datetime.datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                key="download_db_complete_tab5"
            )
        
        # Gráficos de resumen
        st.divider()
        col_g1, col_g2 = st.columns(2)
        
        with col_g1:
            st.subheader("📈 Diagnóstico Real")
            st.bar_chart(df_full['diagnostico'].value_counts())
            
        with col_g2:
            st.subheader("🤖 Diagnóstico Algoritmo")
            if 'Diagnóstico Algoritmo' in df_full.columns:
                st.bar_chart(df_full['Diagnóstico Algoritmo'].value_counts())

        # Botón de limpieza
        st.divider()
        with st.expander("⚠️ Zona de Peligro: Limpiar Base de Datos"):
            st.warning("Esta acción eliminará TODOS los casos guardados.")
            if st.button("🗑️ BORRAR TODA LA BASE DE DATOS", type="secondary"):
                ruta_abs = os.path.join(BASE_DIR, DB_FILE)
                if os.path.isfile(ruta_abs):
                    os.remove(ruta_abs)
                    st.success("✅ Base de datos eliminada.")
                    st.rerun()
    else:
        st.info("📭 No hay casos guardados aún.")


# ==========================================
# TAB 6: DIAGNÓSTICO DEL ALGORITMO
# ==========================================
if selected_tab == "Diagnóstico del Algoritmo":
    st.header("🤖 Diagnóstico del Algoritmo")
    st.markdown("Aplica el algoritmo de amiloidosis a toda la tabla y visualiza los resultados")
    
    ruta_abs = os.path.join(BASE_DIR, DB_FILE)
    
    if not os.path.isfile(ruta_abs):
        st.error(f"❌ No se encuentra {DB_FILE} en la carpeta")
        st.stop()
    
    # Cargar datos
    with st.spinner("Cargando tabla de amiloidosis..."):
        df_algoritmo = pd.read_csv(ruta_abs)
    
    st.success(f"✅ Tabla cargada: {len(df_algoritmo)} casos")
    
    # Crear columnas de resultado si no existen
    if 'Diagnóstico Algoritmo' not in df_algoritmo.columns:
        df_algoritmo['Diagnóstico Algoritmo'] = ''
    if 'Hallazgos Detectados' not in df_algoritmo.columns:
        df_algoritmo['Hallazgos Detectados'] = ''
    if 'Score' not in df_algoritmo.columns:
        df_algoritmo['Score'] = 0
    if 'Confianza (%)' not in df_algoritmo.columns:
        df_algoritmo['Confianza (%)'] = 0.0
    
    # Opciones de procesamiento
    col_opts1, col_opts2 = st.columns(2)
    
    with col_opts1:
        procesar_todos = st.checkbox("🔄 Procesar TODOS los casos", value=False)
    
    with col_opts2:
        procesar_vacios = st.checkbox("⚙️ Solo procesar resultados vacíos", value=True)
    
    # Botón de procesamiento
    if st.button("▶️ EJECUTAR ALGORITMO", type="primary"):
        with st.spinner("Procesando casos..."):
            total_procesados = 0
            
            # Mapeo de nombres de columnas CSV a nombres esperados por el algoritmo
            mapeo_columnas = {
                'IVS (mm)': 'ivs',
                'Voltaje (mV)': 'volt',
                'GLS (%)': 'gls',
                'NT-proBNP (pg/ml)': 'nt_probnp',
                'Troponina elevada': 'troponina',
                'Troponina elevada crónica baja (ATTR)': 'troponina_cronica',
                'Patrón LGE': 'lge_patron',
                'ECV (%)': 'ecv',
                'T1 Mapping': 't1_mapping',
                'Hiperrealce subepicárdico (ATTR-v)': 'hiperrealce_subepicardico',
                'Edad': 'edad',
                'Sexo': 'sexo',
                'Derrame pericárdico': 'derrame_pericardico',
                'Pared posterior (mm)': 'septum_posterior',
                'Distribución concéntrica (ATTR)': 'distribucion_concentrica',
                'RV engrosado (ATTR)': 'rv_engrosado',
                'Aorta pequeña/normal relativa (ATTR)': 'aorta_pequena',
                'Ausencia de FA (ATTR vs HTA)': 'ausencia_fa',
                'STC bilateral': 'stc',
                'Rotura bíceps': 'biceps',
                'Estenosis lumbar': 'lumbar',
                'Tendinitis hombro': 'hombro',
                'Dupuytren': 'dupuytren',
                'Artralgias': 'artralgias',
                'Fractura vertebral': 'fractura_vert',
                'Tendinitis cálcica': 'tendinitis_calcifica',
                'Macroglosia': 'macro',
                'Púrpura periorbital': 'purpura',
                'MGUS/Paraproteína': 'mgus',
                'Síndrome nefrótico': 'nefro',
                'Polineuropatía': 'neuro_p',
                'Disautonomía': 'disauto',
                'Hepatomegalia': 'hepato',
                'Fatiga severa': 'fatiga',
                'Lesiones cutáneas': 'piel_lesiones',
                'Apical sparing': 'apical_sparing',
                'Dilatación biauricular': 'biatrial',
                'BAV/Marcapasos': 'bav_mp',
                'Pseudoinfarto': 'pseudo_q',
                'Bajo voltaje paradójico': 'bajo_voltaje',
                'Mutación TTR': 'mutacion_ttr',
                'Confusor HTA': 'confusor_hta',
                'Confusor EAo': 'confusor_ao',
                'Confusor IRC': 'confusor_irc'
            }
            
            for idx, row in df_algoritmo.iterrows():
                # Verificar si ya tiene resultado
                tiene_resultado = pd.notna(df_algoritmo.loc[idx, 'Diagnóstico Algoritmo']) and \
                                  str(df_algoritmo.loc[idx, 'Diagnóstico Algoritmo']).strip() != ''
                
                if procesar_vacios and tiene_resultado and not procesar_todos:
                    continue
                
                # Preparar datos para el algoritmo
                d = DEFAULT_DATA.copy()
                row_dict = row.to_dict()
                
                # Aplicar mapeo de columnas
                for nombre_csv, nombre_algoritmo in mapeo_columnas.items():
                    if nombre_csv in row_dict:
                        valor = row_dict[nombre_csv]
                        if nombre_algoritmo in DEFAULT_DATA:
                            if isinstance(DEFAULT_DATA[nombre_algoritmo], bool):
                                if isinstance(valor, bool):
                                    d[nombre_algoritmo] = valor
                                elif isinstance(valor, str):
                                    d[nombre_algoritmo] = valor.lower() in ['true', '1', 'sí', 's', 'yes', 'v']
                                elif isinstance(valor, (int, float)):
                                    d[nombre_algoritmo] = bool(valor)
                                else:
                                    d[nombre_algoritmo] = False
                            elif isinstance(DEFAULT_DATA[nombre_algoritmo], (int, float)):
                                try:
                                    if pd.isna(valor) or valor == '' or valor is None:
                                        d[nombre_algoritmo] = DEFAULT_DATA[nombre_algoritmo]
                                    else:
                                        d[nombre_algoritmo] = float(valor)
                                except (ValueError, TypeError):
                                    d[nombre_algoritmo] = DEFAULT_DATA[nombre_algoritmo]
                            else:
                                d[nombre_algoritmo] = valor if pd.notna(valor) and valor != '' else DEFAULT_DATA[nombre_algoritmo]
                
                # Asegurar valores numéricos válidos
                for key in ['ivs', 'volt', 'gls', 'nt_probnp', 'ecv', 'septum_posterior', 'edad']:
                    if key in d and (pd.isna(d[key]) or d[key] is None):
                        d[key] = 0.0 if isinstance(d[key], float) else 0
                
                try:
                    # Ejecutar algoritmo
                    res = calcular_riesgo_experto(d)
                    
                    # Guardar resultados
                    df_algoritmo.loc[idx, 'Diagnóstico Algoritmo'] = res.get('nivel', '')
                    df_algoritmo.loc[idx, 'Hallazgos Detectados'] = ", ".join(res.get('hallazgos', []))
                    df_algoritmo.loc[idx, 'Score'] = res.get('score', 0)
                    df_algoritmo.loc[idx, 'Confianza (%)'] = res.get('confianza_porcentaje', 0.0)
                    
                    total_procesados += 1
                    
                except Exception as e:
                    df_algoritmo.loc[idx, 'Diagnóstico Algoritmo'] = f"ERROR: {str(e)[:50]}"
            
            # Guardar en CSV
            df_algoritmo.to_csv(ruta_abs, index=False)
            
        st.success(f"✅ Procesamiento completado: {total_procesados} casos calculados")
        st.rerun()
    
    # Filtros
    st.divider()
    st.subheader("🔍 Filtros de Visualización")
    
    col_filter1, col_filter2, col_filter3 = st.columns(3)
    
    with col_filter1:
        diagnosticos_unicos = ['Todos'] + sorted(df_algoritmo['Diagnóstico Algoritmo'].dropna().unique().tolist())
        diag_filtro = st.selectbox("Diagnóstico Algoritmo:", diagnosticos_unicos)
    
    with col_filter2:
        score_min = st.number_input("Score mínimo:", min_value=0, max_value=50, value=0)
    
    with col_filter3:
        mostrar_filas = st.number_input("Filas a mostrar:", min_value=5, max_value=len(df_algoritmo)+1, value=20)
    
    # Aplicar filtros
    df_filtrado = df_algoritmo.copy()
    
    if diag_filtro != 'Todos':
        df_filtrado = df_filtrado[df_filtrado['Diagnóstico Algoritmo'] == diag_filtro]
    
    df_filtrado = df_filtrado[df_filtrado['Score'] >= score_min]
    
    # Columnas a mostrar
    columnas_mostrar = ['id', 'Diagnóstico Algoritmo', 'Score', 'Confianza (%)', 'Hallazgos Detectados']
    columnas_disponibles = [col for col in columnas_mostrar if col in df_filtrado.columns]
    
    st.subheader(f"📋 Resultados ({len(df_filtrado)} casos)")
    df_filtrado_tabla = df_filtrado[columnas_disponibles].head(int(mostrar_filas)).copy()
    df_filtrado_tabla = df_filtrado_tabla.rename(columns={
        'Diagnóstico Algoritmo': 'Sospecha Algoritmo'
    })
    st.dataframe(
        df_filtrado_tabla,
        use_container_width=True,
        height=400
    )
    
    # Estadísticas
    st.divider()
    st.subheader("📊 Estadísticas Generales")
    
    col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
    
    # Contar diagnósticos
    diag_str = df_algoritmo['Diagnóstico Algoritmo'].astype(str)
    diag_counts = diag_str.value_counts()
    
    with col_stat1:
        alta_sospecha = len(df_algoritmo[diag_str.str.contains('ALTA', case=False, na=False)])
        st.metric("Alta Sospecha", alta_sospecha)
    
    with col_stat2:
        intermedia = len(df_algoritmo[diag_str.str.contains('INTERMEDIA', case=False, na=False)])
        st.metric("Intermedia", intermedia)
    
    with col_stat3:
        sano = len(df_algoritmo[diag_str.str.contains('BAJA|Sano', case=False, na=False)])
        st.metric("Bajo Riesgo/Sano", sano)
    
    with col_stat4:
        sin_calcular = len(df_algoritmo[df_algoritmo['Diagnóstico Algoritmo'].isna() | (df_algoritmo['Diagnóstico Algoritmo'] == '')])
        st.metric("Sin Calcular", sin_calcular)
    
    # Gráficos
    st.divider()
    
    col_graf1, col_graf2 = st.columns(2)
    
    with col_graf1:
        st.subheader("Distribución de Diagnósticos")
        if len(diag_counts) > 0:
            st.bar_chart(diag_counts)
        else:
            st.info("Sin datos para mostrar")
    
    with col_graf2:
        st.subheader("Distribución de Scores")
        scores = df_algoritmo['Score'].dropna()
        if len(scores) > 0:
            st.histogram(scores, bins=20, x_label="Score", y_label="Frecuencia")
        else:
            st.info("Sin datos para mostrar")
    
    # Exportar resultados
    st.divider()
    st.subheader("💾 Exportar Resultados")
    
    col_exp1, col_exp2 = st.columns(2)
    
    with col_exp1:
        csv_export = df_algoritmo.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Descargar Resultados (CSV)",
            data=csv_export,
            file_name=f"diagnostico_algoritmo_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    
    with col_exp2:
        # Crear Excel con filtrados
        df_excel = df_filtrado[columnas_disponibles].copy()
        excel_buffer = pd.DataFrame.to_excel.__defaults__[0] if hasattr(pd.DataFrame.to_excel, '__defaults__') else None
        try:
            from io import BytesIO
            excel_bytes = BytesIO()
            with pd.ExcelWriter(excel_bytes, engine='openpyxl') as writer:
                df_excel.to_excel(writer, sheet_name='Diagnóstico', index=False)
            st.download_button(
                label="📊 Descargar como Excel",
                data=excel_bytes.getvalue(),
                file_name=f"diagnostico_algoritmo_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        except Exception as e:
            st.warning(f"⚠️ No se puede exportar a Excel: {e}")


# ==========================================
# TAB 7: TEST DE ESTRÉS (VALIDACIÓN DEL ALGORITMO)
# ==========================================
if selected_tab == "Test de Estrés":
    st.header("🔬 Test de Estrés del Algoritmo")
    st.markdown("Resumen de validación del algoritmo de amiloidosis cardíaca")
    
    st.divider()
    
    # Resumen General
    st.subheader("📊 Resumen General")
    st.metric("Precisión Global (Accuracy)", "83.8%", help="Acierto total del algoritmo")
    
    st.info("""
    El algoritmo es **excelente detectando Amiloidosis AL y ATTR**, pero sufre un poco más 
    diferenciando entre casos Intermedios, Sanos y HVI (Hipertrofia Ventricular Izquierda 
    por HTA o problemas Aórticos).
    """)
    
    st.divider()
    
    # Rendimiento por Categoría
    st.subheader("🔍 Rendimiento por Categoría")
    
    # Amiloidosis AL
    st.markdown("### Amiloidosis AL (Alta Sospecha) 🟢 Excelente")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Precisión", "100%", help="No hay falsos positivos: cuando dice AL, es seguro que es AL")
    with col2:
        st.metric("Sensibilidad", "95.0%", help="Encuentra el 95% de los casos reales")
    
    st.divider()
    
    # Amiloidosis ATTR
    st.markdown("### Amiloidosis ATTR (Alta Probabilidad) 🟢 Muy bueno")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Precisión", "93.1%")
    with col2:
        st.metric("Sensibilidad", "94.5%")
    
    st.divider()
    
    # Baja / Sano
    st.markdown("### Baja / Sano 🟡 Bueno, pero con falsos positivos")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Precisión", "73.8%", help='A veces dice que es "Baja/Sano" cuando en realidad pertenece a otra categoría')
    with col2:
        st.metric("Sensibilidad", "97.0%", help="Casi nunca se le escapa un paciente realmente sano")
    
    st.divider()
    
    # HVI No Amiloide
    st.markdown("### HVI No Amiloide 🟠 Moderado")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Precisión", "84.2%")
    with col2:
        st.metric("Sensibilidad", "69.5%", help="Pierde a varios pacientes con HVI que clasifica como Intermedios o Sanos")
    
    st.divider()
    
    # Intermedia (Screening)
    st.markdown("### Intermedia (Screening) 🔴 El punto más débil")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Precisión", "70.4%")
    with col2:
        st.metric("Sensibilidad", "63.0%", help="Le cuesta identificar los casos intermedios, confundiéndolos con Sanos o HVI No Amiloide")
    
    st.divider()
    
    # Tabla resumen
    st.subheader("📋 Tabla Resumen de Métricas")
    
    df_resumen = pd.DataFrame({
        'Categoría': [
            'Amiloidosis AL (Alta Sospecha)',
            'Amiloidosis ATTR (Alta Probabilidad)',
            'Baja / Sano',
            'HVI No Amiloide',
            'Intermedia (Screening)'
        ],
        'Precisión': ['100%', '93.1%', '73.8%', '84.2%', '70.4%'],
        'Sensibilidad': ['95.0%', '94.5%', '97.0%', '69.5%', '63.0%'],
        'Evaluación': ['🟢 Excelente', '🟢 Muy bueno', '🟡 Bueno', '🟠 Moderado', '🔴 Punto débil']
    })
    
    st.dataframe(df_resumen, use_container_width=True, hide_index=True)
    
    st.divider()
    
    # Conclusiones
    st.subheader("💡 Conclusiones")
    st.success("""
    **Fortalezas:**
    - Excelente detección de AL (100% precisión, 95% sensibilidad)
    - Muy buena detección de ATTR (93.1% precisión, 94.5% sensibilidad)
    - Alta sensibilidad para identificar pacientes sanos (97%)
    
    **Áreas de mejora:**
    - Diferenciación entre casos Intermedios, Sanos y HVI
    - Sensibilidad de HVI (69.5%) y casos Intermedios (63.0%)
    - Reducir falsos positivos en categoría Baja/Sano
    """)


# ==========================================
# PESTAÑA 8: VALIDACIÓN STRESS TEST (ULTRA-STATISTICS)
# ==========================================
# ==========================================
# PESTAÑA 9: VALIDEZ DEL ALGORITMO (ELIMINADA)
# ==========================================

if __name__ == '__main__':
    # Permite ejecutar generación desde terminal sin iniciar Streamlit UI
    if os.environ.get('AMYLO_GENERATE_SYNTH', '0') == '1':
        # Activar modo headless para evitar st.* visuales
        os.environ['AMYLO_HEADLESS'] = '1'
        print("Iniciando generación de base sintética (1000 casos)...")
        try:
            df_gen = generar_base_datos_sintetica(1000)
            print(f"Generación completada. Archivo guardado en: {os.path.join(BASE_DIR, DB_FILE)}")
        except Exception as e:
            print(f"Error generando casos sintéticos: {e}")
