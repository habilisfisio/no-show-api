import os
import sys
import logging
import joblib
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from supabase import create_client
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import traceback
import time

# =============================================================================
# CONFIGURAÇÃO DE LOGGING
# =============================================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

logger.info("=" * 70)
logger.info("🚀 INICIANDO NO-SHOW API (versão com logging)")
logger.info("=" * 70)

load_dotenv()

app = FastAPI()
origins = [
    "https://octahealth.com.br",
    "https://www.octahealth.com.br",
    "https://octa-health.lovable.app",
    "https://id-preview--4efb51f1-dd1d-4fe0-a7ea-f3c4174deafa.lovable.app",
    "http://healthcheck.railway.app"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

url = os.environ.get("SUPABASE_URL")
key = os.environ.get("SUPABASE_KEY")
logger.info(f"🔑 SUPABASE_URL configurada: {'sim' if url else 'NÃO'}")
logger.info(f"🔑 SUPABASE_KEY configurada: {'sim' if key else 'NÃO'}")

supabase = create_client(url, key)

# =============================================================================
# CARREGAR MODELO
# =============================================================================
logger.info("📦 Carregando modelo...")
try:
    modelo = joblib.load("models/no_show_pipeline_v2.pkl")
    logger.info("✅ Modelo carregado com sucesso!")
    logger.info(f"   Tipo: {type(modelo)}")

    if hasattr(modelo, 'model'):
        colunas_modelo = modelo.model.exog_names
    elif hasattr(modelo, 'feature_names_in_'):
        colunas_modelo = list(modelo.feature_names_in_)
        if 'const' not in colunas_modelo:
            colunas_modelo = ['const'] + colunas_modelo
    else:
        colunas_modelo = [
            'const', 'patient_total_appointments_past', 'patient_noshow_rate_smooth',
            'patient_noshow_last_30d', 'patient_noshow_streak',
            'dia_semana_Quinta', 'dia_semana_Segunda', 'dia_semana_Sexta',
            'dia_semana_Sábado', 'dia_semana_Terça',
            'faixa_horaria_Manhã', 'faixa_horaria_Tarde', 'faixa_horaria_Noite',
            'agreement_name_Contrato Unimed', 'agreement_name_Judicializacao',
            'agreement_name_Particular', 'agreement_name_Particular (negociacao)',
            'agreement_name_Particular pago por profissional',
            'agreement_name_Unimed Seguros'
        ]
        logger.warning("⚠️ Usando lista fixa de colunas (fallback)")

    logger.info(f"   Colunas esperadas: {len(colunas_modelo)}")
    logger.info(f"   Exemplo: {colunas_modelo[:5]}...")

except Exception as e:
    logger.error(f"❌ Erro ao carregar modelo: {e}")
    traceback.print_exc()
    modelo = None
    colunas_modelo = []

logger.info("=" * 70)
logger.info("✅ API pronta para receber requisições")
logger.info("=" * 70)

# Parâmetros
MEDIA_GLOBAL = 0.1206
ALPHA_SUAVIZACAO = 10
CORTE_NEGOCIO = 0.1079

CONVENIO_MAP = {
    'particular': 'Particular',
    'judicializacao': 'Judicializacao',
    'contrato unimed': 'Contrato Unimed',
    'unimed seguros': 'Unimed Seguros',
    'particular (negociacao)': 'Particular (negociacao)',
    'particular pago por profissional': 'Particular pago por profissional',
    'atendimento custeado pela clinica': 'Atendimento custeado pela Clínica',
}

# =============================================================================
# FUNÇÃO DE FEATURE ENGINEERING
# =============================================================================
def construir_features(appt: dict, history: dict) -> pd.DataFrame:
    logger.info("   [construir_features] Iniciando...")

    try:
        appt_date = pd.to_datetime(appt['data_agendamento'])
        logger.info(f"   [construir_features] Data: {appt_date}")

        if 'hora_inicio' in appt and appt['hora_inicio']:
            if isinstance(appt['hora_inicio'], str):
                hora = pd.to_datetime(appt['hora_inicio'], format='%H:%M:%S').hour
            else:
                hora = appt['hora_inicio'].hour
        else:
            hora = 12
        logger.info(f"   [construir_features] Hora: {hora}")

        total_passado = history.get('total_agendamentos_historico', 0)
        count_faltas_passado = history.get('total_faltas_historico', 0)
        faltas_ultimos_30d = history.get('faltas_ultimos_30d', 0)
        streak_faltas = history.get('sequencia_faltas_atual', 0)
        logger.info(f"   [construir_features] Histórico: total={total_passado}, faltas={count_faltas_passado}")

        if total_passado > 0:
            taxa_raw = count_faltas_passado / total_passado
        else:
            taxa_raw = 0.0
        taxa_suavizada = (count_faltas_passado + MEDIA_GLOBAL * ALPHA_SUAVIZACAO) / (total_passado + ALPHA_SUAVIZACAO)
        logger.info(f"   [construir_features] Taxa suavizada: {taxa_suavizada:.4f}")

        features = {col: 0.0 for col in colunas_modelo}
        features['const'] = 1.0

        features['patient_total_appointments_past'] = float(total_passado)
        features['patient_noshow_rate_smooth'] = float(taxa_suavizada)
        features['patient_noshow_last_30d'] = float(faltas_ultimos_30d)
        features['patient_noshow_streak'] = float(streak_faltas)

        mapa_dias = {0: 'Segunda', 1: 'Terça', 2: 'Quarta', 3: 'Quinta', 4: 'Sexta', 5: 'Sábado', 6: 'Domingo'}
        nome_dia = f"dia_semana_{mapa_dias[appt_date.weekday()]}"
        if nome_dia in features:
            features[nome_dia] = 1.0
            logger.info(f"   [construir_features] Dia: {nome_dia}")

        if hora < 12:
            faixa = 'Manhã'
        elif hora < 18:
            faixa = 'Tarde'
        else:
            faixa = 'Noite'
        nome_faixa = f"faixa_horaria_{faixa}"
        if nome_faixa in features:
            features[nome_faixa] = 1.0
            logger.info(f"   [construir_features] Faixa: {nome_faixa}")

        convenio_original = appt.get('nome_convenio', 'Particular')
        convenio_key = str(convenio_original).strip().lower()
        convenio_mapeado = CONVENIO_MAP.get(convenio_key, convenio_original)
        nome_conv = f"agreement_name_{convenio_mapeado}"
        if nome_conv in features:
            features[nome_conv] = 1.0
            logger.info(f"   [construir_features] Convênio: {nome_conv}")

        preenchidas = sum(1 for v in features.values() if v != 0)
        logger.info(f"   [construir_features] Features preenchidas: {preenchidas}/{len(features)}")

        df = pd.DataFrame([features])
        df = df[colunas_modelo]
        logger.info(f"   [construir_features] DataFrame shape: {df.shape}")
        return df

    except Exception as e:
        logger.error(f"   ❌ Erro em construir_features: {e}")
        traceback.print_exc()
        raise

# =============================================================================
# ENDPOINT DE PREDIÇÃO
# =============================================================================
@app.post("/predict/{agendamento_id}")
async def get_prediction(agendamento_id: str):
    logger.info(f"\n🔵 [ENDPOINT] Recebida requisição para: {agendamento_id}")
    start_time = time.time()

    try:
        logger.info("   [ENDPOINT] Buscando agendamento...")
        appt_query = supabase.table("agendamentos").select("*").eq("id", agendamento_id).single().execute()
        appt = appt_query.data

        if not appt:
            logger.error("   ❌ Agendamento não encontrado")
            raise HTTPException(status_code=404, detail="Appointment not found")

        logger.info(f"   ✅ Agendamento encontrado. Paciente ID: {appt.get('paciente_id')}")

        logger.info("   [ENDPOINT] Buscando histórico...")
        paciente_id = appt.get('paciente_id')
        if not paciente_id:
            logger.error("   ❌ Paciente sem ID")
            raise HTTPException(status_code=400, detail="Paciente ID missing")

        history_query = supabase.table("v_paciente_features").select("*").eq("paciente_id", paciente_id).single().execute()
        history = history_query.data or {}
        logger.info(f"   ✅ Histórico obtido: {len(history)} campos")

        logger.info("   [ENDPOINT] Construindo features...")
        df_input = construir_features(appt, history)
        logger.info(f"   ✅ Features construídas: {df_input.shape}")

        logger.info("   [ENDPOINT] Executando predição...")
        if modelo is None:
            raise Exception("Modelo não carregado")

        if hasattr(modelo, 'predict'):
            probabilidade = float(modelo.predict(df_input)[0])
        else:
            probabilidade = float(modelo.predict_proba(df_input)[0][1])

        logger.info(f"   ✅ Probabilidade calculada: {probabilidade:.4f}")

        if probabilidade >= CORTE_NEGOCIO:
            pred_status = "RISCO DE FALTA"
            if probabilidade >= 0.30:
                nivel_risco = "ALTO"
            else:
                nivel_risco = "MÉDIO"
        else:
            pred_status = "COMPARECE"
            nivel_risco = "BAIXO"

        logger.info(f"   ✅ Classificação: {pred_status} ({nivel_risco})")

        logger.info("   [ENDPOINT] Persistindo...")
        try:
            supabase.table("ai_predicoes").upsert({
                "agendamento_id": agendamento_id,
                "predicao_status": pred_status,
                "probabilidade_risco": round(probabilidade, 2),
                "modelo_versao": "tcc_g_v2"
            }).execute()
            logger.info("   ✅ Upsert OK")

            supabase.table("ai_logs").insert({
                "agendamento_id": agendamento_id,
                "predicao": pred_status,
                "probabilidade": round(probabilidade, 2),
                "modelo_versao": "tcc_g_v2"
            }).execute()
            logger.info("   ✅ Log OK")

        except Exception as e:
            logger.warning(f"   ⚠️ Erro ao persistir: {e}")

        elapsed = time.time() - start_time
        logger.info(f"   ✅ Requisição concluída em {elapsed:.2f}s")

        return {
            "agendamento_id": agendamento_id,
            "status": pred_status,
            "probabilidade": round(probabilidade, 2),
            "nivel_risco": nivel_risco
        }

    except HTTPException as e:
        logger.error(f"   ❌ HTTPException: {e.detail}")
        raise
    except Exception as e:
        logger.error(f"   ❌ Erro inesperado: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")

# =============================================================================
# HEALTH CHECK
# =============================================================================
@app.get("/health")
async def health_check():
    logger.info("🟢 Healthcheck chamado!")
    return {
        "status": "healthy",
        "version": "tcc_g_v2",
        "environment": os.environ.get("RAILWAY_ENVIRONMENT", "local")
    }