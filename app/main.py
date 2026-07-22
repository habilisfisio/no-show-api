import os
import sys
import logging
import joblib
import pandas as pd
import numpy as np
import re
from datetime import datetime
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
logger.info("🚀 INICIANDO NO-SHOW API (v3 - sem suavização bayesiana)")
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

modelo_data = None
modelo = None
colunas_modelo = []
ponto_corte_base = 0.1044
corte_negocio = 0.10

try:
    modelo_data = joblib.load("models/no_show_model_v3.pkl")
    modelo = modelo_data['modelo']
    colunas_modelo = modelo_data['colunas']
    ponto_corte_base = modelo_data.get('ponto_corte_base', 0.1044)
    corte_negocio = modelo_data.get('corte_negocio', 0.10)
    
    logger.info("✅ Modelo carregado com sucesso!")
    logger.info(f"   Tipo: {type(modelo)}")
    logger.info(f"   Colunas esperadas: {len(colunas_modelo)}")
    logger.info(f"   Ponto de corte base: {ponto_corte_base:.4f}")
    logger.info(f"   Corte de negócio: {corte_negocio:.4f}")

except Exception as e:
    logger.error(f"❌ Erro ao carregar modelo: {e}")
    traceback.print_exc()
    modelo = None
    colunas_modelo = []

logger.info("=" * 70)
logger.info("✅ API pronta para receber requisições")
logger.info("=" * 70)

# =============================================================================
# CONSTANTES E MAPEAMENTOS
# =============================================================================

# Mapeamento de convênios para os nomes EXATOS usados nas dummies do modelo
CONVENIO_MAP = {
    'particular': 'Particular',
    'judicializacao': 'JudicializaAAo',      # ← NOME EXATO DO MODELO
    'judicialização': 'JudicializaAAo',      # ← NOME EXATO DO MODELO
    'contrato unimed': 'Contrato Unimed',
    'unimed seguros': 'Unimed Seguros',
    'particular (negociacao)': 'Particular (negociacao)',
    'particular (negociação)': 'Particular (negociacao)',
    'particular pago por profissional': 'Particular pago por profissional',
}

# Mapeamento de dias da semana
DIAS_SEMANA = {
    0: 'Segunda',
    1: 'Terça',
    2: 'Quarta',
    3: 'Quinta',
    4: 'Sexta',
    5: 'Sábado',
    6: 'Domingo'
}

# Mapeamento de faixas horárias
def get_faixa_horaria(hora: int) -> str:
    if hora < 12:
        return 'Manhã'
    elif hora < 18:
        return 'Tarde'
    else:
        return 'Noite'

# =============================================================================
# FUNÇÃO DE FEATURE ENGINEERING
# =============================================================================
def construir_features(appt: dict, history: dict) -> pd.DataFrame:
    """
    Constrói o DataFrame de features para o modelo logístico.
    Inclui valor_servico, eh_bonus, features históricas, dia_semana, faixa_horaria e convênio.
    """
    logger.info("   [construir_features] Iniciando...")

    try:
        # 1. Data e hora
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

        # 2. Valor do serviço e flag de bônus
        valor_servico = 0.0
        eh_bonus = 0

        # Tenta usar valor_procedimento se existir
        if 'valor_procedimento' in appt and appt['valor_procedimento'] is not None:
            try:
                valor_servico = float(appt['valor_procedimento'])
            except:
                valor_servico = 0.0

        # Se valor_procedimento não disponível ou for zero, tenta extrair do nome_procedimento
        if valor_servico == 0.0:
            nome_proc = appt.get('nome_procedimento', '')
            if isinstance(nome_proc, str):
                # Mesma regex usada no treino
                match = re.search(r"(\d{1,3}(?:\.\d{3})*,\d{2}|\d+,\d{2}|\d+)", nome_proc)
                if match:
                    valor_str = match.group(1).replace(".", "").replace(",", ".")
                    try:
                        valor_servico = float(valor_str)
                    except:
                        valor_servico = 0.0

        # Detecta se é bônus (valor zero ou nome contém "bônus")
        nome_proc = appt.get('nome_procedimento', '')
        if (valor_servico == 0.0) or (isinstance(nome_proc, str) and re.search(r'bônus|bonus', nome_proc, re.IGNORECASE)):
            eh_bonus = 1
            valor_servico = 0.0

        logger.info(f"   [construir_features] valor_servico={valor_servico}, eh_bonus={eh_bonus}")

        # 3. Features históricas (SEM suavização bayesiana)
        total_passado = history.get('total_agendamentos_historico', 0)
        count_faltas_passado = history.get('total_faltas_historico', 0)
        faltas_ultimos_30d = history.get('faltas_ultimos_30d', 0)
        streak_faltas = history.get('sequencia_faltas_atual', 0)
        
        logger.info(f"   [construir_features] Histórico: total={total_passado}, faltas={count_faltas_passado}")

        # Taxa de faltas passada (sem suavização)
        if total_passado > 0:
            taxa_faltas = count_faltas_passado / total_passado
        else:
            taxa_faltas = 0.0
        logger.info(f"   [construir_features] Taxa de faltas: {taxa_faltas:.4f}")

        # 4. Inicializar dicionário com zeros
        features = {col: 0.0 for col in colunas_modelo}
        features['const'] = 1.0

        # Preencher features numéricas
        features['valor_servico'] = float(valor_servico)
        features['eh_bonus'] = float(eh_bonus)
        features['patient_total_appointments_past'] = float(total_passado)
        features['patient_noshow_rate_past'] = float(taxa_faltas)  # ← SEM suavização
        features['patient_noshow_last_30d'] = float(faltas_ultimos_30d)
        features['patient_noshow_streak'] = float(streak_faltas)

        # 5. Dia da semana (dummies)
        nome_dia = f"dia_semana_{DIAS_SEMANA[appt_date.weekday()]}"
        if nome_dia in features:
            features[nome_dia] = 1.0
            logger.info(f"   [construir_features] Dia: {nome_dia}")

        # 6. Faixa horária (dummies)
        faixa = get_faixa_horaria(hora)
        nome_faixa = f"faixa_horaria_{faixa}"
        if nome_faixa in features:
            features[nome_faixa] = 1.0
            logger.info(f"   [construir_features] Faixa: {nome_faixa}")

        # 7. Convênio (dummies)
        convenio_original = appt.get('nome_convenio', 'Particular')
        convenio_key = str(convenio_original).strip().lower()
        convenio_mapeado = CONVENIO_MAP.get(convenio_key, convenio_original)
        nome_conv = f"nome_convenio_{convenio_mapeado}"
        if nome_conv in features:
            features[nome_conv] = 1.0
            logger.info(f"   [construir_features] Convênio: {nome_conv}")
        else:
            logger.warning(f"   ⚠️ Convênio '{nome_conv}' não encontrado no modelo")

        # 8. Garantir ordem das colunas
        df_features = pd.DataFrame([features])
        
        # Verificar se todas as colunas existem
        colunas_faltantes = set(colunas_modelo) - set(df_features.columns)
        if colunas_faltantes:
            logger.warning(f"   ⚠️ Colunas faltantes no DataFrame: {colunas_faltantes}")
            for col in colunas_faltantes:
                df_features[col] = 0.0
        
        df_features = df_features[colunas_modelo]  # reordena conforme modelo
        logger.info(f"   [construir_features] DataFrame shape: {df_features.shape}")
        return df_features

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

        # Verifica se o modelo é statsmodels ou sklearn
        if hasattr(modelo, 'predict'):
            probabilidade = float(modelo.predict(df_input)[0])
        else:
            # Fallback para modelos sklearn
            probabilidade = float(modelo.predict_proba(df_input)[0][1])

        logger.info(f"   ✅ Probabilidade calculada: {probabilidade:.4f}")

        # Classificação baseada no corte de negócio
        if probabilidade >= corte_negocio:
            pred_status = "RISCO DE FALTA"
            if probabilidade >= 0.30:
                nivel_risco = "ALTO"
            elif probabilidade >= 0.20:
                nivel_risco = "MÉDIO"
            else:
                nivel_risco = "BAIXO"
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
                "modelo_versao": "tcc_g_v3"
            }).execute()
            logger.info("   ✅ Upsert OK")

            supabase.table("ai_logs").insert({
                "agendamento_id": agendamento_id,
                "predicao": pred_status,
                "probabilidade": round(probabilidade, 2),
                "modelo_versao": "tcc_g_v3"
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
            "nivel_risco": nivel_risco,
            "modelo_versao": "tcc_g_v3"
        }

    except HTTPException as e:
        logger.error(f"   ❌ HTTPException: {e.detail}")
        raise
    except Exception as e:
        logger.error(f"   ❌ Erro inesperado: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")

# =============================================================================
# ENDPOINT DE DIAGNÓSTICO
# =============================================================================
@app.get("/diagnose/{agendamento_id}")
async def diagnose(agendamento_id: str):
    """Endpoint para diagnóstico - retorna todas as features usadas na predição"""
    try:
        appt_query = supabase.table("agendamentos").select("*").eq("id", agendamento_id).single().execute()
        appt = appt_query.data
        
        if not appt:
            raise HTTPException(status_code=404, detail="Appointment not found")
        
        paciente_id = appt.get('paciente_id')
        history_query = supabase.table("v_paciente_features").select("*").eq("paciente_id", paciente_id).single().execute()
        history = history_query.data or {}
        
        df_input = construir_features(appt, history)
        
        if hasattr(modelo, 'predict'):
            probabilidade = float(modelo.predict(df_input)[0])
        else:
            probabilidade = float(modelo.predict_proba(df_input)[0][1])
        
        return {
            "agendamento_id": agendamento_id,
            "features": df_input.to_dict('records')[0],
            "probabilidade": round(probabilidade, 4),
            "colunas_esperadas": colunas_modelo
        }
        
    except Exception as e:
        logger.error(f"❌ Erro no diagnóstico: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# =============================================================================
# HEALTH CHECK
# =============================================================================
@app.get("/health")
async def health_check():
    logger.info("🟢 Healthcheck chamado!")
    return {
        "status": "healthy",
        "version": "tcc_g_v3",
        "environment": os.environ.get("RAILWAY_ENVIRONMENT", "local"),
        "modelo_carregado": modelo is not None,
        "colunas": len(colunas_modelo)
    }