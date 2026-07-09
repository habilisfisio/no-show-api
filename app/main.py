import os
import sys
import logging
import joblib
import pandas as pd
import numpy as np
import re
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
logger.info("🚀 INICIANDO NO-SHOW API (versão com novo modelo)")
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
    # Carrega o modelo treinado (statsmodels ou pipeline)
    modelo = joblib.load("models/no_show_pipeline_v2.pkl")  # ajuste o caminho se necessário
    logger.info("✅ Modelo carregado com sucesso!")
    logger.info(f"   Tipo: {type(modelo)}")

    # Obter lista de colunas esperadas (com base no modelo)
    # Para statsmodels, usamos model.exog_names
    if hasattr(modelo, 'model'):
        colunas_modelo = modelo.model.exog_names
    elif hasattr(modelo, 'feature_names_in_'):
        colunas_modelo = list(modelo.feature_names_in_)
        if 'const' not in colunas_modelo:
            colunas_modelo = ['const'] + colunas_modelo
    else:
        # Fallback: lista fixa extraída do modelo treinado
        colunas_modelo = [
            'const',
            'valor_servico',
            'eh_bonus',
            'patient_total_appointments_past',
            'patient_noshow_rate_smooth',
            'patient_noshow_last_30d',
            'patient_noshow_streak',
            'dia_semana_Quinta',
            'dia_semana_Segunda',
            'dia_semana_Sexta',
            'dia_semana_Sábado',
            'dia_semana_Terça',
            'faixa_horaria_Manhã',
            'faixa_horaria_Tarde',
            'faixa_horaria_Noite',
            'nome_convenio_Contrato Unimed',
            'nome_convenio_Judicializacao',   # atenção: sem acento
            'nome_convenio_Particular',
            'nome_convenio_Particular (negociacao)',
            'nome_convenio_Particular pago por profissional',
            'nome_convenio_Unimed Seguros'
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

# Parâmetros (iguais aos usados no treino)
MEDIA_GLOBAL = 0.1044  # taxa base do treino
ALPHA_SUAVIZACAO = 10
CORTE_NEGOCIO = 0.1002  # obtido do script (recall 80%)

# Mapeamento de convênios para os nomes usados nas dummies do modelo
CONVENIO_MAP = {
    'particular': 'Particular',
    'judicializacao': 'Judicializacao',
    'judicialização': 'Judicializacao',
    'contrato unimed': 'Contrato Unimed',
    'unimed seguros': 'Unimed Seguros',
    'particular (negociacao)': 'Particular (negociacao)',
    'particular (negociação)': 'Particular (negociacao)',
    'particular pago por profissional': 'Particular pago por profissional',
    'atendimento custeado pela clinica': 'Atendimento custeado pela Clínica',
}

# =============================================================================
# FUNÇÃO DE FEATURE ENGINEERING (adaptada para o novo modelo)
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
            valor_servico = 0.0  # força zero

        logger.info(f"   [construir_features] valor_servico={valor_servico}, eh_bonus={eh_bonus}")

        # 3. Features históricas
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

        # 4. Inicializar dicionário com zeros
        features = {col: 0.0 for col in colunas_modelo}
        features['const'] = 1.0

        # Preencher features numéricas
        features['valor_servico'] = float(valor_servico)
        features['eh_bonus'] = float(eh_bonus)
        features['patient_total_appointments_past'] = float(total_passado)
        features['patient_noshow_rate_smooth'] = float(taxa_suavizada)
        features['patient_noshow_last_30d'] = float(faltas_ultimos_30d)
        features['patient_noshow_streak'] = float(streak_faltas)

        # 5. Dia da semana (dummies)
        mapa_dias = {
            0: 'Segunda',
            1: 'Terça',
            2: 'Quarta',
            3: 'Quinta',
            4: 'Sexta',
            5: 'Sábado',
            6: 'Domingo'
        }
        nome_dia = f"dia_semana_{mapa_dias[appt_date.weekday()]}"
        if nome_dia in features:
            features[nome_dia] = 1.0
            logger.info(f"   [construir_features] Dia: {nome_dia}")

        # 6. Faixa horária (dummies)
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

        # 7. Convênio (dummies)
        convenio_original = appt.get('nome_convenio', 'Particular')
        convenio_key = str(convenio_original).strip().lower()
        convenio_mapeado = CONVENIO_MAP.get(convenio_key, convenio_original)
        nome_conv = f"nome_convenio_{convenio_mapeado}"
        if nome_conv in features:
            features[nome_conv] = 1.0
            logger.info(f"   [construir_features] Convênio: {nome_conv}")

        # 8. Garantir ordem das colunas
        df = pd.DataFrame([features])
        df = df[colunas_modelo]  # reordena conforme modelo
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

        # O modelo pode ser um objeto statsmodels ou um pipeline.
        if hasattr(modelo, 'predict'):
            probabilidade = float(modelo.predict(df_input)[0])
        else:
            # Fallback para modelos sklearn que usam predict_proba
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
                "modelo_versao": "tcc_g_v2"   # mantido conforme solicitado
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
        "version": "tcc_g_v2",  # mantido
        "environment": os.environ.get("RAILWAY_ENVIRONMENT", "local")
    }