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
import json

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
    modelo = joblib.load("models/no_show_model.pkl")  # ajuste o caminho se necessário
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
            'patient_noshow_rate_past',
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

# Adicione uma variável global para armazenar os metadados
METADADOS_MODELO = {}

def carregar_metadados():
    global METADADOS_MODELO
    caminho = os.path.join(os.getcwd(), "models/model_metadata.json")
    logger.info(f"🔍 Tentando carregar metadados em: {caminho}")
    
    try:
        with open(caminho, "r") as f:
            METADADOS_MODELO = json.load(f)
        logger.info(f"✅ Metadados carregados: {METADADOS_MODELO}")
    except FileNotFoundError:
        logger.error(f"❌ Erro: Arquivo não encontrado no caminho {caminho}")
    except json.JSONDecodeError:
        logger.error("❌ Erro: O arquivo JSON está corrompido ou mal formatado.")
    except Exception as e:
        logger.error(f"❌ Erro inesperado ao carregar metadados: {type(e).__name__} - {e}")

# Chame a função logo após carregar o modelo no seu main.py
carregar_metadados()

# =============================================================================
# FUNÇÃO DE FEATURE ENGINEERING (adaptada para o novo modelo)
# =============================================================================
# =============================================================================
# FUNÇÃO DE FEATURE ENGINEERING (Alinhada com o modelo v3)
# =============================================================================
def construir_features(appt: dict, history: dict) -> pd.DataFrame:
    """
    Constrói o DataFrame de features para o modelo logístico.
    Garante que os nomes gerados correspondam EXATAMENTE aos nomes do treino.
    """
    logger.info("   [construir_features] Iniciando...")

    try:
        # 1. Inicializar dicionário com ZEROS para todas as colunas que o modelo espera
        # Isso garante que, se uma categoria não existir (ex: não é segunda-feira), ela fica com 0.0
        features = {col: 0.0 for col in colunas_modelo}
        if 'const' in features:
            features['const'] = 1.0

        # 2. Data e hora
        appt_date = pd.to_datetime(appt['data_agendamento'])
        hora = pd.to_datetime(appt.get('hora_inicio', '12:00:00'), format='%H:%M:%S').hour if isinstance(appt.get('hora_inicio'), str) else getattr(appt.get('hora_inicio'), 'hour', 12)

        # 3. Valor do serviço e flag de bônus
        valor_servico = 0.0
        eh_bonus = 0

        if appt.get('valor_procedimento'):
            try: valor_servico = float(appt['valor_procedimento'])
            except: pass

        nome_proc = str(appt.get('nome_procedimento', ''))
        if valor_servico == 0.0:
            match = re.search(r"(\d{1,3}(?:\.\d{3})*,\d{2}|\d+,\d{2}|\d+)", nome_proc)
            if match:
                try: valor_servico = float(match.group(1).replace(".", "").replace(",", "."))
                except: pass

        if (valor_servico == 0.0) or re.search(r'bônus|bonus', nome_proc, re.IGNORECASE):
            eh_bonus = 1
            valor_servico = 0.0

        # Atribuir features numéricas (com os Nomes Exatos do train.py)
        if 'valor_servico' in features: features['valor_servico'] = float(valor_servico)
        if 'eh_bonus' in features: features['eh_bonus'] = float(eh_bonus)

        # 4. Features históricas
        total_passado = float(history.get('total_agendamentos_historico', 0))
        count_faltas_passado = float(history.get('total_faltas_historico', 0))
        
        # A taxa agora se chama 'patient_noshow_rate_past' conforme seu treino
        taxa_raw = count_faltas_passado / total_passado if total_passado > 0 else 0.0

        if 'patient_total_appointments_past' in features: features['patient_total_appointments_past'] = total_passado
        if 'patient_noshow_rate_past' in features: features['patient_noshow_rate_past'] = taxa_raw
        if 'patient_noshow_last_30d' in features: features['patient_noshow_last_30d'] = float(history.get('faltas_ultimos_30d', 0))
        if 'patient_noshow_streak' in features: features['patient_noshow_streak'] = float(history.get('sequencia_faltas_atual', 0))

        # 5. Dia da semana (Mapeado para INGLÊS, igual ao train.py)
        mapa_dias = {0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 3: 'Thursday', 4: 'Friday', 5: 'Saturday', 6: 'Sunday'}
        nome_dia = f"dia_semana_{mapa_dias[appt_date.weekday()]}"
        if nome_dia in features:
            features[nome_dia] = 1.0

        # 6. Faixa horária
        faixa = 'Manhã' if hora < 12 else 'Tarde' if hora < 18 else 'Noite'
        nome_faixa = f"faixa_horaria_{faixa}"
        if nome_faixa in features:
            features[nome_faixa] = 1.0

        # 7. Convênio
        convenio_original = appt.get('nome_convenio', 'Particular')
        convenio_key = str(convenio_original).strip().lower()
        convenio_mapeado = CONVENIO_MAP.get(convenio_key, convenio_original)
        nome_conv = f"nome_convenio_{convenio_mapeado}"
        if nome_conv in features:
            features[nome_conv] = 1.0

        # 8. Empacotar no formato e ordem corretos
        df = pd.DataFrame([features])
        df = df[colunas_modelo] # Garante a mesma ordem do treino
        return df

    except Exception as e:
        logger.error(f"   ❌ Erro em construir_features: {e}")
        traceback.print_exc()
        raise

# =============================================================================
# ENDPOINT DE PREDIÇÃO
# =============================================================================
@app.get("/model/info")
async def get_model_info():
    """Retorna os metadados gerados pelo train.py."""
    return {
        **METADADOS_MODELO,
        "status": "healthy" if modelo is not None else "error",
        "api_version": "tcc_g_v3"
    }

@app.get("/model/features")
async def get_model_features():
    """Retorna a importância baseada nos coeficientes reais do modelo carregado."""
    if modelo is None:
        raise HTTPException(status_code=503, detail="Modelo não disponível")
    
    # Se o JSON de metadados já tiver as importâncias, retorne-as
    if "features_importances" in METADADOS_MODELO:
        return {"importancias": METADADOS_MODELO["features_importances"]}
        
    # Caso contrário, calcula em tempo real como antes
    coefs = modelo.params.drop("const", errors='ignore')
    return {
        "importancias": sorted(
            [{"nome": k, "importancia": float(v)} for k, v in coefs.items()],
            key=lambda x: abs(x['importancia']), reverse=True
        )
    }

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
                "modelo_versao": "tcc_g_v3"   # mantido conforme solicitado
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
        "version": "tcc_g_v3",  # mantido
        "environment": os.environ.get("RAILWAY_ENVIRONMENT", "local")
    }