import os
import sys
import json
import logging
import time
from typing import Dict, Any, Optional, List
from contextlib import asynccontextmanager

import numpy as np
import joblib
import anyio
import onnxruntime as ort
from fastapi import FastAPI, HTTPException, status, Path, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from supabase._async.client import AsyncClient, create_client as create_async_client

# =============================================================================
# 1. LOGGING CORPORATIVO DE MLOPS
# =============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(levelname)s] - %(name)s - %(message)s"
)
logger = logging.getLogger("mlops_noshow_api")

# =============================================================================
# 2. SCHEMAS DE VALIDAÇÃO (PYDANTIC V2)
# =============================================================================
class PredictionResponse(BaseModel):
    agendamento_id: str = Field(..., description="UUID do agendamento avaliado")
    status: str = Field(..., description="Classificação: COMPARECE ou RISCO DE CANCELAMENTO")
    probabilidade: float = Field(..., ge=0.0, le=1.0, description="Probabilidade estimada de ausência")
    nivel_risco: str = Field(..., description="Nível de Risco: ALTO, MÉDIO ou BAIXO")
    modelo_versao: str = Field(..., description="Identificador da versão do modelo ONNX")
    tempo_execucao_ms: float = Field(..., description="Tempo de inferência em milissegundos")

class HealthResponse(BaseModel):
    status: str = Field(..., description="Estado da aplicação")
    onnx_loaded: bool = Field(..., description="Sinaliza se o ONNX está ativo na RAM")
    threshold: float = Field(..., description="Limite de decisão de risco")
    total_colunas: int = Field(..., description="Quantidade de variáveis esperadas pela rede")

# =============================================================================
# 3. GERENCIAMENTO DE CICLO DE VIDA (LIFESPAN VIA APP.STATE)
# =============================================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 [MLOps Lifespan] Inicializando motor ONNX Runtime e recursos...")
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        models_dir = os.path.join(base_dir, "models")
        if not os.path.exists(models_dir):
            models_dir = os.path.join(os.path.dirname(base_dir), "models")

        onnx_path = os.path.join(models_dir, "modelo_rede.onnx")
        scaler_path = os.path.join(models_dir, "scaler.pkl")
        colunas_path = os.path.join(models_dir, "colunas_modelo.json")

        if not os.path.exists(onnx_path):
            raise FileNotFoundError(f"❌ Modelo ONNX não encontrado em: {onnx_path}")

        session_options = ort.SessionOptions()
        session_options.intra_op_num_threads = 2
        session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

        app.state.onnx_session = ort.InferenceSession(onnx_path, session_options)
        app.state.scaler = joblib.load(scaler_path)

        with open(colunas_path, "r", encoding="utf-8") as f:
            app.state.colunas_modelo = json.load(f)

        app.state.threshold = float(os.getenv("THRESHOLD", "0.3623"))

        # Supabase Async Client
        url = os.environ.get("SUPABASE_URL", "")
        key = os.environ.get("SUPABASE_KEY", "")
        if url and key:
            app.state.supabase = await create_async_client(url, key)
            logger.info("✅ Conexão assíncrona Supabase estabelecida.")
        else:
            app.state.supabase = None
            logger.warning("⚠️ SUPABASE_URL ou SUPABASE_KEY não configuradas no .env.")

        logger.info(f"✅ Artefatos de ML carregados na RAM! Features ({len(app.state.colunas_modelo)}), Threshold: {app.state.threshold}")
        yield

    except Exception as e:
        logger.critical(f"❌ Falha crítica ao inicializar artefatos de ML: {e}")
        sys.exit(1)
    finally:
        app.state.onnx_session = None
        app.state.scaler = None
        app.state.supabase = None
        logger.info("🛑 Artefatos desinterligados da memória RAM.")

# =============================================================================
# 4. INICIALIZAÇÃO DA API FASTAPI E CORS
# =============================================================================
app = FastAPI(
    title="No-Show Prediction Engine",
    description="Engine MLOps assíncrono para predição de absenteísmo médico usando ONNX Runtime",
    version="4.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type"],
)

# =============================================================================
# 5. FEATURE ENGINEERING PARITY ENGINE (SEM OVERHEAD DE PANDAS)
# =============================================================================
def limpar_valor_procedimento(val: Any) -> float:
    if isinstance(val, str):
        val = val.strip()
        if not val:
            return 0.0
        val = val.replace(".", "").replace(",", ".")
        try:
            return float(val)
        except ValueError:
            return 0.0
    try:
        return float(val) if val is not None else 0.0
    except (ValueError, TypeError):
        return 0.0

def extrair_e_alinhar_features_otimizado(appt: dict, history: dict, colunas_modelo: List[str], scaler: Any) -> np.ndarray:
    # Day mapping
    raw_date = appt.get('data_agendamento')
    dia_semana_mapeado = "Segunda"
    if raw_date:
        try:
            import datetime
            d = datetime.date.fromisoformat(str(raw_date)[:10])
            dias = ["Segunda", "Terça", "Quarta", "Quinta", "Sexta", "Sábado", "Domingo"]
            dia_semana_mapeado = dias[d.weekday()]
        except Exception:
            dia_semana_mapeado = "Segunda"

    # Hour mapping
    hora_str = str(appt.get('hora_inicio', '12:00:00'))
    try:
        hora = int(hora_str.split(":")[0])
    except Exception:
        hora = 12

    if hora <= 11:
        faixa_horaria = "Manhã"
    elif hora <= 17:
        faixa_horaria = "Tarde"
    else:
        faixa_horaria = "Noite"

    valor_servico = limpar_valor_procedimento(appt.get('valor_procedimento'))
    recorrencia_str = str(appt.get('recorrencia', '')).strip().lower()

    tem_pacote = 1.0 if recorrencia_str in ["diaria", "mensal", "semanal", "quinzenal"] else 0.0
    eh_bonus = 1.0 if valor_servico == 0.0 else 0.0
    eh_evasao = 1.0 if "evas" in str(appt.get('nome_procedimento', '')).lower() else 0.0

    total_passado = float(history.get('total_agendamentos_historico', 0))
    count_cancel_passado = float(history.get('total_cancelamentos_historico', 0))
    cancel_30d = float(history.get('cancelamentos_ultimos_30d', 0))
    streak = float(history.get('sequencia_cancelamentos_atual', 0))
    noshow_rate = (count_cancel_passado / total_passado) if total_passado > 0 else 0.0

    features_dict = {
        "valor_servico": valor_servico,
        "tem_pacote": tem_pacote,
        "eh_bonus": eh_bonus,
        "eh_evasao": eh_evasao,
        "patient_total_appointments_past": total_passado,
        "patient_noshow_rate_past": noshow_rate,
        "patient_noshow_last_30d": cancel_30d,
        "patient_noshow_streak": streak,
        f"dia_semana_{dia_semana_mapeado}": 1.0,
        f"faixa_horaria_{faixa_horaria}": 1.0,
        f"nome_convenio_{str(appt.get('nome_convenio', '')).strip()}": 1.0
    }

    # Vector positioning based on 21 exact columns expected by ONNX
    vector = np.zeros((1, len(colunas_modelo)), dtype=np.float32)
    for idx, col in enumerate(colunas_modelo):
        vector[0, idx] = features_dict.get(col, 0.0)

    scaled_vector = scaler.transform(vector)
    return scaled_vector.astype(np.float32)

def _executar_inferencia_onnx(session: ort.InferenceSession, x_input: np.ndarray) -> float:
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: x_input})
    return float(outputs[0][0][0])

# =============================================================================
# 6. ENDPOINTS DE PRODUÇÃO
# =============================================================================
@app.post("/predict/{agendamento_id}", response_model=PredictionResponse, status_code=status.HTTP_200_OK)
async def predict_no_show(
    request: Request,
    agendamento_id: str = Path(..., description="ID no formato UUID do agendamento")
):
    start_time = time.time()
    supabase: Optional[AsyncClient] = request.app.state.supabase

    if not supabase or not request.app.state.onnx_session:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Serviço de Inteligência Artificial ou Supabase indisponível."
        )

    try:
        # Query 1: Agendamento
        appt_resp = await supabase.table("agendamentos").select("*").eq("id", agendamento_id).single().execute()
        appt = appt_resp.data

        if not appt:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Agendamento '{agendamento_id}' não localizado."
            )

        paciente_id = appt.get('paciente_id')
        if not paciente_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Registro de agendamento não contém 'paciente_id'."
            )

        # Query 2: Histórico
        hist_resp = await supabase.table("v_paciente_features").select("*").eq("paciente_id", paciente_id).single().execute()
        history = hist_resp.data or {}

        # Inference
        x_input = extrair_e_alinhar_features_otimizado(
            appt, history, request.app.state.colunas_modelo, request.app.state.scaler
        )
        
        probabilidade = await anyio.to_thread.run_sync(
            _executar_inferencia_onnx, request.app.state.onnx_session, x_input
        )

        threshold = request.app.state.threshold
        if probabilidade >= threshold:
            pred_status = "RISCO DE CANCELAMENTO"
            nivel_risco = "ALTO" if probabilidade >= 0.70 else "MÉDIO" if probabilidade >= 0.50 else "BAIXO"
        else:
            pred_status = "COMPARECE"
            nivel_risco = "BAIXO"

        elapsed_ms = round((time.time() - start_time) * 1000, 2)

        # Log assíncrono de auditoria no Supabase
        try:
            await supabase.table("ai_predicoes").upsert({
                "agendamento_id": agendamento_id,
                "predicao_status": pred_status,
                "probabilidade_risco": round(probabilidade, 4),
                "modelo_versao": "onnx_v1"
            }).execute()
        except Exception as log_err:
            logger.warning(f"⚠️ Erro ao persistir log de auditoria: {log_err}")

        return PredictionResponse(
            agendamento_id=agendamento_id,
            status=pred_status,
            probabilidade=round(probabilidade, 4),
            nivel_risco=nivel_risco,
            modelo_versao="onnx_v1",
            tempo_execucao_ms=elapsed_ms
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Erro fatal no pipeline ({agendamento_id}): {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Erro interno durante o processamento da predição."
        )

@app.get("/health", response_model=HealthResponse)
async def health_check(request: Request):
    onnx_ok = hasattr(request.app.state, "onnx_session") and request.app.state.onnx_session is not None
    total_colunas = len(request.app.state.colunas_modelo) if hasattr(request.app.state, "colunas_modelo") else 0
    status_str = "healthy" if (onnx_ok and total_colunas > 0) else "unhealthy"

    return HealthResponse(
        status=status_str,
        onnx_loaded=onnx_ok,
        threshold=getattr(request.app.state, "threshold", 0.0),
        total_colunas=total_colunas
    )