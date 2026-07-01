from pydantic import BaseModel
import os
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from supabase import create_client, Client
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()
origins = [
    "https://octahealth.com.br",
    "https://www.octahealth.com.br",
    "https://octa-health.lovable.app",
    "https://id-preview--4efb51f1-dd1d-4fe0-a7ea-f3c4174deafa.lovable.app",
    "http://healthcheck.railway.app"
]

# 2. Add the Middleware with the list
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],  # Allows GET, POST, OPTIONS, etc.
    allow_headers=["*"],  # Allows Content-Type, Authorization, etc.
)

# Credentials from Railway Environment Variables
url = os.environ.get("SUPABASE_URL")
key = os.environ.get("SUPABASE_KEY")
supabase = create_client(url, key)

# Load the pipeline you created in Colab
model = joblib.load("models/no_show_pipeline_v2.pkl")

@app.get("/health")
async def health_check():
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
        
    return {
        "status": "healthy", 
        "version": "tcc_g_v2",
        "environment": os.environ.get("RAILWAY_ENVIRONMENT", "local")
    }

@app.post("/predict/{agendamento_id}")
async def get_prediction(agendamento_id: str):
    print(f"DEBUG: Iniciando predição para {agendamento_id}")

    appt_query = supabase.table("agendamentos").select("*").eq("id", agendamento_id).single().execute()
    appt = appt_query.data
    
    if not appt:
        print("DEBUG: Agendamento não encontrado")
        raise HTTPException(status_code=404, detail="Appointment not found")

    print("DEBUG: Agendamento carregado. Buscando histórico...") # Checkpoint 2

    # 2. Fetch historical features from our SQL View
    history_query = supabase.table("v_paciente_features").select("*").eq("paciente_id", appt['paciente_id']).single().execute()
    history = history_query.data or {"total_agendamentos_historico": 0, "taxa_risco_paciente": 0.0}
    
    # 3. Feature Engineering (Seguindo a lógica do seu TCC_Ana.ipynb)
    # 3. Feature Engineering
    appt_date = pd.to_datetime(appt['data_agendamento'])
    # Cria o dicionário base (gabarito) com todas as colunas zeradas
    features_dict = {col: 0.0 for col in model.model.exog_names}
    features_dict['const'] = 1.0  # Obrigatório para Statsmodels

    # Preenche valores numéricos
    features_dict['valor_servico'] = float(appt.get('valor_procedimento') or 0)
    features_dict['patient_total_appointments_past'] = float(history['total_agendamentos_historico'])
    features_dict['patient_noshow_rate_past'] = float(history['taxa_risco_paciente'])

    # Preenche Dummies de Dia da Semana (Ajuste o nome se necessário)
    # Dica: se o modelo foi treinado em português, 'dia_semana_Monday' não vai funcionar.
    # Verifique o log anterior: se ele espera 'dia_semana_Segunda', use o mapeamento abaixo.
    mapa_dias = {0: 'Segunda', 1: 'Terça', 2: 'Quarta', 3: 'Quinta', 4: 'Sexta', 5: 'Sábado', 6: 'Domingo'}
    nome_dia = f"dia_semana_{mapa_dias[appt_date.weekday()]}"
    if nome_dia in features_dict:
        features_dict[nome_dia] = 1.0

    # Preenche Dummies de Convênio
    convenio = appt.get('nome_convenio')
    nome_conv = f"agreement_name_{convenio}"
    if nome_conv in features_dict:
        features_dict[nome_conv] = 1.0

    # Cria o DataFrame e garante a ordem das colunas
    df_input = pd.DataFrame([features_dict])
    df_input = df_input[model.model.exog_names]

    # 4. Inference
    prediction = int(model.predict(df_input)[0])

    # Statsmodels Logit não tem predict_proba direto como Scikit-Learn
    # Para Logit, a predição já é a probabilidade se usar model.predict()
    probability = float(prediction)
    print(f"DEBUG: Probabilidade de risco: {probability}")

    risk_level = "DESCONHECIDO"
    pred_status = "DESCONHECIDO"
    if probability < 0.20:
        risk_level = "BAIXO"
        pred_status = "COMPARECE"
    elif 0.20 <= probability < 0.50:
        risk_level = "MÉDIO"
        pred_status = "RISCO DE FALTA"
    else:
        risk_level = "ALTO"
        pred_status = "RISCO DE FALTA"

    # 5. PERSISTENCE (Safe Guard against multiple deployments)
    try:
        prediction_entry = {
            "agendamento_id": agendamento_id,
            "predicao_status": pred_status,
            "probabilidade_risco": round(probability, 2),
            "modelo_versao": "tcc_g_v2" # Ensure this is updated in your newest code
        }

        # UPSERT: Update if exists, Insert if not.
        # This prevents the 'no-show-api:v1' and 'tcc_g_v2' from co-existing
        # for the same agendamento_id.
        supabase.table("ai_predicoes").insert(prediction_entry).execute()

        # Logs can remain as inserts to see the 'double hit' happening in real-time
        supabase.table("ai_logs").insert({
            "agendamento_id": agendamento_id,
            "predicao": pred_status,
            "probabilidade": round(probability, 2),
            "modelo_versao": "tcc_g_v2"
        }).execute()

    except Exception as e:
        print(f"Persistence Error: {e}")

    return {
        "agendamento_id": agendamento_id,
        "status": pred_status,
        "probabilidade": round(probability, 2),
        "nivel_risco": risk_level
    }
