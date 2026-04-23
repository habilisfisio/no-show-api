from pydantic import BaseModel
import os
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from supabase import create_client, Client
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # For production, replace "*" with your Vercel URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Credentials from Railway Environment Variables
url = os.environ.get("SUPABASE_URL")
key = os.environ.get("SUPABASE_KEY")
supabase = create_client(url, key)

# Load the pipeline you created in Colab
model = joblib.load("models/no_show_pipeline.pkl")

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
    appt_date = pd.to_datetime(appt['data_agendamento'])

    # Prevenção de erro: split seguro da hora
    hora_str = appt.get('hora_inicio', '00:00:00')
    hora_int = int(hora_str.split(':')[0]) if hora_str else 0

    input_data = {
        "preco_numerico": float(appt.get('valor_procedimento') or 0),
        "hora": hora_int,
        "dia_da_semana": appt_date.weekday(),
        "mes": appt_date.month,
        "eh_fim_de_semana": 1 if appt_date.weekday() >= 5 else 0,
        "total_agendamentos_anterior": history['total_agendamentos_historico'],
        "taxa_risco_paciente": float(history['taxa_risco_paciente']),
        "agreement_name": appt.get('nome_convenio') or "Particular",
        "procedimento_limpo": str(appt.get('nome_procedimento', '')).split('(')[0].strip(),
        "user_name": appt.get('nome_profissional', 'Nao Informado'),
        "eh_primeira_consulta": 1 if history['total_agendamentos_historico'] == 0 else 0
    }

    # 4. Inference
    df_input = pd.DataFrame([input_data])
    print(f"DEBUG: Dados preparados para o modelo: {input_data}")

    prediction = int(model.predict(df_input)[0])
    print(f"DEBUG: Predição realizada com sucesso: {prediction}")
    probability = float(model.predict_proba(df_input)[0][1])
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

    # 5. PERSISTENCE
    try:
        prediction_entry = {
            "agendamento_id": agendamento_id,
            "predicao_status": pred_status,
            "probabilidade_risco": round(probability, 2),
            "modelo_versao": "tcc_g_v1"
        }
        supabase.table("ai_predicoes").insert(prediction_entry).execute()

        supabase.table("ai_logs").insert({
        "agendamento_id": agendamento_id,
        "predicao": pred_status,
        "probabilidade": round(probability, 2)
    }).execute()
    except Exception as e:
        print(f"Logging Error: {e}")

    return {
        "agendamento_id": agendamento_id,
        "status": pred_status,
        "probabilidade": round(probability, 2),
        "nivel_risco": risk_level
    }