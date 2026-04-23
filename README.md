# No-Show Prediction API 🏥

An AI service designed to predict the probability of patient no-shows for medical appointments. This service integrates with Supabase to fetch historical data and persist predictions for real-time monitoring.

## 🚀 Fundamentals: How it Works
1. **Trigger**: An `agendamento_id` is sent to the `/predict` endpoint.
2. **Context Enrichment**: The API fetches specific features (price, time, day of week) and historical patient behavior from a SQL View (`v_paciente_features`).
3. **Inference**: A Scikit-Learn Pipeline processes the data and predicts if the patient will attend or miss.
4. **Persistence**: The result is saved back to `ai_predicoes` (for application use) and `ai_logs` (for future auditing).

---

## 🛠 Setup & Environment

### Prerequisites
- Docker & Docker Compose
- Supabase Project (PostgreSQL)

### Environment Variables
Create a `.env` file in the root directory:
```env
SUPABASE_URL=your_supabase_url
SUPABASE_KEY=your_supabase_anon_or_service_role_key

Installation
Bash

# Build and start the container
docker compose up --build

🏗 Data Architecture
SQL Tables & Views

The API depends on the following database structures:

    v_paciente_features: A view providing the patient's historical no-show rate.

    ai_predicoes: Stores the latest prediction for the frontend.

    ai_logs: Stores historical prediction data for model retraining.

Model Features

The model uses 11 specific features including:

    preco_numerico, hora, dia_da_semana, mes, eh_fim_de_semana

    total_agendamentos_anterior, taxa_risco_paciente, eh_primeira_consulta

    agreement_name, procedimento_limpo, user_name

🔌 API Documentation
POST /predict/{agendamento_id}

Triggers a prediction for a specific appointment.

Response Example:
JSON

{
  "agendamento_id": "uuid-v4-here",
  "status": "RISCO DE FALTA",
  "probabilidade": 0.85
}

⚠️ Important Notes (ML Specialist Advice)

    Library Versions: This project requires scikit-learn==1.6.1 to maintain compatibility with the no_show_pipeline.pkl exported from Colab.

    Data Drift: Predictions should be monitored via the ai_logs table to detect when model performance begins to decrease over time.