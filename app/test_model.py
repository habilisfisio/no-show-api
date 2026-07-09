# test_local.py – Testa o modelo com dados mockados
import joblib
import pandas as pd
import numpy as np
from datetime import datetime

# Carregar modelo
modelo = joblib.load("models/no_show_model_enxuto.pkl")

# Parâmetros do treino (extraídos do Colab)
MEDIA_GLOBAL = 0.1206
ALPHA_SUAVIZACAO = 10
CORTE_NEGOCIO = 0.1079

# =============================================================================
# FUNÇÃO DE FEATURE ENGINEERING (mesma do endpoint)
# =============================================================================
def construir_features(appt: dict, history: dict) -> pd.DataFrame:
    appt_date = pd.to_datetime(appt['data_agendamento'])
    hora = pd.to_datetime(appt['start_time'], format='%H:%M:%S').hour

    total_passado = history.get('total_agendamentos_historico', 0)
    count_faltas_passado = history.get('total_faltas_historico', 0)
    faltas_ultimos_30d = history.get('faltas_ultimos_30d', 0)
    streak_faltas = history.get('sequencia_faltas_atual', 0)

    # Suavização bayesiana
    if total_passado > 0:
        taxa_raw = count_faltas_passado / total_passado
    else:
        taxa_raw = 0.0
    taxa_suavizada = (count_faltas_passado + MEDIA_GLOBAL * ALPHA_SUAVIZACAO) / (total_passado + ALPHA_SUAVIZACAO)

    # Inicializa todas as colunas com zero
    features = {col: 0.0 for col in modelo.model.exog_names}
    features['const'] = 1.0

    # Preenche features numéricas
    features['patient_total_appointments_past'] = float(total_passado)
    features['patient_noshow_rate_smooth'] = float(taxa_suavizada)
    features['patient_noshow_last_30d'] = float(faltas_ultimos_30d)
    features['patient_noshow_streak'] = float(streak_faltas)

    # Dia da semana
    mapa_dias = {0: 'Segunda', 1: 'Terça', 2: 'Quarta', 3: 'Quinta', 4: 'Sexta', 5: 'Sábado', 6: 'Domingo'}
    nome_dia = f"dia_semana_{mapa_dias[appt_date.weekday()]}"
    if nome_dia in features:
        features[nome_dia] = 1.0

    # Faixa horária
    if hora < 12:
        faixa = 'Manhã'
    elif hora < 18:
        faixa = 'Tarde'
    else:
        faixa = 'Noite'
    nome_faixa = f"faixa_horaria_{faixa}"
    if nome_faixa in features:
        features[nome_faixa] = 1.0

    # Convênio
    convenio = appt.get('nome_convenio', 'Particular')
    nome_conv = f"agreement_name_{convenio}"
    if nome_conv in features:
        features[nome_conv] = 1.0

    df = pd.DataFrame([features])
    df = df[modelo.model.exog_names]
    return df

# =============================================================================
# CENÁRIOS DE TESTE
# =============================================================================
print("=" * 60)
print("TESTE DO MODELO COM DADOS MOCKADOS")
print("=" * 60)

# Cenário 1: Paciente com histórico de faltas
appt1 = {
    'data_agendamento': '2025-06-01',
    'start_time': '10:00:00',
    'nome_convenio': 'Particular'
}
history1 = {
    'total_agendamentos_historico': 10,
    'total_faltas_historico': 4,
    'faltas_ultimos_30d': 2,
    'sequencia_faltas_atual': 1
}
df1 = construir_features(appt1, history1)
prob1 = float(modelo.predict(df1)[0])
print(f"Cenário 1 (faltou 4 de 10): Probabilidade = {prob1:.2%} | Risco = {'ALTO' if prob1 >= 0.30 else 'MÉDIO' if prob1 >= 0.1079 else 'BAIXO'}")

# Cenário 2: Paciente novo (sem histórico)
appt2 = {
    'data_agendamento': '2025-06-02',
    'start_time': '15:00:00',
    'nome_convenio': 'Unimed'
}
history2 = {}
df2 = construir_features(appt2, history2)
prob2 = float(modelo.predict(df2)[0])
print(f"Cenário 2 (novo paciente, convênio Unimed): Probabilidade = {prob2:.2%} | Risco = {'ALTO' if prob2 >= 0.30 else 'MÉDIO' if prob2 >= 0.1079 else 'BAIXO'}")

# Cenário 3: Paciente com streak de faltas
appt3 = {
    'data_agendamento': '2025-06-03',
    'start_time': '08:00:00',
    'nome_convenio': 'Particular'
}
history3 = {
    'total_agendamentos_historico': 5,
    'total_faltas_historico': 3,
    'faltas_ultimos_30d': 3,
    'sequencia_faltas_atual': 3
}
df3 = construir_features(appt3, history3)
prob3 = float(modelo.predict(df3)[0])
print(f"Cenário 3 (streak de 3 faltas): Probabilidade = {prob3:.2%} | Risco = {'ALTO' if prob3 >= 0.30 else 'MÉDIO' if prob3 >= 0.1079 else 'BAIXO'}")

print("\n" + "=" * 60)
print("Teste concluído. O modelo está funcionando!")