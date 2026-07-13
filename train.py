import pandas as pd
import numpy as np
import logging
import json
import os
import joblib
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, ConfusionMatrixDisplay
from data.supabase_client import buscar_e_limpar_dados
from sklearn.utils.class_weight import compute_sample_weight

# Configuração de Log
if not os.path.exists('logs'): os.makedirs('logs')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/treino.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 1. CARREGAMENTO E LIMPEZA
df = buscar_e_limpar_dados()
df['data_agendamento'] = pd.to_datetime(df['data_agendamento'])

# Remoção de Evasão e limpeza inicial
evasao_mask = df["nome_procedimento"].astype(str).str.contains("Evas", case=False, na=False)
df = df[~evasao_mask].copy()
df = df[df['status'].isin(['atendido', 'desmarcado', 'falta'])]

# 2. DEFINIÇÃO DO ALVO (NoShow)
df["NoShow"] = df["status"].apply(lambda x: 1 if x == 'desmarcado' else 0)

# --- CORREÇÃO DE VAZAMENTO DE DADOS (Leakage) ---
# Removemos colunas que contêm o resultado final para evitar overfitting (AUC 1.0)
colunas_vazamento = ["status", "predicao_status", "predicao_modelo_versao", "predicao_gerada_em", "data_confirmacao"]
df = df.drop(columns=[c for c in colunas_vazamento if c in df.columns])

# 3. FEATURE ENGINEERING
df["valor_servico"] = df["nome_procedimento"].astype(str).str.extract(r"(\d+,\d{2})")[0].str.replace(",", ".").astype(float).fillna(0)
df["eh_bonus"] = (df["valor_servico"] == 0).astype(int)

# Recorrência
df = df.sort_values(["paciente_id", "nome_procedimento", "data_agendamento"])
def count_recorrencia_6m(group):
    return group.set_index("data_agendamento").rolling("180D")["NoShow"].count().shift(1).fillna(0)
df["eh_pacote_recorrente"] = df.groupby(["paciente_id", "nome_procedimento"]).apply(count_recorrencia_6m).values > 0
df.loc[df["eh_pacote_recorrente"], "valor_servico"] /= 10

# Convênios e Temporais
volume_conv = df["nome_convenio"].value_counts()
df = df[df["nome_convenio"].isin(volume_conv[volume_conv >= 30].index)]
df["dia_semana"] = df['data_agendamento'].dt.day_name()
df["faixa_horaria"] = pd.cut(pd.to_datetime(df['hora_inicio'], format='%H:%M:%S').dt.hour, 
                             bins=[-0.1, 4.9, 11.9, 17.9, 23.9], labels=["Madrugada", "Manhã", "Tarde", "Noite"])
# Mantenha as colunas originais e adicione a combinação
df['dia_horario'] = df['dia_semana'].astype(str) + '_' + df['faixa_horaria'].astype(str)

# Em 'features_cat', certifique-se de adicionar a nova coluna
features_cat = ["dia_semana", "faixa_horaria", "nome_convenio", "dia_horario"]

# Features Históricas
df = df.sort_values(["paciente_id", "data_agendamento"])
df["patient_total_appointments_past"] = df.groupby("paciente_id").cumcount()
df["patient_noshow_count_past"] = df.groupby("paciente_id")["NoShow"].transform(lambda s: s.shift(1).fillna(0).cumsum())
df["patient_noshow_rate_past"] = np.where(df["patient_total_appointments_past"] > 0, df["patient_noshow_count_past"] / df["patient_total_appointments_past"], 0)

def calcular_noshow_30d(g):
    return g.set_index("data_agendamento")["NoShow"].rolling("30D").sum().shift(1).fillna(0)

df["patient_noshow_last_30d"] = df.groupby("paciente_id", group_keys=False).apply(calcular_noshow_30d).values
# Cálculo do streak 100% à prova de vazamento
def calcular_streak_seguro(g):
    g = g.sort_values("data_agendamento")
    # Pega o histórico ANTERIOR à consulta atual
    historico = g["NoShow"].shift(1).fillna(0)
    # Calcula streaks baseados apenas no passado
    return historico.replace(0, np.nan).ffill().groupby(historico.eq(0).cumsum()).cumcount().fillna(0)

df["patient_noshow_streak"] = df.groupby("paciente_id", group_keys=False).apply(calcular_streak_seguro)

# 4. PREPARAÇÃO E TREINO
features_num = ["valor_servico", "eh_bonus", "eh_pacote_recorrente", "patient_total_appointments_past", "patient_noshow_rate_past", "patient_noshow_last_30d", "patient_noshow_streak"]
features_cat = ["dia_semana", "faixa_horaria", "nome_convenio"]
X = pd.get_dummies(df[features_num + features_cat], columns=features_cat, drop_first=True).astype(float).fillna(0)
X = X.replace([np.inf, -np.inf], 0)

# Coloque isso exatamente antes de X = pd.get_dummies(...)
logger.info(f"Colunas usadas no treino: {X.columns.tolist()}")

if 'status' in X.columns:
    logger.error("ALERTA CRÍTICO: A coluna 'status' ainda está no conjunto de features!")

X_train, X_test, y_train, y_test = train_test_split(X, df["NoShow"], test_size=0.2, random_state=42, stratify=df["NoShow"])
ponto_corte = y_train.mean()

# Force o modelo a dar 20x mais importância para o erro na classe de falta
pesos_classes = {0: 1, 1: 40} 

# Aplique no treino
modelo_log = sm.Logit(y_train, sm.add_constant(X_train)).fit_regularized(
    method='l1', 
    alpha=0.5,
    weights=y_train.map(pesos_classes)
)

# Analise quais variáveis são as "culpadas" pelo overfitting
coefs = pd.DataFrame({'feature': X_train.columns, 'coef': modelo_log.params.drop("const")})
logger.info(f"\nCoeficientes do Modelo:\n{coefs.sort_values(by='coef', ascending=False)}")

if not os.path.exists('models'): os.makedirs('models')
joblib.dump(modelo_log, "models/no_show_model.pkl")
logger.info("✅ Modelo exportado para models/no_show_model.pkl")

# Metadados
metadados = {
    "versao": "tcc_v3_logistica_regularizada",
    "algoritmo": "Regressão Logística (L1)",
    "treinado_em": pd.Timestamp.now().isoformat(),
    "threshold": float(ponto_corte)
}
with open("models/model_metadata.json", "w") as f:
    json.dump(metadados, f, indent=4)
logger.info("✅ Metadados exportados para models/model_metadata.json")

# 5. VISUALIZAÇÕES E LOGS DE PERFORMANCE
prob_log = modelo_log.predict(sm.add_constant(X_test))
y_pred_log = (prob_log >= ponto_corte).astype(int)

logger.info(f"AUC Score: {roc_auc_score(y_test, prob_log):.4f}")
logger.info(f"\nRelatório de Classificação:\n{classification_report(y_test, y_pred_log, digits=3)}")

cm = confusion_matrix(y_test, y_pred_log)
fig, ax = plt.subplots(figsize=(6, 5))
ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Compareceu (0)", "Desmarcou (1)"]).plot(ax=ax, cmap="Blues")
plt.savefig("logs/confusion_matrix.png")
logger.info("✅ Matriz de confusão salva em logs/confusion_matrix.png")