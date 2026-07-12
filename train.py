import pandas as pd
import numpy as np
import logging
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, ConfusionMatrixDisplay

# Configuração de Log
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 1. CARREGAMENTO E LIMPEZA
df = pd.read_csv("agendamentos_limpo.csv")
df['data_agendamento'] = pd.to_datetime(df['data_agendamento'])

# Remoção de Evasão (Critério Conceitual do TCC)
evasao_mask = df["procedure_name"].astype(str).str.contains("Evas", case=False, na=False)
df = df[~evasao_mask].copy()
df = df[df['status'].isin(['atendido', 'desmarcado', 'falta'])]

# 2. DEFINIÇÃO DO ALVO (NoShow = 1 apenas para 'desmarcado')
df["NoShow"] = df["status"].apply(lambda x: 1 if x == 'desmarcado' else 0)

# 3. FEATURE ENGINEERING
# Valor e Bônus
df["valor_servico"] = df["procedure_name"].astype(str).str.extract(r"(\d+,\d{2})")[0].str.replace(",", ".").astype(float).fillna(0)
df["eh_bonus"] = (df["valor_servico"] == 0).astype(int)

# Recorrência 6 meses (Pacotes)
df = df.sort_values(["paciente_id", "procedure_name", "data_agendamento"])
def count_recorrencia_6m(group):
    return group.set_index("data_agendamento").rolling("180D")["status"].count().shift(1).fillna(0)
df["eh_pacote_recorrente"] = df.groupby(["paciente_id", "procedure_name"]).apply(count_recorrencia_6m).values > 0
df.loc[df["eh_pacote_recorrente"], "valor_servico"] /= 10

# Convênios e Temporais
volume_conv = df["agreement_name"].value_counts()
df = df[df["agreement_name"].isin(volume_conv[volume_conv >= 30].index)]
df["dia_semana"] = df['data_agendamento'].dt.day_name()
df["faixa_horaria"] = pd.cut(pd.to_datetime(df['hora_inicio'], format='%H:%M:%S').dt.hour, 
                             bins=[-0.1, 4.9, 11.9, 17.9, 23.9], labels=["Madrugada", "Manhã", "Tarde", "Noite"])

# Features Históricas (Com shift para evitar vazamento de dados)
df = df.sort_values(["paciente_id", "data_agendamento"])
df["patient_total_appointments_past"] = df.groupby("paciente_id").cumcount()
df["patient_noshow_count_past"] = df.groupby("paciente_id")["NoShow"].transform(lambda s: s.shift(1).fillna(0).cumsum())
df["patient_noshow_rate_past"] = np.where(df["patient_total_appointments_past"] > 0, df["patient_noshow_count_past"] / df["patient_total_appointments_past"], 0)
df["patient_noshow_last_30d"] = df.groupby("paciente_id", group_keys=False).apply(lambda g: g.set_index("data_agendamento")["NoShow"].rolling("30D").sum().shift(1).fillna(0))
df["patient_noshow_streak"] = df.groupby("paciente_id", group_keys=False).apply(lambda g: g.sort_values("data_agendamento")["NoShow"].replace(0, np.nan).ffill().groupby(g["NoShow"].eq(0).cumsum()).cumcount().fillna(0))

# 4. PREPARAÇÃO E TREINO
features_num = ["valor_servico", "eh_bonus", "eh_pacote_recorrente", "patient_total_appointments_past", 
                "patient_noshow_rate_past", "patient_noshow_last_30d", "patient_noshow_streak"]
features_cat = ["dia_semana", "faixa_horaria", "agreement_name"]
X = pd.get_dummies(df[features_num + features_cat], columns=features_cat, drop_first=True).astype(float).fillna(0)
X = X.replace([np.inf, -np.inf], 0) # Verificação de qualidade

X_train, X_test, y_train, y_test = train_test_split(X, df["NoShow"], test_size=0.2, random_state=42, stratify=df["NoShow"])
ponto_corte = y_train.mean()

# Treino do Modelo
modelo_log = sm.Logit(y_train, sm.add_constant(X_train)).fit(method="bfgs", maxiter=1000, disp=False)

# 5. VISUALIZAÇÃO PACIENTE A PACIENTE
prob_log = modelo_log.predict(sm.add_constant(X_test))
y_pred_log = (prob_log >= ponto_corte).astype(int)

resultados = pd.DataFrame({
    "prob_desmarcar_%": (prob_log * 100).round(1),
    "decisao_modelo": np.where(y_pred_log == 1, "Desmarca", "Comparece"),
    "realidade": np.where(y_test == 1, "Desmarcou", "Compareceu"),
})
resultados["acertou"] = np.where(y_pred_log.values == y_test.values, "Sim", "Não")
resultados["tipo"] = [("TP" if p==1 and r==1 else "FP" if p==1 else "FN" if r==1 else "TN") 
                     for p, r in zip(y_pred_log.values, y_test.values)]
resultados = resultados.sort_values("prob_desmarcar_%", ascending=False)

# Estilização
def cor_tipo(val):
    cores = {"TP": "#d4f4dd", "FN": "#ffd6d3", "FP": "#ffe9c7"}
    return f"background-color: {cores.get(val, '')}"

styler = resultados.head(25).style
styler = styler.map(cor_tipo, subset=["tipo"]) if hasattr(styler, "map") else styler.applymap(cor_tipo, subset=["tipo"])
display(styler.set_caption("Top 25 pacientes por risco de desmarcação"))

# 6. MATRIZ DE CONFUSÃO VISUAL
cm = confusion_matrix(y_test, y_pred_log)
fig, ax = plt.subplots(figsize=(6, 5))
ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Compareceu (0)", "Desmarcou (1)"]).plot(ax=ax, cmap="Blues", colorbar=False)
ax.set_title("Matriz de Confusão — Regressão Logística")
plt.show()

print(f"\nAUC: {roc_auc_score(y_test, prob_log):.4f}")
print(classification_report(y_test, y_pred_log, digits=3))