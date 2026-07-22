import os
import sys
import json
import pytest
import numpy as np
import joblib
import onnxruntime as ort

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))

if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from api import extrair_e_alinhar_features, ml_artifacts, _executar_inferencia_onnx

def test_pipeline_onnx_full_flow():
    models_dir = os.path.join(BASE_DIR, "models")
    if not os.path.exists(models_dir):
        models_dir = os.path.join(CURRENT_DIR, "models")

    # 1. Carrega artefatos
    ml_artifacts["scaler"] = joblib.load(os.path.join(models_dir, "scaler.pkl"))
    with open(os.path.join(models_dir, "colunas_modelo.json"), "r", encoding="utf-8") as f:
        ml_artifacts["colunas_modelo"] = json.load(f)

    onnx_path = os.path.join(models_dir, "modelo_rede.onnx")
    assert os.path.exists(onnx_path), f"Arquivo ONNX ausente em {onnx_path}"

    ml_artifacts["onnx_session"] = ort.InferenceSession(onnx_path)

    # 2. Mock de Agendamento
    mock_appt = {
        "data_agendamento": "2026-08-05",
        "hora_inicio": "14:30:00",
        "valor_procedimento": "150,00",
        "recorrencia": "semanal",
        "nome_procedimento": "Sessão Fisioterapia",
        "nome_convenio": "Contrato Unimed"
    }

    mock_history = {
        "total_agendamentos_historico": 10,
        "total_cancelamentos_historico": 2,
        "cancelamentos_ultimos_30d": 1,
        "sequencia_cancelamentos_atual": 0
    }

    # 3. Executa engenharia de recursos
    X_input = extrair_e_alinhar_features(mock_appt, mock_history)

    # 4. Asserções
    assert X_input.shape == (1, len(ml_artifacts["colunas_modelo"]))
    assert X_input.dtype == np.float32
    assert not np.isnan(X_input).any()

    # 5. Teste de Inferência no ONNX
    prob = _executar_inferencia_onnx(X_input)
    assert 0.0 <= prob <= 1.0