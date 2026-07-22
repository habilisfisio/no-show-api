# script_conversao.py
import os
import json
import logging
import tensorflow as tf
import tf2onnx

logging.basicConfig(level=logging.INFO, format="%(asctime)s - [%(levelname)s] - %(message)s")
logger = logging.getLogger("converter_onnx")

def construir_modelo_tf_keras(n_features: int) -> tf.keras.Model:
    """
    Constrói a arquitetura exatamente idêntica ao treinamento usando tf.keras nativo.
    """
    inputs = tf.keras.Input(shape=(n_features,), name="input_dense", dtype=tf.float32)
    x = tf.keras.layers.Dense(16, activation="relu", name="dense_1")(inputs)
    x = tf.keras.layers.Dropout(0.3, name="dropout_1")(x)
    x = tf.keras.layers.Dense(8, activation="relu", name="dense_2")(x)
    x = tf.keras.layers.Dropout(0.2, name="dropout_2")(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid", name="output_dense")(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="no_show_mlp")
    return model

def converter_keras_para_onnx():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(base_dir, "models")
    
    keras_path = os.path.join(models_dir, "modelo_rede.keras")
    onnx_path = os.path.join(models_dir, "modelo_rede.onnx")
    json_path = os.path.join(models_dir, "colunas_modelo.json")

    if not os.path.exists(keras_path):
        raise FileNotFoundError(f"❌ Modelo Keras não encontrado em: {keras_path}")
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"❌ Arquivo de colunas não encontrado em: {json_path}")

    with open(json_path, "r", encoding="utf-8") as f:
        colunas = json.load(f)
    
    n_features = len(colunas)
    logger.info(f"📦 Construindo modelo tf.keras nativo com {n_features} features...")
    
    model = construir_modelo_tf_keras(n_features)
    
    # Carregamento dos pesos treinados
    model.load_weights(keras_path)
    logger.info("✅ Pesos carregados no grafo tf.keras com sucesso!")

    # Função decorada com @tf.function (GenericFunction)
    @tf.function
    def model_func(x):
        return model(x, training=False)

    input_signature = [tf.TensorSpec((None, n_features), tf.float32, name="input_dense")]

    logger.info("🔄 Convertendo tf.function para ONNX...")
    
    # PASSAGEM CORRETA CONFORME A DOCUMENTAÇÃO OFICIAL DO TF2ONNX:
    # O primeiro argumento DEVE ser a função decorada com @tf.function, e NÃO a ConcreteFunction.
    model_proto, _ = tf2onnx.convert.from_function(
        function=model_func,
        input_signature=input_signature,
        output_path=onnx_path
    )
    
    logger.info(f"✅ Conversão concluída com sucesso! Arquivo ONNX salvo em: {onnx_path}")

if __name__ == "__main__":
    converter_keras_para_onnx()