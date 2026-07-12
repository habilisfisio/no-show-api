import pandas as pd
import unicodedata
import re
# Adicione logs para saber o que está acontecendo
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def padronizar_acentos(texto):
    """
    Converte texto com acentos para uma forma normalizada sem acentos,
    substituindo caracteres como 'á' por 'a', 'ç' por 'c', etc.
    """
    if pd.isna(texto) or not isinstance(texto, str):
        return texto
    # Normaliza para forma NFKD e remove diacríticos
    texto_norm = unicodedata.normalize('NFKD', texto)
    texto_sem_acento = ''.join(c for c in texto_norm if not unicodedata.combining(c))
    # Substitui caracteres especiais que podem não ter sido removidos
    texto_sem_acento = re.sub(r'[^\w\s.,;:!?()\-]', '', texto_sem_acento)
    return texto_sem_acento

def processar_agendamentos(caminho_entrada, caminho_saida, encoding_entrada='latin1'):
    """
    Lê o arquivo CSV de agendamentos, limpa, padroniza e salva para uso em modelo logístico.
    """
    # 1. Leitura do CSV com a codificação correta (geralmente latin1 ou cp1252)
    df = pd.read_csv(caminho_entrada, encoding=encoding_entrada, low_memory=False)

    # 2. Remover linhas completamente vazias
    df.dropna(how='all', inplace=True)

    # 3. Remover colunas que não são úteis para o modelo (metadados, chaves internas, etc.)
    colunas_remover = [
        'id', 'created_at', 'updated_at', 'token_confirmacao',
        'agendamento_pai_id', 'recorrencia_ate', 'plano_paciente_id',
        'data_pagamento'  # geralmente com muitos nulos
    ]
    # Manter apenas colunas que existem no DataFrame
    colunas_para_remover = [col for col in colunas_remover if col in df.columns]
    df.drop(columns=colunas_para_remover, inplace=True, errors='ignore')

    # 4. Padronizar acentos nas colunas de texto (strings)
    # Identificar colunas de tipo object (texto)
    colunas_texto = df.select_dtypes(include=['object']).columns
    for col in colunas_texto:
        df[col] = df[col].apply(padronizar_acentos)

    # 5. Tratar valores nulos de forma simples (substituir por string vazia ou NaN)
    # Para variáveis categóricas, substituir NaN por 'desconhecido' ou similar
    # Para variáveis numéricas, substituir por 0 ou média, mas aqui vamos apenas manter como NaN
    # O modelo logístico pode lidar com NaNs se tratados adequadamente.
    # Vamos preencher campos de texto vazios com '' e numéricos com 0? Melhor deixar como NaN para decisão posterior.
    # Para simplicidade, preenchemos com valores padrão:
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].fillna('')
    for col in df.select_dtypes(include=['int64', 'float64']).columns:
        df[col] = df[col].fillna(0)

    # 6. Salvar o arquivo limpo em UTF-8
    df.to_csv(caminho_saida, index=False, encoding='utf-8-sig')
    print(f"Arquivo processado e salvo em: {caminho_saida}")
    print(f"Linhas: {len(df)}, Colunas: {len(df.columns)}")

if __name__ == "__main__":
    # Exemplo de uso
    entrada = "/content/drive/MyDrive/freela habilis/agendamentos_rows (4).csv"
    saida = "/content/drive/MyDrive/freela habilis/agendamentos_limpo.csv"
    processar_agendamentos(entrada, saida, encoding_entrada='latin1')