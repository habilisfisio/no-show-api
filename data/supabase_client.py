import os
import pandas as pd
import numpy as np
import logging
import unicodedata
import re
from supabase import create_client
from dotenv import load_dotenv

# Configuração de Log
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Credenciais
load_dotenv()
supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))

def padronizar_acentos(texto):
    if pd.isna(texto) or not isinstance(texto, str): return texto
    texto_norm = unicodedata.normalize('NFKD', texto)
    return re.sub(r'[^\w\s.,;:!?()\-]', '', ''.join(c for c in texto_norm if not unicodedata.combining(c)))

def buscar_e_limpar_dados():
    logger.info("📡 Buscando apenas colunas seguras do Supabase...")
    
    # Lista estrita de colunas necessárias para o seu modelo
    # Removemos 'appointment_package_name' pois não está no schema fornecido
    colunas_seguras = [
        "paciente_id", 
        "data_agendamento", 
        "hora_inicio", 
        "status", 
        "nome_procedimento", 
        "nome_convenio"
    ]
    
    # Executa a busca apenas com as colunas necessárias
    response = supabase.table("agendamentos").select(",".join(colunas_seguras)).execute()
    df = pd.DataFrame(response.data)
    
    # Se a lista de colunas for idêntica ao banco, NÃO há necessidade de renomear
    
    # Filtragem de registros (mantém apenas o que serve para o treino)
    if 'status' in df.columns:
        df = df[df['status'].isin(['atendido', 'desmarcado', 'falta'])].copy()

    # Tratamento de acentos e preenchimento de nulos
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].apply(padronizar_acentos).fillna('')
    
    for col in df.select_dtypes(include=['int64', 'float64']).columns:
        df[col] = df[col].fillna(0)
        
    logger.info(f"✅ Dados seguros carregados. Total: {len(df)} linhas.")
    return df

# --- Fluxo de Execução ---
if __name__ == "__main__":
    df = buscar_e_limpar_dados()
    
    # A partir daqui, você pode prosseguir com o seu script de modelagem
    # usando este DataFrame 'df' que já está limpo e padronizado.
    logger.info("Pipeline de ingestão concluído com sucesso.")