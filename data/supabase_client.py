import os
import pandas as pd
import numpy as np
import logging
from supabase import create_client
from dotenv import load_dotenv

# Importação para garantir que o Acento seja removido
def padronizar_acentos(texto):
    if not isinstance(texto, str): return texto
    import unicodedata
    return ''.join(c for c in unicodedata.normalize('NFD', texto) if unicodedata.category(c) != 'Mn')

logger = logging.getLogger(__name__)
load_dotenv()

supabase = create_client(os.environ.get("SUPABASE_URL"), os.environ.get("SUPABASE_KEY"))

def buscar_e_limpar_dados(force_refresh=False):
    caminho_parquet = 'data/agendamentos_processados.parquet'
    
    # 1. TENTA LER LOCALMENTE
    if os.path.exists(caminho_parquet) and not force_refresh:
        logger.info(f"📂 Lendo dados localmente de {caminho_parquet}...")
        return pd.read_parquet(caminho_parquet)

    # 2. SE NÃO EXISTIR, BUSCA NO SUPABASE (PAGINADO)
    logger.info("📡 Arquivo não encontrado ou atualização forçada. Buscando dados do Supabase...")
    
    colunas_seguras = [
        "paciente_id", 
        "data_agendamento", 
        "hora_inicio", 
        "status", 
        "nome_procedimento", 
        "nome_convenio"
    ]
    
    all_data = []
    batch_size = 1000 
    start = 0
    
    while True:
        response = supabase.table("agendamentos")\
            .select(",".join(colunas_seguras))\
            .in_("status", ["atendido", "desmarcado", "falta"])\
            .range(start, start + batch_size - 1)\
            .execute()
        
        if not response.data:
            break
            
        all_data.extend(response.data)
        logger.info(f"📥 Baixando... {len(all_data)} linhas acumuladas.")
        start += batch_size
        
        if len(response.data) < batch_size:
            break
            
    df = pd.DataFrame(all_data)

    # 3. LIMPEZA E PADRONIZAÇÃO
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].apply(padronizar_acentos).fillna('')
    
    for col in df.select_dtypes(include=['int64', 'float64']).columns:
        df[col] = df[col].fillna(0)
        
    # 4. SALVA LOCALMENTE
    if not os.path.exists('data'):
        os.makedirs('data')
    df.to_parquet(caminho_parquet, index=False)
    logger.info(f"💾 Dados salvos em {caminho_parquet}")
    
    logger.info(f"✅ Dados carregados com sucesso. Total Final: {len(df)} linhas.")
    return df

# --- Fluxo de Execução ---
if __name__ == "__main__":
    df = buscar_e_limpar_dados()
    
    # A partir daqui, você pode prosseguir com o seu script de modelagem
    # usando este DataFrame 'df' que já está limpo e padronizado.
    logger.info("Pipeline de ingestão concluído com sucesso.")