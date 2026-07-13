FROM python:3.11-slim

# Variáveis de ambiente para melhor comportamento em containers
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Instala dependências
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir uvicorn

# Copia todo o código fonte para dentro do container
COPY . .

# Comando padrão usando o módulo python para evitar erros de PATH
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]