FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
# Usando porta 8080 fixa para teste local ou permitindo que o Docker a veja
ENV PORT=8080
CMD uvicorn app.main:app --host 0.0.0.0 --port 8080