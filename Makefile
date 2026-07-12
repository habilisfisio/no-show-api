# Comandos simples no seu terminal
clean-data:
	python3 data/processar_agendamentos.py

train-model:
	python3 train.py

all: clean-data train-model