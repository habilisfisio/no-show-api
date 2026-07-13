# Busca dados do Supabase, processa e treina tudo de uma vez
tudo:
	python3 train.py

# Apenas limpa a base de dados se você tiver um script separado
limpar:
	python3 data/supabase_client.py

# Se quiser rodar apenas o treino
treino:
	python3 train.py

# Ajuda
ajuda:
	@echo "Comandos disponíveis:"
	@echo "  make tudo    - Busca, limpa e treina o modelo."
	@echo "  make treino  - Executa apenas o script de treino."