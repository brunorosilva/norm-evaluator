benchmark-all-models:
	python3 main.py

format:
	black .
	isort .