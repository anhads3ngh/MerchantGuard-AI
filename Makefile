.PHONY: install demo test api dashboard clean
install:
	python -m pip install -e '.[dev,dashboard]'
demo:
	python -m merchantguard demo --rows 12000
test:
	python -m pytest
api:
	python -m uvicorn merchantguard.api:app --reload --port 8000
dashboard:
	python -m streamlit run dashboard.py
clean:
	rm -rf .merchantguard .pytest_cache
