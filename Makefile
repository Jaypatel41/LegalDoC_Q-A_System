.PHONY: install index train eval demo all clean

install:
	pip install -r requirements.txt

index:
	python -m scripts.build_index

train:
	python -m scripts.train_intent
	python -m scripts.train_hallucination

eval:
	python -m scripts.run_eval

# offline smoke test (no keys / no heavy deps needed)
smoke:
	USE_STUB_LLM=1 python -m scripts.build_index
	USE_STUB_LLM=1 python -m scripts.run_eval

demo:
	streamlit run app/streamlit_app.py

all: install index train eval

clean:
	rm -rf data/index data/models __pycache__ src/**/__pycache__
