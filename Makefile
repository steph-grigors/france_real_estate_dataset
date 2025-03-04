.DEFAULT_GOAL := default

#################### PACKAGE ACTIONS ###################
reinstall_package:
	@pip uninstall -y real-estate || :
	@pip install -e .

run_work_env:
	echo "Current work environment: $(ENVIRONMENT)"

run_extract:
	python -c 'from real_estate_parquet.interface.main import data_extraction; data_extraction()'

run_clean:
	python -c 'from real_estate_parquet.interface.main import cleaning_in_chunks; cleaning_in_chunks()'

run_preproc:
	python -c 'from real_estate_parquet.interface.main import preprocessing_in_chunks; preprocessing_in_chunks()'

run_train:
	python -c 'from real_estate_parquet.interface.main import train_model; train_model()'

run_evaluate:
	python -c 'from real_estate_parquet.interface.main import evaluate_model; evaluate_model()'

run_predict:
	python -c 'from real_estate_parquet.interface.main import predict; predict()'

run_all: run_clean run_preproc run_train run_evaluate run_predict

run_create_docker_image:
	docker build -t real_estate_api .

run_build_docker_container:
	docker run -d -p 8080:8000 \
  -v "$(pwd)/.env:/prod/.env" \
  -v "$(pwd)/data:/prod/data" \
  -v "$(pwd)/artifacts:/prod/artifacts" \
  real_estate_api

run_api:
	uvicorn real_estate_parquet.api.api:app --reload

################### DATA SOURCES ACTIONS ################

# Environment-based Paths
LOCAL_DATA_PATH = data
LOCAL_REGISTRY_PATH = artifacts

# Create Project Structure
init-directories:
	mkdir -p $(LOCAL_DATA_PATH)/raw_dataset
	mkdir -p $(LOCAL_DATA_PATH)/raw_dataset/raw_dataset_chunks_csv
	mkdir -p $(LOCAL_DATA_PATH)/raw_dataset/raw_dataset_full
	mkdir -p $(LOCAL_DATA_PATH)/cleaned_dataset
	mkdir -p $(LOCAL_DATA_PATH)/merged_dataset
	mkdir -p $(LOCAL_DATA_PATH)/merged_dataset/test_set
	mkdir -p $(LOCAL_DATA_PATH)/merged_dataset/train_set

	mkdir -p $(LOCAL_REGISTRY_PATH)/models
	mkdir -p $(LOCAL_REGISTRY_PATH)/metrics
	mkdir -p $(LOCAL_REGISTRY_PATH)/params
	mkdir -p $(LOCAL_REGISTRY_PATH)/preprocessor

	mkdir -p $(LOCAL_PROJECT_PATH)/interface
	mkdir -p $(LOCAL_PROJECT_PATH)/ml_logic
	mkdir -p $(LOCAL_PROJECT_PATH)/api
	mkdir -p $(LOCAL_PROJECT_PATH)/notebooks


##################### CLEANING #####################

clean:
	@rm -f */version.txt
	@rm -f .coverage
	@rm -fr **/__pycache__ **/*.pyc
	@rm -fr **/build **/dist
	@rm -fr proj-*.dist-info
	@rm -fr proj.egg-info
	@rm -f **/.DS_Store
	@rm -f **/*Zone.Identifier
	@rm -f **/.ipynb_checkpoints
