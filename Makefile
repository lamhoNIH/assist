CONTAINER_IMAGE=$(shell bash scripts/container_image.sh)
PYTHON ?= "python3"
PYTEST_OPTS ?= "-s -vvv"
PYTEST_DIR ?= "tests"
SCRIPT_DIR ?= "scripts"
PREF_SHELL ?= "bash"

GITREF=$(shell git rev-parse --short HEAD)

export NETWORK_ANALYSIS_DIR := analyses/network_analysis
export NETWORK_ANALYSIS_TAG := assist/network_analysis:0.1.0
export MODULE_EXTRACTION_DIR := analyses/module_extraction
export MODULE_EXTRACTION_TAG := assist/module_extraction:0.1.0
export MEMBERSHIP_ANALYSIS_DIR := analyses/module_membership_analysis
export MEMBERSHIP_ANALYSIS_TAG := assist/module_membership_analysis:0.1.0
export DIAGNOSTIC_CORRELATION_DIR := analyses/module_de_diagnostic_correlation
export DIAGNOSTIC_CORRELATION_TAG := assist/module_de_diagnostic_correlation:0.1.0
export NETWORK_EMBEDDING_DIR := analyses/network_embedding
export NETWORK_EMBEDDING_TAG := assist/network_embedding:0.1.0
export MODULE_SUBSELECTION_EDA_DIR := analyses/module_subselection_eda
export MODULE_SUBSELECTION_EDA_TAG := assist/module_subselection_eda:0.1.0
export MODULE_SUBSELECTION_EMBEDDING_DIR := analyses/module_subselection_embedding
export MODULE_SUBSELECTION_EMBEDDING_TAG := assist/module_subselection_embedding:0.1.0
export ML_AND_CRITICAL_GENE_IDENTIFIER_DIR := analyses/ml_and_critical_gene_identifier
export ML_AND_CRITICAL_GENE_IDENTIFIER_TAG := assist/ml_and_critical_gene_identifier:0.1.0

all: network-analysis-image module-extraction-image

network-analysis-image:
	python record_version_info.py > version.txt; \
	cp version.txt $(NETWORK_ANALYSIS_DIR); \
	mkdir $(NETWORK_ANALYSIS_DIR)/src; \
	cp -r src/eda $(NETWORK_ANALYSIS_DIR)/src; \
	cp -r src/preproc $(NETWORK_ANALYSIS_DIR)/src; \
	cd $(NETWORK_ANALYSIS_DIR); \
	find . -name '*.pyc' -delete; \
	docker build -t ${NETWORK_ANALYSIS_TAG} .; \
	rm version.txt; \
	rm -r src
	
module-extraction-image:
	python record_version_info.py > version.txt; \
	cp version.txt $(MODULE_EXTRACTION_DIR); \
	mkdir $(MODULE_EXTRACTION_DIR)/src; \
	cp -r src/eda $(MODULE_EXTRACTION_DIR)/src; \
	cp -r src/preproc $(MODULE_EXTRACTION_DIR)/src; \
	cd $(MODULE_EXTRACTION_DIR); \
	find . -name '*.pyc' -delete; \
	docker build -t ${MODULE_EXTRACTION_TAG} .; \
	rm version.txt; \
	rm -r src
	
membership-analysis-image:
	python record_version_info.py > version.txt; \
	cp version.txt $(MEMBERSHIP_ANALYSIS_DIR); \
	mkdir $(MEMBERSHIP_ANALYSIS_DIR)/src; \
	cp -r src/eda $(MEMBERSHIP_ANALYSIS_DIR)/src; \
	cp -r src/preproc $(MEMBERSHIP_ANALYSIS_DIR)/src; \
	cd $(MEMBERSHIP_ANALYSIS_DIR); \
	find . -name '*.pyc' -delete; \
	docker build -t ${MEMBERSHIP_ANALYSIS_TAG} .; \
	rm version.txt; \
	rm -r src
	
diagnostic-correlation-image:
	python record_version_info.py > version.txt; \
	cp version.txt $(DIAGNOSTIC_CORRELATION_DIR); \
	mkdir $(DIAGNOSTIC_CORRELATION_DIR)/src; \
	cp -r src/eda $(DIAGNOSTIC_CORRELATION_DIR)/src; \
	cp -r src/preproc $(DIAGNOSTIC_CORRELATION_DIR)/src; \
	cd $(DIAGNOSTIC_CORRELATION_DIR); \
	find . -name '*.pyc' -delete; \
	docker build -t ${DIAGNOSTIC_CORRELATION_TAG} .; \
	rm version.txt; \
	rm -r src

network-embedding-image:
	python record_version_info.py > version.txt; \
	cp version.txt $(NETWORK_EMBEDDING_DIR); \
	mkdir $(NETWORK_EMBEDDING_DIR)/src; \
	cp -r src/eda $(NETWORK_EMBEDDING_DIR)/src; \
	cp -r src/preproc $(NETWORK_EMBEDDING_DIR)/src; \
	cp -r src/embedding $(NETWORK_EMBEDDING_DIR)/src; \
	cd $(NETWORK_EMBEDDING_DIR); \
	find . -name '*.pyc' -delete; \
	docker build -t ${NETWORK_EMBEDDING_TAG} .; \
	rm version.txt; \
	rm -r src

module-subselection-eda-image:
	python record_version_info.py > version.txt; \
	cp version.txt $(MODULE_SUBSELECTION_EDA_DIR); \
	mkdir $(MODULE_SUBSELECTION_EDA_DIR)/src; \
	cp -r src/eda $(MODULE_SUBSELECTION_EDA_DIR)/src; \
	cp -r src/preproc $(MODULE_SUBSELECTION_EDA_DIR)/src; \
	cd $(MODULE_SUBSELECTION_EDA_DIR); \
	find . -name '*.pyc' -delete; \
	docker build -t ${MODULE_SUBSELECTION_EDA_TAG} .; \
	rm version.txt; \
	rm -r src

module-subselection-embedding-image:
	python record_version_info.py > version.txt; \
	cp version.txt $(MODULE_SUBSELECTION_EMBEDDING_DIR); \
	mkdir $(MODULE_SUBSELECTION_EMBEDDING_DIR)/src; \
	cp -r src/eda $(MODULE_SUBSELECTION_EMBEDDING_DIR)/src; \
	cp -r src/embedding $(MODULE_SUBSELECTION_EMBEDDING_DIR)/src; \
	cp -r src/preproc $(MODULE_SUBSELECTION_EMBEDDING_DIR)/src; \
	cd $(MODULE_SUBSELECTION_EMBEDDING_DIR); \
	find . -name '*.pyc' -delete; \
	docker build -t ${MODULE_SUBSELECTION_EMBEDDING_TAG} .; \
	rm version.txt; \
	rm -r src
	
ml-and-critical-gene-identifier-image:
	python record_version_info.py > version.txt; \
	cp version.txt $(ML_AND_CRITICAL_GENE_IDENTIFIER_DIR); \
	mkdir $(ML_AND_CRITICAL_GENE_IDENTIFIER_DIR)/src; \
	cp -r src/eda $(ML_AND_CRITICAL_GENE_IDENTIFIER_DIR)/src; \
	cp -r src/models $(ML_AND_CRITICAL_GENE_IDENTIFIER_DIR)/src; \
	cp -r src/preproc $(ML_AND_CRITICAL_GENE_IDENTIFIER_DIR)/src; \
	cd $(ML_AND_CRITICAL_GENE_IDENTIFIER_DIR); \
	find . -name '*.pyc' -delete; \
	docker build -t ${ML_AND_CRITICAL_GENE_IDENTIFIER_TAG} .; \
	rm version.txt; \
	rm -r src

tests-pytest:
#	bash $(SCRIPT_DIR)/run_container_process.sh $(PYTHON) -m "pytest" $(PYTEST_DIR) $(PYTEST_OPTS)
	echo "not implemented"

tests-deployed:
	echo "not implemented"

clean: clean-reactor-image clean-tests clean-app-image

clean-reactor-image:
	docker rmi -f $(CONTAINER_IMAGE)

clean-wasserstein-image:
	bash scripts/remove_images.sh $(PDT_WASSERSTEIN_INIFILE)

clean-tests:
	rm -rf .hypothesis .pytest_cache __pycache__ */__pycache__ tmp.* *junit.xml
