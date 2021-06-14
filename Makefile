CONTAINER_IMAGE=$(shell bash scripts/container_image.sh)
PYTHON ?= "python3"
PYTEST_OPTS ?= "-s -vvv"
PYTEST_DIR ?= "tests"
SCRIPT_DIR ?= "scripts"
PREF_SHELL ?= "bash"

GITREF=$(shell git rev-parse --short HEAD)

export DATA_DIR := /Users/gzheng/projects/niaaa/assist/data

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

all: network-analysis-image module-extraction-image membership-analysis-image \
diagnostic-correlation-image network-embedding-image ml-and-critical-gene-identifier-image

network-analysis-image:
	python record_version_info.py > version.txt; \
	cp version.txt $(NETWORK_ANALYSIS_DIR); \
	mkdir $(NETWORK_ANALYSIS_DIR)/src; \
	cp -r src/eda $(NETWORK_ANALYSIS_DIR)/src; \
	cp -r src/preproc $(NETWORK_ANALYSIS_DIR)/src; \
	cd $(NETWORK_ANALYSIS_DIR); \
	find . -name '*.pyc' -delete; \
	docker build --no-cache -t ${NETWORK_ANALYSIS_TAG} .; \
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
	docker build --no-cache -t ${MODULE_EXTRACTION_TAG} .; \
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
	docker build --no-cache -t ${MEMBERSHIP_ANALYSIS_TAG} .; \
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
	docker build --no-cache -t ${DIAGNOSTIC_CORRELATION_TAG} .; \
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
	docker build --no-cache -t ${NETWORK_EMBEDDING_TAG} .; \
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
	docker build --no-cache -t ${ML_AND_CRITICAL_GENE_IDENTIFIER_TAG} .; \
	rm version.txt; \
	rm -r src

save-standalone-images:
	for module in network_analysis module_extraction module_membership_analysis \
		module_de_diagnostic_correlation network_embedding ml_and_critical_gene_identifier; do \
		mkdir images/standalone/$$module; \
		echo assist/$$module; \
		docker save -o images/standalone/$$module/0.1.0.tar assist/$$module; \
	done

load-standalone-images:
	for module in network_analysis module_extraction module_membership_analysis \
		module_de_diagnostic_correlation network_embedding ml_and_critical_gene_identifier; do \
		echo assist/$$module; \
		docker load --input images/standalone/$$module/0.1.0.tar; \
	done