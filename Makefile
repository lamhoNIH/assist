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
export MODULE_SUBSELECTION_DIR := analyses/module_subselection
export MODULE_SUBSELECTION_TAG := assist/module_subselection:0.1.0

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
	
module-subselection-image:
	python record_version_info.py > version.txt; \
	cp version.txt $(MODULE_SUBSELECTION_DIR); \
	mkdir $(MODULE_SUBSELECTION_DIR)/src; \
	cp -r src/eda $(MODULE_SUBSELECTION_DIR)/src; \
	cp -r src/preproc $(MODULE_SUBSELECTION_DIR)/src; \
	cd $(MODULE_SUBSELECTION_DIR); \
	find . -name '*.pyc' -delete; \
	docker build -t ${MODULE_SUBSELECTION_TAG} .; \
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