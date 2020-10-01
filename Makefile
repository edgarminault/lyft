##### Package params ---------------

PACKAGE_NAME=lyft
FILENAME=trainer

##### GCP Params -------------------

BUCKET_NAME=wagon-ml-edgarminault-gcp

REGION=europe-west4

PYTHON_VERSION=3.7
FRAMEWORK=scikit-learn
RUNTIME_VERSION=1.15

PACKAGE_NAME=lyft
FILENAME=trainer

JOB_NAME=lyft_training_linear_$(shell date +'%Y%m%d_%H%M%S')

# ----------------------------------
#          INSTALL & TEST
# ----------------------------------
install_requirements:
	@pip install -r requirements.txt

check_code:
	@flake8 scripts/* lyft/*.py

black:
	@black scripts/* lyft/*.py

run_locally:
	@python -m ${PACKAGE_NAME}.${FILENAME}

#test:
#	@coverage run -m pytest tests/*.py
#	@coverage report -m --omit=$(VIRTUAL_ENV)/lib/python*

ftest:
	@Write me

clean:
	@rm -f */version.txt
	@rm -f .coverage
	@rm -fr */__pycache__ */*.pyc __pycache__
	@rm -fr build dist
	@rm -fr lyft-*.dist-info
	@rm -fr lyft.egg-info

install:
	@pip install . -U

all: clean install test black check_code


uninstal:
	@python setup.py install --record files.txt
	@cat files.txt | xargs rm -rf
	@rm -f files.txt

count_lines:
	@find ./ -name '*.py' -exec  wc -l {} \; | sort -n| awk \
        '{printf "%4s %s\n", $$1, $$2}{s+=$$0}END{print s}'
	@echo ''
	@find ./scripts -name '*-*' -exec  wc -l {} \; | sort -n| awk \
		        '{printf "%4s %s\n", $$1, $$2}{s+=$$0}END{print s}'
	@echo ''
	@find ./tests -name '*.py' -exec  wc -l {} \; | sort -n| awk \
        '{printf "%4s %s\n", $$1, $$2}{s+=$$0}END{print s}'
	@echo ''

# ----------------------------------
#      UPLOAD PACKAGE TO PYPI
# ----------------------------------
build:
	@python setup.py sdist bdist_wheel

pypi_test:
	@twine upload -r testpypi dist/* -u lologibus2

pypi:
	@twine upload dist/* -u lologibus2

# ----------------------------------
#           MODELS ON GCP
# ----------------------------------

gcp_train:
	@gcloud ai-platform jobs submit training ${JOB_NAME} \
	--job-dir gs://${BUCKET_NAME}/${BUCKET_TRAINING_FOLDER}  \
	--package-path ${PACKAGE_NAME} \
	--module-name ${PACKAGE_NAME}.${FILENAME} \
	--python-version=${PYTHON_VERSION} \
	--runtime-version=${RUNTIME_VERSION} \
	--region ${REGION} \
	--stream-logs

