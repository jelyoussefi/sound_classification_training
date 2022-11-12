#----------------------------------------------------------------------------------------------------------------------
# Flags
#----------------------------------------------------------------------------------------------------------------------
SHELL:=/bin/bash
CURRENT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))

DOCKER_IMAGE_NAME=sound_classification_training_image
export DOCKER_BUILDKIT=1
#----------------------------------------------------------------------------------------------------------------------
# Targets
#----------------------------------------------------------------------------------------------------------------------
default: train 
.PHONY:  test

install_prerequisite:
	@$(call msg, Installing Prerequisite  ...)
	@pip3 install torch==1.13.0+cu116 torchvision==0.14.0+cu116 torchaudio==0.13.0+cu116 -f https://download.pytorch.org/whl/cu116/torch_stable.html	

train:
	@$(call msg, Training the Audio Classification Model   ...)
	@rm -rf ./model/*
	@python3 ./train.py -o "./model"

valid:
	@$(call msg, Validating the Audio Classification Model   ...)
	@python3 ./valid.py -m ./model/model.pth


docker-build:
	@$(call msg, Building docker image ${DOCKER_IMAGE_NAME} ...)
	@docker build  -t ${DOCKER_IMAGE_NAME} .


docker-train: docker-build
	@$(call msg, Traing with docker image ${DOCKER_IMAGE_NAME} ...)
	@docker run -it --rm -a stdout -a stderr -v ${CURRENT_DIR}:/workspace -v /data:/data -w /workspace --gpus all ${DOCKER_IMAGE_NAME}  \
		make
#----------------------------------------------------------------------------------------------------------------------
# helper functions
#----------------------------------------------------------------------------------------------------------------------
define msg
	tput setaf 2 && \
	for i in $(shell seq 1 120 ); do echo -n "-"; done; echo  "" && \
	echo "         "$1 && \
	for i in $(shell seq 1 120 ); do echo -n "-"; done; echo "" && \
	tput sgr0
endef

