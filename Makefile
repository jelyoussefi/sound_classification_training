#----------------------------------------------------------------------------------------------------------------------
# Flags
#----------------------------------------------------------------------------------------------------------------------
SHELL:=/bin/bash
CURRENT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))


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
	@python3 ./train.py

valid:
	@$(call msg, Validating the Audio Classification Model   ...)
	@python3 ./valid.py -m ./mnhn_model.pth

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

