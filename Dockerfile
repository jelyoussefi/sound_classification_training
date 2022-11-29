FROM nvidia/cuda:11.7.0-cudnn8-runtime-ubuntu20.04
ARG DEBIAN_FRONTEND=noninteractive
RUN apt update -y
RUN apt-get  install -y software-properties-common python3 python3-pip
RUN pip3 install \
	torch==1.13.0+cu117 \
	torchvision==0.14.0+cu117 \
	torchaudio==0.13.0+cu117 -f https://download.pytorch.org/whl/cu117/torch_stable.html	
RUN pip3 install tensorboard==2.10.1 pandas==1.4.4 ipython==8.0.1 matplotlib librosa sox
RUN apt-get install -y libsndfile1-dev
RUN pip3 install torchsummary
RUN pip3 install tqdm 
#RUN pip3 install torch-ort-infer[openvino]
#RUN python3 -m torch_ort.configure
