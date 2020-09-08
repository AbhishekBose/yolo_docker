FROM nvidia/cuda:10.1-cudnn7-devel-ubuntu18.04
MAINTAINER abhishek bose 

RUN apt-get update -y && apt-get install -y python3-pip python3-dev libsm6 libxext6 libxrender-dev


RUN \
	apt-get install -y \
	wget \
	unzip \
	ffmpeg \ 
	git

RUN pip3 install numpy
RUN pip3 install opencv-python
RUN pip3 install requests
RUN pip3 install numba
RUN pip3 install imutils

WORKDIR home/

RUN git clone https://github.com/pjreddie/darknet
WORKDIR darknet/

RUN sed -i 's/GPU=.*/GPU=1/' Makefile 
RUN sed -i 's/CUDNN=.*/CUDNN=1/' Makefile && \
	make


RUN wget https://pjreddie.com/media/files/yolov3.weights -P weights/

WORKDIR /home


