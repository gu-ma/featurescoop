# CPU image for the API

ARG BASE_IMAGE=python:3.8
FROM $BASE_IMAGE

RUN pip install -U pip
RUN pip install flask flask_restful flask_cors
RUN pip install matplotlib==3.3.2
RUN pip install moviepy==1.0.3
RUN pip install numpy==1.19.3
RUN pip install Pillow==8.1.0
RUN pip install termcolor
RUN pip install tqdm
RUN pip install scenedetect==0.5.4.1
RUN pip install imageio==2.9.0
RUN pip install opencv-python-headless